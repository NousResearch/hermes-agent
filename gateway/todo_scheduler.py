"""
Todo Scheduler — usage-aware deferred task queue for Discord.

Tasks submitted via ``/todo`` are queued in a JSON file and processed
by up to 3 worker slots.  Each slot is gated by a Home Assistant
binary sensor:

    binary_sensor.claude_usage_induce_lvl1  →  slot 1
    binary_sensor.claude_usage_induce_lvl2  →  slot 2
    binary_sensor.claude_usage_induce_lvl3  →  slot 3

When a sensor is **on**, the corresponding slot is unlocked and may
pick up the oldest pending task, dispatch it as a real message in the
task's Discord thread, and wait for the agent session to complete
before looping back.

Concurrency notes:

* All workers share a single ``asyncio.Lock`` to serialise queue
  reads/writes — no two workers can claim the same task.
* Workers run as ``asyncio.Task``s inside the gateway event loop.
* A worker only picks a new task if its sensor is still **on**.

Scheduling behaviour:

* The main loop wakes on an ``asyncio.Event`` instead of a fixed sleep.
* ``notify(delay=N)`` sets the event after *N* seconds (default 0).
* After a ``/todo`` enqueue the handler calls ``notify(delay=2)``.
* After a worker finishes a task it calls ``notify()`` immediately.
* HA sensors are polled at most once every ``HA_MIN_INTERVAL`` (300 s)
  unless there are pending tasks — then every tick.
* Maximum sleep between ticks is ``MAX_POLL`` (300 s) to guarantee
  periodic HA checks.

HA status reporting:

* The scheduler reports its operational status back to HA via
  ``input_text.todo_scheduler_status`` — a JSON blob with queue depth,
  active workers, and per-slot state.  Updated every tick that has
  pending or running tasks, plus on task enqueue/complete/fail.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_STATE_DIR = Path(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.claude-code"))
) / "state"
TODO_QUEUE_FILE = _STATE_DIR / "todo_queue.json"

# HA sensor entity IDs (1-indexed to match slot numbers)
HA_SENSORS = {
    1: "binary_sensor.claude_usage_induce_lvl1",
    2: "binary_sensor.claude_usage_induce_lvl2",
    3: "binary_sensor.claude_usage_induce_lvl3",
}

# HA entity for status reporting
HA_STATUS_ENTITY = "input_text.todo_scheduler_status"

# Per-slot worker status entities (on/off/unknown)
HA_WORKER_ENTITIES = {
    1: "binary_sensor.todo_worker_slot_1",
    2: "binary_sensor.todo_worker_slot_2",
    3: "binary_sensor.todo_worker_slot_3",
}


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

class TodoQueue:
    """Async-safe FIFO queue backed by a JSON file.

    Every public method acquires ``_lock`` so concurrent workers never
    see a half-written file or claim the same task.

    Use ``TodoQueue.instance()`` to get the shared singleton so that
    the slash-command handler and the scheduler share the same
    ``asyncio.Lock``.
    """

    _singleton: Optional["TodoQueue"] = None

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        _STATE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def instance(cls) -> "TodoQueue":
        """Return (or create) the module-wide singleton."""
        if cls._singleton is None:
            cls._singleton = cls()
        return cls._singleton

    # -- public API --

    async def add(self, task: dict) -> str:
        """Append *task* to the queue.  Returns the generated task ID."""
        async with self._lock:
            tasks = self._read()
            task_id = uuid.uuid4().hex[:8]
            task.update(
                id=task_id,
                status="pending",
                created_at=time.time(),
            )
            tasks.append(task)
            self._write(tasks)
            return task_id

    async def claim_next(self, worker_id: str) -> Optional[dict]:
        """Atomically claim the oldest *pending* task.  Returns ``None``
        when the queue has nothing to do."""
        async with self._lock:
            tasks = self._read()
            for task in tasks:
                if task["status"] == "pending":
                    task["status"] = "running"
                    task["worker_id"] = worker_id
                    task["started_at"] = time.time()
                    self._write(tasks)
                    return dict(task)  # return a copy
            return None

    async def complete(self, task_id: str) -> None:
        async with self._lock:
            tasks = self._read()
            for task in tasks:
                if task["id"] == task_id:
                    task["status"] = "completed"
                    task["completed_at"] = time.time()
                    break
            # Prune completed tasks older than 24 h
            cutoff = time.time() - 86_400
            tasks = [
                t for t in tasks
                if not (
                    t["status"] == "completed"
                    and t.get("completed_at", 0) < cutoff
                )
            ]
            self._write(tasks)

    async def fail(self, task_id: str, error: str = "") -> None:
        async with self._lock:
            tasks = self._read()
            for task in tasks:
                if task["id"] == task_id:
                    task["status"] = "failed"
                    task["error"] = error
                    task["failed_at"] = time.time()
                    break
            self._write(tasks)

    async def pending_count(self) -> int:
        async with self._lock:
            return sum(1 for t in self._read() if t["status"] == "pending")

    async def snapshot(self) -> dict:
        """Return a summary of queue state for status reporting."""
        async with self._lock:
            tasks = self._read()
            pending = sum(1 for t in tasks if t["status"] == "pending")
            running = sum(1 for t in tasks if t["status"] == "running")
            completed = sum(1 for t in tasks if t["status"] == "completed")
            failed = sum(1 for t in tasks if t["status"] == "failed")
            return {
                "pending": pending,
                "running": running,
                "completed": completed,
                "failed": failed,
                "total": len(tasks),
            }

    # -- internal --

    def _read(self) -> list[dict]:
        try:
            if TODO_QUEUE_FILE.exists():
                return json.loads(TODO_QUEUE_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("[TodoQueue] Read error: %s", exc)
        return []

    def _write(self, tasks: list[dict]) -> None:
        try:
            TODO_QUEUE_FILE.write_text(json.dumps(tasks, indent=2))
        except OSError as exc:
            logger.error("[TodoQueue] Write error: %s", exc)


# ---------------------------------------------------------------------------
# HA sensor helpers
# ---------------------------------------------------------------------------

async def _check_ha_sensor(sensor_id: str, ha_url: str, ha_token: str) -> bool:
    """Return True when a HA binary_sensor is **on**."""
    try:
        import aiohttp
    except ImportError:
        # Fallback: synchronous urllib (always available)
        return _check_ha_sensor_sync(sensor_id, ha_url, ha_token)

    url = f"{ha_url}/api/states/{sensor_id}"
    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("state") == "on"
                logger.debug("[TodoScheduler] HA sensor %s HTTP %s", sensor_id, resp.status)
    except Exception as exc:
        logger.warning("[TodoScheduler] HA sensor check failed (%s): %s", sensor_id, exc)
    return False


def _check_ha_sensor_sync(sensor_id: str, ha_url: str, ha_token: str) -> bool:
    """Blocking fallback when aiohttp is unavailable."""
    import urllib.request, urllib.error
    url = f"{ha_url}/api/states/{sensor_id}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("state") == "on"
    except Exception as exc:
        logger.warning("[TodoScheduler] HA sensor sync check failed (%s): %s", sensor_id, exc)
    return False


async def _get_unlocked_slots(ha_url: str, ha_token: str) -> set[int]:
    """Return the set of slot numbers whose sensor is currently **on**."""
    results = await asyncio.gather(
        *[
            _check_ha_sensor(sensor, ha_url, ha_token)
            for sensor in HA_SENSORS.values()
        ],
        return_exceptions=True,
    )
    unlocked: set[int] = set()
    for slot_id, result in zip(HA_SENSORS.keys(), results):
        if result is True:
            unlocked.add(slot_id)
    return unlocked


async def _set_ha_state(
    entity_id: str, state: str, ha_url: str, ha_token: str,
    attributes: dict | None = None,
) -> bool:
    """Set a HA entity state.  Returns True on success."""
    url = f"{ha_url}/api/states/{entity_id}"
    headers = {
        "Authorization": f"Bearer {ha_token}",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {"state": state}
    if attributes:
        payload["attributes"] = attributes

    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, headers=headers, json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                return resp.status in (200, 201)
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("[TodoScheduler] HA state set failed (%s): %s", entity_id, exc)
    # sync fallback
    try:
        import urllib.request
        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status in (200, 201)
    except Exception as exc:
        logger.debug("[TodoScheduler] HA state set sync failed (%s): %s", entity_id, exc)
    return False


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class TodoWorker:
    """A single numbered worker slot.  Picks up one task at a time."""

    def __init__(self, slot_id: int, queue: TodoQueue, runner: Any, scheduler: Any = None) -> None:
        self.slot_id = slot_id
        self.queue = queue
        self.runner = runner  # GatewayRunner
        self.scheduler = scheduler  # TodoScheduler (for HA status reports)
        self._current_task_id: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        self._in_error: bool = False  # True after a failed task

    @property
    def is_busy(self) -> bool:
        return self._current_task_id is not None

    @property
    def ha_state(self) -> str:
        """Return HA-style state: on (busy), off (idle), unknown (error)."""
        if self._in_error:
            return "unknown"
        return "on" if self.is_busy else "off"

    async def _report_ha_status(self, event: str) -> None:
        """Delegate to scheduler's _report_status if available."""
        if self.scheduler:
            try:
                await self.scheduler._report_status(event=event)
            except Exception:
                pass

    async def run_once(self, ha_url: str, ha_token: str) -> bool:
        """Try to pick up and fully process one task.

        Returns True if a task was dispatched (regardless of success).
        """
        # Re-check sensor right before claiming work
        sensor = HA_SENSORS[self.slot_id]
        if not await _check_ha_sensor(sensor, ha_url, ha_token):
            return False

        task = await self.queue.claim_next(f"slot-{self.slot_id}")
        if task is None:
            return False

        self._current_task_id = task["id"]
        self._in_error = False  # clear error on new task pickup
        logger.info(
            "[TodoScheduler] Slot %d claimed task %s: %.80s",
            self.slot_id, task["id"], task.get("prompt", ""),
        )

        # Report: agent thread starting
        await self._report_ha_status("agent_thread_start")

        try:
            if task.get("no_agent") and task.get("script"):
                # no_agent mode: run script directly, zero inference cost
                await self._run_script_task(task)
            else:
                await self._dispatch_and_wait(task)
            await self.queue.complete(task["id"])
            self._in_error = False
            logger.info("[TodoScheduler] Slot %d completed task %s", self.slot_id, task["id"])
            # Report: agent thread ended successfully
            await self._report_ha_status("agent_thread_end")
        except asyncio.CancelledError:
            # Gateway shutting down — mark as failed so it can be
            # retried on restart.
            await self.queue.fail(task["id"], "cancelled (gateway shutdown)")
            raise
        except Exception as exc:
            logger.error(
                "[TodoScheduler] Slot %d failed task %s: %s",
                self.slot_id, task["id"], exc, exc_info=True,
            )
            await self.queue.fail(task["id"], str(exc)[:300])
            self._in_error = True
            # Report: agent thread ended with error
            await self._report_ha_status("agent_thread_error")
            # Notify in thread
            try:
                adapter = self._get_discord_adapter()
                if adapter:
                    await adapter.send(
                        chat_id=task["thread_id"],
                        content=f"❌ Task failed: {str(exc)[:200]}",
                        metadata={"thread_id": task["thread_id"]},
                    )
            except Exception:
                pass
        finally:
            self._current_task_id = None

        return True

    # -- internals --

    def _get_discord_adapter(self) -> Any:
        from gateway.platforms.base import Platform
        return self.runner.adapters.get(Platform.DISCORD)

    async def _run_script_task(self, task: dict) -> None:
        """Run a no_agent script task and deliver stdout to Discord.

        - Non-empty stdout → post to deliver_channel_id (or thread)
        - Empty stdout → silent (watchdog pattern)
        - Non-zero exit → raise so run_once marks it failed
        - Timeout (5 min) → raise
        """
        script_path = os.path.expanduser(task["script"])
        # script_path may contain args (e.g. "/path/script.py --dry-run")
        import shlex as _shlex
        _script_parts = _shlex.split(script_path)
        _actual_file = _script_parts[0]
        if not os.path.isfile(_actual_file):
            raise FileNotFoundError(f"Script not found: {_actual_file}")

        # Determine interpreter
        import shlex
        script_parts = shlex.split(script_path)
        actual_script = script_parts[0]
        script_args = script_parts[1:]

        if actual_script.endswith((".sh", ".bash")):
            cmd = ["bash", actual_script] + script_args
        else:
            cmd = ["python3", "-u", actual_script] + script_args

        timeout = 300  # 5 min default
        thread_id = task.get("thread_id", "")

        logger.info(
            "[TodoScheduler] Slot %d running script: %s (timeout=%ds)",
            self.slot_id, script_path, timeout,
        )

        # Notify in thread that script is running
        adapter = self._get_discord_adapter()
        if adapter and thread_id:
            await adapter.send(
                chat_id=thread_id,
                content=f"⚡ Running script (slot {self.slot_id})…",
                metadata={"thread_id": thread_id},
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "TODO_TASK_ID": task.get("id", "")},
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise RuntimeError(f"Script timed out after {timeout}s: {script_path}")

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            error_msg = stderr_text[:500] or f"Exit code {proc.returncode}"
            raise RuntimeError(f"Script exited {proc.returncode}: {error_msg}")

        if stderr_text:
            logger.debug(
                "[TodoScheduler] Script %s stderr: %s",
                task["id"], stderr_text[:200],
            )

        if not stdout_text:
            # Empty stdout → silent (watchdog found nothing to report)
            logger.debug(
                "[TodoScheduler] Script %s produced no output — silent",
                task["id"],
            )
            if adapter and thread_id:
                await adapter.send(
                    chat_id=thread_id,
                    content="✅ Script completed (no output).",
                    metadata={"thread_id": thread_id},
                )
            return

        # Deliver stdout
        deliver_channel = task.get("deliver_channel_id")
        if deliver_channel:
            await self._deliver_to_channel(deliver_channel, stdout_text, task)

        # Also post result in the task thread
        if adapter and thread_id:
            # Truncate for thread message
            result_text = stdout_text[:1900] if len(stdout_text) > 1900 else stdout_text
            await adapter.send(
                chat_id=thread_id,
                content=f"✅ Script completed.\n```\n{result_text}\n```",
                metadata={"thread_id": thread_id},
            )

    async def _deliver_to_channel(
        self, channel_id: str, content: str, task: dict,
    ) -> None:
        """Post content to a Discord channel via bot HTTP API."""
        adapter = self._get_discord_adapter()
        if not adapter:
            logger.warning(
                "[TodoScheduler] No Discord adapter — cannot deliver script output",
            )
            return

        bot_token = (
            getattr(getattr(adapter, "config", None), "token", None)
            or os.environ.get("DISCORD_BOT_TOKEN", "")
        )
        if not bot_token:
            logger.error("[TodoScheduler] No Discord bot token for delivery")
            return

        import aiohttp

        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        headers = {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json",
        }

        # Truncate to Discord's 2000-char limit
        if len(content) > 1950:
            content = content[:1950] + "\n…(truncated)"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json={"content": content}, headers=headers,
                ) as resp:
                    if resp.status not in (200, 201):
                        body = await resp.text()
                        logger.error(
                            "[TodoScheduler] Delivery failed HTTP %d: %s",
                            resp.status, body[:300],
                        )
                    else:
                        logger.info(
                            "[TodoScheduler] Delivered script output to channel %s (%d chars)",
                            channel_id, len(content),
                        )
        except Exception as exc:
            logger.error(
                "[TodoScheduler] Delivery error: %s", exc, exc_info=True,
            )

    async def _dispatch_and_wait(self, task: dict) -> None:
        """Inject the task prompt into its Discord thread and wait for completion."""
        from gateway.platforms.base import MessageEvent, MessageType
        from gateway.session import build_session_key

        adapter = self._get_discord_adapter()
        if adapter is None:
            raise RuntimeError("Discord adapter not available")

        thread_id = task["thread_id"]

        # Notify thread that work is starting
        await adapter.send(
            chat_id=thread_id,
            content=f"🚀 Picked up by worker slot {self.slot_id}. Processing now…",
            metadata={"thread_id": thread_id},
        )

        # Build the MessageEvent to inject
        source = adapter.build_source(
            chat_id=thread_id,
            chat_name=task.get("chat_name", ""),
            chat_type="thread",
            user_id=task.get("user_id", ""),
            user_name=task.get("user_name", ""),
            thread_id=thread_id,
            guild_id=task.get("guild_id"),
            parent_chat_id=task.get("parent_channel_id"),
        )

        parent_id = task.get("parent_channel_id", "")
        _skills = (
            adapter._resolve_channel_skills(thread_id, parent_id or None)
            if hasattr(adapter, "_resolve_channel_skills")
            else None
        )
        _channel_prompt = (
            adapter._resolve_channel_prompt(thread_id, parent_id or None)
            if hasattr(adapter, "_resolve_channel_prompt")
            else None
        )

        event = MessageEvent(
            text=task["prompt"],
            message_type=MessageType.TEXT,
            source=source,
            auto_skill=_skills,
            channel_prompt=_channel_prompt,
        )

        # Compute session key for completion tracking
        session_key = build_session_key(
            source,
            group_sessions_per_user=adapter.config.extra.get(
                "group_sessions_per_user", True
            ),
            thread_sessions_per_user=adapter.config.extra.get(
                "thread_sessions_per_user", False
            ),
        )

        # Dispatch — this spawns a background task and returns immediately
        await adapter.handle_message(event)

        # Wait for the agent session to finish
        await self._wait_for_session(session_key, adapter, timeout=1800)

    async def _wait_for_session(
        self,
        session_key: str,
        adapter: Any,
        timeout: int = 1800,
    ) -> None:
        """Poll until the session is no longer tracked as active."""
        deadline = time.time() + timeout
        await asyncio.sleep(2)  # let the session register

        while time.time() < deadline:
            adapter_active = session_key in adapter._active_sessions
            runner_active = session_key in self.runner._running_agents

            if not adapter_active and not runner_active:
                return  # done

            await asyncio.sleep(3)

        logger.warning(
            "[TodoScheduler] Slot %d timed out (%ds) waiting for session %s",
            self.slot_id, timeout, session_key,
        )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class TodoScheduler:
    """Top-level coordinator.  Spawned as a single ``asyncio.Task`` by
    ``GatewayRunner.start()``.

    The main loop sleeps on an ``asyncio.Event`` so it can be woken
    immediately by external callers (task enqueue, worker completion)
    without busy-polling.  The maximum sleep is ``MAX_POLL`` seconds
    to guarantee periodic HA sensor checks.
    """

    MAX_POLL = 300          # max seconds between ticks (5 min HA guarantee)
    HA_MIN_INTERVAL = 300   # minimum seconds between HA sensor polls when idle
    _instance: Optional["TodoScheduler"] = None

    def __init__(self, runner: Any) -> None:
        TodoScheduler._instance = self
        self.runner = runner
        self.queue = TodoQueue.instance()
        self.workers: dict[int, TodoWorker] = {
            i: TodoWorker(i, self.queue, runner, scheduler=self) for i in range(1, 4)
        }
        self._ha_url = ""
        self._ha_token = ""
        self._running = False
        self._worker_tasks: dict[int, asyncio.Task] = {}
        # Event-driven wake
        self._wake: asyncio.Event = asyncio.Event()
        self._wake_handle: Optional[asyncio.TimerHandle] = None
        # HA sensor cache
        self._cached_slots: set[int] = set()
        self._last_ha_check: float = 0.0

    # -- external API --

    def notify(self, delay: float = 0) -> None:
        """Wake the scheduler loop after *delay* seconds.

        Called by:
        - ``/todo`` handler (delay=2 — let Discord thread settle)
        - Worker completion (delay=0 — immediate re-check)
        """
        if delay <= 0:
            self._wake.set()
            return
        # Schedule a delayed wake; cancel any earlier pending one
        loop = asyncio.get_event_loop()
        if self._wake_handle is not None:
            self._wake_handle.cancel()
        self._wake_handle = loop.call_later(delay, self._wake.set)

    # -- config --

    def _load_ha_config(self) -> None:
        for env_path in (
            Path.home() / ".claude-code" / ".env",
            Path.home() / ".hermes" / ".env",
        ):
            if not env_path.exists():
                continue
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip("\"'")
                if key == "HASS_TOKEN" and not self._ha_token:
                    self._ha_token = val
                elif key == "HASS_URL" and not self._ha_url:
                    self._ha_url = val

        self._ha_url = self._ha_url or os.environ.get(
            "HASS_URL", "http://172.24.127.105:8123"
        )
        self._ha_token = self._ha_token or os.environ.get("HASS_TOKEN", "")

    # -- HA status reporting --

    async def _report_status(self, event: str = "tick") -> None:
        """Push current scheduler status to HA as input_text entity."""
        if not self._ha_token:
            return
        try:
            snap = await self.queue.snapshot()
            active_slots = []
            for sid, worker in self.workers.items():
                if worker.is_busy:
                    active_slots.append(sid)

            status = {
                "queue_pending": snap["pending"],
                "queue_running": snap["running"],
                "queue_completed": snap["completed"],
                "queue_failed": snap["failed"],
                "active_workers": len(active_slots),
                "active_slots": active_slots,
                "unlocked_slots": sorted(self._cached_slots),
                "max_workers": 3,
                "event": event,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            state_str = json.dumps(status, separators=(",", ":"))
            # HA input_text max 255 chars — truncate if needed
            if len(state_str) > 255:
                state_str = state_str[:255]
            await _set_ha_state(
                HA_STATUS_ENTITY, state_str, self._ha_url, self._ha_token,
                attributes={
                    "friendly_name": "Todo Scheduler Status",
                    "queue_pending": snap["pending"],
                    "active_workers": len(active_slots),
                    "unlocked_slots": sorted(self._cached_slots),
                },
            )

            # Per-slot worker status (on/off/unknown)
            for sid, entity_id in HA_WORKER_ENTITIES.items():
                worker = self.workers[sid]
                attrs = {
                    "friendly_name": f"Todo Worker Slot {sid}",
                    "task_id": worker._current_task_id or "",
                    "slot_id": sid,
                    "sensor_unlocked": sid in self._cached_slots,
                }
                await _set_ha_state(
                    entity_id, worker.ha_state,
                    self._ha_url, self._ha_token,
                    attributes=attrs,
                )
        except Exception as exc:
            logger.debug("[TodoScheduler] Status report failed: %s", exc)

    # -- main loop --

    async def run(self) -> None:
        """Entry point — runs until ``stop()`` or cancellation."""
        self._load_ha_config()
        if not self._ha_token:
            logger.warning("[TodoScheduler] No HASS_TOKEN — scheduler disabled")
            return

        self._running = True
        logger.info("[TodoScheduler] Started (HA: %s)", self._ha_url)

        try:
            while self._running:
                try:
                    await self._tick()
                except Exception as exc:
                    logger.error("[TodoScheduler] Tick error: %s", exc, exc_info=True)

                # Wait for wake event OR timeout (guarantees periodic HA check)
                self._wake.clear()
                try:
                    await asyncio.wait_for(
                        self._wake.wait(), timeout=self.MAX_POLL
                    )
                    logger.debug("[TodoScheduler] Woken by event")
                except asyncio.TimeoutError:
                    logger.debug("[TodoScheduler] Woken by MAX_POLL timeout")
        except asyncio.CancelledError:
            logger.info("[TodoScheduler] Shutting down")
            self._running = False
            # Cancel any running worker tasks
            for task in self._worker_tasks.values():
                if not task.done():
                    task.cancel()
            # Final status report
            await self._report_status(event="shutdown")
            raise

    async def _tick(self) -> None:
        """One scheduler tick: check sensors, launch idle workers, report status."""
        pending = await self.queue.pending_count()

        # Decide whether to refresh HA sensors
        now = time.time()
        elapsed = now - self._last_ha_check
        need_ha_refresh = (
            elapsed >= self.HA_MIN_INTERVAL  # 5 min guarantee
            or pending > 0                    # always refresh when tasks waiting
            or self._last_ha_check == 0       # first tick
        )

        if need_ha_refresh:
            self._cached_slots = await _get_unlocked_slots(self._ha_url, self._ha_token)
            self._last_ha_check = now
            logger.debug(
                "[TodoScheduler] HA refresh: slots=%s, pending=%d",
                self._cached_slots, pending,
            )

        # Report status if there's activity
        has_activity = pending > 0 or any(w.is_busy for w in self.workers.values())
        if has_activity or need_ha_refresh:
            await self._report_status(event="tick")

        if not pending:
            return  # nothing to do

        if not self._cached_slots:
            return  # all sensors off

        # Clean up finished worker tasks
        any_finished = False
        for slot_id in list(self._worker_tasks):
            task = self._worker_tasks[slot_id]
            if task.done():
                # Harvest exceptions so they don't go unhandled
                try:
                    task.result()
                except Exception:
                    pass
                del self._worker_tasks[slot_id]
                any_finished = True

        if any_finished:
            await self._report_status(event="worker_done")

        for slot_id in sorted(self._cached_slots):
            worker = self.workers[slot_id]

            # Skip if this worker already has a running task
            if slot_id in self._worker_tasks and not self._worker_tasks[slot_id].done():
                continue

            if worker.is_busy:
                continue

            # Spawn worker — it will claim a task, process it fully, then return
            coro = self._run_worker(worker)
            self._worker_tasks[slot_id] = asyncio.create_task(coro)
            logger.debug("[TodoScheduler] Launched slot %d worker task", slot_id)

    async def _run_worker(self, worker: TodoWorker) -> None:
        """Wrapper that calls ``run_once`` then immediately wakes the scheduler
        so the next pending task is picked up without waiting for MAX_POLL."""
        try:
            dispatched = await worker.run_once(self._ha_url, self._ha_token)
            if dispatched:
                await self._report_status(event="task_complete")
                # Immediately wake to pick up next task
                self.notify(delay=0)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Errors already logged inside run_once
            self.notify(delay=0)

    def stop(self) -> None:
        self._running = False
        self._wake.set()  # unblock the wait
