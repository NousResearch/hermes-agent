"""
Recurring Jobs — automatic repeated task injection into the /todo queue.

Jobs are defined in ``~/.claude-code/state/recurring_jobs.json`` and
checked on a configurable tick (default 60 s).  Each job has:

* A **schedule** (minimum interval between enqueues)
* A **security scope** — which services/data the worker may touch
* **Additional instructions** and **bail-out / expected-response** rules
* An **error notification thread** for bail-out alerts

When a job is due, the recurring scheduler:
1. Creates a Discord thread (or reuses a template)
2. Enqueues the task into the /todo queue
3. Records ``last_enqueued_at`` so the cooldown is respected

Error handling:
* Workers are instructed to return a JSON error code on bail-out
* The recurring scheduler monitors completed tasks for error codes
* Errors are logged to ``~/.claude-code/logs/recurring_errors.log``
* Notifications are sent to the configured Discord thread
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERMES_HOME = Path(
    os.environ.get("HERMES_HOME", os.path.expanduser("~/.claude-code"))
)
_STATE_DIR = _HERMES_HOME / "state"
_LOGS_DIR = _HERMES_HOME / "logs"

RECURRING_JOBS_FILE = _STATE_DIR / "recurring_jobs.json"
RECURRING_ERRORS_LOG = _LOGS_DIR / "recurring_errors.log"

# Default check interval (seconds)
CHECK_INTERVAL = 60

# ---------------------------------------------------------------------------
# Schedule parsing
# ---------------------------------------------------------------------------

SCHEDULE_ALIASES: dict[str, int] = {
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
    "daily": 86400,
    "weekly": 604800,
    "hourly": 3600,
}


def parse_schedule(schedule: str) -> int:
    """Convert a schedule string to seconds.

    Accepts aliases ('15m', 'daily', etc.) or raw seconds as a string.
    """
    schedule = schedule.strip().lower()
    if schedule in SCHEDULE_ALIASES:
        return SCHEDULE_ALIASES[schedule]
    # Try "every Xm/h/d/w" format
    if schedule.startswith("every "):
        schedule = schedule[6:].strip()
    # Numeric with suffix
    for suffix, mult in [("m", 60), ("h", 3600), ("d", 86400), ("w", 604800)]:
        if schedule.endswith(suffix):
            try:
                return int(schedule[:-1]) * mult
            except ValueError:
                pass
    # Raw seconds
    try:
        return int(schedule)
    except ValueError:
        raise ValueError(f"Cannot parse schedule: {schedule!r}")


# ---------------------------------------------------------------------------
# Security scope template
# ---------------------------------------------------------------------------

DEFAULT_SECURITY_SCOPE = {
    "allowed_services": ["read-only"],
    "allowed_data": ["read-only"],
    "forbidden_actions": [
        "delete files outside working directory",
        "modify system configuration",
        "install or uninstall packages",
        "send messages to channels other than the task thread",
        "create or delete Discord channels/threads",
        "modify Home Assistant automations or scripts",
        "access credentials or tokens directly",
    ],
}


def build_worker_prompt(job: dict) -> str:
    """Build the full prompt injected into the /todo task.

    Includes the user's instructions, security scope, expected response
    format, and bail-out instructions.
    """
    scope = job.get("security_scope", DEFAULT_SECURITY_SCOPE)
    instructions = job.get("instructions", "")
    expected_response = job.get("expected_response", "")
    bail_out = job.get("bail_out_instructions", "")

    parts: list[str] = []

    # Main task
    parts.append(f"## Recurring Task: {job.get('name', 'unnamed')}\n")
    if instructions:
        parts.append(f"### Instructions\n{instructions}\n")

    # Security scope
    parts.append("### Security Scope (MANDATORY — do NOT exceed these boundaries)")
    allowed_svc = scope.get("allowed_services", ["read-only"])
    allowed_data = scope.get("allowed_data", ["read-only"])
    forbidden = scope.get("forbidden_actions", [])
    parts.append(f"**Allowed services:** {', '.join(allowed_svc)}")
    parts.append(f"**Allowed data access:** {', '.join(allowed_data)}")
    if forbidden:
        parts.append("**Forbidden actions:**")
        for action in forbidden:
            parts.append(f"  - {action}")
    parts.append("")

    # Expected response
    if expected_response:
        parts.append(f"### Expected Response\n{expected_response}\n")

    # Bail-out instructions
    parts.append("### Bail-Out Protocol")
    if bail_out:
        parts.append(bail_out)
    else:
        parts.append(
            "If you encounter ANY of the following, you MUST bail out immediately:"
        )
        parts.append("- An unexpected state that doesn't match expected conditions")
        parts.append("- A permission error or access denial")
        parts.append("- Data that looks corrupted or inconsistent")
        parts.append("- Any action that would exceed the security scope above")
    parts.append("")
    parts.append(
        "**To bail out**, end your response with a JSON error block on its own line:"
    )
    parts.append("```json")
    parts.append(
        '{"recurring_error": true, "code": "ERROR_CODE", '
        '"message": "Human-readable description", '
        '"job_id": "' + job.get("id", "unknown") + '", '
        '"severity": "warning|error|critical"}'
    )
    parts.append("```")
    parts.append("")
    parts.append("Error codes:")
    parts.append("- `SCOPE_VIOLATION` — would exceed security scope")
    parts.append("- `UNEXPECTED_STATE` — conditions don't match expectations")
    parts.append("- `PERMISSION_DENIED` — access denied to required resource")
    parts.append("- `DATA_CORRUPTION` — data looks wrong or inconsistent")
    parts.append("- `SERVICE_UNAVAILABLE` — required service is down")
    parts.append("- `TIMEOUT` — operation took too long")
    parts.append("- `UNKNOWN_ERROR` — catch-all for unclassified failures")
    parts.append("")
    parts.append(
        "**Do NOT attempt to fix errors yourself.** Bail out and let the "
        "error notification system alert the owner."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Error parser + logger
# ---------------------------------------------------------------------------

def parse_error_from_response(response_text: str) -> Optional[dict]:
    """Extract a recurring_error JSON block from agent response text.

    Returns the parsed dict if found, else None.
    """
    if "recurring_error" not in response_text:
        return None
    # Search for JSON blocks
    import re
    # Try fenced code blocks first
    for match in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL):
        try:
            data = json.loads(match.group(1).strip())
            if isinstance(data, dict) and data.get("recurring_error"):
                return data
        except (json.JSONDecodeError, TypeError):
            continue
    # Try bare JSON on a line
    for line in response_text.splitlines():
        line = line.strip()
        if line.startswith("{") and "recurring_error" in line:
            try:
                data = json.loads(line)
                if isinstance(data, dict) and data.get("recurring_error"):
                    return data
            except (json.JSONDecodeError, TypeError):
                continue
    return None


def log_error(job: dict, error: dict) -> None:
    """Append an error entry to the recurring errors log."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "job_id": job.get("id", "unknown"),
        "job_name": job.get("name", "unnamed"),
        "error_code": error.get("code", "UNKNOWN_ERROR"),
        "severity": error.get("severity", "error"),
        "message": error.get("message", ""),
    }
    with open(RECURRING_ERRORS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    logger.warning(
        "[RecurringJobs] Error from job %s: %s — %s",
        job.get("id"), error.get("code"), error.get("message"),
    )


# ---------------------------------------------------------------------------
# Job storage
# ---------------------------------------------------------------------------

def _read_jobs() -> list[dict]:
    """Read recurring jobs from the JSON file."""
    try:
        if RECURRING_JOBS_FILE.exists():
            return json.loads(RECURRING_JOBS_FILE.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("[RecurringJobs] Read error: %s", exc)
    return []


def _write_jobs(jobs: list[dict]) -> None:
    """Write recurring jobs to the JSON file."""
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        RECURRING_JOBS_FILE.write_text(json.dumps(jobs, indent=2))
    except OSError as exc:
        logger.error("[RecurringJobs] Write error: %s", exc)


def add_job(job: dict) -> str:
    """Add a new recurring job. Returns the generated job ID."""
    jobs = _read_jobs()
    job_id = uuid.uuid4().hex[:8]
    job["id"] = job_id
    job.setdefault("enabled", True)
    job.setdefault("created_at", time.time())
    job.setdefault("last_enqueued_at", 0)
    job.setdefault("enqueue_count", 0)
    job.setdefault("error_count", 0)
    job.setdefault("security_scope", DEFAULT_SECURITY_SCOPE)
    jobs.append(job)
    _write_jobs(jobs)
    return job_id


def remove_job(job_id: str) -> bool:
    """Remove a recurring job by ID. Returns True if found and removed."""
    jobs = _read_jobs()
    before = len(jobs)
    jobs = [j for j in jobs if j["id"] != job_id]
    if len(jobs) < before:
        _write_jobs(jobs)
        return True
    return False


def update_job(job_id: str, updates: dict) -> bool:
    """Update fields on a recurring job. Returns True if found."""
    jobs = _read_jobs()
    for job in jobs:
        if job["id"] == job_id:
            job.update(updates)
            _write_jobs(jobs)
            return True
    return False


def get_job(job_id: str) -> Optional[dict]:
    """Get a recurring job by ID."""
    for job in _read_jobs():
        if job["id"] == job_id:
            return dict(job)
    return None


def list_jobs() -> list[dict]:
    """List all recurring jobs."""
    return _read_jobs()


# ---------------------------------------------------------------------------
# Recurring Scheduler
# ---------------------------------------------------------------------------

class RecurringScheduler:
    """Background task that checks recurring jobs and enqueues them.

    Runs alongside the TodoScheduler as another asyncio task.
    """

    _instance: Optional["RecurringScheduler"] = None

    def __init__(self, adapter: Any = None) -> None:
        self.runner = adapter  # GatewayRunner — access Discord via .adapters
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        RecurringScheduler._instance = self

    def _get_discord_adapter(self) -> Any:
        """Get the Discord adapter from the GatewayRunner."""
        if not self.runner or not hasattr(self.runner, "adapters"):
            return None
        try:
            from gateway.config import Platform
            return self.runner.adapters.get(Platform.DISCORD)
        except ImportError:
            return None

    async def start(self) -> None:
        """Start the recurring scheduler background task."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run_loop(), name="recurring-scheduler")
        logger.info("[RecurringJobs] Scheduler started")

    async def stop(self) -> None:
        """Stop the recurring scheduler."""
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("[RecurringJobs] Scheduler stopped")

    async def _run_loop(self) -> None:
        """Main loop — check jobs every CHECK_INTERVAL seconds."""
        while not self._stop_event.is_set():
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("[RecurringJobs] Tick error: %s", exc)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=CHECK_INTERVAL
                )
                break  # stop_event was set
            except asyncio.TimeoutError:
                pass  # normal — just loop again

    async def _tick(self) -> None:
        """Check all recurring jobs and enqueue any that are due."""
        jobs = _read_jobs()
        now = time.time()
        dirty = False

        for job in jobs:
            if not job.get("enabled", True):
                continue

            try:
                interval = parse_schedule(job.get("schedule", "1d"))
            except ValueError:
                logger.warning(
                    "[RecurringJobs] Bad schedule for job %s: %s",
                    job.get("id"), job.get("schedule"),
                )
                continue

            last = job.get("last_enqueued_at", 0)
            if now - last < interval:
                continue  # Not due yet

            # no_agent jobs don't use the /todo queue, skip active-task check
            if not job.get("no_agent"):
                # Check if a task from this job is already pending or running
                if await self._has_active_task(job["id"]):
                    logger.debug(
                        "[RecurringJobs] Job %s still has an active task, skipping",
                        job["id"],
                    )
                    continue

            # Due — run or enqueue
            if job.get("no_agent") and job.get("script"):
                # no_agent mode: run the script directly, zero inference cost
                logger.info(
                    "[RecurringJobs] Job %s (%s) is due — running script (no_agent)",
                    job["id"], job.get("name", "unnamed"),
                )
                try:
                    await self._run_script_job(job)
                    job["last_enqueued_at"] = now
                    job["enqueue_count"] = job.get("enqueue_count", 0) + 1
                    dirty = True
                except Exception as exc:
                    logger.exception(
                        "[RecurringJobs] Script job %s failed: %s",
                        job["id"], exc,
                    )
                    job["error_count"] = job.get("error_count", 0) + 1
                    dirty = True
            else:
                # Standard mode: enqueue into /todo for agent processing
                logger.info(
                    "[RecurringJobs] Job %s (%s) is due — enqueuing",
                    job["id"], job.get("name", "unnamed"),
                )
                try:
                    await self._enqueue_job(job)
                    job["last_enqueued_at"] = now
                    job["enqueue_count"] = job.get("enqueue_count", 0) + 1
                    dirty = True
                except Exception as exc:
                    logger.exception(
                        "[RecurringJobs] Failed to enqueue job %s: %s",
                        job["id"], exc,
                    )

        if dirty:
            _write_jobs(jobs)

        # Check completed tasks for error codes
        await self._check_for_errors()

    async def _has_active_task(self, job_id: str) -> bool:
        """Check if this recurring job already has a pending/running task."""
        from gateway.todo_scheduler import TodoQueue
        queue = TodoQueue.instance()
        async with queue._lock:
            tasks = queue._read()
            return any(
                t.get("recurring_job_id") == job_id
                and t["status"] in ("pending", "running")
                for t in tasks
            )

    # ------------------------------------------------------------------
    # no_agent script execution
    # ------------------------------------------------------------------

    async def _run_script_job(self, job: dict) -> None:
        """Run a no_agent script job and deliver stdout to Discord.

        - Non-empty stdout → post to deliver_channel_id
        - Empty stdout → silent (watchdog pattern)
        - Non-zero exit → error notification to notify_thread_id
        - Timeout (5 min) → error notification
        """
        script_path = os.path.expanduser(job["script"])
        if not os.path.isfile(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Determine interpreter
        if script_path.endswith((".sh", ".bash")):
            cmd = ["bash", script_path]
        else:
            cmd = ["python3", "-u", script_path]

        timeout = job.get("script_timeout", 300)  # 5 min default

        logger.info(
            "[RecurringJobs] Running script: %s (timeout=%ds)",
            script_path, timeout,
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "RECURRING_JOB_ID": job.get("id", "")},
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            error_info = {
                "code": "TIMEOUT",
                "message": f"Script timed out after {timeout}s: {script_path}",
                "severity": "error",
            }
            log_error(job, error_info)
            await self._notify_error(job, error_info, {"id": "script", "thread_id": ""})
            return

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            # Non-zero exit → error
            error_msg = stderr_text[:500] or f"Exit code {proc.returncode}"
            error_info = {
                "code": "SCRIPT_FAILED",
                "message": f"Script exited {proc.returncode}: {error_msg}",
                "severity": "error",
            }
            log_error(job, error_info)
            await self._notify_error(job, error_info, {"id": "script", "thread_id": ""})
            return

        if stderr_text:
            logger.debug(
                "[RecurringJobs] Script %s stderr: %s",
                job["id"], stderr_text[:200],
            )

        if not stdout_text:
            # Empty stdout → silent (watchdog found nothing to report)
            logger.debug(
                "[RecurringJobs] Script %s produced no output — silent",
                job["id"],
            )
            return

        # Deliver stdout to the configured channel
        deliver_channel = job.get("deliver_channel_id")
        if deliver_channel:
            await self._deliver_to_channel(deliver_channel, stdout_text, job)
        else:
            logger.info(
                "[RecurringJobs] Script %s output (no deliver channel): %s",
                job["id"], stdout_text[:200],
            )

    async def _deliver_to_channel(
        self, channel_id: str, content: str, job: dict,
    ) -> None:
        """Post content to a Discord channel via bot HTTP API."""
        adapter = self._get_discord_adapter()
        if not adapter:
            logger.warning(
                "[RecurringJobs] No Discord adapter — cannot deliver script output",
            )
            return

        bot_token = (
            getattr(getattr(adapter, "config", None), "token", None)
            or os.environ.get("DISCORD_BOT_TOKEN", "")
        )
        if not bot_token:
            logger.error("[RecurringJobs] No Discord bot token for delivery")
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
                            "[RecurringJobs] Delivery failed HTTP %d: %s",
                            resp.status, body[:300],
                        )
        except Exception as exc:
            logger.exception(
                "[RecurringJobs] Delivery error for job %s: %s",
                job.get("id"), exc,
            )

    async def _enqueue_job(self, job: dict) -> None:
        """Create a Discord thread and enqueue the task."""
        prompt = build_worker_prompt(job)

        # Create Discord thread if we have an adapter
        thread_id = None
        adapter = self._get_discord_adapter()
        if adapter:
            thread_name = f"🔄 {job.get('name', 'Recurring Task')}"[:100]
            try:
                thread_id = await self._create_thread(thread_name, prompt, job, adapter)
            except Exception as exc:
                logger.error(
                    "[RecurringJobs] Thread creation failed for job %s: %s",
                    job["id"], exc,
                )
                raise

        if not thread_id:
            logger.error(
                "[RecurringJobs] No thread_id for job %s — cannot enqueue",
                job["id"],
            )
            return

        # Enqueue into the /todo queue
        from gateway.todo_scheduler import TodoQueue, TodoScheduler

        queue = TodoQueue.instance()
        task_id = await queue.add({
            "prompt": prompt,
            "thread_id": str(thread_id),
            "parent_channel_id": job.get(
                "parent_channel_id", "1501081034832351302"
            ),
            "guild_id": job.get("guild_id", "793984343667900468"),
            "user_id": job.get("user_id", "234311476607844353"),
            "user_name": job.get("user_name", "bradcarnage"),
            "guild_name": job.get(
                "guild_name", "Brad's Computer Stuff & Things"
            ),
            "chat_name": f"Brad's Computer Stuff & Things / {job.get('name', 'Recurring')}",
            "recurring_job_id": job["id"],
        })

        logger.info(
            "[RecurringJobs] Enqueued task %s for job %s in thread %s",
            task_id, job["id"], thread_id,
        )

        # Wake the todo scheduler
        if TodoScheduler._instance is not None:
            TodoScheduler._instance.notify(delay=2)

    async def _create_thread(
        self, name: str, prompt: str, job: dict,
        adapter: Any = None,
    ) -> Optional[str]:
        """Create a Discord thread for the recurring task."""
        if not adapter:
            return None

        parent_channel_id = job.get(
            "parent_channel_id", "1501081034832351302"
        )

        try:
            # Use the adapter's HTTP session to create a thread
            import aiohttp
            bot_token = (
                getattr(getattr(adapter, "config", None), "token", None)
                or os.environ.get("DISCORD_BOT_TOKEN", "")
            )
            if not bot_token:
                logger.error("[RecurringJobs] No Discord bot token available")
                return None

            url = (
                f"https://discord.com/api/v10/channels/"
                f"{parent_channel_id}/threads"
            )
            headers = {
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
            }
            payload = {
                "name": name,
                "type": 11,  # PUBLIC_THREAD
                "auto_archive_duration": 1440,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, headers=headers
                ) as resp:
                    if resp.status in (200, 201):
                        data = await resp.json()
                        thread_id = data["id"]

                        # Send waiting message + prompt in thread
                        msg_url = (
                            f"https://discord.com/api/v10/channels/"
                            f"{thread_id}/messages"
                        )
                        user_id = job.get("user_id", "234311476607844353")
                        await session.post(
                            msg_url,
                            json={
                                "content": (
                                    f"<@{user_id}>\n"
                                    "🔄 **Recurring task enqueued.**\n"
                                    "⏳ Waiting for free compute time."
                                )
                            },
                            headers=headers,
                        )
                        # Echo the prompt
                        await session.post(
                            msg_url,
                            json={
                                "content": (
                                    f"**Prompt:**\n"
                                    f"{prompt[:1900]}"
                                )
                            },
                            headers=headers,
                        )

                        # Track the thread
                        if hasattr(adapter, "_threads"):
                            adapter._threads.mark(thread_id)

                        return thread_id
                    else:
                        body = await resp.text()
                        logger.error(
                            "[RecurringJobs] Thread creation HTTP %d: %s",
                            resp.status, body[:300],
                        )
                        return None
        except Exception as exc:
            logger.exception("[RecurringJobs] Thread creation error: %s", exc)
            return None

    async def _check_for_errors(self) -> None:
        """Scan recently completed tasks from recurring jobs for error codes."""
        from gateway.todo_scheduler import TodoQueue

        queue = TodoQueue.instance()
        async with queue._lock:
            tasks = queue._read()

        # Only check completed tasks from recurring jobs that haven't been
        # error-checked yet
        for task in tasks:
            if not task.get("recurring_job_id"):
                continue
            if task.get("_error_checked"):
                continue
            if task["status"] not in ("completed", "failed"):
                continue

            # Mark as checked regardless of outcome
            task["_error_checked"] = True

            # For failed tasks, always notify
            if task["status"] == "failed":
                job = get_job(task["recurring_job_id"])
                if job:
                    error_info = {
                        "code": "TASK_FAILED",
                        "message": task.get("error", "Task failed"),
                        "severity": "error",
                    }
                    log_error(job, error_info)
                    await self._notify_error(job, error_info, task)

            # For completed tasks, check response for error codes
            # NOTE: The _response field must be set by the TodoScheduler
            # worker when it finishes. Currently NOT populated — this is
            # a future enhancement. Failed tasks are still caught above.
            response = task.get("_response", "")
            if response:
                error = parse_error_from_response(response)
                if error:
                    job = get_job(task["recurring_job_id"])
                    if job:
                        log_error(job, error)
                        job["error_count"] = job.get("error_count", 0) + 1
                        update_job(job["id"], {"error_count": job["error_count"]})
                        await self._notify_error(job, error, task)

        # Persist the _error_checked flags
        async with queue._lock:
            all_tasks = queue._read()
            checked_ids = {
                t["id"] for t in tasks if t.get("_error_checked")
            }
            for t in all_tasks:
                if t["id"] in checked_ids:
                    t["_error_checked"] = True
            queue._write(all_tasks)

    async def _notify_error(
        self, job: dict, error: dict, task: dict
    ) -> None:
        """Send error notification to the configured Discord thread."""
        notify_thread = job.get(
            "notify_thread_id", "1501660002199535616"
        )  # Default: this agent thread

        severity = error.get("severity", "error")
        emoji = {"warning": "⚠️", "error": "🚨", "critical": "🔴"}.get(
            severity, "🚨"
        )

        message = (
            f"{emoji} **Recurring Job Error**\n"
            f"**Job:** {job.get('name', 'unknown')} (`{job.get('id', '?')}`)\n"
            f"**Code:** `{error.get('code', 'UNKNOWN')}`\n"
            f"**Severity:** {severity}\n"
            f"**Message:** {error.get('message', 'No details')}\n"
            f"**Task:** `{task.get('id', '?')}` in <#{task.get('thread_id', '?')}>"
        )

        adapter = self._get_discord_adapter()
        if adapter:
            try:
                await adapter.send(
                    chat_id=str(notify_thread),
                    content=message,
                    metadata={"thread_id": str(notify_thread)},
                )
            except Exception as exc:
                logger.error(
                    "[RecurringJobs] Failed to send error notification: %s",
                    exc,
                )
        else:
            logger.warning(
                "[RecurringJobs] No adapter — error notification:\n%s",
                message,
            )
