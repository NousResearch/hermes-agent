"""Turn-queue dispatch policy: human-platform priority and the busy explainer.

When several agent lanes share one gateway process (messaging turns, TUI/desktop sessions,
automation wakes, cron-driven agent jobs), every queued turn's claim races the same event
loop, the same off-loop claim executor and the same GIL. A long autonomous turn therefore
delays a person's first message in a brand-new chat with no visibility at all: an admission
committed at T+10s started at T+35m and the chat saw nothing in between (incident 2026-11-07,
est-38ov8). Two knobs answer it from one module:

* ``gateway.autonomous_turn_cap`` — how many NON-human-class turns may hold the pipeline at
  once. Human-class admissions never wait for the cap, so a saturated autonomous backlog can
  delay an autonomous successor but never a person. The cap is the priority mechanism: it
  bounds the resource contention instead of reordering any session's FIFO (arrival order
  inside a session stays inviolate, #28503).
* ``gateway.busy_explainer_after_seconds`` — after a human message's admission is durable
  and its turn is still QUEUED this long, one short receipt is posted to the chat
  ("saved, waiting for a free slot, oldest holder X"), refreshed at most every
  ``gateway.busy_explainer_refresh_seconds`` up to ``gateway.busy_explainer_max_messages``.
  Silence is only acceptable while the person can see something running — once the row is
  claimed, the existing long-running heartbeat owns the narration.

Class comes from the admission's ``principal_id`` (``messaging:[profile, platform, ...]``);
``gateway.turn_priority_human_platforms`` overrides the human set (e.g. add ``local`` to
treat TUI/desktop as human). Everything here is pure/config-resolution — the seams
(``session_authority._drain``, ``session_ingress.admit_message``) stay thin.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import time
from typing import Any, FrozenSet, Optional

logger = logging.getLogger(__name__)

HUMAN = "human"
AUTONOMOUS = "autonomous"

# Surfaces a person types on. Anything else reaching the gateway (local, api_server,
# webhook, msgraph_webhook, relay) is autonomous-class: it may be a script, a bot wake or a
# cron agent, and none of them may outrank — or starve — a human turn. Plugin platforms are
# matched by their lowercase value string, so adding a chat plugin only means adding it here.
DEFAULT_HUMAN_PLATFORMS = frozenset({
    "telegram", "discord", "whatsapp", "whatsapp_cloud", "slack", "signal", "mattermost",
    "matrix", "email", "sms", "dingtalk", "feishu", "wecom", "wecom_callback", "weixin",
    "bluebubbles", "qqbot", "yuanbao",
    # plugin platforms shipped under plugins/platforms
    "imessage", "teams", "google_chat", "line", "irc",
})

EXPLAINER_AFTER_S = 30.0
EXPLAINER_REFRESH_S = 120.0
EXPLAINER_MAX = 2


class TurnQueuePolicy:
    """Resolved ``gateway.turn_priority*`` / ``gateway.busy_explainer*`` settings."""

    __slots__ = ("enabled", "autonomous_cap", "human_platforms",
                 "explainer_after_s", "explainer_refresh_s", "explainer_max")

    def __init__(self, *, enabled: bool, autonomous_cap: Optional[int],
                 human_platforms: FrozenSet[str], explainer_after_s: float,
                 explainer_refresh_s: float, explainer_max: int):
        self.enabled = enabled
        self.autonomous_cap = autonomous_cap
        self.human_platforms = human_platforms
        self.explainer_after_s = max(0.0, explainer_after_s)
        self.explainer_refresh_s = max(1.0, explainer_refresh_s)
        self.explainer_max = max(0, explainer_max)

    def class_of_principal(self, principal_id: Optional[str]) -> str:
        """Class a durable-admission principal. Unknown shapes are AUTONOMOUS: a principal
        that cannot name a human platform may not claim the human lane."""
        platform = principal_platform(principal_id)
        if platform is None:
            return AUTONOMOUS
        return HUMAN if platform in self.human_platforms else AUTONOMOUS

    def class_of_platform_value(self, platform: Optional[str]) -> str:
        return HUMAN if (platform or "") in self.human_platforms else AUTONOMOUS


def principal_platform(principal_id: Optional[str]) -> Optional[str]:
    """The platform stamped in a durable admission principal
    (``messaging:["profile","platform",...]``); None when not parseable."""
    if not principal_id or not principal_id.startswith("messaging:"):
        return None
    try:
        parts = json.loads(principal_id[len("messaging:"):])
    except (ValueError, TypeError):
        return None
    if isinstance(parts, list) and len(parts) >= 2 and isinstance(parts[1], str):
        return parts[1].strip().lower()
    return None


def _positive_int(raw: Any) -> Optional[int]:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = int(str(raw).strip(), 10) if isinstance(raw, str) else int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _positive_float(raw: Any, default: float) -> float:
    if raw is None or isinstance(raw, bool):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _int_default(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(0, value)


def _config_picker(config: Any):
    """Key reader for the three shapes a gateway config reaches code in: the raw root dict
    (nested ``gateway`` section, or already the gateway section itself), a GatewayConfig
    instance (fields on the object), or None (every default)."""
    if config is None:
        return lambda key: None
    if isinstance(config, dict):
        nested = config.get("gateway")
        base = nested if isinstance(nested, dict) else {}

        def pick(key: str):
            if key in base:
                return base[key]
            if key in config:
                return config[key]
            return config.get(f"gateway.{key}")
        return pick

    def attr(key: str):
        nested = getattr(config, "gateway", None)
        if nested is not None and hasattr(nested, key):
            return getattr(nested, key)
        return getattr(config, key, None)
    return attr


def resolve_policy(config: Any) -> TurnQueuePolicy:
    """Read ``turn_priority*`` / ``busy_explainer*`` from a gateway config (any shape).

    The feature is ON by default with the cap unset: the explainer (visibility) ships on,
    the cap (priority enforcement) is opt-in per seat, because the right number depends on
    the host's cores and lane mix — the same defaulting law as ``max_concurrent_sessions``.
    """
    pick = _config_picker(config)
    mode = pick("turn_priority")
    enabled = str(mode).strip().lower() != "off" if mode is not None else True
    platforms = pick("turn_priority_human_platforms")
    if isinstance(platforms, (list, tuple, set, frozenset)) and platforms:
        human = frozenset(str(p).strip().lower() for p in platforms if str(p).strip())
    else:
        human = DEFAULT_HUMAN_PLATFORMS
    return TurnQueuePolicy(
        enabled=enabled,
        autonomous_cap=_positive_int(pick("autonomous_turn_cap")),
        human_platforms=human,
        explainer_after_s=_positive_float(pick("busy_explainer_after_seconds"), EXPLAINER_AFTER_S),
        explainer_refresh_s=_positive_float(pick("busy_explainer_refresh_seconds"), EXPLAINER_REFRESH_S),
        explainer_max=_int_default(pick("busy_explainer_max_messages"), EXPLAINER_MAX),
    )


# --------------------------------------------------------------------------- gate

class AutonomousGate:
    """Counts in-flight autonomous-class turns per event loop; humans never wait.

    One gate per running loop (the drain, the waiter and the release all share it). Not a
    fairness device — a ceiling: it keeps N long autonomous turns from owning every thread
    and every GIL slice the loop's claim path needs.
    """

    def __init__(self, cap: int):
        self.cap = cap
        self._count = 0
        self._free = None  # asyncio.Event, created lazily on the running loop

    def _event(self):
        if self._free is None:
            self._free = asyncio.Event()
        return self._free

    @property
    def in_flight(self) -> int:
        return self._count

    async def acquire(self) -> None:
        while self._count >= self.cap:
            await self._event().wait()
            self._event().clear()
        self._count += 1

    def release(self) -> None:
        if self._count > 0:
            self._count -= 1
        self._event().set()


_gates: "dict[tuple[int, int], AutonomousGate]" = {}
_gates_lock = threading.Lock()


def gate_for(loop: asyncio.AbstractEventLoop, cap: int) -> AutonomousGate:
    key = (id(loop), cap)
    with _gates_lock:
        gate = _gates.get(key)
        if gate is None:
            gate = _gates[key] = AutonomousGate(cap)
        return gate


def drop_gates_for_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Test/shutdown helper: forget gates keyed to a dead loop."""
    with _gates_lock:
        for key in [k for k in _gates if k[0] == id(loop)]:
            _gates.pop(key, None)


@contextlib.asynccontextmanager
async def turn_capacity(policy: TurnQueuePolicy, klass: str):
    """Hold an autonomous slot across claim→execute→settle when *klass* says so.

    Cancelling (a drain stopped mid-wait) releases cleanly because ``release`` runs in
    ``finally``; a human-class turn takes the no-op path and never touches the gate.
    """
    if not (policy.enabled and policy.autonomous_cap) or klass != AUTONOMOUS:
        yield None
        return
    gate = gate_for(asyncio.get_running_loop(), policy.autonomous_cap)
    await gate.acquire()
    try:
        yield gate
    finally:
        gate.release()


def _format_age(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h" if not minutes else f"{hours}h{minutes}m"


def compose_queue_wait_message(
    *, waited_s: float, running_count: int = 0, oldest_wait_s: float = 0.0,
) -> str:
    """Receipt for a human message whose turn is admitted but still queued.

    Detail beyond the count/age is deliberately absent: the gateway cannot know what
    another lane's turn is doing, and a wrong guess reads as a lie. Once the row is
    claimed the running-turn heartbeat takes over the narration — this says nothing about
    a turn that is visibly working.
    """
    from agent.i18n import t
    detail = ""
    if running_count > 0:
        detail = t("gateway.busy.explainer_holders",
                   count=running_count, oldest=_format_age(oldest_wait_s))
    return t("gateway.busy.explainer_queued", detail=detail)


# --------------------------------------------------------------------------- explainer

async def run_busy_explainer(
    policy: TurnQueuePolicy, *, db, admission_id: str, event, runner, adapter,
    started_ts: float, sleep=asyncio.sleep, row_reader=None, holder_probe=None,
) -> int:
    """Post the queue-wait receipt(s) for one admitted human message; returns sends.

    Reads the row before every send so a settled turn can never be narrated late (the
    answer is on its way: an ack then reads as a lie). Sends via the runner's busy-reply
    path (thread metadata included, retry-bounded). Every failure is quiet by design —
    an explainer must never turn into a failed turn. ``sleep``/``row_reader``/
    ``holder_probe`` are seams for tests and for a clock-frozen drain replay.
    """
    if row_reader is None:
        from hermes_state_runtime import get_session_admission
        def row_reader():
            return get_session_admission(db, admission_id=admission_id)
    if holder_probe is None:
        holder_probe = _default_holder_probe(runner)
    sent = 0
    delay = policy.explainer_after_s
    while sent < policy.explainer_max:
        try:
            await sleep(delay)
        except asyncio.CancelledError:
            return sent
        delay = policy.explainer_refresh_s
        try:
            row = row_reader()
        except Exception:
            return sent
        if row is None or row.get("status") != "queued":
            return sent  # started: the running-turn heartbeat owns that narration;
        # terminal/refused/deleted: the real reply (or its absence) already speaks. The
        # explainer never double-talks about a turn that is visibly working.
        running_count, oldest_s = holder_probe()
        message = compose_queue_wait_message(
            waited_s=max(0.0, time.time() - started_ts),
            running_count=running_count, oldest_wait_s=oldest_s,
        )
        try:
            await runner._send_busy_reply(event, adapter, message)
        except Exception:
            logger.debug("busy explainer send failed for admission %s", admission_id)
            return sent
        sent += 1
    return sent


def _default_holder_probe(runner):
    """(turns running in this process, seconds the oldest has been running)."""
    def probe():
        try:
            ts_map = getattr(runner, "_running_agents_ts", None) or {}
            stamps = [v for v in (ts_map.get(k) for k in list(ts_map))
                      if isinstance(v, (int, float))]
            if not stamps:
                running = getattr(runner, "_running_agents", None) or {}
                return (max(0, len(running)), 0.0)
            return (len(stamps), max(0.0, time.time() - min(stamps)))
        except Exception:
            return (0, 0.0)
    return probe


def schedule_busy_explainer(authority, event, admission_id: str, policy: TurnQueuePolicy,
                            *, started_ts=None, adapter=None) -> "asyncio.Task | None":
    """Spawn the explainer for one admitted human message; None when not eligible.

    Eligibility: the message looked idle at admission (``_busy_explainer_eligible`` — a
    mid-run follow-up already got the busy ack), is not internal or a command, names a
    human platform, and the after-delay is > 0. One task per admission; the task dies with
    the process if its turn never returns, so it is tracked like any adapter background.
    """
    if not policy.enabled:
        return None
    if policy.explainer_after_s <= 0 or policy.explainer_max <= 0:
        return None
    if getattr(event, "internal", False) or event.get_command():
        return None
    if getattr(event, "_busy_explainer_eligible", False) is not True:
        return None
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", None)
    if policy.class_of_platform_value(platform) != HUMAN:
        return None
    runner = getattr(authority, "runner", None)
    if runner is None:
        return None
    if adapter is None:
        try:
            adapter = runner._delivery_adapter_for(source)
        except Exception:
            adapter = None
    if adapter is None:
        return None
    task = asyncio.create_task(run_busy_explainer(
        policy, db=authority.db, admission_id=admission_id, event=event,
        runner=runner, adapter=adapter,
        started_ts=started_ts if started_ts is not None else __import__("time").time(),
    ))

    def _log_failure(done):
        exc = done.exception() if not done.cancelled() else None
        if exc is not None:
            logger.warning("busy explainer task died: %r", exc)

    task.add_done_callback(_log_failure)
    tasks = getattr(runner, "_busy_explainer_tasks", None)
    if tasks is None:
        tasks = runner._busy_explainer_tasks = set()
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task
