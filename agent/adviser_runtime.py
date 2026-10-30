"""Per-adviser review queue, worker, retry ladder, and the session-level roster.

Port of oh-my-pi's ``advisor/runtime.ts`` (queue/drain/retry/quota/halt) and the
capture/cadence half of ``session/session-advisors.ts``, wired to native Hermes
turn boundaries instead of an agent-core ``turn_end`` callback.

Lifecycle:

* :class:`AdviserSession` owns the roster for one :class:`~run_agent.AIAgent`,
  reads ``adviser:`` settings from ``config.yaml`` and the ``WATCHDOG.yml``
  roster from the session cwd, and mirrors omp's cadence (``reviewMode`` /
  ``reviewInterval`` / ``syncBacklog``).
* Each configured adviser gets one :class:`AdviserRuntime`: a background thread
  drains captured delta batches through that adviser's private model loop. The
  primary turn NEVER parks on a failing adviser: catch-up waits are released
  immediately once the adviser is failing.
* Advice delivery (steer / interrupt / continuation / card) lives in
  :mod:`agent.adviser_delivery`; admission and dedupe live in
  :mod:`agent.adviser_emission`.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent.adviser_config import (
    DEFAULT_TOOL_NAMES,
    AdviserConfig,
    discover_advisers,
    discover_watchdog_files,
)
from agent.adviser_delta import DeltaTracker, PendingDelta
from agent.adviser_delivery import append_card, deliver
from agent.adviser_emission import (
    DEFAULT_BUDGET_PER_UPDATE,
    SEVERITIES,
    EmissionGuard,
    meets_severity,
)
from agent.adviser_model import (
    MAX_QUARANTINE_RETRIES,
    AdviserContextOverflow,
    AdviserModel,
    AdviserOutputQuarantined,
    AdviserUsageLimit,
    AdviseTool,
    classify_error,
)
from agent.adviser_prompts import (
    format_active_repo,
    format_context_files,
    format_memory_instructions,
    render_system_prompt,
)
from agent.adviser_recorder import AdviserRecorder

logger = logging.getLogger(__name__)

RETRY_DELAY_S = 1.0
DROP_AFTER_FAILURES = 3
HALT_AFTER_DROPPED_BACKLOGS = 3
MAX_OVERFLOW_RECOVERIES = 1

STATUS_RUNNING = "running"
STATUS_PAUSED = "paused"
STATUS_QUOTA = "quota_exhausted"
STATUS_ERROR = "error"

#: Rough token estimate used only for stale tool-result eviction.
_CHARS_PER_TOKEN = 4

SETTING_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "provider": "",
    "model": "",
    "review_mode": "turn",
    "review_interval": 1,
    "sync_backlog": "off",
    "immune_turns": 3,
    "continue_severity": "concern",
    "max_notes_per_update": DEFAULT_BUDGET_PER_UPDATE,
    "max_continuations": 2,
    "evict_stale_results": True,
    "timeout_s": 120.0,
    "temperature": 0.0,
}


@dataclass
class AdviserSettings:
    enabled: bool = False
    provider: str = ""
    model: str = ""
    review_mode: str = "turn"
    review_interval: int = 1
    sync_backlog: str = "off"
    immune_turns: int = 3
    continue_severity: str = "concern"
    max_notes_per_update: int = DEFAULT_BUDGET_PER_UPDATE
    max_continuations: int = 2
    evict_stale_results: bool = True
    timeout_s: float = 120.0
    temperature: float = 0.0

    @classmethod
    def from_config(cls, raw: Any) -> AdviserSettings:
        section = raw if isinstance(raw, dict) else {}
        values = dict(SETTING_DEFAULTS)
        values.update({key: section[key] for key in SETTING_DEFAULTS if key in section})
        return cls(
            enabled=bool(values["enabled"]),
            provider=str(values["provider"] or "").strip(),
            model=str(values["model"] or "").strip(),
            review_mode=str(values["review_mode"] if values["review_mode"] in ("turn", "agent-end") else "turn"),
            review_interval=_min_int(values["review_interval"], 1, 1000),
            sync_backlog=str(values["sync_backlog"] if str(values["sync_backlog"]) in
                             ("off", "1", "3", "5", "strict") else "off"),
            immune_turns=_min_int(values["immune_turns"], 0, 100),
            continue_severity=str(values["continue_severity"] if values["continue_severity"] in SEVERITIES
                                  else "concern"),
            max_notes_per_update=_min_int(values["max_notes_per_update"], 1, 32),
            max_continuations=_min_int(values["max_continuations"], 0, 10),
            evict_stale_results=bool(values["evict_stale_results"]),
            timeout_s=float(values["timeout_s"] or 120.0),
            temperature=float(values["temperature"] or 0.0),
        )


def _min_int(value: Any, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, number))


def load_settings() -> AdviserSettings:
    """``adviser:`` from the active profile config; unreadable config means disabled."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly() or {}
    except Exception:
        logger.debug("adviser settings unavailable", exc_info=True)
        return AdviserSettings()
    return AdviserSettings.from_config(config.get("adviser"))


def is_nested_agent(agent: Any) -> bool:
    """Delegated subagents, detached forks and unattended runs never carry an adviser.

    The reviewer is interactive-session assistance: a subagent multiplying reviews
    multiplies spend with no one watching, and a background-review fork binding an
    adviser would re-enter the review machinery it exists to avoid.
    """
    if int(getattr(agent, "_delegate_depth", 0) or 0) > 0:
        return True
    if bool(getattr(agent, "_persist_disabled", False)):
        return True
    return str(getattr(agent, "platform", "") or "") in ("subagent", "cron")


def _estimate_tokens(message: Any) -> int:
    content = message.get("content") if isinstance(message, dict) else None
    size = len(content) if isinstance(content, str) else 0
    for call in (message.get("tool_calls") if isinstance(message, dict) else None) or ():
        size += len(str(call))
    return max(0, size // _CHARS_PER_TOKEN)


def evict_stale_tool_results(messages: list[dict[str, Any]], *, min_tokens: int = 50) -> int:
    """Blank stale read/search results from reviews before the latest one.

    Implements omp's margin rule: a cut is taken only when the tokens it saves
    outweigh the prefix it forces the provider to rewrite. Returns messages
    evicted.
    """
    protect_from = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user" and not message.get("synthetic"):
            protect_from = index
            break
    if protect_from <= 1:
        return 0
    protect_from = max(1, protect_from)

    def _candidate(message: Any) -> bool:
        return (
            isinstance(message, dict)
            and message.get("role") == "tool"
            and message.get("_adviser_pruned") is None
            and _estimate_tokens(message) >= min_tokens
            and message.get("name") in ("read_file", "search_files")
        )

    non_candidate = sum(
        _estimate_tokens(message) for message in messages[protect_from:]
    )
    saved = stub = 0
    best_margin = 0
    best_index = -1
    for index in range(protect_from - 1, 0, -1):
        message = messages[index]
        tokens = _estimate_tokens(message)
        if not _candidate(message):
            non_candidate += tokens
            continue
        stub_tokens = len(f"[Stale result elided - {tokens} tokens]") // _CHARS_PER_TOKEN
        saved += tokens - stub_tokens
        stub += stub_tokens
        margin = saved - (non_candidate + stub)
        if margin > best_margin:
            best_margin, best_index = margin, index
    if best_index < 0:
        return 0
    evicted = 0
    for index in range(best_index, protect_from):
        message = messages[index]
        if not _candidate(message):
            continue
        tokens = _estimate_tokens(message)
        message["content"] = f"[Stale result elided - {tokens} tokens]"
        message["_adviser_pruned"] = True
        evicted += 1
    return evicted


@dataclass
class _RuntimeState:
    pending: list[PendingDelta] = field(default_factory=list)
    held: list[PendingDelta] = field(default_factory=list)
    backlog: int = 0
    failing: bool = False
    halted: bool = False
    quota: bool = False
    reviewed: bool = False
    epoch: int = 0
    consecutive_failures: int = 0
    consecutive_quarantines: int = 0
    dropped_backlogs: int = 0


class AdviserRuntime:
    """One adviser's queue and worker; safe to call from the turn thread."""

    def __init__(self, *, config: AdviserConfig, settings: AdviserSettings,
                 model: AdviserModel, recorder: AdviserRecorder,
                 advise: AdviseTool) -> None:
        self.config = config
        self.name = config.name
        self.slug = config.slug
        self.advise = advise
        self.review_mode = config.review_mode or settings.review_mode
        self.review_interval = max(1, config.review_interval or settings.review_interval)
        self.sync_backlog = config.sync_backlog or settings.sync_backlog
        self.evict_stale = settings.evict_stale_results
        self._model = model
        self._recorder = recorder
        self._state = _RuntimeState()
        self._tracker = DeltaTracker()
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._eligible_updates = 0
        self._covered_turn: Optional[int] = None

    # ---- status -----------------------------------------------------------
    @property
    def backlog(self) -> int:
        with self._lock:
            return self._state.backlog

    @property
    def status(self) -> str:
        with self._lock:
            if self._state.halted or self._state.dropped_backlogs >= HALT_AFTER_DROPPED_BACKLOGS:
                return STATUS_ERROR
            if self._state.quota:
                return STATUS_QUOTA
            return STATUS_RUNNING

    @property
    def failing(self) -> bool:
        with self._lock:
            return self._state.failing

    def seed(self, messages: list[Any]) -> None:
        with self._lock:
            self._tracker.seed(messages)
            self._state.pending.clear()
            self._state.held.clear()
            self._state.backlog = 0

    # ---- capture ----------------------------------------------------------
    def capture(self, messages: list[Any], *, wip: bool, dispatch: bool = True,
                turn: Optional[int] = None, terminal: bool = False) -> None:
        with self._lock:
            if self._stop.is_set() or self._state.quota or self._state.halted:
                return
        try:
            delta = self._tracker.capture(messages, wip=wip)
        except Exception:
            logger.warning("adviser delta render failed", exc_info=True)
            with self._lock:
                self._state.failing = True
            self._release_failure_waiters()
            return
        if delta is None:
            return
        delta.terminal = bool(terminal)
        if delta.reset:
            with self._lock:
                self._model.reset()
                self._recorder.commit_review()
        with self._lock:
            if dispatch:
                self._pending_append(delta, turn)
            else:
                self._state.held.append(delta)
            self._wake.notify_all()

    def _pending_append(self, latest: Optional[PendingDelta], turn: Optional[int]) -> None:
        parts = [*self._state.held, *([latest] if latest is not None else [])]
        self._state.held = []
        if not parts:
            return
        merged = parts[-1] if len(parts) == 1 else PendingDelta(
            text="\n\n".join(part.text for part in parts),
            wip=parts[-1].wip,
            turns=1,
            reset=any(part.reset for part in parts),
            terminal=parts[-1].terminal,
        )
        self._state.pending.append(merged)
        self._state.backlog += 1
        self._covered_turn = turn
        self._ensure_worker()

    def flush_held(self) -> None:
        with self._lock:
            if self._stop.is_set() or self._state.quota or self._state.halted:
                return
            if not self._state.held:
                return
            self._pending_append(None, self._covered_turn)
            self._wake.notify_all()

    # ---- worker -----------------------------------------------------------
    def _ensure_worker(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        from agent.memory_provider import spawn_context_thread

        self._stop.clear()
        self._thread = spawn_context_thread(
            self._drain, name=f"adviser-{self.slug or 'default'}")
        self._thread.start()

    def _drain(self) -> None:
        while True:
            with self._lock:
                while not self._state.pending and not self._stop.is_set():
                    self._wake.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                batch = self._state.pending.pop(0)
                self._state.pending.clear()
                epoch = self._state.epoch
            self._run_batch(batch, epoch)

    def _run_batch(self, batch: PendingDelta, epoch: int) -> None:
        overflow_retries = 0
        while True:
            with self._lock:
                if self._stop.is_set() or self._state.epoch != epoch:
                    return
                self._state.consecutive_failures = max(0, self._state.consecutive_failures)
            self._recorder.begin_review()
            try:
                if self.evict_stale:
                    evict_stale_tool_results(self._model.messages)
                self._model.run([batch.text], wip=batch.wip, covered_turn=self._covered_turn,
                                terminal=batch.terminal)
            except Exception as exc:  # health: allow BLE001 -- provider/loop failures are classified below
                logger.debug("adviser review failed", exc_info=True)
                with self._lock:
                    if self._stop.is_set() or self._state.epoch != epoch:
                        self._recorder.abandon_review()
                        return
                    self._state.failing = True
                self._release_failure_waiters()
                decision = self._handle_failure(exc, batch, epoch)
                if decision == "retry":
                    continue
                if decision == "overflow":
                    if overflow_retries >= MAX_OVERFLOW_RECOVERIES:
                        decision = "drop"
                    else:
                        overflow_retries += 1
                        with self._lock:
                            self._model.reset()
                            self._recorder.commit_review()
                        continue
                self._recorder.abandon_review()
                if decision == "drop":
                    self._drop_batch(batch, exc)
                return
            with self._lock:
                self._state.consecutive_failures = 0
                self._state.failing = False
                self._state.reviewed = True
                self._state.backlog = max(0, self._state.backlog - 1)
                self._wake.notify_all()
            self._recorder.commit_review()
            return

    def _handle_failure(self, exc: BaseException, batch: PendingDelta, epoch: int) -> str:
        kind = classify_error(exc)
        if isinstance(exc, AdviserOutputQuarantined):
            with self._lock:
                self._state.consecutive_quarantines += 1
                exhausted = self._state.consecutive_quarantines >= MAX_QUARANTINE_RETRIES
                if not exhausted:
                    self._model.reset()
            if exhausted:
                logger.warning("adviser %s: quarantine retries exhausted", self.name)
                return "drop"
            return "retry"
        if kind == "quota":
            with self._lock:
                self._state.quota = True
                self._model.reset()
            logger.warning("adviser %s paused on provider quota/rate limit", self.name)
            return "drop"
        if kind == "overflow":
            return "overflow"
        if kind == "permanent":
            logger.warning("adviser %s: permanent provider rejection: %s", self.name, exc)
            return "drop"
        with self._lock:
            self._state.consecutive_failures += 1
            failures = self._state.consecutive_failures
        if failures >= DROP_AFTER_FAILURES:
            return "drop"
        if self._stop.wait(RETRY_DELAY_S):
            return "drop"
        return "retry"

    def _drop_batch(self, batch: PendingDelta, exc: BaseException) -> None:
        permanent = classify_error(exc) == "permanent"
        with self._lock:
            self._state.consecutive_failures = 0
            self._state.backlog = max(0, self._state.backlog - 1)
            self._state.dropped_backlogs += 1
            dropped = self._state.dropped_backlogs
            if permanent or dropped >= HALT_AFTER_DROPPED_BACKLOGS:
                self._state.halted = True
                self._state.pending.clear()
                self._state.held.clear()
            self._wake.notify_all()
        logger.warning("adviser %s dropped a review after failures (%d): %s", self.name, dropped, exc)

    def _release_failure_waiters(self) -> None:
        with self._lock:
            self._wake.notify_all()

    # ---- catch-up ---------------------------------------------------------
    def wait_for_catchup(self, timeout: Optional[float], threshold: int = 1) -> bool:
        """Block until backlog < threshold; a failing/halted adviser releases immediately."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._lock:
            while True:
                if (self._stop.is_set() or self._state.quota or self._state.halted
                        or self._state.backlog < threshold or self._state.failing):
                    return self._state.backlog < threshold
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._wake.wait(timeout=remaining)

    # ---- lifecycle --------------------------------------------------------
    def dispose(self) -> None:
        with self._lock:
            self._stop.set()
            self._state.epoch += 1
            self._state.pending.clear()
            self._state.held.clear()
            self._state.backlog = 0
            self._wake.notify_all()
        self._thread = None

    def reset(self) -> None:
        with self._lock:
            self._state.epoch += 1
            self._state.pending.clear()
            self._state.held.clear()
            self._state.backlog = 0
            self._state.failing = False
            self._state.quota = False
            self._state.halted = False
            self._state.dropped_backlogs = 0
            self._state.consecutive_failures = 0
            self._state.consecutive_quarantines = 0
            self._tracker.reset()
            self._model.reset()
            self._wake.notify_all()


class AdviserSession:
    """The roster and turn-boundary policy attached to one ``AIAgent``."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.settings = AdviserSettings()
        self._lock = threading.RLock()
        self._runtimes: list[AdviserRuntime] = []
        self._built_for: Optional[tuple] = None
        self._last_refresh = 0.0
        self._turn_live = False
        self._turn_completed = 0
        self._continuation: Optional[dict[str, Any]] = None
        self._continuations_sent = 0
        self._immune_start: Optional[int] = None
        self._warnings: list[str] = []
        self._pending_cards: list[tuple[str, Optional[str], Optional[int]]] = []
        self.continue_severity = "concern"
        self._last_status: dict[str, str] = {}

    # ---- lifecycle --------------------------------------------------------
    def refresh(self, *, force: bool = False) -> None:
        if not force and self._built_for is not None and time.monotonic() - self._last_refresh < 2.0:
            return
        settings = load_settings()
        cwd = str(getattr(self.agent, "session_cwd", "") or "") or str(getattr(self.agent, "_cwd", "") or "")
        discovered = discover_advisers(cwd) if cwd else None
        signature = (
            settings.enabled, settings.provider, settings.model, settings.review_mode,
            settings.review_interval, settings.sync_backlog, settings.immune_turns,
            settings.continue_severity, settings.max_notes_per_update,
            settings.max_continuations, settings.evict_stale_results, settings.timeout_s,
            settings.temperature, cwd,
            tuple((a.slug, a.name, a.model, a.provider,
                   None if a.tools is None else tuple(a.tools), a.review_mode,
                   a.review_interval, a.sync_backlog, a.instructions, a.enabled,
                   a.max_notes_per_update) for a in (discovered.advisers if discovered else ())),
        )
        with self._lock:
            if not force and signature == self._built_for:
                return
            self._teardown()
            self.settings = settings
            self.continue_severity = settings.continue_severity
            self._built_for = signature
            self._last_refresh = time.monotonic()
            if not settings.enabled:
                return
            self._warnings = list(discovered.warnings) if discovered else []
            roster = list(discovered.advisers) if discovered and discovered.advisers else [
                AdviserConfig(name="default", slug="")
            ]
            shared = discovered.shared_instructions if discovered else None
            shared_budget = discovered.shared_max_notes_per_update if discovered else None
            for config in roster:
                runtime = self._build(config, shared, shared_budget, cwd)
                if runtime is not None:
                    self._runtimes.append(runtime)

    def _build(self, config: AdviserConfig, shared: Optional[str], shared_budget: Optional[int],
               cwd: str) -> Optional[AdviserRuntime]:
        if config.enabled is False:
            self._last_status[config.slug] = STATUS_PAUSED
            return None
        budget = min(32, config.max_notes_per_update or shared_budget or self.settings.max_notes_per_update)
        blocks = self._system_blocks(config, shared, budget, cwd)
        tools = self._tool_schemas(config)
        advise = AdviseTool(
            lambda note, severity, turn, terminal, slug=config.slug, name=config.name:
                self._deliver_note(slug, name, note, severity, turn, terminal),
            EmissionGuard(budget_per_update=budget),
        )
        recorder = AdviserRecorder(session_id=str(getattr(self.agent, "session_id", "") or ""),
                                   slug=config.slug, cwd=cwd)
        model = AdviserModel(
            system_blocks=blocks,
            tools=tools,
            advise=advise,
            run_tool=self._run_tool,
            call_model=self._call_model(config),
            on_record=recorder.record,
        )
        runtime = AdviserRuntime(
            config=config,
            settings=self.settings,
            model=model,
            recorder=recorder,
            advise=advise,
        )
        self._last_status[config.slug] = STATUS_RUNNING
        return runtime

    def _system_blocks(self, config: AdviserConfig, shared: Optional[str], budget: int, cwd: str) -> list[str]:
        watchdog = discover_watchdog_files(cwd) if cwd else ""
        active_repo = format_active_repo(cwd) if cwd else None
        if active_repo:
            watchdog = f"{watchdog}\n\n{active_repo}" if watchdog else active_repo
        memory = format_memory_instructions(self._memory_instructions())
        return render_system_prompt(
            max_notes=budget,
            context_prompt=format_context_files(cwd) if cwd else None,
            memory_prompt=memory,
            watchdog_prompt=watchdog or None,
            shared_instructions=shared,
            instructions=config.instructions,
        )

    def _memory_instructions(self) -> Optional[str]:
        try:
            manager = getattr(self.agent, "_memory_manager", None)
            for provider in getattr(manager, "providers", None) or ():
                text = getattr(provider, "system_prompt_instructions", None)
                if isinstance(text, str) and text.strip():
                    return text
        except Exception:
            logger.debug("adviser memory instructions unavailable", exc_info=True)
        return None

    @staticmethod
    def _tool_schemas(config: AdviserConfig) -> list[dict[str, Any]]:
        from tools.registry import registry

        names = config.tools if config.tools is not None else list(DEFAULT_TOOL_NAMES)
        schemas: list[dict[str, Any]] = []
        for name in names:
            schema = registry.get_schema(name)
            if not isinstance(schema, dict):
                continue
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(schema.get("description") or ""),
                    "parameters": schema.get("parameters") or {"type": "object", "properties": {}},
                },
            })
        return schemas

    def _run_tool(self, name: str, args: dict[str, Any]) -> str:
        from tools.registry import registry

        return str(registry.dispatch(name, args))

    def _call_model(self, config: AdviserConfig) -> Callable[..., Any]:
        provider = (config.provider or self.settings.provider or "").strip() or None
        model = (config.model or self.settings.model or "").strip() or None

        def call(**kwargs: Any) -> Any:
            from agent.auxiliary_client import call_llm

            return call_llm(
                provider=provider,
                model=model,
                messages=kwargs.get("messages") or [],
                tools=kwargs.get("tools"),
                temperature=self.settings.temperature,
                timeout=self.settings.timeout_s,
            )

        return call

    def _teardown(self) -> None:
        for runtime in self._runtimes:
            try:
                runtime.dispose()
            except Exception:
                logger.debug("adviser runtime dispose failed", exc_info=True)
        self._runtimes = []
        self._last_status = {}
        self._continuation = None
        self._continuations_sent = 0
        self._immune_start = None
        self._warnings = []
        self._pending_cards = []

    def dispose(self) -> None:
        with self._lock:
            self._teardown()
            self._built_for = None
            self._turn_live = False

    def reset(self) -> None:
        """Conversation boundary (``/new``, branch): re-prime every adviser."""
        with self._lock:
            for runtime in self._runtimes:
                runtime.reset()
            self._continuation = None
            self._continuations_sent = 0
            self._immune_start = None
            self._turn_completed = 0

    # ---- turn boundaries --------------------------------------------------
    def bind_turn(self) -> None:
        if is_nested_agent(self.agent):
            with self._lock:
                self._turn_live = False
                self._continuation = None
            return
        with self._lock:
            self._turn_live = True
            self._continuation = None
            self._continuations_sent = 0
        self.refresh()

    def unbind_turn(self) -> None:
        with self._lock:
            self._turn_live = False
            self._continuation = None
            cards, self._pending_cards = self._pending_cards, []
        for text, severity, turn in cards:
            append_card(self.agent, text, severity=severity,
                        metadata={"preserved": True, "turn": turn} if turn is not None else {"preserved": True})

    def on_step(self, messages: list[Any], *, terminal: bool) -> None:
        """Capture one boundary; terminal boundaries may dispatch and wait for catch-up."""
        if is_nested_agent(self.agent):
            return
        with self._lock:
            if not self.settings.enabled or not self._runtimes:
                return
            runtimes = list(self._runtimes)
            turn = self._turn_completed + (1 if terminal else 0)
        for runtime in runtimes:
            eligible = not (runtime.review_mode == "agent-end" and not terminal)
            dispatch = False
            if eligible:
                runtime._eligible_updates += 1
                dispatch = runtime._eligible_updates % runtime.review_interval == 0
            runtime.capture(messages, wip=not terminal, dispatch=dispatch, turn=turn,
                            terminal=terminal)
        if terminal:
            with self._lock:
                self._turn_completed += 1
                immune = self._immune_start
                immune_turns = self.settings.immune_turns
                if immune is not None and self._turn_completed >= immune + immune_turns:
                    self._immune_start = None
            for runtime in runtimes:
                runtime.flush_held()
                runtime.advise.flush_deferred_notes()
            self._wait_for_catchup(runtimes)

    def _wait_for_catchup(self, runtimes: list[AdviserRuntime]) -> None:
        for runtime in runtimes:
            mode = runtime.sync_backlog
            if mode == "off":
                continue
            if mode == "strict":
                runtime.wait_for_catchup(None, 1)
            else:
                runtime.wait_for_catchup(30.0, int(mode))

    # ---- delivery ---------------------------------------------------------
    def _deliver_note(self, slug: str, name: str, note: str, severity: Optional[str],
                      turn: Optional[int], terminal: bool) -> None:
        notes = [(note, severity, None if slug == "" else name)]
        with self._lock:
            runtime = next((item for item in self._runtimes if item.slug == slug), None)
            terminal = bool(terminal) or not self._turn_live
        try:
            channel = deliver(self, notes, turn=turn, terminal=terminal)
        except Exception:
            logger.warning("adviser delivery failed", exc_info=True)
            return
        if channel == "steer":
            with self._lock:
                self._immune_start = self._turn_completed + 1
        if runtime is not None:
            self._last_status[slug] = runtime.status

    def set_continuation(self, text: str, *, source: str = "") -> bool:
        """Queue one adviser continuation for the stop gate; bounded per turn."""
        with self._lock:
            if self._continuation is not None or self._continuations_sent >= self.settings.max_continuations:
                return False
            self._continuations_sent += 1
            self._continuation = {"text": text, "source": source}
            return True

    def take_continuation(self) -> Optional[dict[str, Any]]:
        with self._lock:
            pending, self._continuation = self._continuation, None
            return pending

    def turn_live(self) -> bool:
        return bool(self._turn_live)

    def interrupt_immune_active(self) -> bool:
        with self._lock:
            start = self._immune_start
            if start is None or self.settings.immune_turns <= 0:
                return False
            return self._turn_completed < start + self.settings.immune_turns

    def preserve_card(self, text: str, *, severity: Optional[str] = None,
                      turn: Optional[int] = None) -> None:
        """Append the card now when idle; during the live turn queue it for ``unbind_turn``.

        A terminal-boundary review finishes before the final answer row is appended;
        writing the card immediately would put advice about the answer *above* it.
        """
        with self._lock:
            if self._turn_live:
                self._pending_cards.append((text, severity, turn))
                return
        append_card(self.agent, text, severity=severity,
                    metadata={"preserved": True, "turn": turn} if turn is not None else {"preserved": True})

    # ---- introspection ----------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            runtimes = list(self._runtimes)
            enabled = self.settings.enabled
            warnings = list(self._warnings)
        return {
            "enabled": enabled,
            "continue_severity": self.continue_severity,
            "sync_backlog": self.settings.sync_backlog,
            "review_mode": self.settings.review_mode,
            "review_interval": self.settings.review_interval,
            "advisers": [
                {
                    "name": runtime.name,
                    "slug": runtime.slug,
                    "status": runtime.status,
                    "backlog": runtime.backlog,
                    "eligible_updates": runtime._eligible_updates,
                    "review_mode": runtime.review_mode,
                    "review_interval": runtime.review_interval,
                    "sync_backlog": runtime.sync_backlog,
                }
                for runtime in runtimes
            ],
            "warnings": warnings,
        }


__all__ = [
    "HALT_AFTER_DROPPED_BACKLOGS",
    "SETTING_DEFAULTS",
    "STATUS_ERROR",
    "STATUS_PAUSED",
    "STATUS_QUOTA",
    "STATUS_RUNNING",
    "AdviserRuntime",
    "AdviserSession",
    "AdviserSettings",
    "evict_stale_tool_results",
    "is_nested_agent",
    "load_settings",
]
