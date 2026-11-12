"""Per-session gateway state in one container (replaces ~19 session_key-keyed dicts on
GatewayRunner that bred boundary drift and wholesale-reset races).  Scopes follow where each dict
was CLEARED: ``turn`` at the end of every turn; ``conversation`` at conversation boundaries (/new,
/resume, auto-reset, expiry); ``persistent`` fields have their own lifecycles."""

from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

# /fast stores "priority" or None (explicit normal), so key PRESENCE decides, not truthiness.
_UNSET_TIER = object()
SERVICE_TIER_UNSET = _UNSET_TIER  # public alias


@dataclass
class TurnState:
    """State scoped to one running gateway turn.  ``lease_tokens`` is NOT touched by
    ``clear()``: ``_release_turn_lease`` owns it (release exactly once)."""

    agent: Any = None  # running AIAgent (or _AGENT_PENDING_SENTINEL); None = idle
    # The MessageEvent that opened the running turn and the live TurnContext: a successful busy
    # redirect re-anchors both to the redirecting message (#115001).
    event: Any = None
    ctx: Any = None
    # The queued follow-up the turn is running now (its own sender's turn, see ``running_event``).
    follow_up_event: Any = None
    started_ts: float = 0.0  # 0.0 = not running
    lease: Any = None  # cross-process active-session slot lease
    busy_ack_ts: float = 0.0  # debounce; 0.0 = never acked
    # Held turn-lease tokens keyed by acquiring run generation: release/rebind resolve the
    # token for their own generation, so a displaced turn's unwind frees only its own lease and
    # never a successor's (an evicted turn and its replacement may both hold one briefly).
    lease_tokens: dict[int, Any] = field(default_factory=dict)
    # Agents taken out of the cache while this turn still ran them (a one-turn ``model_override``
    # agent): ``_release_running_agent_state`` releases them once the slot clears.
    retired_agents: list[Any] = field(default_factory=list)

    def running_event(self) -> Any:
        """The event whose input the worker is running now: a drained queued follow-up, else the
        opener. A steer or redirect folds into THIS one, so it clears the actor approvals read."""
        return self.follow_up_event if self.follow_up_event is not None else self.event

    @contextmanager
    def running_follow_up(self, event: Any) -> Iterator[None]:
        """Run *event* (a queued follow-up) as the slot's running turn; restore the outer one on
        exit unless the slot was cleared or rebound meanwhile."""
        previous, self.follow_up_event = self.follow_up_event, event
        try:
            yield
        finally:
            if self.follow_up_event is event:
                self.follow_up_event = previous

    def clear(self) -> None:
        """Reset the per-turn slot.  The caller pops ``lease`` first to release it."""
        self.agent = self.lease = self.event = self.ctx = self.follow_up_event = None
        self.started_ts = self.busy_ack_ts = 0.0


@dataclass
class ConversationState:
    """State scoped to one conversation (survives turns, not boundaries)."""

    model_override: Optional[dict[str, Any]] = None  # /model per-session override
    one_turn_restore: Optional[dict[str, Any]] = None  # /model --once snapshot
    reasoning_override: Optional[dict[str, Any]] = None  # /reasoning override
    service_tier_override: Any = _UNSET_TIER  # /fast: "priority" or None; _UNSET_TIER = absent
    last_resolved_model: str = ""  # last successfully-resolved non-empty model
    queued_events: list[Any] = field(default_factory=list)  # /queue overflow FIFO (head in adapter)
    sidecar_notes: list[str] = field(default_factory=list)  # one-shot must-deliver notes
    ephemeral_pin: Optional[tuple[Any, ...]] = None  # pinned session-context (change_key, text, redact_pii)
    # (channel_prompt, parent_chat_id) of the last non-internal turn; internal events reuse it
    channel_pin: Optional[tuple[Optional[str], Optional[str]]] = None
    vc_last: Optional[str] = None  # last voice-channel context delivered

    def clear(self) -> None:
        """Reset every field to its default, so new fields are cleared automatically."""
        self.__dict__.update(ConversationState().__dict__)


@dataclass
class PersistentState:
    """State with its own lifecycle — NOT cleared wholesale by turn or boundary resets
    (approvals/update prompts ARE cleared, individually, by the boundary security funnel)."""

    approvals: Optional[dict[str, Any]] = None  # {"command": ..., "pattern_key": ...}
    update_prompt_pending: bool = False  # /update prompt awaiting a reply
    native_image_paths: list[str] = field(default_factory=list)  # consumed one-shot
    # Legacy runner-level pending text (flushed on shutdown); not the adapter-level one.
    pending_command_text: Optional[str] = None
    run_generation: int = 0  # monotonic; NEVER reset (stale-run detection depends on it)
    # Consecutive hygiene compression failures (the in-agent ladder is unreachable: hygiene builds
    # a FRESH AIAgent per run).  Reset on success; process-local, mirrored to the DB by run.py.
    # Monotonic run-generation counter (#28686). NEVER reset: clearing it would break stale-run detection.
    # The in-agent compressor escalates repeat timeouts via ContextCompressor._consecutive_timeout_failures,
    # but hygiene builds a FRESH AIAgent per run and bind_session_state() zeroes that counter, so the
    # in-agent ladder is structurally unreachable from the gateway. Tracking the streak here — outside the
    # per-run agent — lets hygiene escalate its cooldown instead of retrying on a flat interval forever.
    # Reset on a successful compression, not by turn/boundary resets. PROCESS-LOCAL, deliberately:
    # `PersistentState` means "survives turn and boundary resets", NOT "survives a restart" — this field has
    # no disk flush (unlike `pending_command_text` above, #72680), so a gateway restart drops escalation
    # back to rung 1 while the DB-backed deadline itself survives (#74136). Keying on `session_key` rather
    # than `session_id` is what buys correctness across compaction ROTATION (the sid changes, the chat does
    # not). hermes_agent.gateway.run mirrors this value to the DB keyed by session_key so the same semantics also survive
    # gateway restarts.
    hygiene_failure_streak: int = 0


@dataclass
class SessionState:
    """All per-session gateway state, grouped by lifecycle scope."""

    turn: TurnState = field(default_factory=TurnState)
    conversation: ConversationState = field(default_factory=ConversationState)
    persistent: PersistentState = field(default_factory=PersistentState)


# --- Legacy dict-view adapters: tests read/write the old dict attributes directly
# (``runner._running_agents = {}``); each view is a LIVE MutableMapping over one field.


class _FieldSpec(NamedTuple):
    """One legacy dict: scope attr, field name, default factory, presence test."""
    scope: str
    name: str
    default: Callable[[], Any]
    is_present: Callable[[Any], bool]


def _spec(scope: str, name: str, default: Any) -> _FieldSpec:
    """``default`` is a type (presence = truthiness) or a sentinel (presence = ``is not``)."""
    if isinstance(default, type):
        return _FieldSpec(scope, name, default, bool)
    return _FieldSpec(scope, name, lambda: default, lambda v: v is not default)


class _RunnerView(MutableMapping):
    """Shared plumbing: live view over ``runner._sessions``, dict-comparable."""

    __slots__ = ("_runner",)

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def _sessions(self) -> dict[str, SessionState]:
        return self._runner.__dict__.get("_sessions") or {}

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __eq__(self, other: object) -> bool:  # Mapping has no __eq__; tests compare to dicts
        if isinstance(other, (dict, MutableMapping)):
            return dict(self.items()) == dict(other)
        return NotImplemented


class SessionFieldView(_RunnerView):
    """Live dict-like view of one SessionState field across sessions."""

    __slots__ = ("_spec",)

    def __init__(self, runner: Any, spec: _FieldSpec) -> None:
        super().__init__(runner)
        self._spec = spec

    def _value(self, state: SessionState) -> Any:
        return getattr(getattr(state, self._spec.scope), self._spec.name)

    def _set(self, state: SessionState, value: Any) -> None:
        setattr(getattr(state, self._spec.scope), self._spec.name, value)

    def _present(self, key: Any) -> Optional[SessionState]:
        """The session state for ``key`` if its field is present, else None."""
        state = self._sessions().get(key)
        return state if state is not None and self._spec.is_present(self._value(state)) else None

    def _held(self, key: str) -> SessionState:
        state = self._present(key)
        if state is None:
            raise KeyError(key)
        return state

    def __getitem__(self, key: str) -> Any:
        return self._value(self._held(key))

    def __setitem__(self, key: str, value: Any) -> None:
        self._set(self._runner._session_state(key), value)

    def __delitem__(self, key: str) -> None:
        self._set(self._held(key), self._spec.default())

    def __iter__(self) -> Iterator[str]:
        return (k for k in list(self._sessions()) if self._present(k) is not None)

    def __contains__(self, key: object) -> bool:
        return self._present(key) is not None

    def clear(self) -> None:  # avoid MutableMapping's popitem loop
        for state in list(self._sessions().values()):
            self._set(state, self._spec.default())

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SessionFieldView({self._spec.scope}.{self._spec.name}, {dict(self.items())!r})"


# One spec per legacy dict attribute.
LEGACY_FIELD_SPECS: dict[str, _FieldSpec] = {
    "_running_agents": _spec("turn", "agent", None),
    "_session_model_overrides": _spec("conversation", "model_override", None),
    "_pending_messages": _spec("persistent", "pending_command_text", None),
}


def _legacy_property(make_view: Callable[[Any], MutableMapping], doc: str) -> property:
    """Dict-shaped @property over a live view; the setter takes a plain dict (test pattern
    ``runner._X = {...}``): reset the field on every session, then apply the entries."""

    def fset(self: Any, mapping: Optional[dict[Any, Any]]) -> None:
        view = make_view(self)
        view.clear()
        view.update(mapping or {})

    return property(make_view, fset, lambda self: make_view(self).clear(), doc=doc)


def legacy_dict_property(attr_name: str) -> property:
    """Legacy dict-shaped @property for one migrated attribute."""
    spec = LEGACY_FIELD_SPECS[attr_name]
    return _legacy_property(
        lambda self: SessionFieldView(self, spec),
        f"Legacy dict view over SessionState.{spec.scope}.{spec.name} (for pre-SessionState tests).",
    )
