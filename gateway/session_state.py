"""Per-session gateway state in one container (replaces ~19 session_key-keyed dicts on
GatewayRunner that bred boundary drift and wholesale-reset races).  Scopes follow where each dict
was CLEARED: ``turn`` at the end of every turn; ``conversation`` at conversation boundaries (/new,
/resume, auto-reset, expiry); ``persistent`` fields have their own lifecycles."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

# /fast stores "priority" or None (explicit normal), so key PRESENCE decides, not truthiness.
_UNSET_TIER = object()
SERVICE_TIER_UNSET = _UNSET_TIER  # public alias


@dataclass
class TurnState:
    """State scoped to one running gateway turn.  ``lease_token`` / ``lease_generation``
    are NOT touched by ``clear()``: ``_release_turn_lease`` owns them (release exactly once)."""

    agent: Any = None  # running AIAgent (or _AGENT_PENDING_SENTINEL); None = idle
    started_ts: float = 0.0  # 0.0 = not running
    lease: Any = None  # cross-process active-session slot lease
    busy_ack_ts: float = 0.0  # debounce; 0.0 = never acked
    # Held turn-lease token + acquiring generation: release/rebind match only when the
    # generation is current, so a stale unwind can never free a newer turn's lease.
    lease_token: Any = None
    lease_generation: Optional[int] = None

    def clear(self) -> None:
        """Reset the per-turn slot.  The caller pops ``lease`` first to release it."""
        self.agent = self.lease = None
        self.started_ts = self.busy_ack_ts = 0.0


@dataclass
class ConversationState:
    """State scoped to one conversation (survives turns, not boundaries)."""

    model_override: Optional[Dict[str, Any]] = None  # /model per-session override
    one_turn_restore: Optional[Dict[str, Any]] = None  # /model --once snapshot
    reasoning_override: Optional[Dict[str, Any]] = None  # /reasoning override
    service_tier_override: Any = _UNSET_TIER  # /fast: "priority" or None; _UNSET_TIER = absent
    last_resolved_model: str = ""  # last successfully-resolved non-empty model
    queued_events: List[Any] = field(default_factory=list)  # /queue overflow FIFO (head in adapter)
    sidecar_notes: List[str] = field(default_factory=list)  # one-shot must-deliver notes
    ephemeral_pin: Optional[Tuple[Any, ...]] = None  # pinned session-context (change_key, text)
    vc_last: Optional[str] = None  # last voice-channel context delivered

    def clear(self) -> None:
        """Reset every field to its default, so new fields are cleared automatically."""
        self.__dict__.update(ConversationState().__dict__)


@dataclass
class PersistentState:
    """State with its own lifecycle — NOT cleared wholesale by turn or boundary resets
    (approvals/update prompts ARE cleared, individually, by the boundary security funnel)."""

    approvals: Optional[Dict[str, Any]] = None  # {"command": ..., "pattern_key": ...}
    update_prompt_pending: bool = False  # /update prompt awaiting a reply
    native_image_paths: List[str] = field(default_factory=list)  # consumed one-shot
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
    # not). gateway.run mirrors this value to the DB keyed by session_key so the same semantics also survive
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

    def _sessions(self) -> Dict[str, SessionState]:
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


class TurnLeaseTokenView(_RunnerView):
    """Legacy (session_key, generation)-keyed view of ``_turn_lease_tokens``; the lease registry
    serializes acquisition per session, so one ``TurnState`` slot per key equals the old dict."""

    __slots__ = ()

    def _held(self, key: Any) -> TurnState:
        """TurnState for a currently-held (session_key, generation) or raise KeyError."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        state = self._sessions().get(key[0])
        if state is None or state.turn.lease_token is None or state.turn.lease_generation != key[1]:
            raise KeyError(key)
        return state.turn

    def __getitem__(self, key: Any) -> Any:
        return self._held(key).lease_token

    def __setitem__(self, key: Any, value: Any) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        turn = self._runner._session_state(key[0]).turn
        turn.lease_token, turn.lease_generation = value, key[1]

    def __delitem__(self, key: Any) -> None:
        turn = self._held(key)
        turn.lease_token = turn.lease_generation = None

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        return ((k, s.turn.lease_generation) for k, s in list(self._sessions().items())
                if s.turn.lease_token is not None)

    def clear(self) -> None:  # avoid MutableMapping's popitem loop
        for key in list(self):
            del self[key]


# One spec per legacy dict attribute.
LEGACY_FIELD_SPECS: Dict[str, _FieldSpec] = {
    "_running_agents": _spec("turn", "agent", None),
    "_running_agents_ts": _spec("turn", "started_ts", float),
    "_active_session_leases": _spec("turn", "lease", None),
    "_busy_ack_ts": _spec("turn", "busy_ack_ts", float),
    "_session_model_overrides": _spec("conversation", "model_override", None),
    "_pending_one_turn_model_restores": _spec("conversation", "one_turn_restore", None),
    "_session_reasoning_overrides": _spec("conversation", "reasoning_override", None),
    "_session_service_tier_overrides": _spec("conversation", "service_tier_override", _UNSET_TIER),
    "_last_resolved_model": _spec("conversation", "last_resolved_model", str),
    "_queued_events": _spec("conversation", "queued_events", list),
    "_pending_turn_sidecar_notes": _spec("conversation", "sidecar_notes", list),
    "_session_ephemeral_pin": _spec("conversation", "ephemeral_pin", None),
    "_session_vc_last": _spec("conversation", "vc_last", None),
    "_pending_approvals": _spec("persistent", "approvals", None),
    "_update_prompt_pending": _spec("persistent", "update_prompt_pending", bool),
    "_pending_native_image_paths_by_session": _spec("persistent", "native_image_paths", list),
    "_pending_messages": _spec("persistent", "pending_command_text", None),
    "_session_run_generation": _spec("persistent", "run_generation", int),
}


def _legacy_property(make_view: Callable[[Any], MutableMapping], doc: str) -> property:
    """Dict-shaped @property over a live view; the setter takes a plain dict (test pattern
    ``runner._X = {...}``): reset the field on every session, then apply the entries."""

    def fset(self: Any, mapping: Optional[Dict[Any, Any]]) -> None:
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


def legacy_lease_token_property() -> property:
    """Legacy (session_key, generation)-keyed view of held turn-lease tokens."""
    return _legacy_property(
        TurnLeaseTokenView, "Legacy (session_key, generation)-keyed turn-lease token view."
    )
