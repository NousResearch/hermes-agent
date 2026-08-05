"""Per-session gateway state consolidated into one container.

GatewayRunner historically carried ~19 separate ``Dict[str, ...]`` attributes
keyed by session_key, each with its own ad-hoc lifecycle.  Three failure
classes grew out of that shape:

1. Boundary drift — every conversation boundary carried a hand-copied
   pop-list that went stale when a new dict was added (#48031, #58403,
   #10702, #35809).  Mitigated by the ``_CONVERSATION_SCOPED_STATE`` registry,
   now structurally fixed: the fields live in one ``ConversationState``
   dataclass with a single ``clear()``.
2. Turn-release drift — ad-hoc ``del self._running_agents[key]`` sites that
   popped different subsets of the turn dicts.  Mitigated by
   ``_release_running_agent_state``, now ``TurnState.clear()``.
3. Wholesale-reset races — lazy-init paths like
   ``self._session_reasoning_overrides = {}`` replaced the ENTIRE dict,
   discarding concurrent sessions' entries when raced.  Structurally
   impossible now: state is per-session, resets touch one field of one
   ``SessionState``.

Scopes (placement follows where each dict is CLEARED today):

- ``SessionState.turn`` — reset at end of every running turn.
- ``SessionState.conversation`` — reset at conversation boundaries
  (/new, /resume, auto-reset, expiry, compression-exhausted reset).
- ``SessionState.persistent`` — own lifecycles (approval resolution, update
  prompt answer, native-image consumption); ``run_generation`` is monotonic
  and NEVER reset (#28686).

Entries in ``GatewayRunner._sessions`` are never evicted (matching the old
dicts, most of which also leaked empty/stale entries for dead sessions —
see the migration table in the consolidating commit).  Follow-up work may
add eviction of fully-default SessionStates.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Presence-sensitive sentinel: /fast stores "priority" or None (explicit
# normal), so key PRESENCE — not value truthiness — decides whether the
# override applies.  ``_UNSET_TIER`` means "no override recorded".
_UNSET_TIER = object()

# Public alias for callers outside this module.
SERVICE_TIER_UNSET = _UNSET_TIER


@dataclass
class TurnState:
    """State scoped to one running gateway turn.

    Cleared by ``GatewayRunner._release_running_agent_state`` (via
    ``clear()``) at every site that ends a running turn.  ``lease_token`` /
    ``lease_generation`` are deliberately NOT cleared here — they are owned
    by ``_release_turn_lease`` (#64934), which must release the registry
    lease exactly once per acquiring turn.
    """

    # Running AIAgent instance (or _AGENT_PENDING_SENTINEL); None = idle.
    agent: Any = None
    # Turn start timestamp (0.0 = not running).
    started_ts: float = 0.0
    # Cross-process active-session slot lease (None = none held).
    lease: Any = None
    # Last busy-ack timestamp (debounce; 0.0 = never acked).
    busy_ack_ts: float = 0.0
    # Held turn-lease token + the run generation that acquired it.  The old
    # ``_turn_lease_tokens`` dict was keyed by (session_key, generation) so a
    # stale unwind could never free a newer turn's lease; the pair encoding
    # preserves that: release/rebind only match when generation is current.
    lease_token: Any = None
    lease_generation: Optional[int] = None

    def clear(self) -> None:
        """Reset the per-turn slot (agent / start ts / lease / busy-ack).

        Mirrors the exact clear set of the old _release_running_agent_state:
        _running_agents, _running_agents_ts, _active_session_leases (popped
        by the caller so it can call ``lease.release()``), _busy_ack_ts.
        """
        self.agent = None
        self.started_ts = 0.0
        self.lease = None
        self.busy_ack_ts = 0.0


@dataclass
class ConversationState:
    """State scoped to one conversation (survives turns, not boundaries)."""

    # /model per-session override (model/provider/api_key/base_url/api_mode).
    model_override: Optional[Dict[str, Any]] = None
    # /model --once restore snapshot.
    one_turn_restore: Optional[Dict[str, Any]] = None
    # /reasoning per-session override.
    reasoning_override: Optional[Dict[str, Any]] = None
    # /fast per-session override: "priority" or None; _UNSET_TIER = absent.
    service_tier_override: Any = _UNSET_TIER
    # Last successfully-resolved non-empty model (#35314 recovery).
    last_resolved_model: str = ""
    # /queue overflow FIFO (adapter slot holds the head).
    queued_events: List[Any] = field(default_factory=list)
    # Per-turn must-deliver sidecar notes (one-shot).
    sidecar_notes: List[str] = field(default_factory=list)
    # Pinned session-context bytes: (change_key, text).
    ephemeral_pin: Optional[Tuple[Any, ...]] = None
    # Last voice-channel context delivered (None = never delivered).
    vc_last: Optional[str] = None

    def clear(self) -> None:
        """Reset every conversation-scoped field to its default.

        The structural successor of the ``_CONVERSATION_SCOPED_STATE``
        pop-loop: adding a field here means every boundary clears it
        automatically.
        """
        self.model_override = None
        self.one_turn_restore = None
        self.reasoning_override = None
        self.service_tier_override = _UNSET_TIER
        self.last_resolved_model = ""
        self.queued_events = []
        self.sidecar_notes = []
        self.ephemeral_pin = None
        self.vc_last = None


@dataclass
class PersistentState:
    """State with its own lifecycle — NOT cleared by turn or boundary resets
    wholesale (approvals/update prompts ARE cleared by the boundary
    *security* funnel, but individually, matching the old behavior)."""

    # Pending exec approval ({"command": ..., "pattern_key": ...}).
    approvals: Optional[Dict[str, Any]] = None
    # /update prompt awaiting a user response.
    update_prompt_pending: bool = False
    # Image paths staged for native (inline) attachment; consumed one-shot.
    native_image_paths: List[str] = field(default_factory=list)
    # Legacy runner-level pending message text (write-mostly; flushed to
    # disk on shutdown — see #72680).  NOTE: distinct from the adapter-level
    # ``_pending_messages`` (Dict[str, MessageEvent]) in gateway/base.py,
    # which is a different store that happens to share the old name.
    pending_command_text: Optional[str] = None
    # Monotonic run-generation counter (#28686).  NEVER reset: clearing it
    # would break stale-run detection.
    run_generation: int = 0
    # Consecutive session-hygiene compression failures for this session
    # (#79624).  The in-agent compressor escalates repeat timeouts via
    # ContextCompressor._consecutive_timeout_failures, but hygiene builds a
    # FRESH AIAgent per run and bind_session_state() zeroes that counter, so
    # the in-agent ladder is structurally unreachable from the gateway.
    # Tracking the streak here — outside the per-run agent — lets hygiene
    # escalate its cooldown instead of retrying on a flat interval forever.
    # Reset on a successful compression, not by turn/boundary resets.
    hygiene_failure_streak: int = 0


@dataclass
class SessionState:
    """All per-session gateway state, grouped by lifecycle scope."""

    turn: TurnState = field(default_factory=TurnState)
    conversation: ConversationState = field(default_factory=ConversationState)
    persistent: PersistentState = field(default_factory=PersistentState)


# ---------------------------------------------------------------------------
# Legacy dict-view adapters.
#
# Dozens of tests construct bare runners (object.__new__) and read/write the
# old dict attributes directly (``runner._running_agents = {}``,
# ``assert key in runner._pending_approvals``...).  These views keep that
# surface working: each is a LIVE MutableMapping over one SessionState field
# across all sessions.  Production code in gateway/run.py accesses state via
# ``self._session_state(key).<scope>.<field>`` — the views exist for tests
# and for the thin remaining mixin/adapter call sites documented in the
# consolidating commit.
# ---------------------------------------------------------------------------


class _FieldSpec:
    """Describes one legacy dict: scope attr, field name, absence test."""

    __slots__ = ("scope", "name", "default", "is_present")

    def __init__(
        self,
        scope: str,
        name: str,
        default: Callable[[], Any],
        is_present: Callable[[Any], bool],
    ) -> None:
        self.scope = scope
        self.name = name
        self.default = default
        self.is_present = is_present


def _present_not_none(v: Any) -> bool:
    return v is not None


def _present_nonzero(v: Any) -> bool:
    return bool(v)


def _present_not_unset(v: Any) -> bool:
    return v is not _UNSET_TIER


class SessionFieldView(MutableMapping):
    """Live dict-like view of one SessionState field across sessions."""

    __slots__ = ("_runner", "_spec")

    def __init__(self, runner: Any, spec: _FieldSpec) -> None:
        self._runner = runner
        self._spec = spec

    # -- internals --------------------------------------------------------
    def _sessions(self) -> Dict[str, SessionState]:
        return self._runner.__dict__.get("_sessions") or {}

    def _value(self, state: SessionState) -> Any:
        return getattr(getattr(state, self._spec.scope), self._spec.name)

    # -- mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        state = self._sessions().get(key)
        if state is None:
            raise KeyError(key)
        value = self._value(state)
        if not self._spec.is_present(value):
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        state = self._runner._session_state(key)
        setattr(getattr(state, self._spec.scope), self._spec.name, value)

    def __delitem__(self, key: str) -> None:
        state = self._sessions().get(key)
        if state is None or not self._spec.is_present(self._value(state)):
            raise KeyError(key)
        setattr(
            getattr(state, self._spec.scope),
            self._spec.name,
            self._spec.default(),
        )

    def __iter__(self) -> Iterator[str]:
        for key, state in list(self._sessions().items()):
            if self._spec.is_present(self._value(state)):
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __contains__(self, key: object) -> bool:
        state = self._sessions().get(key)  # type: ignore[arg-type]
        return state is not None and self._spec.is_present(self._value(state))

    def clear(self) -> None:  # avoid MutableMapping's popitem loop
        for state in list(self._sessions().values()):
            setattr(
                getattr(state, self._spec.scope),
                self._spec.name,
                self._spec.default(),
            )

    # Mapping doesn't provide __eq__; tests compare against plain dicts.
    def __eq__(self, other: object) -> bool:
        if isinstance(other, (dict, MutableMapping)):
            return dict(self.items()) == dict(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SessionFieldView({self._spec.scope}.{self._spec.name}, {dict(self.items())!r})"


class TurnLeaseTokenView(MutableMapping):
    """Legacy view of ``_turn_lease_tokens``: keyed by (session_key, generation).

    The pair now lives on ``TurnState.lease_token`` / ``lease_generation``;
    at most one held token exists per session key (the lease registry
    serializes acquisition per session), so the single slot is equivalent to
    the old tuple-keyed dict.
    """

    __slots__ = ("_runner",)

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def _sessions(self) -> Dict[str, SessionState]:
        return self._runner.__dict__.get("_sessions") or {}

    @staticmethod
    def _split(key: Any) -> Tuple[str, int]:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        return key[0], key[1]

    def __getitem__(self, key: Any) -> Any:
        session_key, generation = self._split(key)
        state = self._sessions().get(session_key)
        if (
            state is None
            or state.turn.lease_token is None
            or state.turn.lease_generation != generation
        ):
            raise KeyError(key)
        return state.turn.lease_token

    def __setitem__(self, key: Any, value: Any) -> None:
        session_key, generation = self._split(key)
        state = self._runner._session_state(session_key)
        state.turn.lease_token = value
        state.turn.lease_generation = generation

    def __delitem__(self, key: Any) -> None:
        session_key, generation = self._split(key)
        state = self._sessions().get(session_key)
        if (
            state is None
            or state.turn.lease_token is None
            or state.turn.lease_generation != generation
        ):
            raise KeyError(key)
        state.turn.lease_token = None
        state.turn.lease_generation = None

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for key, state in list(self._sessions().items()):
            if state.turn.lease_token is not None:
                yield (key, state.turn.lease_generation)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (dict, MutableMapping)):
            return dict(self.items()) == dict(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result


# One spec per legacy dict attribute (see the consolidating commit's
# migration table for declared-line / clear-path provenance).
LEGACY_FIELD_SPECS: Dict[str, _FieldSpec] = {
    "_running_agents": _FieldSpec("turn", "agent", lambda: None, _present_not_none),
    "_running_agents_ts": _FieldSpec("turn", "started_ts", float, _present_nonzero),
    "_active_session_leases": _FieldSpec("turn", "lease", lambda: None, _present_not_none),
    "_busy_ack_ts": _FieldSpec("turn", "busy_ack_ts", float, _present_nonzero),
    "_session_model_overrides": _FieldSpec(
        "conversation", "model_override", lambda: None, _present_not_none
    ),
    "_pending_one_turn_model_restores": _FieldSpec(
        "conversation", "one_turn_restore", lambda: None, _present_not_none
    ),
    "_session_reasoning_overrides": _FieldSpec(
        "conversation", "reasoning_override", lambda: None, _present_not_none
    ),
    "_session_service_tier_overrides": _FieldSpec(
        "conversation", "service_tier_override", lambda: _UNSET_TIER, _present_not_unset
    ),
    "_last_resolved_model": _FieldSpec(
        "conversation", "last_resolved_model", str, _present_nonzero
    ),
    "_queued_events": _FieldSpec("conversation", "queued_events", list, _present_nonzero),
    "_pending_turn_sidecar_notes": _FieldSpec(
        "conversation", "sidecar_notes", list, _present_nonzero
    ),
    "_session_ephemeral_pin": _FieldSpec(
        "conversation", "ephemeral_pin", lambda: None, _present_not_none
    ),
    "_session_vc_last": _FieldSpec(
        "conversation", "vc_last", lambda: None, _present_not_none
    ),
    "_pending_approvals": _FieldSpec(
        "persistent", "approvals", lambda: None, _present_not_none
    ),
    "_update_prompt_pending": _FieldSpec(
        "persistent", "update_prompt_pending", bool, _present_nonzero
    ),
    "_pending_native_image_paths_by_session": _FieldSpec(
        "persistent", "native_image_paths", list, _present_nonzero
    ),
    "_pending_messages": _FieldSpec(
        "persistent", "pending_command_text", lambda: None, _present_not_none
    ),
    "_session_run_generation": _FieldSpec(
        "persistent", "run_generation", int, _present_nonzero
    ),
}


def legacy_dict_property(attr_name: str) -> property:
    """Build a legacy dict-shaped @property for one migrated attribute.

    Getter returns a live :class:`SessionFieldView`; setter accepts a plain
    dict (the ubiquitous test pattern ``runner._X = {...}``), resetting the
    field on every known session and then applying the given entries.
    """
    spec = LEGACY_FIELD_SPECS[attr_name]

    def fget(self: Any) -> SessionFieldView:
        return SessionFieldView(self, spec)

    def fset(self: Any, mapping: Optional[Dict[str, Any]]) -> None:
        view = SessionFieldView(self, spec)
        view.clear()
        for key, value in (mapping or {}).items():
            view[key] = value

    def fdel(self: Any) -> None:
        # ``del runner._busy_ack_ts`` in older tests simulated a runner
        # without the attribute; the closest equivalent is "no entries".
        SessionFieldView(self, spec).clear()

    return property(
        fget,
        fset,
        fdel,
        doc=(
            f"Legacy dict view over SessionState.{spec.scope}.{spec.name} "
            "(kept for tests that access the pre-SessionState attribute)."
        ),
    )


def legacy_lease_token_property() -> property:
    """Legacy (session_key, generation)-keyed view of held turn-lease tokens."""

    def fget(self: Any) -> TurnLeaseTokenView:
        return TurnLeaseTokenView(self)

    def fset(self: Any, mapping: Optional[Dict[Tuple[str, int], Any]]) -> None:
        view = TurnLeaseTokenView(self)
        for key in list(view):
            del view[key]
        for key, value in (mapping or {}).items():
            view[key] = value

    def fdel(self: Any) -> None:
        view = TurnLeaseTokenView(self)
        for key in list(view):
            del view[key]

    return property(
        fget,
        fset,
        fdel,
        doc="Legacy (session_key, generation)-keyed turn-lease token view.",
    )
