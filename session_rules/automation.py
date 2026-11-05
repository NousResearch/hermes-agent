"""The automation-state codecs the store applies when compression moves a session's goal, heartbeat and loop state to its child: each state class with its fields and its JSON codec; and the rewrite of an automation admission a completion batch folds in.

Each state class carries its fields and its JSON codec up to ``from_json`` (``GoalContract`` up to ``from_dict``); the prompt, scheduling and display methods live with the commands. Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


DEFAULT_MAX_TURNS = 20


# Quality gates: deterministic shell commands that must pass before the judge may declare DONE. A
# failed gate short-circuits the judge — its output IS the continuation prompt, so the agent works
# on concrete evidence instead of a vibe check.
DEFAULT_GATE_TIMEOUT_SECONDS = 300
DEFAULT_GATE_MAX_RETRIES = 3


# The five contract fields, in display order (after OpenAI Codex's "strong goal" guidance: what
# "done" means, how to prove it, what must not regress, what is in bounds, when to stop and ask).
# A bare free-form goal stays fully supported — empty fields are omitted from every prompt.
_CONTRACT_FIELDS = ("outcome", "verification", "constraints", "boundaries", "stop_when")


@dataclass
class GoalContract:
    """Optional structured completion contract; empty fields are omitted everywhere."""
    outcome: str = ""
    verification: str = ""
    constraints: str = ""
    boundaries: str = ""
    stop_when: str = ""

    def is_empty(self) -> bool:
        return not any(getattr(self, f).strip() for f in _CONTRACT_FIELDS)

    def to_dict(self) -> dict[str, str]:
        return {f: getattr(self, f) for f in _CONTRACT_FIELDS}

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> GoalContract:
        if not isinstance(data, dict):
            return cls()
        return cls(**{f: str(data.get(f) or "").strip() for f in _CONTRACT_FIELDS})


@dataclass
class GoalGate:
    """A deterministic shell command that must pass before a goal can be done.

    Gates run at turn boundary BEFORE the LLM judge; a failing gate short-circuits judging and its
    bounded output becomes the continuation prompt.
    """
    command: str
    timeout_seconds: int = DEFAULT_GATE_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_GATE_MAX_RETRIES
    attempts: int = 0
    last_exit_code: Optional[int] = None
    last_output_tail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> GoalGate:
        if not isinstance(data, dict):
            return cls(command="")
        return cls(
            command=str(data.get("command") or ""),
            timeout_seconds=int(data.get("timeout_seconds") or DEFAULT_GATE_TIMEOUT_SECONDS),
            max_retries=int(data.get("max_retries") or DEFAULT_GATE_MAX_RETRIES),
            attempts=int(data.get("attempts") or 0),
            last_exit_code=(int(data["last_exit_code"]) if data.get("last_exit_code") is not None else None),
            last_output_tail=str(data.get("last_output_tail") or ""),
        )


@dataclass
class GoalState:
    """Serializable goal state stored per session."""
    goal: str
    status: str = "active"          # active | paused | done | cleared
    turns_used: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: Optional[str] = None        # "done" | "blocked" | "continue" | "wait" | "skipped"
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None       # why we auto-paused (budget, etc.)
    consecutive_parse_failures: int = 0       # judge-output parse failures in a row
    # Tracked separately from parse failures: a broken API key returns 401 every call and must
    # auto-pause instead of burning the budget on an unreachable judge.
    consecutive_transport_failures: int = 0   # judge API/transport errors in a row
    # User-added criteria (/subgoal). Both the judge and continuation prompts include them.
    subgoals: list[str] = field(default_factory=list)
    # Wait barrier (judge ``wait`` verdict or ``/goal wait``): parks the loop instead of re-poking the
    # agent into busy-work. pid → until exit; session → until that process_registry session's OWN
    # trigger fires (exit OR watch_patterns match — preferred for watchers that signal mid-run);
    # until → wall-clock deadline. While ANY is active evaluate_after_turn returns
    # should_continue=False without burning a turn; cleared lazily when satisfied or by unwait/pause/
    # resume/clear. Defaults empty so old state_meta rows load unchanged.
    waiting_on_pid: Optional[int] = None
    waiting_on_session: Optional[str] = None
    waiting_until: float = 0.0
    # Live delegation batches when a timed WAIT was set because of them; the barrier lifts as soon
    # as that count drops (a batch returned), not only when the timer runs out.
    waiting_on_delegations: int = 0
    waiting_reason: Optional[str] = None
    waiting_since: float = 0.0
    contract: GoalContract = field(default_factory=GoalContract)
    # /goal gate add <cmd>: ALL must pass before the judge may declare done.
    gates: list[GoalGate] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> GoalState:
        data = json.loads(raw)
        raw_subgoals = data.get("subgoals") or []
        ints = {k: int(data.get(k) or 0) for k in ("turns_used", "consecutive_parse_failures", "consecutive_transport_failures", "waiting_on_delegations")}
        floats = {k: float(data.get(k) or 0.0) for k in ("created_at", "last_turn_at", "waiting_until", "waiting_since")}
        return cls(
            goal=data.get("goal", ""),
            status=data.get("status", "active"),
            max_turns=int(data.get("max_turns") or DEFAULT_MAX_TURNS),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
            subgoals=[str(s).strip() for s in raw_subgoals if str(s).strip()] if isinstance(raw_subgoals, list) else [],
            waiting_on_pid=(int(data["waiting_on_pid"]) if data.get("waiting_on_pid") else None),
            waiting_on_session=(str(data["waiting_on_session"]) if data.get("waiting_on_session") else None),
            waiting_reason=data.get("waiting_reason"),
            contract=GoalContract.from_dict(data.get("contract")),
            gates=[
                GoalGate.from_dict(g) for g in (data.get("gates") or [])
                if isinstance(g, dict) and str(g.get("command") or "").strip()
            ],
            **ints, **floats,
        )


# field -> (coercer, default used when the stored value is missing/falsy)
_STATE_FIELDS = {
    "prompt": (str, ""), "interval_seconds": (int, 0), "status": (str, "active"),
    "created_at": (float, 0.0), "last_fired_at": (float, 0.0), "fire_count": (int, 0),
}


@dataclass
class HeartbeatState:
    """Serializable per-session heartbeat."""

    prompt: str
    interval_seconds: int
    status: str = "active"          # active | paused | cleared
    created_at: float = 0.0
    last_fired_at: float = 0.0
    fire_count: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> HeartbeatState:
        data = json.loads(raw)
        return cls(**{name: coerce(data.get(name) or default) for name, (coerce, default) in _STATE_FIELDS.items()})


# Backstop tick budget so an unattended loop can't run forever. 0 = unlimited; config loops.max_ticks.
DEFAULT_MAX_TICKS = 100


@dataclass
class LoopState:
    """Serializable /loop state stored per session."""

    prompt: str
    status: str = "active"            # active | paused | done | cleared
    mode: str = "interval"            # interval | self_paced
    interval_seconds: float = 0.0     # fixed cadence (mode == "interval")
    current_delay: float = 0.0        # live cadence (self-paced backoff)
    times: int = 0                    # user cap (--times N); 0 = none
    until: str = ""                   # judged stop condition; "" = none
    max_ticks: int = DEFAULT_MAX_TICKS  # config backstop; 0 = unlimited
    ticks_fired: int = 0
    created_at: float = 0.0
    last_fired_at: float = 0.0
    next_due_at: float = 0.0
    # True between "wakeup injected" and "that turn's response evaluated": stops a tick from
    # double-firing mid-turn and tells the post-turn hook the turn that just ended was ours.
    awaiting_response: bool = False
    last_response_digest: str = ""    # self-paced change detection
    paused_reason: Optional[str] = None
    last_stop_reason: Optional[str] = None
    # Gateway routing (platform / chat_id / chat_type / thread_id) captured at creation so the
    # idle watcher can inject ticks into the right chat. Empty for CLI/TUI (own schedulers).
    route: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> LoopState:
        data = json.loads(raw)
        route = data.get("route")
        kwargs: dict[str, Any] = {
            "prompt": data.get("prompt", ""),
            "status": data.get("status", "active"),
            "mode": data.get("mode", "interval"),
            "paused_reason": data.get("paused_reason"),
            "last_stop_reason": data.get("last_stop_reason"),
            "route": route if isinstance(route, dict) else {},
        }
        # Remaining scalar fields: missing key -> dataclass default; present-but-falsy -> type zero.
        casts = {"str": str, "int": int, "float": float, "bool": bool}
        for f in fields(cls):
            if f.name not in kwargs:
                kwargs[f.name] = casts[f.type](data.get(f.name, f.default) or casts[f.type]())
        return cls(**kwargs)


#: The text a completion folded into a later batch keeps: its output rides the batch.
FOLDED_COMPLETION_TEXT = '[folded into a later completion batch]'


def automation_descriptor(payload: dict) -> dict:
    """The automation descriptor an admission payload carries (local or native-text), ``{}`` for none."""
    return payload.get('local_automation_v1') or (payload.get('native_text_v1') or {}).get('automation') or {}


def folded_payload(payload: dict) -> dict:
    """*payload* (a decoded copy) of a completion a later batch folds in: it keeps only its own identity, its text
    and earlier identities riding the batch, so no fold stores them twice."""
    payload['text'] = FOLDED_COMPLETION_TEXT
    automation_descriptor(payload).pop('identities', None)
    return payload
