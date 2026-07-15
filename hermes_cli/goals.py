"""Persistent session goals — the Ralph loop for Hermes.

A goal is a free-form user objective that stays active across turns. The
primary model records its structured outcome through the todo tool. Hermes
then continues mechanically until that model reports verified completion,
an optional owner-configured turn budget is exhausted, or the user
pauses/clears the goal. A budget of ``0`` means no automatic cross-turn
pause; per-turn model, tool, permission, and resource boundaries still apply.

State is persisted in SessionDB's ``state_meta`` table keyed by
``goal:<session_id>`` so ``/resume`` picks it up.

Design notes / invariants:

- The continuation prompt is just a normal user message appended to the
  session via ``run_conversation``. No system-prompt mutation, no toolset
  swap — prompt caching stays intact.
- Missing model-authored outcome state is fail-OPEN to ``continue``. Runtime
  bookkeeping must never infer completion from response text.
- When a real user message arrives mid-loop it preempts the continuation
  prompt. The primary model can record the new structured outcome in that
  same turn.
- This module has zero hard dependency on ``cli.HermesCLI`` or the gateway
  runner — both wire the same ``GoalManager`` in.

Nothing in this module touches the agent's system prompt or toolset.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20


CONTINUATION_PROMPT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Continue working toward this goal. Take the next concrete step. "
    "Before ending this turn, call the todo tool with goal_outcome: continue "
    "unless you have concrete verification, complete only with that evidence, "
    "or blocked only after every safe available approach is exhausted and "
    "specific user or external input is genuinely required."
)

# Used when the goal carries a structured completion contract. The contract
# block tells the agent exactly what "done" means, how to prove it, what not
# to break, what's in scope, and when to stop and ask — so it targets the
# verification surface instead of declaring victory loosely.
CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Completion contract:\n"
    "{contract_block}\n\n"
    "Continue working toward the outcome above. Take the next concrete step. "
    "Stay within the stated boundaries and do not violate the constraints. "
    "Before claiming the goal is done, satisfy the Verification criterion and "
    "show the concrete evidence (command output, file contents, test result). "
    "If you hit the stated stop condition or are otherwise blocked and need "
    "user input, say so clearly. Before ending this turn, call the todo tool "
    "with goal_outcome: continue, complete, or blocked under those same rules."
)

# Used when the user has added one or more /subgoal criteria. Surfaced
# to the agent verbatim so it sees what to target on the next turn.
CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Additional criteria the user added mid-loop:\n"
    "{subgoals_block}\n\n"
    "Continue working toward the goal AND all additional criteria. Take "
    "the next concrete step. If you believe the goal and every "
    "additional criterion are complete, verify that result. Before ending "
    "this turn, call the todo tool with goal_outcome: continue unless verified, "
    "complete only with concrete evidence, or blocked only after every safe "
    "available approach is exhausted."
)


PRIMARY_MODEL_GOAL_KICKOFF_PROMPT_TEMPLATE = (
    "[Begin working toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Create or update a concrete todo plan when the work has multiple steps, "
    "then execute the next safe steps now. If one approach fails, revise the "
    "plan and try the remaining safe approaches. Before ending this turn, call "
    "the todo tool with goal_outcome: continue unless you have concrete "
    "verification, complete only with that evidence, or blocked only after "
    "every safe available approach is exhausted and specific user or external "
    "input is genuinely required."
)


PRIMARY_MODEL_GOAL_KICKOFF_WITH_CONTRACT_TEMPLATE = (
    "[Begin working toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Completion contract:\n"
    "{contract_block}\n\n"
    "Create or update a concrete todo plan, then execute the next safe steps "
    "within this contract now. Before ending this turn, call the todo tool with "
    "goal_outcome: continue unless the Verification criterion is satisfied, "
    "complete only with concrete evidence, or blocked only after every safe "
    "available approach is exhausted and the stop condition genuinely applies."
)


PRIMARY_MODEL_DRAFT_PROMPT_TEMPLATE = (
    "[Author your standing-goal workspace]\n"
    "Goal: {goal}\n\n"
    "Using your full task context, create the concrete step-by-step todo plan "
    "and author a structured goal_contract through the todo tool. The contract "
    "must define outcome, verification, constraints, boundaries, and the true "
    "condition that requires human input. Then begin the first safe concrete "
    "step immediately. Before ending the turn, also record goal_outcome through "
    "the todo tool. You are the sole semantic authority: do not delegate "
    "planning or completion judgment to an auxiliary model."
)


# ──────────────────────────────────────────────────────────────────────
# Completion contract
# ──────────────────────────────────────────────────────────────────────

# The five contract fields, in display order. Adapted from OpenAI Codex's
# "strong goal" guidance: a durable objective works best when it names what
# "done" means, how to prove it, what must not regress, what tools/paths are
# in bounds, and when to stop and ask. A bare free-form goal (no contract)
# stays fully supported — every field defaults empty and is simply omitted
# from the prompts when unset.
_CONTRACT_FIELDS = ("outcome", "verification", "constraints", "boundaries", "stop_when")

# Human labels for rendering and for the inline `field: value` parser.
_CONTRACT_LABELS = {
    "outcome": "Outcome",
    "verification": "Verification",
    "constraints": "Constraints",
    "boundaries": "Boundaries",
    "stop_when": "Stop when blocked",
}

# Inline-input aliases the user may type before a value, mapped to the
# canonical field name. e.g. `verify: tests pass` or `done when: ...`.
_CONTRACT_ALIASES = {
    "outcome": "outcome",
    "goal": "outcome",
    "done": "outcome",
    "done when": "outcome",
    "verification": "verification",
    "verify": "verification",
    "verified by": "verification",
    "evidence": "verification",
    "proof": "verification",
    "constraints": "constraints",
    "constraint": "constraints",
    "preserve": "constraints",
    "must not": "constraints",
    "do not change": "constraints",
    "boundaries": "boundaries",
    "boundary": "boundaries",
    "scope": "boundaries",
    "allowed": "boundaries",
    "files": "boundaries",
    "stop when": "stop_when",
    "stop_when": "stop_when",
    "blocked": "stop_when",
    "stop if blocked": "stop_when",
    "give up when": "stop_when",
}


@dataclass
class GoalContract:
    """Optional structured completion contract for a goal.

    Each field is free-form prose supplied by the user or authored by the
    primary model through the todo tool. Empty fields are omitted everywhere
    — a goal with no contract behaves exactly like the original free-form
    goal. The contract is woven into the continuation prompt so the primary
    model targets the verification surface and respects constraints.
    """

    outcome: str = ""
    verification: str = ""
    constraints: str = ""
    boundaries: str = ""
    stop_when: str = ""

    def is_empty(self) -> bool:
        return not any(getattr(self, f).strip() for f in _CONTRACT_FIELDS)

    def to_dict(self) -> Dict[str, str]:
        return {f: getattr(self, f) for f in _CONTRACT_FIELDS}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "GoalContract":
        if not isinstance(data, dict):
            return cls()
        return cls(**{f: str(data.get(f) or "").strip() for f in _CONTRACT_FIELDS})

    def render_block(self) -> str:
        """Render non-empty contract fields as a labelled block. Empty
        contract → empty string (callers skip the section entirely)."""
        lines = []
        for f in _CONTRACT_FIELDS:
            val = getattr(self, f).strip()
            if val:
                lines.append(f"- {_CONTRACT_LABELS[f]}: {val}")
        return "\n".join(lines)


def parse_contract(text: str) -> Tuple[str, GoalContract]:
    """Split user-typed goal text into a headline + structured contract.

    Supports inline ``field: value`` lines so power users can type a full
    contract in one shot, e.g.::

        Migrate auth to JWT
        verify: the auth test suite passes
        constraints: keep the public /login response shape unchanged
        boundaries: only touch services/auth and its tests
        stop when: a schema change needs product sign-off

    The first non-field line(s) become the goal headline; recognized
    ``field:`` lines populate the contract. Lines for the same field are
    joined. Unrecognized prefixes stay part of the headline, so a plain
    free-form goal with an incidental colon (``Fix bug: the parser``)
    is NOT mangled — only lines whose prefix matches a known alias are
    pulled out. Returns ``(headline, contract)``.
    """
    if not text:
        return "", GoalContract()

    headline_parts: List[str] = []
    fields: Dict[str, List[str]] = {f: [] for f in _CONTRACT_FIELDS}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        matched = False
        if ":" in line:
            prefix, _, value = line.partition(":")
            key = _CONTRACT_ALIASES.get(prefix.strip().lower())
            if key is not None and value.strip():
                fields[key].append(value.strip())
                matched = True
        if not matched:
            headline_parts.append(line)

    headline = " ".join(headline_parts).strip()
    contract = GoalContract(
        **{f: " ".join(v).strip() for f, v in fields.items()}
    )
    # If a headline was given but no explicit `outcome:` field, the headline
    # IS the outcome — don't duplicate it into the contract block (the goal
    # text already carries it), so leave outcome empty in that case.
    return headline, contract


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class GoalState:
    """Serializable goal state stored per session."""

    goal: str
    status: str = "active"          # active | paused | done | cleared
    turns_used: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: Optional[str] = None        # model-owned structured outcome
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None       # why we auto-paused (budget, etc.)
    generation_id: str = ""
    active_model_turn_id: Optional[str] = None
    pending_model_outcome: Optional[str] = None
    pending_model_reason: Optional[str] = None
    pending_model_turn_id: Optional[str] = None
    pending_model_generation_id: Optional[str] = None
    # User-added criteria appended mid-loop via the /subgoal command.
    # The continuation prompt includes them so the primary model treats them
    # as additional completion criteria. Backwards-compatible: defaults to
    # empty so old state_meta rows load unchanged.
    subgoals: List[str] = field(default_factory=list)
    # Wait barrier: when the agent is blocked on long-running async work
    # (CI poller, build, test run, deploy, rate-limit cooldown) the goal loop
    # PARKS instead of being re-poked every turn into busy-work. Two barrier
    # kinds, set explicitly through ``/goal wait``:
    #   • ``waiting_on_pid`` — park until that process exits.
    #   • ``waiting_on_session`` — park until that process_registry session's
    #     OWN trigger fires: it exits, OR (if it has watch_patterns) its
    #     pattern matches. Covers long-lived watchers/servers that signal
    #     mid-run via a trigger and may never exit. Preferred over raw pid
    #     when the agent set up a watch_patterns/notify_on_complete process.
    #   • ``waiting_until``  — park until this wall-clock epoch (time backoff).
    # While ANY is active, ``evaluate_after_turn`` short-circuits to
    # should_continue=False without burning a turn. The barrier auto-clears
    # when the pid exits / the trigger fires / the deadline passes, then the
    # next turn resumes normal model-authored outcome handling. Cleared by that,
    # ``/goal unwait``, pause, resume, or clear. Backwards-compatible: old
    # state_meta rows load with no barrier.
    waiting_on_pid: Optional[int] = None
    waiting_on_session: Optional[str] = None
    waiting_until: float = 0.0
    waiting_reason: Optional[str] = None
    waiting_since: float = 0.0
    # Optional structured completion contract (outcome / verification /
    # constraints / boundaries / stop_when). Empty by default; a goal with
    # no contract behaves exactly like the original free-form goal.
    contract: GoalContract = field(default_factory=GoalContract)

    def to_json(self) -> str:
        data = asdict(self)
        # asdict already recursed GoalContract into a plain dict.
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "GoalState":
        data = json.loads(raw)
        raw_max_turns = data.get("max_turns")
        max_turns = (
            DEFAULT_MAX_TURNS
            if raw_max_turns is None or isinstance(raw_max_turns, bool)
            else int(raw_max_turns)
        )
        if max_turns < 0:
            max_turns = DEFAULT_MAX_TURNS
        raw_subgoals = data.get("subgoals") or []
        subgoals: List[str] = []
        if isinstance(raw_subgoals, list):
            subgoals = [str(s).strip() for s in raw_subgoals if str(s).strip()]
        return cls(
            goal=data.get("goal", ""),
            status=data.get("status", "active"),
            turns_used=int(data.get("turns_used", 0) or 0),
            max_turns=max_turns,
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_turn_at=float(data.get("last_turn_at", 0.0) or 0.0),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
            generation_id=str(data.get("generation_id") or ""),
            active_model_turn_id=data.get("active_model_turn_id"),
            pending_model_outcome=data.get("pending_model_outcome"),
            pending_model_reason=data.get("pending_model_reason"),
            pending_model_turn_id=data.get("pending_model_turn_id"),
            pending_model_generation_id=data.get("pending_model_generation_id"),
            subgoals=subgoals,
            waiting_on_pid=(int(data["waiting_on_pid"]) if data.get("waiting_on_pid") else None),
            waiting_on_session=(str(data["waiting_on_session"]) if data.get("waiting_on_session") else None),
            waiting_until=float(data.get("waiting_until", 0.0) or 0.0),
            waiting_reason=data.get("waiting_reason"),
            waiting_since=float(data.get("waiting_since", 0.0) or 0.0),
            contract=GoalContract.from_dict(data.get("contract")),
        )

    # --- contract helpers -------------------------------------------------

    def has_contract(self) -> bool:
        return self.contract is not None and not self.contract.is_empty()

    # --- subgoals helpers -------------------------------------------------

    def render_subgoals_block(self) -> str:
        """Render the subgoals as a numbered ``- N. text`` block. Empty
        when no subgoals exist."""
        if not self.subgoals:
            return ""
        return "\n".join(f"- {i}. {text}" for i, text in enumerate(self.subgoals, start=1))


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────


def _meta_key(session_id: str) -> str:
    return f"goal:{session_id}"


_DB_CACHE: Dict[str, Any] = {}


def _get_session_db() -> Optional[Any]:
    """Return a SessionDB instance for the current HERMES_HOME.

    SessionDB has no built-in singleton, but opening a new connection per
    /goal call would thrash the file. We cache one instance per
    ``hermes_home`` path so profile switches still pick up the right DB.
    Defensive against import/instantiation failures so tests and
    non-standard launchers can still use the GoalManager.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:  # pragma: no cover
        logger.debug("GoalManager: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


def load_goal(session_id: str) -> Optional[GoalState]:
    """Load the goal for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        logger.debug("GoalManager: get_meta failed: %s", exc)
        return None
    if not raw:
        return None
    try:
        return GoalState.from_json(raw)
    except Exception as exc:
        logger.warning("GoalManager: could not parse stored goal for %s: %s", session_id, exc)
        return None


def save_goal(session_id: str, state: GoalState) -> None:
    """Persist a goal to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("GoalManager: set_meta failed: %s", exc)


def _clear_model_turn_authority(state: GoalState) -> None:
    """Clear all authority and pending writes associated with one model turn."""

    state.active_model_turn_id = None
    state.pending_model_outcome = None
    state.pending_model_reason = None
    state.pending_model_turn_id = None
    state.pending_model_generation_id = None


def _atomic_mutate_goal(
    session_id: str,
    fallback_state: Optional[GoalState],
    mutate: Callable[
        [Optional[GoalState]],
        Tuple[Optional[GoalState], Any, bool],
    ],
) -> Tuple[Optional[GoalState], Any]:
    """Read, validate, and write one goal row in a single DB transaction.

    Goal outcomes can be written by tool-worker threads while CLI/TUI/gateway
    code holds an older ``GoalManager`` instance.  A plain load-then-save would
    let either side overwrite the other's fields.  SessionDB's write primitive
    starts ``BEGIN IMMEDIATE`` and retries cross-process contention, so using it
    here makes the durable row -- not any manager cache -- authoritative.

    ``mutate`` returns ``(state, result, changed)``.  The fallback preserves the
    historical in-memory behavior for unusual launchers where SessionDB is not
    available, while production always takes the transactional path.
    """

    db = _get_session_db()
    execute_write = getattr(db, "_execute_write", None) if db is not None else None
    if not callable(execute_write):
        state, result, _changed = mutate(fallback_state)
        return state, result

    key = _meta_key(session_id)

    def _do(conn):
        row = conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (key,)
        ).fetchone()
        raw = None if row is None else row[0]
        state: Optional[GoalState] = None
        if raw:
            try:
                state = GoalState.from_json(raw)
            except Exception as exc:
                logger.warning(
                    "GoalManager: could not parse stored goal for %s: %s",
                    session_id,
                    exc,
                )
        state, result, changed = mutate(state)
        if changed and state is not None:
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, state.to_json()),
            )
        return state, result

    return execute_write(_do)


def clear_goal(session_id: str) -> None:
    """Mark a goal cleared in the DB (preserved for audit, status=cleared)."""
    state = load_goal(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_goal(session_id, state)


def migrate_goal_to_session(old_session_id: str, new_session_id: str, *, reason: str = "") -> bool:
    """Carry a persistent /goal from a parent session to its continuation.

    Context compression rotates ``session_id`` to a fresh child session,
    but ``load_goal`` does a flat ``goal:<session_id>`` lookup with no
    parent-lineage walk — so an active goal silently dies at the
    compaction boundary (#33618). Copy the goal onto the new session and
    archive the old row as ``cleared`` so exactly one active goal row
    exists per logical conversation (avoids the "two active goals"
    hazard of a pure copy).

    Returns True when a goal was migrated, False when there was nothing
    to migrate or the DB was unavailable. Best-effort and never raises —
    a failure here must not block compression.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        db = _get_session_db()
        execute_write = getattr(db, "_execute_write", None) if db is not None else None
        if not callable(execute_write):
            return False

        old_key = _meta_key(old_session_id)
        new_key = _meta_key(new_session_id)

        def _do(conn) -> bool:
            old_row = conn.execute(
                "SELECT value FROM state_meta WHERE key = ?", (old_key,)
            ).fetchone()
            if old_row is None or not old_row[0]:
                return False
            state = GoalState.from_json(old_row[0])
            if state.status == "cleared":
                return False
            # Don't clobber a goal already set on the child (e.g. a resumed
            # lineage that re-established its own goal).
            if conn.execute(
                "SELECT 1 FROM state_meta WHERE key = ?", (new_key,)
            ).fetchone() is not None:
                return False

            # A compression/session migration is a new authority generation.
            # A worker from the parent turn must be rejected on BOTH rows even
            # if it finishes after the child session becomes current.
            state.generation_id = uuid.uuid4().hex
            _clear_model_turn_authority(state)
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?)",
                (new_key, state.to_json()),
            )

            parent = GoalState.from_json(old_row[0])
            parent.status = "cleared"
            _clear_model_turn_authority(parent)
            conn.execute(
                "UPDATE state_meta SET value = ? WHERE key = ?",
                (parent.to_json(), old_key),
            )
            return True

        migrated = bool(execute_write(_do))
        if not migrated:
            return False
        logger.debug(
            "GoalManager: migrated goal %s -> %s (%s)",
            old_session_id, new_session_id, reason or "rotation",
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("GoalManager: goal migration failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────
# Mechanical wait-barrier checks
# ──────────────────────────────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` is currently alive.

    Delegates to ``gateway.status._pid_exists`` — the canonical,
    cross-platform, footgun-safe liveness check (psutil with a ctypes /
    POSIX fallback). Critically this avoids ``os.kill(pid, 0)``, which on
    Windows is NOT a no-op: it routes to ``CTRL_C_EVENT`` and hard-kills the
    target's console process group (bpo-14484). Any error resolves to False
    (treat unknown as dead) so a stale barrier never wedges the loop — the
    worst case is the goal resumes one turn early, which is safe.
    """
    if not pid or pid <= 0:
        return False
    try:
        from gateway.status import _pid_exists

        return bool(_pid_exists(int(pid)))
    except Exception:
        pass
    # Last-resort fallback if gateway.status is unavailable: psutil directly.
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        return False


def _session_waiting(session_id: str) -> bool:
    """Whether a goal parked on a process_registry session should stay parked.

    Delegates to ``process_registry.is_session_waiting`` — True while the
    session is running and (if it has watch_patterns) its trigger hasn't fired.
    Fail-safe: any import/registry error yields False (don't wait) so a stale
    barrier can never wedge the loop.
    """
    if not session_id:
        return False
    try:
        from tools.process_registry import process_registry

        return bool(process_registry.is_session_waiting(session_id))
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# GoalManager — the orchestration surface CLI + gateway talk to
# ──────────────────────────────────────────────────────────────────────


class GoalManager:
    """Per-session goal state + continuation decisions.

    The CLI and gateway each hold one ``GoalManager`` per live session.

    Methods:

    - ``set(goal)`` — start a new standing goal.
    - ``clear()`` — remove the active goal.
    - ``pause()`` / ``resume()`` — explicit user controls.
    - ``status()`` — printable one-liner.
    - ``evaluate_after_turn(last_response)`` — apply the primary model's
      structured outcome and return the next mechanical decision.
    - ``next_continuation_prompt()`` — the canonical user-role message to
      feed back into ``run_conversation``.
    """

    def __init__(self, session_id: str, *, default_max_turns: int = DEFAULT_MAX_TURNS):
        self.session_id = session_id
        if isinstance(default_max_turns, bool):
            raise ValueError("default_max_turns must be a non-negative integer")
        self.default_max_turns = int(default_max_turns)
        if self.default_max_turns < 0:
            raise ValueError("default_max_turns must be non-negative")
        self._state: Optional[GoalState] = load_goal(session_id)
        if self._state is not None and not self._state.generation_id:
            # One-time mechanical migration for pre-generation goal rows. Any
            # old pending outcome lacks provenance and is therefore discarded.
            self._state.generation_id = uuid.uuid4().hex
            self._clear_pending_model_outcome()
            save_goal(self.session_id, self._state)

    def _clear_pending_model_outcome(self) -> None:
        if self._state is None:
            return
        self._state.pending_model_outcome = None
        self._state.pending_model_reason = None
        self._state.pending_model_turn_id = None
        self._state.pending_model_generation_id = None

    def _mutate_durable(
        self,
        mutate: Callable[
            [Optional[GoalState]],
            Tuple[Optional[GoalState], Any, bool],
        ],
    ) -> Any:
        """Apply a read/validate/write mutation against the durable row."""

        state, result = _atomic_mutate_goal(
            self.session_id,
            self._state,
            mutate,
        )
        self._state = state
        return result

    def _reload_durable(self) -> Optional[GoalState]:
        self._state = load_goal(self.session_id)
        return self._state

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[GoalState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def has_contract(self) -> bool:
        return self._state is not None and self._state.has_contract()

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status in {"cleared",}:
            return "No active goal. Set one with /goal <text>."
        turns = (
            f"{s.turns_used} turns, no automatic turn cap"
            if s.max_turns == 0
            else f"{s.turns_used}/{s.max_turns} turns"
        )
        sub = f", {len(s.subgoals)} subgoal{'s' if len(s.subgoals) != 1 else ''}" if s.subgoals else ""
        con = ", contract" if self.has_contract() else ""
        meta = f"{turns}{sub}{con}"
        if s.status == "active":
            if s.waiting_on_session and _session_waiting(s.waiting_on_session):
                wr = s.waiting_reason or f"session {s.waiting_on_session}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_on_pid and _pid_alive(s.waiting_on_pid):
                wr = s.waiting_reason or f"pid {s.waiting_on_pid}"
                return f"⏳ Goal (parked on {wr}, {meta}): {s.goal}"
            if s.waiting_until and time.time() < s.waiting_until:
                remaining = int(s.waiting_until - time.time())
                wr = s.waiting_reason or f"{remaining}s"
                return f"⏳ Goal (parked {remaining}s — {wr}, {meta}): {s.goal}"
            return f"⊙ Goal (active, {meta}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {meta}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({meta}): {s.goal}"
        return f"Goal ({s.status}, {meta}): {s.goal}"

    # --- mutation -----------------------------------------------------

    def set(self, goal: str, *, max_turns: Optional[int] = None, contract: Optional[GoalContract] = None) -> GoalState:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal text is empty")
        if isinstance(max_turns, bool):
            raise ValueError("max_turns must be a non-negative integer")
        resolved_max_turns = (
            self.default_max_turns
            if max_turns is None
            else int(max_turns)
        )
        if resolved_max_turns < 0:
            raise ValueError("max_turns must be non-negative")
        state = GoalState(
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=resolved_max_turns,
            created_at=time.time(),
            last_turn_at=0.0,
            generation_id=uuid.uuid4().hex,
            contract=contract if contract is not None else GoalContract(),
        )
        self._state = state
        save_goal(self.session_id, state)
        return state

    def set_contract(self, contract: GoalContract) -> Optional[GoalState]:
        """Attach or replace the completion contract on the active goal.

        Returns the updated state, or None when there is no goal to attach to.
        """
        def _mutate(state):
            if state is None:
                return state, None, False
            state.contract = contract or GoalContract()
            return state, state, True

        return self._mutate_durable(_mutate)

    def pause(self, reason: str = "user-paused") -> Optional[GoalState]:
        def _mutate(state):
            if state is None:
                return state, None, False
            state.status = "paused"
            state.paused_reason = reason
            # A wait barrier is meaningless once paused — drop it.
            state.waiting_on_pid = None
            state.waiting_on_session = None
            state.waiting_until = 0.0
            state.waiting_reason = None
            state.waiting_since = 0.0
            _clear_model_turn_authority(state)
            return state, state, True

        return self._mutate_durable(_mutate)

    def resume(self, *, reset_budget: bool = True) -> Optional[GoalState]:
        def _mutate(state):
            if state is None:
                return state, None, False
            state.status = "active"
            state.paused_reason = None
            # Resuming starts fresh — clear any stale barrier.
            state.waiting_on_pid = None
            state.waiting_on_session = None
            state.waiting_until = 0.0
            state.waiting_reason = None
            state.waiting_since = 0.0
            # Resume is a fresh authority generation. A worker from before the
            # pause can no longer mutate this resumed goal.
            state.generation_id = uuid.uuid4().hex
            _clear_model_turn_authority(state)
            if reset_budget:
                state.turns_used = 0
            return state, state, True

        return self._mutate_durable(_mutate)

    def clear(self) -> None:
        def _mutate(state):
            if state is None:
                return state, False, False
            state.status = "cleared"
            _clear_model_turn_authority(state)
            return state, True, True

        self._mutate_durable(_mutate)
        self._state = None

    def mark_done(self, reason: str) -> None:
        def _mutate(state):
            if state is None:
                return state, None, False
            state.status = "done"
            state.last_verdict = "done"
            state.last_reason = reason
            _clear_model_turn_authority(state)
            return state, None, True

        self._mutate_durable(_mutate)

    def begin_model_turn(self, turn_id: str) -> str:
        """Bind a fresh model turn to the current goal generation.

        Any unconsumed outcome from an interrupted/empty prior turn is cleared
        before the new turn starts. The returned generation id must accompany
        model-authored goal writes from this turn.
        """

        turn_id = str(turn_id or "").strip()
        if not turn_id:
            raise ValueError("goal model turn requires a turn id")

        def _mutate(state):
            if state is None or state.status != "active":
                return state, "", False
            if not state.generation_id:
                state.generation_id = uuid.uuid4().hex
            # Starting a new turn revokes every unconsumed write from the prior
            # turn, while preserving the standing goal generation itself.
            _clear_model_turn_authority(state)
            state.active_model_turn_id = turn_id
            return state, state.generation_id, True

        return str(self._mutate_durable(_mutate) or "")

    def abandon_model_turn(
        self,
        *,
        originating_turn_id: str,
        goal_generation_id: str,
    ) -> bool:
        """Revoke one exact turn without applying or inferring an outcome."""

        turn_id = str(originating_turn_id or "").strip()
        generation_id = str(goal_generation_id or "").strip()

        def _mutate(state):
            if (
                state is None
                or state.status != "active"
                or not turn_id
                or not generation_id
                or state.generation_id != generation_id
                or state.active_model_turn_id != turn_id
            ):
                return state, False, False
            _clear_model_turn_authority(state)
            return state, True, True

        return bool(self._mutate_durable(_mutate))

    def record_model_outcome(
        self,
        outcome: str,
        reason: str,
        *,
        originating_turn_id: str,
        goal_generation_id: str,
    ) -> bool:
        """Persist the primary model's structured decision for this turn.

        This deliberately contains no semantic inference.  The model chooses
        the outcome through the todo tool; the runtime validates the enum and
        later applies it mechanically.
        """
        turn_id = str(originating_turn_id or "").strip()
        generation_id = str(goal_generation_id or "").strip()
        outcome = str(outcome or "").strip().lower()
        if outcome not in {"continue", "complete", "blocked"}:
            raise ValueError("goal outcome must be continue, complete, or blocked")
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("goal outcome requires a reason")

        def _mutate(state):
            if (
                state is None
                or state.status != "active"
                or not turn_id
                or not generation_id
                or generation_id != state.generation_id
                or turn_id != state.active_model_turn_id
            ):
                return state, False, False
            state.pending_model_outcome = outcome
            state.pending_model_reason = reason
            state.pending_model_turn_id = turn_id
            state.pending_model_generation_id = generation_id
            return state, True, True

        return bool(self._mutate_durable(_mutate))

    def record_model_contract(
        self,
        value: Dict[str, Any],
        *,
        originating_turn_id: str,
        goal_generation_id: str,
    ) -> bool:
        """Persist a completion contract authored by the primary model.

        The runtime performs schema conversion only; it does not invent,
        rewrite, rank, or complete any semantic field.
        """
        turn_id = str(originating_turn_id or "").strip()
        generation_id = str(goal_generation_id or "").strip()
        if not isinstance(value, dict):
            raise ValueError("goal contract must be an object")
        if set(value) - set(_CONTRACT_FIELDS):
            raise ValueError("goal contract contains unknown fields")
        contract = GoalContract.from_dict(value)
        if contract.is_empty():
            raise ValueError("goal contract must contain at least one field")

        def _mutate(state):
            if (
                state is None
                or state.status != "active"
                or not turn_id
                or not generation_id
                or generation_id != state.generation_id
                or turn_id != state.active_model_turn_id
            ):
                return state, False, False
            state.contract = contract
            return state, True, True

        return bool(self._mutate_durable(_mutate))

    # --- /subgoal user controls ---------------------------------------

    def add_subgoal(self, text: str) -> str:
        """Append a user-added criterion to the active goal. Requires
        ``has_goal()``; raises ``RuntimeError`` otherwise.

        Returns the cleaned text so the caller can show it back to the user.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("subgoal text is empty")

        def _mutate(state):
            if state is None or state.status not in {"active", "paused"}:
                raise RuntimeError("no active goal")
            state.subgoals.append(text)
            return state, text, True

        return str(self._mutate_durable(_mutate))

    def remove_subgoal(self, index_1based: int) -> str:
        """Remove a subgoal by 1-based index. Returns the removed text."""
        idx = int(index_1based) - 1

        def _mutate(state):
            if state is None or state.status not in {"active", "paused"}:
                raise RuntimeError("no active goal")
            if idx < 0 or idx >= len(state.subgoals):
                raise IndexError(f"index out of range (1..{len(state.subgoals)})")
            removed = state.subgoals.pop(idx)
            return state, removed, True

        return str(self._mutate_durable(_mutate))

    def clear_subgoals(self) -> int:
        """Wipe all subgoals. Returns the previous count."""
        def _mutate(state):
            if state is None or state.status not in {"active", "paused"}:
                raise RuntimeError("no active goal")
            prev = len(state.subgoals)
            state.subgoals = []
            return state, prev, True

        return int(self._mutate_durable(_mutate))

    def render_subgoals(self) -> str:
        """Public helper for the /subgoal slash command."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.subgoals:
            return "(no subgoals — use /subgoal <text> to add criteria)"
        return self._state.render_subgoals_block()

    # --- /goal wait barrier -------------------------------------------

    def wait_on(self, pid: int, reason: str = "") -> GoalState:
        """Park the goal loop on a background process PID.

        While the PID is alive, ``evaluate_after_turn`` returns
        ``should_continue=False`` without burning a turn — the loop quiesces
        instead of re-poking the agent into busy work. The barrier auto-clears
        when the process exits. Requires an
        active goal. For a process with a watch_patterns/notify_on_complete
        trigger, prefer ``wait_on_session`` so a mid-run trigger (not just
        exit) releases the barrier.
        """
        pid = int(pid)
        if pid <= 0:
            raise ValueError("pid must be a positive integer")

        def _mutate(state):
            if state is None or state.status != "active":
                raise RuntimeError("no active goal to park")
            state.waiting_on_pid = pid
            state.waiting_on_session = None
            state.waiting_until = 0.0
            state.waiting_reason = (reason or "").strip() or None
            state.waiting_since = time.time()
            return state, state, True

        return self._mutate_durable(_mutate)

    def wait_on_session(self, session_id: str, reason: str = "") -> GoalState:
        """Park the goal loop on a process_registry session's OWN trigger.

        Unlike ``wait_on`` (which releases only on PID exit), this releases
        when the session's trigger fires: it exits, OR — if it was started
        with ``watch_patterns`` — its pattern matches. This is the right
        barrier for a long-lived watcher/server/poller that signals mid-run
        and may never exit. Requires an active goal.
        """
        session_id = str(session_id or "").strip()
        if not session_id:
            raise ValueError("session_id must be a non-empty string")


        def _mutate(state):
            if state is None or state.status != "active":
                raise RuntimeError("no active goal to park")
            state.waiting_on_session = session_id
            state.waiting_on_pid = None
            state.waiting_until = 0.0
            state.waiting_reason = (reason or "").strip() or None
            state.waiting_since = time.time()
            return state, state, True

        return self._mutate_durable(_mutate)

    def wait_for_seconds(self, seconds: int, reason: str = "") -> GoalState:
        """Park the goal loop until ``seconds`` from now have elapsed.

        Time-based counterpart to ``wait_on`` — for backoff / cooldown waits
        where there's no process to track (e.g. the agent is rate-limited).
        The barrier auto-clears once the deadline passes. Requires an active
        goal.
        """
        seconds = int(seconds)
        if seconds <= 0:
            raise ValueError("seconds must be a positive integer")
        now = time.time()

        def _mutate(state):
            if state is None or state.status != "active":
                raise RuntimeError("no active goal to park")
            state.waiting_on_pid = None
            state.waiting_on_session = None
            state.waiting_until = now + seconds
            state.waiting_reason = (reason or "").strip() or None
            state.waiting_since = now
            return state, state, True

        return self._mutate_durable(_mutate)

    def stop_waiting(self) -> bool:
        """Clear any active wait barrier (pid / session / time). Returns True
        if one was cleared."""
        def _mutate(state):
            if state is None:
                return state, False, False
            if (
                state.waiting_on_pid is None
                and state.waiting_on_session is None
                and not state.waiting_until
            ):
                return state, False, False
            state.waiting_on_pid = None
            state.waiting_on_session = None
            state.waiting_until = 0.0
            state.waiting_reason = None
            state.waiting_since = 0.0
            return state, True, True

        return bool(self._mutate_durable(_mutate))

    def is_waiting(self) -> bool:
        """True iff a barrier is set AND not yet satisfied.

        Session barrier: active until the process exits or its watch-pattern
        trigger fires. Pid barrier: active while the process is alive. Time
        barrier: active until the deadline passes. Side effect: a satisfied
        barrier is cleared here (lazy auto-clear) so the next evaluation
        resumes normal model-authored outcome handling.
        """
        s = self._state
        if s is None:
            return False
        if s.waiting_on_session is not None:
            if _session_waiting(s.waiting_on_session):
                return True
            self.stop_waiting()  # session exited or trigger fired
            return False
        if s.waiting_on_pid is not None:
            if _pid_alive(s.waiting_on_pid):
                return True
            self.stop_waiting()  # process gone
            return False
        if s.waiting_until:
            if time.time() < s.waiting_until:
                return True
            self.stop_waiting()  # deadline passed
            return False
        return False

    # --- the main entry point called after every turn -----------------

    def evaluate_after_turn(
        self,
        last_response: str,
        *,
        originating_turn_id: str,
        goal_generation_id: str,
        user_initiated: bool = True,
        background_processes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Apply the primary model's structured outcome and update goal state.

        Response prose and background-process metadata are intentionally not
        interpreted: they cannot make semantic goal decisions.  The turn and
        generation ids are mechanical authority.  Only the exact durable turn
        may consume its pending model-authored outcome.

        Decision keys:
          - ``status``: current goal status after update
          - ``should_continue``: bool — caller should fire another turn
          - ``continuation_prompt``: str or None
          - ``verdict``: "done" | "blocked" | "continue" | "waiting" |
            "inactive"
          - ``reason``: str
          - ``message``: user-visible one-liner to print/send
        """
        del last_response, user_initiated, background_processes
        turn_id = str(originating_turn_id or "").strip()
        generation_id = str(goal_generation_id or "").strip()

        def _mutate(state):
            if state is None or state.status != "active":
                return state, {
                    "status": state.status if state else None,
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "inactive",
                    "reason": "no active goal",
                    "message": "",
                }, False

            if (
                not turn_id
                or not generation_id
                or state.generation_id != generation_id
                or state.active_model_turn_id != turn_id
            ):
                return state, {
                    "status": state.status,
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "stale",
                    "reason": "turn authority does not match the active goal turn",
                    "message": "",
                }, False

            # A manually established wait barrier is a mechanical pause. It
            # does not burn a turn or ask any runtime component to interpret
            # intent. A satisfied barrier is cleared in this same transaction.
            waiting = False
            target = ""
            if state.waiting_on_session is not None:
                waiting = _session_waiting(state.waiting_on_session)
                target = f"session {state.waiting_on_session}"
            elif state.waiting_on_pid is not None:
                waiting = _pid_alive(state.waiting_on_pid)
                target = f"pid {state.waiting_on_pid}"
            elif state.waiting_until:
                waiting = time.time() < state.waiting_until
                remaining = max(0, int(state.waiting_until - time.time()))
                target = f"{remaining}s remaining"

            if waiting:
                reason = state.waiting_reason or target
                _clear_model_turn_authority(state)
                return state, {
                    "status": "active",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "waiting",
                    "reason": reason,
                    "message": f"⏳ Goal parked — waiting on {target}: {reason}",
                }, True
            if target:
                state.waiting_on_pid = None
                state.waiting_on_session = None
                state.waiting_until = 0.0
                state.waiting_reason = None
                state.waiting_since = 0.0

            state.turns_used += 1
            state.last_turn_at = time.time()

            # The primary model owns the decision through todo.goal_outcome.
            # Missing exact-turn state fails open to continuation; no response
            # text, keyword, or external classifier may infer completion.
            pending_is_exact = (
                state.pending_model_turn_id == turn_id
                and state.pending_model_generation_id == generation_id
            )
            model_outcome = (
                state.pending_model_outcome if pending_is_exact else None
            ) or "continue"
            reason = (
                state.pending_model_reason if pending_is_exact else None
            ) or "primary model has not recorded completion"
            _clear_model_turn_authority(state)

            if model_outcome not in {"continue", "complete", "blocked"}:
                model_outcome = "continue"
                reason = "invalid model-authored goal outcome; continuing"

            verdict = "done" if model_outcome == "complete" else model_outcome
            state.last_verdict = verdict
            state.last_reason = reason

            if verdict == "done":
                state.status = "done"
                return state, {
                    "status": "done",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "done",
                    "reason": reason,
                    "message": f"✓ Goal achieved: {reason}",
                }, True

            if verdict == "blocked":
                state.status = "paused"
                state.paused_reason = reason
                return state, {
                    "status": "paused",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "blocked",
                    "reason": reason,
                    "message": f"⏸ Goal blocked after model-exhausted approaches: {reason}",
                }, True

            if state.max_turns > 0 and state.turns_used >= state.max_turns:
                state.status = "paused"
                state.paused_reason = (
                    f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
                )
                return state, {
                    "status": "paused",
                    "should_continue": False,
                    "continuation_prompt": None,
                    "verdict": "continue",
                    "reason": reason,
                    "message": (
                        f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used. "
                        "Use /goal resume to keep going, or /goal clear to stop."
                    ),
                }, True

            progress = (
                f"{state.turns_used} turns; no automatic turn cap"
                if state.max_turns == 0
                else f"{state.turns_used}/{state.max_turns}"
            )
            return state, {
                "status": "active",
                "should_continue": True,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"↻ Continuing toward goal ({progress}): {reason}"
                ),
            }, True

        decision = self._mutate_durable(_mutate)
        if decision.get("should_continue"):
            decision["continuation_prompt"] = self.next_continuation_prompt()
        return decision

    def next_continuation_prompt(self) -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        # Contract takes priority: it carries the verification surface and
        # constraints the agent must target. Subgoals fold in as extra
        # criteria appended to the contract block.
        if self._state.has_contract():
            contract_block = self._state.contract.render_block()
            if self._state.subgoals:
                extra = "\n".join(
                    f"- Extra criterion {i}: {text}"
                    for i, text in enumerate(self._state.subgoals, start=1)
                )
                contract_block = f"{contract_block}\n{extra}"
            return CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE.format(
                goal=self._state.goal,
                contract_block=contract_block,
            )
        if self._state.subgoals:
            return CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
                goal=self._state.goal,
                subgoals_block=self._state.render_subgoals_block(),
            )
        return CONTINUATION_PROMPT_TEMPLATE.format(goal=self._state.goal)

    def next_kickoff_prompt(self) -> Optional[str]:
        """Return the primary-model first-turn prompt for the active goal."""
        if not self._state or self._state.status != "active":
            return None
        if self._state.has_contract():
            contract_block = self._state.contract.render_block()
            if self._state.subgoals:
                extra = "\n".join(
                    f"- Extra criterion {i}: {text}"
                    for i, text in enumerate(self._state.subgoals, start=1)
                )
                contract_block = f"{contract_block}\n{extra}"
            return PRIMARY_MODEL_GOAL_KICKOFF_WITH_CONTRACT_TEMPLATE.format(
                goal=self._state.goal,
                contract_block=contract_block,
            )
        return PRIMARY_MODEL_GOAL_KICKOFF_PROMPT_TEMPLATE.format(
            goal=self._state.goal
        )

    def render_contract(self) -> str:
        """Public helper for the /goal show + /goal draft slash commands."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.has_contract():
            return "(no completion contract — set one with /goal draft <objective> or inline field: value lines)"
        return self._state.contract.render_block()


# ──────────────────────────────────────────────────────────────────────
# Kanban worker goal loop
# ──────────────────────────────────────────────────────────────────────

# Continuation prompt fed back to a kanban goal-mode worker that has not
# yet completed/blocked its task. The card's own acceptance criteria are
# the goal — the worker already has the full task body in its first turn,
# so we keep this short and point it back at the lifecycle contract.
KANBAN_GOAL_CONTINUATION_TEMPLATE = (
    "[Continuing toward this open kanban task]\n"
    "Reason: {reason}\n\n"
    "Take the next concrete step toward completing the task. When the work "
    "is genuinely finished, call kanban_complete with a summary. If you are "
    "blocked and need human input, call kanban_block with a reason. Do not "
    "stop without calling one of them."
)


def run_kanban_goal_loop(
    *,
    task_id: str,
    goal_text: str,
    run_turn,
    task_status_fn,
    block_fn,
    max_turns: int = DEFAULT_MAX_TURNS,
    first_response: str = "",
    log=None,
) -> Dict[str, Any]:
    """Drive a kanban worker through a Ralph-style goal loop.

    The dispatcher spawns a goal-mode worker exactly like a normal worker
    (``hermes -p <profile> chat -q "work kanban task <id>"``). The worker's
    first turn has already run by the time this is called; ``first_response``
    is that turn's reply. From here we:

    1. Check whether the worker already terminated the task (called
       ``kanban_complete`` / ``kanban_block``). If so, stop — nothing to do.
    2. Otherwise feed a continuation prompt and run another turn IN THE SAME
       SESSION via ``run_turn``. Only the primary worker can close or block
       the task through its structured lifecycle tools.
    3. When the turn budget is exhausted and the worker still hasn't
       terminated the task, ``block_fn`` is invoked so the card lands in a
       sticky ``blocked`` state for human review (NOT a silent exit).

    This function performs NO SessionDB persistence — a worker process is
    ephemeral, so the turn budget lives in a local counter. It is fully
    decoupled from the CLI for testability: callers inject ``run_turn``
    (str -> str), ``task_status_fn`` (() -> str|None), and ``block_fn``
    (reason: str -> None).

    Returns a decision dict: ``{"outcome", "turns_used", "reason"}`` where
    outcome is one of ``"completed_by_worker"``, ``"blocked_budget"``,
    ``"blocked_by_worker"``, or ``"stopped"``.
    """

    def _log(msg: str) -> None:
        if log is not None:
            try:
                log(msg)
            except Exception:
                pass

    del goal_text, first_response

    max_turns = int(max_turns or DEFAULT_MAX_TURNS)
    if max_turns < 1:
        max_turns = DEFAULT_MAX_TURNS

    # The first turn already consumed one unit of budget.
    turns_used = 1
    while True:
        # Did the worker terminate the task itself this turn?
        try:
            status = task_status_fn()
        except Exception as exc:
            _log(f"kanban goal loop: status check failed ({exc}); stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": "status check failed"}

        if status == "done":
            _log(f"kanban goal loop: task {task_id} completed by worker after {turns_used} turn(s)")
            return {"outcome": "completed_by_worker", "turns_used": turns_used, "reason": "worker completed the task"}
        if status == "blocked":
            _log(f"kanban goal loop: task {task_id} blocked by worker after {turns_used} turn(s)")
            return {"outcome": "blocked_by_worker", "turns_used": turns_used, "reason": "worker blocked the task"}
        if status not in ("running", "ready"):
            # Reclaimed / archived / unexpected — let the dispatcher own it.
            _log(f"kanban goal loop: task {task_id} status={status!r}; stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": f"status={status}"}

        # Still open: only the primary worker can close or block its task via
        # the structured lifecycle tools.  No auxiliary model interprets its
        # prose or overrides its completion decision.
        reason = "task remains open; call kanban_complete or kanban_block"
        _log(f"kanban goal loop: turn {turns_used}/{max_turns} status={status}")
        prompt = KANBAN_GOAL_CONTINUATION_TEMPLATE.format(reason=reason)

        # Budget check BEFORE spending another turn.
        if turns_used >= max_turns:
            _log(f"kanban goal loop: task {task_id} exhausted {turns_used}/{max_turns} turns; blocking")
            try:
                block_fn(
                    f"Goal-mode worker exhausted its turn budget "
                    f"({turns_used}/{max_turns}) without completing the task. "
                    f"Last lifecycle state: {status}."
                )
            except Exception as exc:
                _log(f"kanban goal loop: block_fn failed ({exc})")
            return {"outcome": "blocked_budget", "turns_used": turns_used, "reason": "turn budget exhausted"}

        # Run another turn in the same session.
        try:
            run_turn(prompt)
        except Exception as exc:
            _log(f"kanban goal loop: run_turn failed ({exc}); stopping")
            return {"outcome": "stopped", "turns_used": turns_used, "reason": f"run_turn error: {type(exc).__name__}"}
        turns_used += 1


__all__ = [
    "GoalState",
    "GoalContract",
    "GoalManager",
    "parse_contract",
    "PRIMARY_MODEL_GOAL_KICKOFF_PROMPT_TEMPLATE",
    "PRIMARY_MODEL_GOAL_KICKOFF_WITH_CONTRACT_TEMPLATE",
    "PRIMARY_MODEL_DRAFT_PROMPT_TEMPLATE",
    "CONTINUATION_PROMPT_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_CONTRACT_TEMPLATE",
    "KANBAN_GOAL_CONTINUATION_TEMPLATE",
    "DEFAULT_MAX_TURNS",
    "load_goal",
    "save_goal",
    "clear_goal",
    "migrate_goal_to_session",
    "run_kanban_goal_loop",
]
