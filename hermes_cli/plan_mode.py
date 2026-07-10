"""Code-enforced PLAN MODE for Hermes.

Plan mode is a per-session state in which the agent may plan and inspect but
may NOT mutate: every tool in ``MUTATING_TOOL_NAMES`` is blocked, the only
exception being ``write_file``/``patch`` writing the plan markdown itself
(under ``.hermes/plans/``). The agent writes a plan, calls ``plan_ready`` to
request approval via a clarify, and only an explicit **Approve** lifts the
restriction. This is the enforcement layer under the prompt-level ``plan``
skill (skills/software-development/plan/SKILL.md).

State is persisted in SessionDB's ``state_meta`` table keyed by
``plan:<session_id>`` (same mechanism as ``hermes_cli/goals.py``) so it
survives ``/resume``, restarts, and compaction.

Enforcement is two layers, fail-closed:

1. **Toolset restriction** (``session_disabled_toolsets`` / ``apply_session_toolset_policy``):
   entering plan mode adds the mutating toolsets to the session's
   ``disabled_toolsets`` so they vanish from the model's view, and adds the
   ``plan`` toolset to ``enabled_toolsets`` — the latter both exposes the
   ``plan_ready`` tool and busts the gateway agent-cache signature (which
   keys on ``enabled_toolsets``) so the restriction takes effect on the next
   turn's agent rebuild.

2. **Dispatch guard** (``tool_block_reason``): the tool-executor calls this
   before running any tool. If the session is in ``planning`` /
   ``pending_approval`` and the tool is mutating (and not a plan-file write),
   it is blocked with a structured message the model can react to. If plan
   state cannot be read for a session that has a plan row, the guard
   fails CLOSED (blocks the mutating tool).

Nothing here touches the agent's system prompt.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

# Lifecycle statuses. ``planning`` and ``pending_approval`` are both
# RESTRICTED (mutations blocked); ``approved`` lifts the restriction.
STATUS_PLANNING = "planning"
STATUS_PENDING_APPROVAL = "pending_approval"
STATUS_APPROVED = "approved"
# ``off`` is the tombstone written by ``/plan exit`` — plan mode was left and
# the pending plan discarded. It is deliberately a persisted row (not a
# deleted one) so ``plan_mode: always`` does not immediately re-enter the
# session into planning after an explicit exit.
STATUS_OFF = "off"
_RESTRICTED_STATUSES = frozenset({STATUS_PLANNING, STATUS_PENDING_APPROVAL})

# The ``plan`` toolset (containing ``plan_ready``) is injected into
# ``enabled_toolsets`` while plan mode is active.
PLAN_TOOLSET = "plan"

# ``write_file``/``patch`` are mutating but must stay available so the agent
# can write the plan markdown; the dispatch guard allows them ONLY for paths
# inside the plans dir.
PLAN_FILE_TOOLS = frozenset({"write_file", "patch"})

# Where plans live (mirrors the plan skill's save location). Backend-aware
# relative path, so we match the segment anywhere in the target path.
PLANS_DIR_SEGMENT = ".hermes/plans"

# config.yaml top-level key. ``plan_mode: always`` starts new sessions in
# planning.
CONFIG_KEY = "plan_mode"
CONFIG_ALWAYS = "always"


# ──────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────


@dataclass
class PlanState:
    """Serializable plan-mode state stored per session."""

    status: str = STATUS_PLANNING           # planning | pending_approval | approved
    plan_path: Optional[str] = None         # path of the plan markdown, once written
    entered_at: float = 0.0
    entered_by: str = "user"                # user | config | agent
    approval_clarify_id: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "PlanState":
        data = json.loads(raw)
        return cls(
            status=str(data.get("status", STATUS_PLANNING) or STATUS_PLANNING),
            plan_path=data.get("plan_path"),
            entered_at=float(data.get("entered_at", 0.0) or 0.0),
            entered_by=str(data.get("entered_by", "user") or "user"),
            approval_clarify_id=data.get("approval_clarify_id"),
        )

    def is_restricted(self) -> bool:
        return self.status in _RESTRICTED_STATUSES


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta) — mirrors hermes_cli/goals.py
# ──────────────────────────────────────────────────────────────────────


def _meta_key(session_id: str) -> str:
    return f"plan:{session_id}"


_DB_CACHE: Dict[str, Any] = {}


def _get_session_db() -> Optional[Any]:
    """Return a cached SessionDB instance for the current HERMES_HOME.

    One instance per ``hermes_home`` path (profile switches pick up the right
    DB). Defensive against import/instantiation failures so tests and
    non-standard launchers still work.
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB

        home = str(get_hermes_home())
    except Exception as exc:  # pragma: no cover
        logger.debug("PlanManager: SessionDB bootstrap failed (%s)", exc)
        return None

    cached = _DB_CACHE.get(home)
    if cached is not None:
        return cached
    try:
        db = SessionDB()
    except Exception as exc:  # pragma: no cover
        logger.debug("PlanManager: SessionDB() raised (%s)", exc)
        return None
    _DB_CACHE[home] = db
    return db


# Sentinels distinguishing a clean absence from an unreadable row, so the
# dispatch guard can fail closed on the latter without punishing every
# session when the DB is simply unavailable.
_READ_NONE = "none"          # no plan row (or DB unavailable) → not in plan mode
_READ_ACTIVE = "active"      # plan row present and parsed
_READ_UNREADABLE = "unreadable"  # a plan row exists but could not be read/parsed


def _read_plan(session_id: str) -> Tuple[str, Optional[PlanState]]:
    """Faithful read used by the fail-closed dispatch guard.

    Returns ``(_READ_NONE, None)`` when there is no plan row (or the DB is
    unavailable — blocking every session on a momentary DB outage is a worse
    failure than the threat this guards), ``(_READ_ACTIVE, state)`` when a row
    parsed, and ``(_READ_UNREADABLE, None)`` when a row exists but its
    ``get_meta`` raised or its JSON would not parse (→ fail closed).
    """
    if not session_id:
        return _READ_NONE, None
    db = _get_session_db()
    if db is None:
        return _READ_NONE, None
    try:
        raw = db.get_meta(_meta_key(session_id))
    except Exception as exc:
        # The row may exist but the read itself failed — fail closed.
        logger.debug("PlanManager: get_meta failed: %s", exc)
        return _READ_UNREADABLE, None
    if not raw:
        return _READ_NONE, None
    try:
        return _READ_ACTIVE, PlanState.from_json(raw)
    except Exception as exc:
        logger.warning(
            "PlanManager: could not parse stored plan for %s: %s", session_id, exc
        )
        return _READ_UNREADABLE, None


def load_plan(session_id: str) -> Optional[PlanState]:
    """Load the plan state for a session, or None if none/unreadable."""
    _kind, state = _read_plan(session_id)
    return state


def save_plan(session_id: str, state: PlanState) -> None:
    """Persist plan state to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        return
    try:
        db.set_meta(_meta_key(session_id), state.to_json())
    except Exception as exc:
        logger.debug("PlanManager: set_meta failed: %s", exc)


def clear_plan(session_id: str) -> None:
    """Leave plan mode by writing the ``off`` tombstone.

    We keep an explicit row rather than deleting it so ``plan_mode: always``
    does not treat the session as brand-new and force it back into planning
    right after an intentional exit. ``off`` is not a restricted status, so
    the toolset policy and dispatch guard both fall through to unrestricted.
    """
    if not session_id:
        return
    save_plan(session_id, PlanState(status=STATUS_OFF, entered_at=time.time()))


# ──────────────────────────────────────────────────────────────────────
# Toolset policy (layer a)
# ──────────────────────────────────────────────────────────────────────


def _plan_blockable_tools() -> frozenset:
    """Mutating tools the dispatch guard blocks, minus the plan-file tools."""
    from agent.tool_guardrails import MUTATING_TOOL_NAMES

    return MUTATING_TOOL_NAMES - PLAN_FILE_TOOLS


def session_disabled_toolsets() -> List[str]:
    """Toolsets to disable while plan mode is active.

    Derived from ``toolsets.py`` rather than hardcoded: a toolset is disabled
    IFF every tool it resolves to is a blockable mutating tool (mutating and
    not a plan-file tool). Toolsets that mix in read-only tools (``file``,
    ``skills``, ``browser``) stay enabled; the dispatch guard does the
    per-call enforcement for their mutating members. Platform bundles
    (``hermes-*``) and posture toolsets are skipped — they re-list shared core
    tools without owning them, so whole-disabling them is meaningless (and the
    subtraction in ``model_tools`` special-cases them anyway).
    """
    try:
        from toolsets import TOOLSETS, get_toolset, resolve_toolset
    except Exception:  # pragma: no cover
        return []
    blockable = _plan_blockable_tools()
    out: List[str] = []
    for name in TOOLSETS:
        if name.startswith("hermes-"):
            continue
        meta = get_toolset(name) or {}
        if meta.get("posture"):
            continue
        try:
            resolved = set(resolve_toolset(name))
        except Exception:  # pragma: no cover
            continue
        if resolved and resolved <= blockable:
            out.append(name)
    return sorted(out)


def apply_session_toolset_policy(
    session_id: str,
    enabled_toolsets: Optional[List[str]],
    disabled_toolsets: Optional[List[str]],
) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Fold plan-mode restrictions into a session's toolset selection.

    Called at every agent-build site. When the session is NOT in a restricted
    plan status this is a pass-through. When it IS:

    - adds the mutating toolsets to ``disabled_toolsets`` (subtracted at
      tool-build time even when they arrive via a platform bundle), and
    - adds the ``plan`` toolset to ``enabled_toolsets`` (exposes ``plan_ready``
      AND changes the gateway agent-cache signature so the restricted agent is
      rebuilt). ``enabled_toolsets`` of ``None`` means "all toolsets"; we leave
      it None (``plan`` is already reachable) to avoid collapsing the session
      down to only the plan toolset.

    Returns the possibly-updated ``(enabled_toolsets, disabled_toolsets)``.
    """
    state = load_plan(session_id)
    if state is None or not state.is_restricted():
        return enabled_toolsets, disabled_toolsets

    new_disabled = list(disabled_toolsets or [])
    for ts in session_disabled_toolsets():
        if ts not in new_disabled:
            new_disabled.append(ts)

    new_enabled = enabled_toolsets
    if enabled_toolsets is not None and PLAN_TOOLSET not in enabled_toolsets:
        new_enabled = list(enabled_toolsets) + [PLAN_TOOLSET]

    return new_enabled, new_disabled


# ──────────────────────────────────────────────────────────────────────
# Dispatch guard (layer b)
# ──────────────────────────────────────────────────────────────────────


def is_plan_path(path: Optional[str]) -> bool:
    """True when ``path`` targets the plans dir (``.hermes/plans/``).

    Normalises separators and matches the ``.hermes/plans`` segment anywhere
    in the path so relative, backend-workspace, and absolute forms all work.
    """
    if not path or not isinstance(path, str):
        return False
    normalized = path.replace("\\", "/")
    return PLANS_DIR_SEGMENT in normalized


_BLOCK_MESSAGE = (
    "Blocked: plan mode is active — mutating tools are disabled while you plan. "
    "Write your plan to a file under .hermes/plans/, then call plan_ready to "
    "request approval. To leave plan mode without a plan, the user can run "
    "/plan exit (which DISCARDS the pending plan — it does not approve it)."
)


def tool_block_reason(
    session_id: str,
    tool_name: str,
    args: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Return a block message if plan mode forbids this tool, else None.

    Fail-closed: if a plan row exists but cannot be read/parsed, a mutating
    tool is blocked. A clean absence (or an unavailable DB) is treated as "not
    in plan mode" so a momentary outage does not wedge every session.
    """
    from agent.tool_guardrails import MUTATING_TOOL_NAMES

    # Only mutating tools are ever affected — read-only tools always pass.
    if tool_name not in MUTATING_TOOL_NAMES:
        return None

    kind, state = _read_plan(session_id)

    if kind == _READ_UNREADABLE:
        # A plan row exists but we could not read it — fail closed.
        return _BLOCK_MESSAGE
    if kind != _READ_ACTIVE or state is None or not state.is_restricted():
        return None

    # In a restricted status: allow writing the plan file itself.
    if tool_name in PLAN_FILE_TOOLS:
        target = None
        if isinstance(args, Mapping):
            target = args.get("path")
        if is_plan_path(target):
            return None
        return (
            "Blocked: plan mode is active — file writes are only allowed for the "
            "plan markdown under .hermes/plans/. Save your plan there, then call "
            "plan_ready."
        )

    return _BLOCK_MESSAGE


# ──────────────────────────────────────────────────────────────────────
# config default
# ──────────────────────────────────────────────────────────────────────


def _config_plan_mode() -> Optional[str]:
    """Read the top-level ``plan_mode`` config key (lowercased), or None."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception as exc:  # pragma: no cover
        logger.debug("PlanManager: load_config failed: %s", exc)
        return None
    val = cfg.get(CONFIG_KEY)
    if isinstance(val, str):
        return val.strip().lower()
    return None


def ensure_default_for_session(session_id: str) -> Optional[PlanState]:
    """Materialise a planning row for a fresh session when ``plan_mode: always``.

    No-op when a plan row already exists (any status), when the config default
    is not ``always``, or when there is no session id. Returns the created
    state, or None.
    """
    if not session_id:
        return None
    if _config_plan_mode() != CONFIG_ALWAYS:
        return None
    kind, _state = _read_plan(session_id)
    if kind != _READ_NONE:
        return None
    state = PlanState(
        status=STATUS_PLANNING,
        entered_at=time.time(),
        entered_by="config",
    )
    save_plan(session_id, state)
    return state


# ──────────────────────────────────────────────────────────────────────
# Manager (mirrors hermes_cli/goals.GoalManager where sensible)
# ──────────────────────────────────────────────────────────────────────


class PlanManager:
    """Per-session plan-mode state + transitions.

    The CLI and gateway each hold one ``PlanManager`` per live session.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._state: Optional[PlanState] = load_plan(session_id)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[PlanState]:
        return self._state

    def is_active(self) -> bool:
        """True while plan mode restricts the session (planning / pending)."""
        return self._state is not None and self._state.is_restricted()

    def status_line(self) -> str:
        s = self._state
        if s is None:
            return "Plan mode is off. Enter it with /plan."
        if s.status == STATUS_PLANNING:
            where = f" — plan at {s.plan_path}" if s.plan_path else ""
            return f"📝 Plan mode: planning{where}. Mutations are blocked; call plan_ready when done."
        if s.status == STATUS_PENDING_APPROVAL:
            return "⏳ Plan mode: awaiting your approval to execute."
        if s.status == STATUS_APPROVED:
            return "✅ Plan approved — execution unlocked."
        if s.status == STATUS_OFF:
            return "Plan mode is off. Enter it with /plan."
        return f"Plan mode: {s.status}."

    # --- transitions --------------------------------------------------

    def enter(self, *, entered_by: str = "user") -> PlanState:
        """Enter (or reset to) planning. Idempotent-ish: re-entering a
        planning/pending session keeps the existing plan_path."""
        prev = self._state
        state = PlanState(
            status=STATUS_PLANNING,
            plan_path=prev.plan_path if prev is not None else None,
            entered_at=time.time(),
            entered_by=entered_by,
        )
        self._state = state
        save_plan(self.session_id, state)
        return state

    def set_plan_path(self, path: str) -> Optional[PlanState]:
        if self._state is None:
            return None
        self._state.plan_path = path
        save_plan(self.session_id, self._state)
        return self._state

    def request_approval(self, clarify_id: Optional[str] = None) -> Optional[PlanState]:
        """Move planning → pending_approval (the clarify is in flight)."""
        if self._state is None:
            return None
        self._state.status = STATUS_PENDING_APPROVAL
        self._state.approval_clarify_id = clarify_id
        save_plan(self.session_id, self._state)
        return self._state

    def approve(self) -> Optional[PlanState]:
        """Approve execution: lifts the restriction on the next rebuild."""
        if self._state is None:
            return None
        self._state.status = STATUS_APPROVED
        self._state.approval_clarify_id = None
        save_plan(self.session_id, self._state)
        return self._state

    def keep_planning(self) -> Optional[PlanState]:
        """Reject an approval request: back to planning, stays restricted."""
        if self._state is None:
            return None
        self._state.status = STATUS_PLANNING
        self._state.approval_clarify_id = None
        save_plan(self.session_id, self._state)
        return self._state

    def exit(self) -> bool:
        """Leave plan mode entirely, DISCARDING any pending/unapproved plan.

        CRITICAL SEMANTIC: ``/plan exit`` NEVER approves — it throws the plan
        away and lifts the restriction. (Shipped-bug lesson: an "exit" that
        silently approved a half-baked plan let unreviewed mutations run.)
        Returns True if plan mode was active.
        """
        had = self._state is not None
        clear_plan(self.session_id)
        self._state = None
        return had
