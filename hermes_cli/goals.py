"""Persistent session goals — the Ralph loop for Hermes.

A goal is a free-form user objective that stays active across turns. After
each turn completes, a small judge call asks an auxiliary model "is this
goal satisfied by the assistant's last response?". If not, Hermes feeds a
continuation prompt back into the same session and keeps working until the
goal is done, turn budget is exhausted, the user pauses/clears it, or the
user sends a new message (which takes priority and pauses the goal loop).

Checklist mode (added 2026-05): when a goal is set, a Phase-A "decompose"
call asks the judge to write an extremely detailed checklist of concrete
completion criteria for that goal. On every subsequent turn (Phase B) the
judge evaluates the agent's most recent output against EACH pending item
and may flip pending → completed | impossible, or append new items it
discovers along the way. The goal is done only when every checklist item
is in a terminal status. This is much harsher than the freeform
"is the goal done?" prompt and gives users a visible, verifiable progress
surface via /subgoal. A bounded read_file tool loop lets the judge inspect
the dumped conversation history when the snippet alone isn't enough to
rule.

State is persisted in SessionDB's ``state_meta`` table keyed by
``goal:<session_id>`` so ``/resume`` picks it up.

Design notes / invariants:

- The continuation prompt is just a normal user message appended to the
  session via ``run_conversation``. No system-prompt mutation, no toolset
  swap — prompt caching stays intact.
- Judge failures are fail-OPEN: ``continue``. A broken judge must not wedge
  progress; the turn budget is the backstop.
- When a real user message arrives mid-loop it preempts the continuation
  prompt and also pauses the goal loop for that turn (we still re-judge
  after, so if the user's message happens to complete the goal the judge
  will say ``done``).
- Stickiness: once an item is marked completed or impossible, only the user
  (via /subgoal undo) can flip it back. Judge updates that try to regress
  terminal items are silently ignored.
- This module has zero hard dependency on ``cli.HermesCLI`` or the gateway
  runner — both wire the same ``GoalManager`` in.

Nothing in this module touches the agent's system prompt or toolset.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20
DEFAULT_JUDGE_TIMEOUT = 60.0
# Cap how much of the last response we send to the judge inline. The judge
# can read the dumped conversation file via read_file if it needs more.
_JUDGE_RESPONSE_SNIPPET_CHARS = 4000
# After this many consecutive judge *parse* failures (empty output / non-JSON),
# the loop auto-pauses and points the user at the goal_judge config. API /
# transport errors do NOT count toward this — those are transient. This guards
# against small models (e.g. deepseek-v4-flash) that cannot follow the strict
# JSON reply contract; without it the loop runs until the turn budget is
# exhausted with every reply shaped like `judge returned empty response` or
# `judge reply was not JSON`.
DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES = 3
# Bound the Phase-B judge tool loop: if the judge keeps calling read_file
# without ever emitting a verdict, cap it so we don't burn the model's budget.
DEFAULT_MAX_JUDGE_TOOL_CALLS = 5
# Cap a single read_file response so a judge that tries to read 100k lines
# doesn't blow up its own context. Judge can paginate if needed.
_JUDGE_READ_FILE_MAX_LINES = 400
_JUDGE_READ_FILE_MAX_CHARS = 32_000


# Status constants ────────────────────────────────────────────────────
ITEM_PENDING = "pending"
ITEM_COMPLETED = "completed"
ITEM_IMPOSSIBLE = "impossible"
TERMINAL_ITEM_STATUSES = frozenset({ITEM_COMPLETED, ITEM_IMPOSSIBLE})
VALID_ITEM_STATUSES = frozenset({ITEM_PENDING, ITEM_COMPLETED, ITEM_IMPOSSIBLE})

ITEM_MARKERS = {
    ITEM_COMPLETED: "[x]",
    ITEM_IMPOSSIBLE: "[!]",
    ITEM_PENDING: "[ ]",
}

ADDED_BY_JUDGE = "judge"
ADDED_BY_USER = "user"


# ──────────────────────────────────────────────────────────────────────
# Continuation prompt
# ──────────────────────────────────────────────────────────────────────

CONTINUATION_PROMPT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Continue working toward this goal. Take the next concrete step. "
    "If you believe the goal is complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly and stop."
)

CONTINUATION_PROMPT_WITH_CHECKLIST_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Checklist progress ({done}/{total} done):\n"
    "{checklist}\n\n"
    "Work on the unchecked items above. Do not declare items done yourself "
    "— a judge marks them based on evidence in your output. If an item is "
    "genuinely impossible in this environment, explain why so the judge can "
    "mark it impossible. If you are blocked on a remaining item and need "
    "user input, say so clearly and stop."
)


# ──────────────────────────────────────────────────────────────────────
# Phase-A: decompose prompts
# ──────────────────────────────────────────────────────────────────────

DECOMPOSE_SYSTEM_PROMPT = (
    "You are a strict judge for an autonomous agent. Your first job, before "
    "judging anything, is to break the user's stated goal into an EXTREMELY "
    "DETAILED checklist of concrete, verifiable completion criteria. Each "
    "item must be specific enough that a third party reading the agent's "
    "output could decide unambiguously whether that item was achieved.\n\n"
    "Be exhaustive. Bias toward MORE items, not fewer. Include sub-items, "
    "edge cases, quality bars, deployment steps, verification checks, and "
    "anything the user would reasonably expect from a goal of this type. "
    "If the user said 'build me a website' you should be enumerating "
    "homepage exists, navigation links work, content is non-placeholder, "
    "mobile responsive, accessibility tags present, deployed somewhere "
    "publicly accessible, domain/URL is functional, etc. Better to "
    "over-specify and let a few items get marked impossible than to "
    "under-specify and let the agent declare victory early.\n\n"
    "Submit your checklist by calling the ``submit_checklist`` tool. Do "
    "not reply with prose or JSON in your message body — call the tool. "
    "The system will not see anything you write outside the tool call."
)

DECOMPOSE_USER_PROMPT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Produce the harshest, most detailed checklist of completion criteria "
    "you can. Aim for at least 5 items; more is better when warranted. "
    "Each item should be a single verifiable statement of fact about the "
    "finished work."
)


# ──────────────────────────────────────────────────────────────────────
# Phase-B: evaluate prompts
# ──────────────────────────────────────────────────────────────────────

EVALUATE_SYSTEM_PROMPT_FREEFORM = (
    "You are a strict judge evaluating whether an autonomous agent has "
    "achieved a user's stated goal. You receive the goal text and the "
    "agent's most recent response. Your only job is to decide whether "
    "the goal is fully satisfied based on that response.\n\n"
    "A goal is DONE only when:\n"
    "- The response explicitly confirms the goal was completed, OR\n"
    "- The response clearly shows the final deliverable was produced, OR\n"
    "- The response explains the goal is unachievable / blocked / needs "
    "user input (treat this as DONE with reason describing the block).\n\n"
    "Otherwise the goal is NOT done — CONTINUE.\n\n"
    "Reply ONLY with a single JSON object on one line:\n"
    '{"done": <true|false>, "reason": "<one-sentence rationale>"}'
)

EVALUATE_SYSTEM_PROMPT_CHECKLIST = (
    "You are a strict judge evaluating an autonomous agent's progress on "
    "a user's goal that has a detailed checklist of completion criteria. "
    "For EACH currently-pending checklist item, decide whether the "
    "available evidence shows the item is satisfied.\n\n"
    "Be strict but not absurd. Default to leaving items pending UNLESS "
    "evidence is reasonably clear. Reasonable evidence includes:\n"
    "- The agent's most recent response describing or showing the work\n"
    "- Tool call results visible in the conversation history (file writes, "
    "command output, web requests, etc.)\n"
    "- A clear statement by the agent that the work was done, when "
    "supported by tool output earlier in the conversation\n\n"
    "Do NOT require the agent to re-prove items it has already established "
    "in earlier turns. If a tool call earlier in the conversation already "
    "wrote a file, you do not need fresh `ls` output every turn — once "
    "established, it's done.\n\n"
    "Flip pending → completed when the response or recent tool calls show "
    "the item is satisfied. Flip pending → impossible only when the work "
    "demonstrates the item cannot be achieved in this environment (NOT "
    "merely that the agent didn't try). Vague intentions ('I will do X "
    "next') do NOT count as completion.\n\n"
    "STICKINESS: items already marked completed or impossible are frozen. "
    "Do not include them in your updates. Only the user can revert them.\n\n"
    "TOOLS:\n"
    "- ``read_file(path, offset, limit)``: inspect the dumped conversation "
    "history file whose path is given in the user message. Use this when "
    "the snippet alone isn't enough to rule. Each call costs tokens, so "
    "only read when needed.\n"
    "- ``update_checklist(updates, new_items, reason)``: issue your "
    "verdict. Call this exactly once per turn when you are ready to rule. "
    "Calling it ENDS the evaluation.\n\n"
    "You MUST call one of these tools every turn. Do not reply with "
    "prose or JSON in your message body — the system will not see "
    "anything written outside tool calls. When you cite evidence, "
    "reference the agent's actual output specifically."
)

EVALUATE_USER_PROMPT_CHECKLIST_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Current checklist (each item is numbered, 1-based — use these "
    "exact 1-based numbers as the ``index`` field in your updates):\n{checklist_block}\n\n"
    "Agent's most recent response (snippet):\n{response}\n\n"
    "Conversation history file (call read_file on this path if you need "
    "more context — pagination supported via offset/limit):\n{history_path}\n\n"
    "Evaluate each pending item. Cite specific evidence."
)

EVALUATE_USER_PROMPT_FREEFORM_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "Is the goal satisfied?"
)


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────


@dataclass
class ChecklistItem:
    """One concrete completion criterion attached to a goal."""

    text: str
    status: str = ITEM_PENDING            # pending | completed | impossible
    added_by: str = ADDED_BY_JUDGE        # judge | user
    added_at: float = 0.0
    completed_at: Optional[float] = None
    evidence: Optional[str] = None        # judge's rationale on flip

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChecklistItem":
        text = str(data.get("text", "")).strip()
        if not text:
            text = "(empty item)"
        status = str(data.get("status", ITEM_PENDING)).strip().lower()
        if status not in VALID_ITEM_STATUSES:
            status = ITEM_PENDING
        added_by = str(data.get("added_by", ADDED_BY_JUDGE)).strip().lower()
        if added_by not in (ADDED_BY_JUDGE, ADDED_BY_USER):
            added_by = ADDED_BY_JUDGE
        return cls(
            text=text,
            status=status,
            added_by=added_by,
            added_at=float(data.get("added_at", 0.0) or 0.0),
            completed_at=(
                float(data["completed_at"])
                if data.get("completed_at") is not None
                else None
            ),
            evidence=data.get("evidence"),
        )


@dataclass
class GoalState:
    """Serializable goal state stored per session."""

    goal: str
    status: str = "active"                    # active | paused | done | cleared
    turns_used: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    created_at: float = 0.0
    last_turn_at: float = 0.0
    last_verdict: Optional[str] = None        # "done" | "continue" | "skipped"
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None       # why we auto-paused (budget, etc.)
    consecutive_parse_failures: int = 0       # judge-output parse failures in a row
    # Checklist mode (added 2026-05). Both fields default safely so old
    # state_meta rows load unchanged.
    checklist: List[ChecklistItem] = field(default_factory=list)
    decomposed: bool = False                  # has Phase-A run for this goal?

    def to_json(self) -> str:
        data = asdict(self)
        # asdict already serializes ChecklistItem via dataclass recursion.
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "GoalState":
        data = json.loads(raw)
        raw_checklist = data.get("checklist") or []
        checklist: List[ChecklistItem] = []
        if isinstance(raw_checklist, list):
            for item in raw_checklist:
                if isinstance(item, dict):
                    try:
                        checklist.append(ChecklistItem.from_dict(item))
                    except Exception:
                        continue
        return cls(
            goal=data.get("goal", ""),
            status=data.get("status", "active"),
            turns_used=int(data.get("turns_used", 0) or 0),
            max_turns=int(data.get("max_turns", DEFAULT_MAX_TURNS) or DEFAULT_MAX_TURNS),
            created_at=float(data.get("created_at", 0.0) or 0.0),
            last_turn_at=float(data.get("last_turn_at", 0.0) or 0.0),
            last_verdict=data.get("last_verdict"),
            last_reason=data.get("last_reason"),
            paused_reason=data.get("paused_reason"),
            consecutive_parse_failures=int(data.get("consecutive_parse_failures", 0) or 0),
            checklist=checklist,
            decomposed=bool(data.get("decomposed", False)),
        )

    # --- checklist helpers ------------------------------------------------

    def checklist_counts(self) -> Tuple[int, int, int, int]:
        """Return (total, completed, impossible, pending)."""
        total = len(self.checklist)
        completed = sum(1 for it in self.checklist if it.status == ITEM_COMPLETED)
        impossible = sum(1 for it in self.checklist if it.status == ITEM_IMPOSSIBLE)
        pending = total - completed - impossible
        return total, completed, impossible, pending

    def all_terminal(self) -> bool:
        """True iff at least one item exists and every item is in a terminal status."""
        if not self.checklist:
            return False
        return all(it.status in TERMINAL_ITEM_STATUSES for it in self.checklist)

    def render_checklist(self, *, numbered: bool = False) -> str:
        if not self.checklist:
            return "(empty)"
        lines = []
        for i, item in enumerate(self.checklist, start=1):
            marker = ITEM_MARKERS.get(item.status, "[?]")
            prefix = f"{i}. {marker}" if numbered else f"  {marker}"
            line = f"{prefix} {item.text}"
            if item.status == ITEM_IMPOSSIBLE and item.evidence:
                line += f" (impossible: {item.evidence})"
            lines.append(line)
        return "\n".join(lines)


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


def clear_goal(session_id: str) -> None:
    """Mark a goal cleared in the DB (preserved for audit, status=cleared)."""
    state = load_goal(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_goal(session_id, state)


# ──────────────────────────────────────────────────────────────────────
# Conversation-history dump (read by the judge tool loop)
# ──────────────────────────────────────────────────────────────────────


def _goals_dump_dir() -> Optional[Path]:
    """Return ``<HERMES_HOME>/goals`` (created on first use), or None on error."""
    try:
        from hermes_constants import get_hermes_home

        home = Path(get_hermes_home())
    except Exception as exc:
        logger.debug("goals dump dir: get_hermes_home failed: %s", exc)
        return None
    try:
        path = home / "goals"
        path.mkdir(parents=True, exist_ok=True)
        return path
    except Exception as exc:
        logger.debug("goals dump dir: mkdir failed: %s", exc)
        return None


def _safe_session_filename(session_id: str) -> str:
    """Make a session_id safe for use as a filename component."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", session_id or "unknown")
    # Bound length to keep filesystem happy.
    return cleaned[:128] or "unknown"


def conversation_dump_path(session_id: str) -> Optional[Path]:
    """Where the dumped messages JSON for ``session_id`` lives."""
    base = _goals_dump_dir()
    if base is None:
        return None
    return base / f"{_safe_session_filename(session_id)}.json"


def dump_conversation(session_id: str, messages: List[Dict[str, Any]]) -> Optional[Path]:
    """Write ``messages`` to the goals/ dump file. Returns the path on success."""
    if not session_id or not messages:
        return None
    path = conversation_dump_path(session_id)
    if path is None:
        return None
    try:
        # Best-effort: messages may contain non-JSON-serializable objects from
        # provider-specific adapter shims. Fall through with default=str.
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(messages, fh, ensure_ascii=False, indent=2, default=str)
        return path
    except Exception as exc:
        logger.debug("dump_conversation: write failed: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Judge: parsing helpers
# ──────────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort extraction of a single JSON object from a possibly-prosey reply."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]
    try:
        data = json.loads(text)
    except Exception:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except Exception:
            return None
    return data if isinstance(data, dict) else None


def _parse_judge_response(raw: str) -> Tuple[bool, str, bool]:
    """Parse the freeform judge's reply. Fail-open to ``(False, "<reason>", parse_failed)``.

    Returns ``(done, reason, parse_failed)``. ``parse_failed`` is True when the
    judge returned output that couldn't be interpreted as the expected JSON
    verdict (empty body, prose, malformed JSON). Callers use that flag to
    auto-pause after N consecutive parse failures so a weak judge model
    doesn't silently burn the turn budget.
    """
    if not raw:
        return False, "judge returned empty response", True

    data = _extract_json_object(raw)
    if data is None:
        return False, f"judge reply was not JSON: {_truncate(raw, 200)!r}", True

    done_val = data.get("done")
    if isinstance(done_val, str):
        done = done_val.strip().lower() in ("true", "yes", "1", "done")
    else:
        done = bool(done_val)
    reason = str(data.get("reason") or "").strip()
    if not reason:
        reason = "no reason provided"
    return done, reason, False


def _parse_decompose_response(raw: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Parse a Phase-A decompose reply. Returns (items, parse_failed)."""
    if not raw:
        return [], True
    data = _extract_json_object(raw)
    if data is None:
        return [], True
    raw_items = data.get("checklist")
    if not isinstance(raw_items, list):
        return [], True
    out: List[Dict[str, Any]] = []
    for item in raw_items:
        if isinstance(item, dict):
            text = str(item.get("text", "")).strip()
            if text:
                out.append({"text": text})
        elif isinstance(item, str):
            text = item.strip()
            if text:
                out.append({"text": text})
    return out, False


def _parse_evaluate_response(raw: str) -> Tuple[Dict[str, Any], bool]:
    """Parse a Phase-B checklist eval reply. Returns (parsed, parse_failed).

    parsed = {"updates": [...], "new_items": [...], "reason": str}
    """
    if not raw:
        return {"updates": [], "new_items": [], "reason": "judge returned empty response"}, True
    data = _extract_json_object(raw)
    if data is None:
        return (
            {
                "updates": [],
                "new_items": [],
                "reason": f"judge reply was not JSON: {_truncate(raw, 200)!r}",
            },
            True,
        )
    updates = data.get("updates") or []
    new_items = data.get("new_items") or []
    reason = str(data.get("reason") or "").strip() or "no reason provided"
    norm_updates = []
    if isinstance(updates, list):
        for upd in updates:
            if not isinstance(upd, dict):
                continue
            try:
                # Judge sees the checklist rendered with 1-based indices
                # (matches the /subgoal CLI). Convert to 0-based here so the
                # apply layer can index ``state.checklist`` directly.
                idx_1based = int(upd.get("index"))
            except (TypeError, ValueError):
                continue
            idx = idx_1based - 1
            status = str(upd.get("status", "")).strip().lower()
            if status not in TERMINAL_ITEM_STATUSES:
                # Phase-B only accepts terminal flips. Pending → pending is a no-op.
                continue
            evidence = str(upd.get("evidence") or "").strip() or None
            norm_updates.append({"index": idx, "status": status, "evidence": evidence})
    norm_new = []
    if isinstance(new_items, list):
        for it in new_items:
            if isinstance(it, dict):
                text = str(it.get("text", "")).strip()
                if text:
                    norm_new.append({"text": text})
            elif isinstance(it, str):
                text = it.strip()
                if text:
                    norm_new.append({"text": text})
    return {"updates": norm_updates, "new_items": norm_new, "reason": reason}, False


# ──────────────────────────────────────────────────────────────────────
# Judge: read_file tool for the judge's bounded tool loop
# ──────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────
# Judge tool schemas: read_file (history inspection) +
# submit_checklist (Phase A) + update_checklist (Phase B)
#
# Forcing the judge to emit through tool calls is dramatically more
# reliable than asking it to reply with JSON text. Most providers
# enforce the schema server-side, so weak/small judge models can no
# longer drift into prose, markdown fences, or empty bodies.
# ──────────────────────────────────────────────────────────────────────


_JUDGE_READ_FILE_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": (
            "Read a portion of the dumped conversation history JSON file. "
            "Use this when the snippet alone isn't enough to rule. Returns "
            "lines from the file with 1-based line numbers. Pagination "
            "supported via offset and limit. Reads beyond a built-in cap "
            "are truncated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Absolute path to the conversation history file. "
                        "You were given this in the user message."
                    ),
                },
                "offset": {
                    "type": "integer",
                    "description": "1-indexed starting line number (default 1).",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Max lines to return (default {_JUDGE_READ_FILE_MAX_LINES})."
                    ),
                    "default": _JUDGE_READ_FILE_MAX_LINES,
                },
            },
            "required": ["path"],
        },
    },
}


_JUDGE_SUBMIT_CHECKLIST_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_checklist",
        "description": (
            "Submit the harsh, detailed completion-criteria checklist you "
            "decomposed the goal into. Each item is one verifiable "
            "completion criterion. Bias toward more items, not fewer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": (
                        "List of checklist items. Each item is a single "
                        "verifiable statement of fact about the finished "
                        "work. Aim for at least 5 items; more is better "
                        "when warranted."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The completion-criterion text.",
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            "required": ["items"],
        },
    },
}


_JUDGE_UPDATE_CHECKLIST_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "update_checklist",
        "description": (
            "Issue your verdict on the current checklist. For each "
            "currently-pending item, decide whether the agent's most "
            "recent response (and conversation history if you read it) "
            "shows the item is satisfied. You may also append new items "
            "the original decomposition missed. Call this exactly once "
            "when you are ready to rule — calling it ends the evaluation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "array",
                    "description": (
                        "Per-item rulings. Use the 1-based ``index`` shown "
                        "in the checklist. ``status`` must be 'completed' "
                        "(clear evidence the item is done) or 'impossible' "
                        "(item cannot be achieved in this environment). "
                        "Items already in a terminal status are frozen — "
                        "do not include them."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "1-based checklist index.",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["completed", "impossible"],
                            },
                            "evidence": {
                                "type": "string",
                                "description": (
                                    "One-sentence specific citation of why "
                                    "this item is done or impossible. "
                                    "Reference the agent's actual output."
                                ),
                            },
                        },
                        "required": ["index", "status", "evidence"],
                    },
                },
                "new_items": {
                    "type": "array",
                    "description": (
                        "Optional: completion criteria the original "
                        "decomposition missed. Stay strict — only add "
                        "items that genuinely belong as completion "
                        "criteria for this goal."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "The new criterion text.",
                            },
                        },
                        "required": ["text"],
                    },
                },
                "reason": {
                    "type": "string",
                    "description": "One-sentence overall rationale for this round of updates.",
                },
            },
            "required": ["updates", "new_items", "reason"],
        },
    },
}


def _judge_read_file(
    path: str,
    *,
    offset: int = 1,
    limit: int = _JUDGE_READ_FILE_MAX_LINES,
    allowed_path: Optional[Path] = None,
) -> str:
    """Bounded read of the dumped conversation file. Returns JSON-serializable text.

    Restricted to ``allowed_path`` when provided — the judge cannot use this
    tool to read arbitrary files.
    """
    if not path:
        return json.dumps({"error": "path is required"})
    try:
        target = Path(path).resolve()
    except Exception as exc:
        return json.dumps({"error": f"path resolve failed: {exc}"})

    if allowed_path is not None:
        try:
            allowed = allowed_path.resolve()
        except Exception:
            allowed = allowed_path
        if target != allowed:
            return json.dumps({
                "error": (
                    f"read_file is restricted to the conversation dump path. "
                    f"Allowed: {allowed}"
                )
            })

    if not target.exists():
        return json.dumps({"error": f"file not found: {target}"})
    try:
        offset = max(1, int(offset or 1))
        limit = max(1, min(int(limit or _JUDGE_READ_FILE_MAX_LINES), _JUDGE_READ_FILE_MAX_LINES))
    except (TypeError, ValueError):
        return json.dumps({"error": "offset and limit must be integers"})

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except Exception as exc:
        return json.dumps({"error": f"read failed: {exc}"})

    total = len(lines)
    start = offset - 1
    end = min(start + limit, total)
    slice_lines = lines[start:end]
    out = "".join(slice_lines)
    if len(out) > _JUDGE_READ_FILE_MAX_CHARS:
        out = out[:_JUDGE_READ_FILE_MAX_CHARS] + "\n… [truncated by judge read cap]"
    return json.dumps({
        "path": str(target),
        "total_lines": total,
        "offset": offset,
        "returned": len(slice_lines),
        "next_offset": end + 1 if end < total else None,
        "content": out,
    }, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────────
# Judge: phase-A (decompose) and phase-B (evaluate)
# ──────────────────────────────────────────────────────────────────────


def _get_judge_client() -> Tuple[Optional[Any], str]:
    """Return (client, model) or (None, '') when unavailable."""
    try:
        from agent.auxiliary_client import get_text_auxiliary_client
    except Exception as exc:
        logger.debug("goal judge: auxiliary client import failed: %s", exc)
        return None, ""
    try:
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception as exc:
        logger.debug("goal judge: get_text_auxiliary_client failed: %s", exc)
        return None, ""
    if client is None or not model:
        return None, ""
    return client, model


def _extract_tool_call(msg: Any, tool_name: str) -> Optional[Dict[str, Any]]:
    """Find a tool call by name on a chat-completions message. Returns
    ``{"id", "name", "arguments": <dict>}`` or None.

    Robust to provider shims that return tool_calls as objects or dicts
    and arguments as JSON strings or already-parsed dicts.
    """
    tool_calls = getattr(msg, "tool_calls", None) or []
    for tc in tool_calls:
        try:
            tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None) or "tc-?"
            fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
            if fn is None:
                continue
            fn_name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "")
            if fn_name != tool_name:
                continue
            fn_args_raw = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else "")
            if isinstance(fn_args_raw, str):
                try:
                    args = json.loads(fn_args_raw) if fn_args_raw else {}
                except Exception:
                    args = {}
            elif isinstance(fn_args_raw, dict):
                args = fn_args_raw
            else:
                args = {}
            return {"id": tc_id, "name": fn_name, "arguments": args}
        except Exception:
            continue
    return None


def _serialize_assistant_tool_calls(msg: Any) -> List[Dict[str, Any]]:
    """Convert a provider-shim tool_calls list into plain-dict form for
    inclusion in subsequent ``messages=[...]`` payloads."""
    out: List[Dict[str, Any]] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        try:
            tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None) or "tc-?"
            fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
            fn_name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else "")
            fn_args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else "")
            if not isinstance(fn_args, str):
                try:
                    fn_args = json.dumps(fn_args)
                except Exception:
                    fn_args = "{}"
            out.append({
                "id": tc_id,
                "type": "function",
                "function": {"name": fn_name or "", "arguments": fn_args},
            })
        except Exception:
            continue
    return out


def _call_judge_with_tool_choice(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    forced_tool_name: Optional[str],
    timeout: float,
    max_tokens: int = 1500,
) -> Tuple[Optional[Any], Optional[str]]:
    """Call the judge with a forced tool choice, falling back to ``auto``
    if the provider rejects ``required`` / a specific function choice.

    Returns ``(response, error)``. On success, ``error`` is None.
    """
    # First attempt: force the specific tool. Most modern providers
    # support {"type": "function", "function": {"name": "..."}}.
    primary_choice: Any
    if forced_tool_name:
        primary_choice = {"type": "function", "function": {"name": forced_tool_name}}
    else:
        primary_choice = "required"

    attempts: List[Any] = [primary_choice, "required", "auto"]
    last_err: Optional[str] = None
    for choice in attempts:
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=choice,
                temperature=0,
                max_tokens=max_tokens,
                timeout=timeout,
            ), None
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            # Only retry on errors that look like the provider rejecting the
            # tool_choice shape. Network errors etc. should bail immediately.
            msg = str(exc).lower()
            if not any(token in msg for token in (
                "tool_choice", "tool choice", "required", "function call",
                "unsupported", "not supported", "invalid", "400",
            )):
                return None, last_err
            logger.debug("goal judge: tool_choice=%r rejected (%s); falling back", choice, exc)
            continue
    return None, last_err or "all tool_choice fallbacks failed"


def decompose_goal(
    goal: str,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Phase-A: ask the judge to break the goal into a checklist via a
    forced ``submit_checklist`` tool call.

    Returns ``(items, error)``. On any failure, returns ``([], reason)``
    so the caller can fall back to freeform mode.
    """
    if not goal.strip():
        return [], "empty goal"

    client, model = _get_judge_client()
    if client is None:
        return [], "auxiliary client unavailable"

    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": DECOMPOSE_USER_PROMPT_TEMPLATE.format(
                goal=_truncate(goal, 4000)
            ),
        },
    ]

    resp, err = _call_judge_with_tool_choice(
        client,
        model=model,
        messages=messages,
        tools=[_JUDGE_SUBMIT_CHECKLIST_TOOL_SCHEMA],
        forced_tool_name="submit_checklist",
        timeout=timeout,
        max_tokens=2000,
    )
    if resp is None:
        logger.info("goal decompose: API call failed (%s)", err)
        return [], f"decompose error: {err}"

    try:
        msg = resp.choices[0].message
    except Exception:
        return [], "decompose response malformed"

    tc = _extract_tool_call(msg, "submit_checklist")
    if tc is None:
        # Provider responded but didn't call the tool. Try parsing content
        # as a last-ditch backstop so a fully-broken provider doesn't
        # silently leave the user with no checklist at all.
        content = getattr(msg, "content", "") or ""
        items, parse_failed = _parse_decompose_response(content)
        if parse_failed or not items:
            logger.info(
                "goal decompose: no submit_checklist tool call AND no parseable JSON (raw=%r)",
                _truncate(content, 200),
            )
            return [], "decompose: judge did not call submit_checklist"
        logger.info("goal decompose: fell back to JSON-content parser (%d items)", len(items))
        return items, None

    raw_items = tc["arguments"].get("items") or []
    items: List[Dict[str, Any]] = []
    if isinstance(raw_items, list):
        for entry in raw_items:
            if isinstance(entry, dict):
                text = str(entry.get("text", "")).strip()
                if text:
                    items.append({"text": text})
            elif isinstance(entry, str):
                text = entry.strip()
                if text:
                    items.append({"text": text})

    if not items:
        logger.info("goal decompose: submit_checklist returned empty items list")
        return [], "decompose: empty checklist"

    logger.info("goal decompose: produced %d checklist items via tool call", len(items))
    return items, None


def judge_goal_freeform(
    goal: str,
    last_response: str,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
) -> Tuple[str, str, bool]:
    """Legacy freeform judge — kept for goals with no checklist.

    Returns ``(verdict, reason, parse_failed)`` where verdict is ``"done"``,
    ``"continue"``, or ``"skipped"``.
    """
    if not goal.strip():
        return "skipped", "empty goal", False
    if not last_response.strip():
        return "continue", "empty response (nothing to evaluate)", False

    client, model = _get_judge_client()
    if client is None:
        return "continue", "auxiliary client unavailable", False

    prompt = EVALUATE_USER_PROMPT_FREEFORM_TEMPLATE.format(
        goal=_truncate(goal, 2000),
        response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVALUATE_SYSTEM_PROMPT_FREEFORM},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
            timeout=timeout,
        )
    except Exception as exc:
        logger.info("goal judge: API call failed (%s) — falling through to continue", exc)
        return "continue", f"judge error: {type(exc).__name__}", False

    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = ""

    done, reason, parse_failed = _parse_judge_response(raw)
    verdict = "done" if done else "continue"
    logger.info("goal judge (freeform): verdict=%s reason=%s", verdict, _truncate(reason, 120))
    return verdict, reason, parse_failed


def evaluate_checklist(
    state: GoalState,
    last_response: str,
    *,
    history_path: Optional[Path],
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
    max_tool_calls: int = DEFAULT_MAX_JUDGE_TOOL_CALLS,
) -> Tuple[Dict[str, Any], bool]:
    """Phase-B: judge evaluates each pending checklist item via forced
    tool calls.

    The judge has two tools available:
      - ``read_file``: inspect the dumped conversation history
      - ``update_checklist``: issue the verdict (terminates the loop)

    ``tool_choice="required"`` forces one of them every iteration. We loop
    until ``update_checklist`` is called or ``max_tool_calls`` is exhausted.

    Returns ``(parsed, parse_failed)`` where parsed is
    ``{"updates": [...], "new_items": [...], "reason": str}``.
    Falls open on transport errors: empty updates/new_items, parse_failed=False.
    """
    client, model = _get_judge_client()
    if client is None:
        return ({"updates": [], "new_items": [], "reason": "auxiliary client unavailable"}, False)

    # Render checklist with 1-based indices the judge addresses via the
    # update_checklist tool's ``index`` field.
    checklist_block = state.render_checklist(numbered=True)

    user_prompt = EVALUATE_USER_PROMPT_CHECKLIST_TEMPLATE.format(
        goal=_truncate(state.goal, 2000),
        checklist_block=checklist_block,
        response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
        history_path=str(history_path) if history_path else "(unavailable — judge from snippet only)",
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": EVALUATE_SYSTEM_PROMPT_CHECKLIST},
        {"role": "user", "content": user_prompt},
    ]

    # Build the toolbox: read_file is only useful when we actually have a
    # history file to read, so we omit it otherwise to keep the schema lean.
    tools: List[Dict[str, Any]] = [_JUDGE_UPDATE_CHECKLIST_TOOL_SCHEMA]
    if history_path is not None:
        tools.insert(0, _JUDGE_READ_FILE_TOOL_SCHEMA)

    reads_left = max(0, int(max_tool_calls)) if history_path is not None else 0

    # Bound the overall loop generously — the judge will normally finish in
    # one or two passes (read_file once, then update_checklist; or just
    # update_checklist directly).
    for iteration in range(reads_left + 2):
        # When out of read budget, drop read_file from the toolbox so the
        # judge MUST emit update_checklist.
        loop_tools = tools if reads_left > 0 else [_JUDGE_UPDATE_CHECKLIST_TOOL_SCHEMA]
        # Forcing update_checklist directly when reads are exhausted gives
        # us the strongest guarantee of termination.
        forced = "update_checklist" if reads_left <= 0 else None

        resp, err = _call_judge_with_tool_choice(
            client,
            model=model,
            messages=messages,
            tools=loop_tools,
            forced_tool_name=forced,
            timeout=timeout,
            max_tokens=1500,
        )
        if resp is None:
            logger.info("goal judge (checklist): API call failed (%s)", err)
            return (
                {
                    "updates": [],
                    "new_items": [],
                    "reason": f"judge error: {err}",
                },
                False,
            )

        try:
            msg = resp.choices[0].message
        except Exception:
            return (
                {"updates": [], "new_items": [], "reason": "judge response malformed"},
                True,
            )

        # Did the judge call update_checklist? If yes, we're done.
        update_tc = _extract_tool_call(msg, "update_checklist")
        if update_tc is not None:
            parsed = _normalize_update_args(update_tc["arguments"])
            logger.info(
                "goal judge (checklist): updates=%d new_items=%d reason=%s",
                len(parsed.get("updates") or []),
                len(parsed.get("new_items") or []),
                _truncate(parsed.get("reason", ""), 120),
            )
            return parsed, False

        # Did the judge call read_file? If yes, run it and feed the result back.
        read_tc = _extract_tool_call(msg, "read_file")
        if read_tc is not None and reads_left > 0:
            args = read_tc["arguments"]
            tool_result = _judge_read_file(
                str(args.get("path", "")),
                offset=args.get("offset", 1),
                limit=args.get("limit", _JUDGE_READ_FILE_MAX_LINES),
                allowed_path=history_path,
            )
            messages.append({
                "role": "assistant",
                "content": getattr(msg, "content", "") or "",
                "tool_calls": _serialize_assistant_tool_calls(msg),
            })
            messages.append({
                "role": "tool",
                "tool_call_id": read_tc["id"],
                "name": "read_file",
                "content": tool_result,
            })
            reads_left -= 1
            continue

        # Neither tool was called. Try parsing the content body as a last-
        # ditch backstop, then bail.
        content = getattr(msg, "content", "") or ""
        if content.strip():
            parsed, parse_failed = _parse_evaluate_response(content)
            if not parse_failed:
                logger.info(
                    "goal judge (checklist): fell back to JSON-content parser "
                    "updates=%d new_items=%d",
                    len(parsed.get("updates") or []),
                    len(parsed.get("new_items") or []),
                )
                return parsed, False
        logger.info(
            "goal judge (checklist): judge emitted neither read_file nor "
            "update_checklist (iteration=%d, content=%r) — bailing",
            iteration, _truncate(content, 120),
        )
        return (
            {
                "updates": [],
                "new_items": [],
                "reason": "judge did not call update_checklist",
            },
            True,
        )

    # Loop exhausted without an update_checklist call.
    return (
        {
            "updates": [],
            "new_items": [],
            "reason": "judge tool-loop exhausted without verdict",
        },
        True,
    )


def _normalize_update_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the ``update_checklist`` tool arguments.

    Performs the same 1-based → 0-based conversion and terminal-status
    filter as ``_parse_evaluate_response``. Returns the canonical
    ``{updates, new_items, reason}`` shape callers expect.
    """
    raw_updates = args.get("updates") or []
    raw_new = args.get("new_items") or []
    reason = str(args.get("reason") or "").strip() or "no reason provided"

    norm_updates: List[Dict[str, Any]] = []
    if isinstance(raw_updates, list):
        for upd in raw_updates:
            if not isinstance(upd, dict):
                continue
            try:
                idx_1based = int(upd.get("index"))
            except (TypeError, ValueError):
                continue
            status = str(upd.get("status", "")).strip().lower()
            if status not in TERMINAL_ITEM_STATUSES:
                continue
            evidence = str(upd.get("evidence") or "").strip() or None
            norm_updates.append({
                "index": idx_1based - 1,  # 1-based → 0-based for apply layer
                "status": status,
                "evidence": evidence,
            })

    norm_new: List[Dict[str, Any]] = []
    if isinstance(raw_new, list):
        for it in raw_new:
            if isinstance(it, dict):
                text = str(it.get("text", "")).strip()
                if text:
                    norm_new.append({"text": text})
            elif isinstance(it, str):
                text = it.strip()
                if text:
                    norm_new.append({"text": text})

    return {"updates": norm_updates, "new_items": norm_new, "reason": reason}


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
    - ``add_subgoal(text)`` — user appends a checklist item.
    - ``mark_subgoal(index, status)`` — user flips an item (override).
    - ``remove_subgoal(index)`` — user deletes an item.
    - ``clear_checklist()`` — user wipes the checklist; next turn re-decomposes.
    - ``evaluate_after_turn(last_response, agent=None)`` — call the judge,
      update state, return a decision dict.
    - ``next_continuation_prompt()`` — the canonical user-role message to
      feed back into ``run_conversation``.
    """

    def __init__(self, session_id: str, *, default_max_turns: int = DEFAULT_MAX_TURNS):
        self.session_id = session_id
        self.default_max_turns = int(default_max_turns or DEFAULT_MAX_TURNS)
        self._state: Optional[GoalState] = load_goal(session_id)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[GoalState]:
        return self._state

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_goal(self) -> bool:
        return self._state is not None and self._state.status in ("active", "paused")

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status in ("cleared",):
            return "No active goal. Set one with /goal <text>."
        turns = f"{s.turns_used}/{s.max_turns} turns"
        cl_total, cl_done, cl_imp, _ = s.checklist_counts()
        cl_text = ""
        if cl_total:
            cl_text = f", {cl_done + cl_imp}/{cl_total} done"
        if s.status == "active":
            return f"⊙ Goal (active, {turns}{cl_text}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {turns}{cl_text}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({turns}{cl_text}): {s.goal}"
        return f"Goal ({s.status}, {turns}{cl_text}): {s.goal}"

    def render_checklist(self) -> str:
        """Public helper for the /subgoal slash command."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.checklist:
            return "(checklist empty — judge will populate it on the next turn)"
        return self._state.render_checklist(numbered=True)

    # --- mutation -----------------------------------------------------

    def set(self, goal: str, *, max_turns: Optional[int] = None) -> GoalState:
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("goal text is empty")
        state = GoalState(
            goal=goal,
            status="active",
            turns_used=0,
            max_turns=int(max_turns) if max_turns else self.default_max_turns,
            created_at=time.time(),
            last_turn_at=0.0,
            checklist=[],
            decomposed=False,
        )
        self._state = state
        save_goal(self.session_id, state)
        return state

    def pause(self, reason: str = "user-paused") -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        save_goal(self.session_id, self._state)
        return self._state

    def resume(self, *, reset_budget: bool = True) -> Optional[GoalState]:
        if not self._state:
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        if reset_budget:
            self._state.turns_used = 0
        save_goal(self.session_id, self._state)
        return self._state

    def clear(self) -> None:
        if self._state is None:
            return
        self._state.status = "cleared"
        save_goal(self.session_id, self._state)
        self._state = None

    def mark_done(self, reason: str) -> None:
        if not self._state:
            return
        self._state.status = "done"
        self._state.last_verdict = "done"
        self._state.last_reason = reason
        save_goal(self.session_id, self._state)

    # --- /subgoal user controls ---------------------------------------

    def add_subgoal(self, text: str) -> ChecklistItem:
        """User appends a new checklist item. Requires an active or paused goal."""
        if self._state is None:
            raise RuntimeError("no active goal")
        text = (text or "").strip()
        if not text:
            raise ValueError("subgoal text is empty")
        item = ChecklistItem(
            text=text,
            status=ITEM_PENDING,
            added_by=ADDED_BY_USER,
            added_at=time.time(),
        )
        self._state.checklist.append(item)
        save_goal(self.session_id, self._state)
        return item

    def mark_subgoal(self, index_1based: int, status: str) -> ChecklistItem:
        """User overrides an item's status.

        ``status`` may be ``completed``, ``impossible``, or ``pending``
        (the last only as an undo flow). Stickiness rules do NOT apply to
        user actions — the user is the only authority that can revert
        terminal items.
        """
        if self._state is None:
            raise RuntimeError("no active goal")
        status = (status or "").strip().lower()
        if status not in VALID_ITEM_STATUSES:
            raise ValueError(
                f"status must be one of {sorted(VALID_ITEM_STATUSES)}; got {status!r}"
            )
        idx = int(index_1based) - 1
        if idx < 0 or idx >= len(self._state.checklist):
            raise IndexError(
                f"index out of range (1..{len(self._state.checklist)})"
            )
        item = self._state.checklist[idx]
        item.status = status
        if status in TERMINAL_ITEM_STATUSES:
            item.completed_at = time.time()
            if not item.evidence:
                item.evidence = "marked by user"
        else:
            item.completed_at = None
            # Don't wipe judge-supplied evidence on undo — useful audit trail.
        save_goal(self.session_id, self._state)
        return item

    def remove_subgoal(self, index_1based: int) -> ChecklistItem:
        if self._state is None:
            raise RuntimeError("no active goal")
        idx = int(index_1based) - 1
        if idx < 0 or idx >= len(self._state.checklist):
            raise IndexError(
                f"index out of range (1..{len(self._state.checklist)})"
            )
        removed = self._state.checklist.pop(idx)
        save_goal(self.session_id, self._state)
        return removed

    def clear_checklist(self) -> None:
        """Wipe the checklist and reset decomposed=False so the judge re-decomposes."""
        if self._state is None:
            return
        self._state.checklist = []
        self._state.decomposed = False
        save_goal(self.session_id, self._state)

    # --- the main entry point called after every turn -----------------

    def evaluate_after_turn(
        self,
        last_response: str,
        *,
        user_initiated: bool = True,
        agent: Any = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the judge and update state. Return a decision dict.

        ``user_initiated`` distinguishes a real user prompt (True) from a
        continuation prompt we fed ourselves (False). Both increment
        ``turns_used`` because both consume model budget.

        ``messages`` is the agent's full conversation list for this session.
        When provided, it's dumped to ``<HERMES_HOME>/goals/<sid>.json`` so
        the Phase-B judge's read_file tool can inspect history. Optional —
        when missing, the judge runs from the snippet only.

        ``agent`` is a back-compat path — when ``messages`` is None we try
        to extract them from common AIAgent attribute names. Most callers
        should pass ``messages`` directly because AIAgent does not store
        the message list as a public instance attribute.

        Decision keys:
          - ``status``: current goal status after update
          - ``should_continue``: bool — caller should fire another turn
          - ``continuation_prompt``: str or None
          - ``verdict``: "done" | "continue" | "skipped" | "inactive" | "decompose"
          - ``reason``: str
          - ``message``: user-visible one-liner to print/send
        """
        state = self._state
        if state is None or state.status != "active":
            return {
                "status": state.status if state else None,
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "inactive",
                "reason": "no active goal",
                "message": "",
            }

        # Count the turn that just finished.
        state.turns_used += 1
        state.last_turn_at = time.time()

        # ── Phase A: decompose (first call after /goal set) ───────────
        if not state.decomposed:
            items, err = decompose_goal(state.goal)
            state.decomposed = True
            decompose_message = ""
            if items:
                now = time.time()
                for entry in items:
                    state.checklist.append(
                        ChecklistItem(
                            text=entry["text"],
                            status=ITEM_PENDING,
                            added_by=ADDED_BY_JUDGE,
                            added_at=now,
                        )
                    )
                state.last_verdict = "decompose"
                state.last_reason = f"decomposed into {len(items)} items"
                decompose_message = (
                    f"⊙ Goal checklist created ({len(items)} items). "
                    f"Use /subgoal to view or edit it."
                )
                save_goal(self.session_id, state)
                return {
                    "status": "active",
                    "should_continue": True,
                    "continuation_prompt": self.next_continuation_prompt(),
                    "verdict": "decompose",
                    "reason": state.last_reason,
                    "message": decompose_message,
                }
            # Decompose failed — fall through to freeform mode below.
            logger.info("goal: decompose failed (%s) — falling back to freeform judge", err)
            state.last_reason = f"decompose failed: {err}"

        # ── Phase B: evaluate ────────────────────────────────────────
        verdict, reason, parse_failed = self._evaluate_state_phase_b(
            state, last_response, agent=agent, messages=messages
        )
        state.last_verdict = verdict
        state.last_reason = reason

        # Track consecutive judge parse failures. Reset on any usable reply,
        # including API / transport errors (parse_failed=False) so a flaky
        # network doesn't trip the auto-pause meant for bad judge models.
        if parse_failed:
            state.consecutive_parse_failures += 1
        else:
            state.consecutive_parse_failures = 0

        if verdict == "done":
            state.status = "done"
            save_goal(self.session_id, state)
            return {
                "status": "done",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "done",
                "reason": reason,
                "message": f"✓ Goal achieved: {reason}",
            }

        # Auto-pause when the judge model can't produce the expected JSON
        # verdict N turns in a row.
        if state.consecutive_parse_failures >= DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES:
            state.status = "paused"
            state.paused_reason = (
                f"judge model returned unparseable output {state.consecutive_parse_failures} turns in a row"
            )
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — the judge model ({state.consecutive_parse_failures} turns) "
                    "isn't returning the required JSON verdict. Route the judge to a stricter "
                    "model in ~/.hermes/config.yaml:\n"
                    "  auxiliary:\n"
                    "    goal_judge:\n"
                    "      provider: openrouter\n"
                    "      model: google/gemini-3-flash-preview\n"
                    "Then /goal resume to continue."
                ),
            }

        if state.turns_used >= state.max_turns:
            state.status = "paused"
            state.paused_reason = f"turn budget exhausted ({state.turns_used}/{state.max_turns})"
            save_goal(self.session_id, state)
            return {
                "status": "paused",
                "should_continue": False,
                "continuation_prompt": None,
                "verdict": "continue",
                "reason": reason,
                "message": (
                    f"⏸ Goal paused — {state.turns_used}/{state.max_turns} turns used. "
                    "Use /goal resume to keep going, or /goal clear to stop."
                ),
            }

        save_goal(self.session_id, state)
        cl_total, cl_done, cl_imp, _ = state.checklist_counts()
        progress = ""
        if cl_total:
            progress = f" — {cl_done + cl_imp}/{cl_total} done"
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self.next_continuation_prompt(),
            "verdict": "continue",
            "reason": reason,
            "message": (
                f"↻ Continuing toward goal ({state.turns_used}/{state.max_turns}{progress}): {reason}"
            ),
        }

    def _evaluate_state_phase_b(
        self,
        state: GoalState,
        last_response: str,
        *,
        agent: Any = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, str, bool]:
        """Run the right kind of Phase-B evaluation given current state.

        With a non-empty checklist: harsh per-item evaluation with a bounded
        read_file tool loop.

        With an empty checklist (e.g. decompose failed twice): fall back to
        the legacy freeform judge so the goal still has a way to terminate.
        """
        if not last_response.strip():
            return "continue", "empty response (nothing to evaluate)", False

        if state.checklist:
            # Dump conversation history if we have one. Prefer explicit
            # ``messages`` arg (most reliable); fall back to extracting from
            # the agent instance for back-compat.
            history_path: Optional[Path] = None
            msgs: List[Dict[str, Any]] = []
            if messages:
                msgs = list(messages)
            elif agent is not None:
                msgs = self._extract_agent_messages(agent)
            if msgs:
                history_path = dump_conversation(self.session_id, msgs)
                if history_path is None:
                    logger.debug(
                        "goal: conversation dump failed for session %s",
                        self.session_id,
                    )
            else:
                logger.debug(
                    "goal: no messages available for session %s — judge will run from snippet only",
                    self.session_id,
                )

            parsed, parse_failed = evaluate_checklist(
                state, last_response, history_path=history_path
            )
            self._apply_checklist_updates(state, parsed)

            if state.all_terminal():
                return "done", parsed.get("reason") or "all checklist items terminal", parse_failed
            return "continue", parsed.get("reason") or "checklist progress", parse_failed

        # No checklist — freeform fallback.
        verdict, reason, parse_failed = judge_goal_freeform(state.goal, last_response)
        return verdict, reason, parse_failed

    # --- internal helpers ---------------------------------------------

    @staticmethod
    def _extract_agent_messages(agent: Any) -> List[Dict[str, Any]]:
        """Best-effort extraction of the agent's conversation history.

        Tries common attribute names so we don't tightly couple to AIAgent.
        Returns an empty list when nothing is available.
        """
        for attr in ("messages", "conversation_history", "_messages", "history"):
            try:
                msgs = getattr(agent, attr, None)
                if isinstance(msgs, list) and msgs:
                    return msgs
            except Exception:
                continue
        return []

    @staticmethod
    def _apply_checklist_updates(state: GoalState, parsed: Dict[str, Any]) -> None:
        """Apply judge updates with stickiness: never regress terminal items."""
        now = time.time()
        for upd in parsed.get("updates") or []:
            try:
                idx = int(upd["index"])
            except (KeyError, TypeError, ValueError):
                continue
            if idx < 0 or idx >= len(state.checklist):
                continue
            item = state.checklist[idx]
            if item.status in TERMINAL_ITEM_STATUSES:
                # Stickiness: judge cannot regress a terminal item.
                continue
            new_status = upd.get("status")
            if new_status not in TERMINAL_ITEM_STATUSES:
                continue
            item.status = new_status
            item.completed_at = now
            evidence = upd.get("evidence")
            if evidence:
                item.evidence = evidence

        for new_item in parsed.get("new_items") or []:
            text = (new_item.get("text") or "").strip()
            if not text:
                continue
            state.checklist.append(
                ChecklistItem(
                    text=text,
                    status=ITEM_PENDING,
                    added_by=ADDED_BY_JUDGE,
                    added_at=now,
                )
            )

    # --- continuation prompt ------------------------------------------

    def next_continuation_prompt(self) -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        if not self._state.checklist:
            return CONTINUATION_PROMPT_TEMPLATE.format(goal=self._state.goal)
        cl_total, cl_done, cl_imp, _ = self._state.checklist_counts()
        return CONTINUATION_PROMPT_WITH_CHECKLIST_TEMPLATE.format(
            goal=self._state.goal,
            done=cl_done + cl_imp,
            total=cl_total,
            checklist=self._state.render_checklist(numbered=False),
        )


# Public name kept for back-compat with the previous freeform-only API.
def judge_goal(
    goal: str,
    last_response: str,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
) -> Tuple[str, str, bool]:
    """Back-compat wrapper — defers to the freeform judge."""
    return judge_goal_freeform(goal, last_response, timeout=timeout)


__all__ = [
    "ChecklistItem",
    "GoalState",
    "GoalManager",
    "CONTINUATION_PROMPT_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_CHECKLIST_TEMPLATE",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_MAX_JUDGE_TOOL_CALLS",
    "ITEM_PENDING",
    "ITEM_COMPLETED",
    "ITEM_IMPOSSIBLE",
    "ITEM_MARKERS",
    "TERMINAL_ITEM_STATUSES",
    "VALID_ITEM_STATUSES",
    "load_goal",
    "save_goal",
    "clear_goal",
    "judge_goal",
    "judge_goal_freeform",
    "decompose_goal",
    "evaluate_checklist",
    "conversation_dump_path",
    "dump_conversation",
]
