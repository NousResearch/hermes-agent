"""Persistent session goals — the Ralph loop for Hermes.

A goal is a free-form user objective that stays active across turns. After
each turn completes, a small judge call asks an auxiliary model "is this
goal satisfied by the assistant's last response?". If not, Hermes feeds a
continuation prompt back into the same session and keeps working until the
goal is done, turn budget is exhausted, the user pauses/clears it, or the
user sends a new message (which takes priority and pauses the goal loop).

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
- This module has zero hard dependency on ``cli.HermesCLI`` or the gateway
  runner — both wire the same ``GoalManager`` in.

Nothing in this module touches the agent's system prompt or toolset.
"""

from __future__ import annotations

import json
import logging
import posixpath
import re
import time
from datetime import datetime, timezone
from pathlib import PurePosixPath
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

DEFAULT_MAX_TURNS = 20
DEFAULT_JUDGE_TIMEOUT = 30.0
# Judge output budget. The freeform judge returns a one-line JSON verdict, but
# reasoning models (deepseek-v4, qwq, etc.) burn tokens on hidden reasoning
# before emitting the visible JSON — and the first /goal turn's prompt is
# larger than later turns, which pushes total reply length past tight caps.
# 200 tokens (the original default) reliably truncated the JSON on reasoning
# models, leaving '{"done": true, "reason": "The agent successfully' and
# triggering the auto-pause. 4096 covers reasoning + verdict on every model
# we've live-tested; override via auxiliary.goal_judge.max_tokens for
# specifically constrained setups.
DEFAULT_JUDGE_MAX_TOKENS = 4096
# Cap how much of the last response + recent messages we send to the judge.
_JUDGE_RESPONSE_SNIPPET_CHARS = 4000
# After this many consecutive judge *parse* failures (empty output / non-JSON),
# the loop auto-pauses and points the user at the goal_judge config. API /
# transport errors do NOT count toward this — those are transient. This guards
# against small models (e.g. deepseek-v4-flash) that cannot follow the strict
# JSON reply contract; without it the loop runs until the turn budget is
# exhausted with every reply shaped like `judge returned empty response` or
# `judge reply was not JSON`.
DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES = 3


CONTINUATION_PROMPT_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Continue working toward this goal. Take the next concrete step. "
    "If you believe the goal is complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly and stop."
)

# Used when the user has added one or more /subgoal criteria. Surfaced
# to the agent verbatim so it sees what to target on the next turn,
# and surfaced to the judge so the verdict considers them too.
CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "[Continuing toward your standing goal]\n"
    "Goal: {goal}\n\n"
    "Additional criteria the user added mid-loop:\n"
    "{subgoals_block}\n\n"
    "Continue working toward the goal AND all additional criteria. Take "
    "the next concrete step. If you believe the goal and every "
    "additional criterion are complete, state so explicitly and stop. "
    "If you are blocked and need input from the user, say so clearly "
    "and stop."
)

# GCW-bound goals need stricter continuation than a normal free-form goal:
# the next turn must re-read GCW's machine state before doing more work, and
# must not let the /goal loop invent completion from chat-only progress.
GCW_CONTINUATION_EVIDENCE_BLOCK = (
    "\n\nGCW evidence-aware supervisor requirements:\n"
    "- Treat GCW status/ledger/phase artifacts as the source of truth.\n"
    "- Before continuing or declaring progress, read back the canonical issue "
    "URL, status.json, ledger-updates.jsonl, phase report/handoff paths, "
    "worker/process state, validator/closeout evidence, and missing gates.\n"
    "- Only GCW terminal state `done` plus required evidence can satisfy the "
    "goal successfully. `blocked`, `needs_user`, or `approval_required` may "
    "stop only as a truthful non-success handoff with owner input/risk "
    "acceptance named. `partial`, stale workers, missing artifacts, PR-only "
    "evidence, or missing validator/closeout gates must continue.\n"
    "- /subgoal criteria are judge-visible reminders/candidate gates unless a "
    "later GCW artifact explicitly promotes them into validator/AC gates."
)


JUDGE_SYSTEM_PROMPT = (
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
    '{\"done\": <true|false>, \"reason\": \"<one-sentence rationale>\"}'
)


JUDGE_USER_PROMPT_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "Current time: {current_time}\n\n"
    "Is the goal satisfied?"
)

# Used when the user has added /subgoal criteria. The judge must
# evaluate ALL of them being met, not just the original goal.
JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE = (
    "Goal:\n{goal}\n\n"
    "Additional criteria the user added mid-loop (all must also be "
    "satisfied for the goal to be DONE):\n{subgoals_block}\n\n"
    "Agent's most recent response:\n{response}\n\n"
    "Current time: {current_time}\n\n"
    "Decision: For each numbered criterion above, find concrete "
    "evidence in the agent's response that the criterion is "
    "satisfied. Do not accept generic phrases like 'all requirements "
    "met' or 'implying it was done' — require specific evidence (a "
    "file contents excerpt, an output line, a command result). If "
    "ANY criterion lacks specific evidence in the response, the goal "
    "is NOT done — return CONTINUE.\n\n"
    "Is the goal AND every additional criterion satisfied?"
)

GCW_JUDGE_EVIDENCE_BLOCK = (
    "\n\nGCW-bound goal rule: If this goal involves GCW, /gcw, GaleHarness, "
    "status.json, ledger-updates.jsonl, phase reports/handoffs, or GitHub "
    "Issue closeout, require a concrete evidence summary in the agent's "
    "response. The summary must identify the issue/run, GCW status/phase, "
    "ledger/status readback, worker/process or terminal artifact evidence, "
    "validator/closeout state, missing gates, terminal candidate, and evidence "
    "links/paths. Mark DONE as successful only when the response shows GCW "
    "terminal state `done` and required evidence. Also mark DONE for a "
    "truthful terminal non-success handoff when the response explicitly reports "
    "`blocked`, `needs_user`, or `approval_required` and identifies the owner "
    "input/risk acceptance required. A `partial` state may stop only when the "
    "response cites GCW artifact evidence that defines it as an owner-facing "
    "final handoff; otherwise partial must continue. The reason for any "
    "non-success stop must say it is blocked/needs user/partial handoff, not "
    "completed. If it reports ordinary `partial`, stale worker, missing "
    "report/handoff, PR-only, issue-only, local-only evidence, or any missing "
    "validator/closeout gate, return CONTINUE."
)


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
    last_verdict: Optional[str] = None        # "done" | "continue" | "skipped"
    last_reason: Optional[str] = None
    paused_reason: Optional[str] = None       # why we auto-paused (budget, etc.)
    consecutive_parse_failures: int = 0       # judge-output parse failures in a row
    # User-added criteria appended mid-loop via the /subgoal command.
    # When non-empty the judge prompt and continuation prompt both
    # include them so the agent works toward them and the judge factors
    # them into the verdict. Backwards-compatible: defaults to empty so
    # old state_meta rows load unchanged.
    subgoals: List[str] = field(default_factory=list)
    # Optional GCW Epic supervision contract persisted with the goal. When set,
    # /goal resume can emit deterministic readback before the normal goal loop
    # continues. This is candidate/readback state only; GCW validator and
    # completion guard artifacts remain the closeout authorities.
    epic_goal_supervision: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "GoalState":
        data = json.loads(raw)
        raw_subgoals = data.get("subgoals") or []
        subgoals: List[str] = []
        if isinstance(raw_subgoals, list):
            subgoals = [str(s).strip() for s in raw_subgoals if str(s).strip()]
        raw_epic_goal_supervision = data.get("epic_goal_supervision")
        epic_goal_supervision = None
        if isinstance(raw_epic_goal_supervision, dict):
            try:
                epic_goal_supervision = normalize_epic_goal_supervision(raw_epic_goal_supervision)
            except Exception as exc:
                logger.warning("GoalState: ignoring invalid epic_goal_supervision: %s", exc)
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
            subgoals=subgoals,
            epic_goal_supervision=epic_goal_supervision,
        )

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


def clear_goal(session_id: str) -> None:
    """Mark a goal cleared in the DB (preserved for audit, status=cleared)."""
    state = load_goal(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_goal(session_id, state)


# ──────────────────────────────────────────────────────────────────────
# Epic GCW goal supervision contract
# ──────────────────────────────────────────────────────────────────────

EPIC_GOAL_SUPERVISION_SCHEMA_VERSION = "epic_goal_supervision.v1"
GCW_FORMAL_GATES_SCHEMA_VERSION = "gcw-formal-gates.v1"

ALLOWED_GITHUB_EVIDENCE_REPOS = {
    "wangrenzhu-ola/ai-infra-demand-pool",
    "wangrenzhu-ola/hermes-agent",
    "wangrenzhu-ola/infra-hermes-core-skills",
}

EPIC_SUPERVISOR_DECISIONS = {
    "continue_current_story",
    "stop_with_report",
    "dispatch_next_story",
    "blocked",
    "needs_user",
    "approval_required",
    "partial",
    "parent_closeout_candidate",
}

EPIC_STORY_STATES = {
    "running",
    "blocked",
    "needs_user",
    "approval_required",
    "partial",
    "completed_candidate",
    "completed",
}

EVIDENCE_REF_TYPES = {
    "status_json",
    "workflow_json",
    "ledger",
    "phase_report",
    "phase_handoff",
    "worker",
    "validator",
    "completion_guard",
    "pr",
    "ci",
    "issue_comment",
    "active_readback",
    "formal_gate_export",
    "hierarchy_readback",
    "dogfood",
}

EVIDENCE_REF_STATUSES = {"present", "missing", "stale", "pass", "fail", "unknown"}
_REQUIRED_STORY_CLOSEOUT_EVIDENCE = {"validator", "completion_guard"}
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HEADER_BOUNDARY = r"(?=(?:\s+[\"']?\b(?:authorization|cookie|set-cookie)\b[\"']?\s*[:=])|[\r\n,\]\}]|$)"
_AUTH_HEADER_RE = re.compile(
    r"(?i)\b(authorization)\b[\"']?\s*[:=]\s*(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|.*?)" + _HEADER_BOUNDARY
)
_COOKIE_HEADER_RE = re.compile(
    r"(?i)\b(cookie|set-cookie)\b[\"']?\s*[:=]\s*(?:\"[^\"\r\n]*\"|'[^'\r\n]*'|.*?)" + _HEADER_BOUNDARY
)
_SECRET_VALUE_RE = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|session[_-]?id|password|secret)\b[\"']?"
    r"\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^\s,;&\]}]+)"
)
_ENV_SECRET_VALUE_RE = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|ACCESS[_-]?TOKEN|REFRESH[_-]?TOKEN|TOKEN|SESSION[_-]?ID|PASSWORD|SECRET))\b[\"']?"
    r"\s*[:=]\s*(\"[^\"]*\"|'[^']*'|[^\s,;&\]}]+)"
)
_BARE_AUTH_TOKEN_RE = re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}")
_URL_SECRET_QUERY_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|session[_-]?id|password|secret)=)[^&\s]+"
)


class EpicGoalContractError(ValueError):
    """Raised when an Epic goal supervision contract is invalid."""


def sanitize_evidence_text(value: Any, *, limit: int = 1000) -> str:
    """Return publication-safe bounded text for evidence refs and reports."""
    text = "" if value is None else str(value)
    text = _ANSI_ESCAPE_RE.sub("", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = _AUTH_HEADER_RE.sub(lambda m: f"{m.group(1)}=<redacted>", text)
    text = _COOKIE_HEADER_RE.sub(lambda m: f"{m.group(1)}=<redacted>", text)
    text = _ENV_SECRET_VALUE_RE.sub(lambda m: f"{m.group(1)}=<redacted>", text)
    text = _SECRET_VALUE_RE.sub(lambda m: f"{m.group(1)}=<redacted>", text)
    text = _BARE_AUTH_TOKEN_RE.sub(lambda m: f"{m.group(1)} <redacted>", text)
    text = _URL_SECRET_QUERY_RE.sub(lambda m: f"{m.group(1)}<redacted>", text)
    text = text.replace("```", "`\u200b``")
    if len(text) > limit:
        text = text[:limit] + "… [truncated]"
    return text


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _canonical_repo_from_github_url(url: str) -> Optional[str]:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        return None
    decoded_parts = [unquote(p) for p in parsed.path.split("/") if p]
    if any(part in {".", ".."} or part.startswith("/") for part in decoded_parts):
        return None
    parts = decoded_parts
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}"


def _validate_issue_url(value: Any, *, field_name: str, repo: Optional[str] = None) -> str:
    url = sanitize_evidence_text(value, limit=500).strip()
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if (
        parsed.scheme != "https"
        or parsed.netloc.lower() != "github.com"
        or parsed.params
        or parsed.query
        or parsed.fragment
        or len(parts) != 4
        or parts[2] != "issues"
        or not parts[3].isdigit()
        or any(unquote(p) in {".", ".."} or unquote(p).startswith("/") for p in parts)
    ):
        raise EpicGoalContractError(f"{field_name} must be a canonical GitHub issue URL")
    actual_repo = f"{parts[0]}/{parts[1]}"
    if repo and actual_repo != repo:
        raise EpicGoalContractError(f"{field_name} repo {actual_repo!r} does not match {repo!r}")
    if actual_repo not in ALLOWED_GITHUB_EVIDENCE_REPOS:
        raise EpicGoalContractError(f"{field_name} repo {actual_repo!r} is not allowlisted")
    return url


def _normalize_metadata(raw: Any, *, limit: int = 20) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in list(raw.items())[:limit]:
        clean_key = sanitize_evidence_text(key, limit=80)
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[clean_key] = sanitize_evidence_text(value, limit=500) if isinstance(value, str) else value
        elif isinstance(value, list):
            out[clean_key] = [sanitize_evidence_text(v, limit=200) for v in value[:10]]
        elif isinstance(value, dict):
            out[clean_key] = _normalize_metadata(value, limit=10)
        else:
            out[clean_key] = sanitize_evidence_text(value, limit=200)
    return out


def _normalize_file_evidence_uri(parsed: Any) -> str:
    """Return a canonical local file:// evidence URI or raise."""
    decoded_path = unquote(parsed.path or "")
    if (
        parsed.netloc
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not decoded_path.startswith("/")
        or decoded_path.startswith("//")
    ):
        raise EpicGoalContractError("file evidence refs must use a normalized absolute local file:// path")
    if ".." in PurePosixPath(decoded_path).parts:
        raise EpicGoalContractError("file evidence refs must use a normalized absolute local file:// path")
    normalized_path = posixpath.normpath(decoded_path)
    if normalized_path != decoded_path:
        raise EpicGoalContractError("file evidence refs must use a normalized absolute local file:// path")
    return "file://" + quote(normalized_path, safe="/")


def _normalize_parent_terminal_candidate(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    raise EpicGoalContractError("parent_terminal_candidate must be a boolean or null")


def normalize_evidence_ref(raw: Any, *, required: bool = False) -> Dict[str, Any]:
    """Normalize and validate a machine-readable, publication-safe evidence ref."""
    if not isinstance(raw, dict):
        raise EpicGoalContractError("evidence ref must be an object")
    ref_type = sanitize_evidence_text(raw.get("type"), limit=80).strip()
    if ref_type not in EVIDENCE_REF_TYPES:
        raise EpicGoalContractError(f"unsupported evidence ref type: {ref_type!r}")
    uri = sanitize_evidence_text(raw.get("uri"), limit=1000).strip()
    parsed = urlparse(uri)
    if parsed.scheme == "https":
        repo = _canonical_repo_from_github_url(uri)
        if repo not in ALLOWED_GITHUB_EVIDENCE_REPOS:
            raise EpicGoalContractError(f"evidence ref GitHub repo {repo!r} is not allowlisted")
    elif parsed.scheme == "file":
        uri = _normalize_file_evidence_uri(parsed)
    else:
        raise EpicGoalContractError("evidence ref uri must be file:// or allowlisted https://github.com")
    status = sanitize_evidence_text(raw.get("status") or "unknown", limit=40).strip()
    if status not in EVIDENCE_REF_STATUSES:
        raise EpicGoalContractError(f"unsupported evidence ref status: {status!r}")
    return {
        "type": ref_type,
        "uri": uri,
        "status": status,
        "required": bool(raw.get("required", required)),
        "observed_at": sanitize_evidence_text(raw.get("observed_at") or _now_iso(), limit=80),
        "metadata": _normalize_metadata(raw.get("metadata")),
    }


def normalize_epic_goal_supervision(raw: Any) -> Dict[str, Any]:
    """Validate and normalize an ``epic_goal_supervision.v1`` contract.

    The returned dict is safe to persist in Goal state, phase handoffs, and
    human-facing reports. It intentionally emits candidate/decision state only;
    GCW validator and completion guard remain the closeout authorities.
    """
    if not isinstance(raw, dict):
        raise EpicGoalContractError("Epic goal supervision contract must be an object")
    if raw.get("schema_version") != EPIC_GOAL_SUPERVISION_SCHEMA_VERSION:
        raise EpicGoalContractError("unsupported epic goal supervision schema_version")
    source_repo = sanitize_evidence_text(raw.get("source_repo"), limit=200).strip()
    if source_repo not in ALLOWED_GITHUB_EVIDENCE_REPOS:
        raise EpicGoalContractError(f"source_repo {source_repo!r} is not allowlisted")

    parent_epic_issue = _validate_issue_url(raw.get("parent_epic_issue"), field_name="parent_epic_issue", repo=source_repo)
    goal_issue = _validate_issue_url(raw.get("goal_issue"), field_name="goal_issue", repo=source_repo)
    feature_issue = _validate_issue_url(raw.get("feature_issue"), field_name="feature_issue", repo=source_repo)
    ordered_story_issues = [
        _validate_issue_url(v, field_name="ordered_story_issues[]", repo=source_repo)
        for v in (raw.get("ordered_story_issues") or [])
    ]
    if not ordered_story_issues:
        raise EpicGoalContractError("ordered_story_issues must be non-empty")
    active_story_issue = _validate_issue_url(raw.get("active_story_issue"), field_name="active_story_issue", repo=source_repo)
    if active_story_issue not in ordered_story_issues:
        raise EpicGoalContractError("active_story_issue must appear in ordered_story_issues")
    blocked_story_issues = [
        _validate_issue_url(v, field_name="blocked_story_issues[]", repo=source_repo)
        for v in (raw.get("blocked_story_issues") or [])
    ]

    story_statuses: Dict[str, Dict[str, Any]] = {}
    for key, value in (raw.get("story_statuses") or {}).items():
        if not isinstance(value, dict):
            raise EpicGoalContractError("story_statuses values must be objects")
        story_key = sanitize_evidence_text(key, limit=40).strip()
        state = sanitize_evidence_text(value.get("state") or "running", limit=40).strip()
        if state not in EPIC_STORY_STATES:
            raise EpicGoalContractError(f"unsupported story state: {state!r}")
        refs = [normalize_evidence_ref(ref) for ref in (value.get("last_evidence_refs") or [])]
        story_statuses[story_key] = {
            "state": state,
            "run_id": sanitize_evidence_text(value.get("run_id"), limit=200),
            "run_dir": sanitize_evidence_text(value.get("run_dir"), limit=1000),
            "artifact_base": sanitize_evidence_text(value.get("artifact_base"), limit=1000),
            "last_evidence_refs": refs,
            "missing_gates": [sanitize_evidence_text(g, limit=200) for g in (value.get("missing_gates") or [])],
            "metadata": _normalize_metadata(value.get("metadata")),
        }

    out = {
        "schema_version": EPIC_GOAL_SUPERVISION_SCHEMA_VERSION,
        "parent_epic_issue": parent_epic_issue,
        "goal_issue": goal_issue,
        "feature_issue": feature_issue,
        "source_repo": source_repo,
        "ordered_story_issues": ordered_story_issues,
        "active_story_issue": active_story_issue,
        "blocked_story_issues": blocked_story_issues,
        "story_statuses": story_statuses,
        "hierarchy_evidence_refs": [normalize_evidence_ref(ref) for ref in (raw.get("hierarchy_evidence_refs") or [])],
        "formal_gate_exports": [normalize_evidence_ref(ref) for ref in (raw.get("formal_gate_exports") or [])],
        "missing_gates": [sanitize_evidence_text(g, limit=200) for g in (raw.get("missing_gates") or [])],
        "next_allowed_action": sanitize_evidence_text(raw.get("next_allowed_action") or "continue_current_story", limit=80),
        "parent_terminal_candidate": _normalize_parent_terminal_candidate(raw.get("parent_terminal_candidate")),
        "resume_readback_required": bool(raw.get("resume_readback_required", True)),
        "updated_at": sanitize_evidence_text(raw.get("updated_at") or _now_iso(), limit=80),
        "metadata": _normalize_metadata(raw.get("metadata")),
    }
    if out["next_allowed_action"] not in EPIC_SUPERVISOR_DECISIONS:
        raise EpicGoalContractError("next_allowed_action is not a supported Epic supervisor decision")
    return out


def evaluate_epic_goal_resume_readback(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the next Epic supervisor action from normalized evidence.

    This is a readback/candidate decision helper. It never marks the parent Epic
    closed and never treats self-report, PR-only, or local-artifact-only refs as
    sufficient Story closeout evidence.
    """
    c = normalize_epic_goal_supervision(contract)
    active_url = c["active_story_issue"]
    active_number = active_url.rstrip("/").split("/")[-1]
    story = c["story_statuses"].get(active_number) or {}
    state = story.get("state") or "running"
    parent_missing_gates = list(c.get("missing_gates") or [])
    story_missing_gates = list(story.get("missing_gates") or [])
    refs = story.get("last_evidence_refs") or []
    passed_required_types = {
        ref["type"]
        for ref in refs
        if ref.get("type") in _REQUIRED_STORY_CLOSEOUT_EVIDENCE and _is_trusted_closeout_ref(ref)
    }
    missing_required_evidence = sorted(_REQUIRED_STORY_CLOSEOUT_EVIDENCE - passed_required_types)

    if state in {"blocked", "needs_user", "approval_required", "partial"}:
        decision = state
    elif story_missing_gates or missing_required_evidence or state not in {"completed", "completed_candidate"}:
        decision = "continue_current_story"
    else:
        idx = c["ordered_story_issues"].index(active_url)
        if idx < len(c["ordered_story_issues"]) - 1:
            decision = "dispatch_next_story"
        else:
            decision = "stop_with_report" if parent_missing_gates else "parent_closeout_candidate"

    return {
        "schema_version": "epic_goal_resume_readback.v1",
        "decision": decision,
        "active_story_issue": active_url,
        "active_story_state": state,
        "missing_gates": parent_missing_gates + story_missing_gates,
        "missing_required_evidence": missing_required_evidence,
        "evidence_refs": refs,
        "parent_terminal_candidate": decision == "parent_closeout_candidate",
        "observed_at": _now_iso(),
    }


def _is_trusted_closeout_ref(ref: Dict[str, Any]) -> bool:
    """Return True only for validator/completion refs backed by closeout authority."""
    if ref.get("status") != "pass":
        return False
    uri = ref.get("uri") or ""
    parsed = urlparse(uri)
    if parsed.scheme != "https" or _canonical_repo_from_github_url(uri) not in ALLOWED_GITHUB_EVIDENCE_REPOS:
        return False
    parts = [unquote(p) for p in parsed.path.split("/") if p]
    if len(parts) >= 5 and parts[2:4] == ["actions", "runs"] and parts[4].isdigit():
        return True
    if len(parts) >= 4 and parts[2] in {"check-runs", "checks"} and parts[3].isdigit():
        return True
    # Trust comes from constrained allowlisted HTTPS closeout-authority
    # provenance; local files, issue/PR/comment URLs, and arbitrary repo paths
    # cannot self-assert closeout authority through contract metadata.
    return False


def read_epic_goal_resume_readback_json(path: str) -> Dict[str, Any]:
    """Load an epic_goal_supervision.v1 JSON file and return readback JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return evaluate_epic_goal_resume_readback(json.load(f))


def read_epic_goal_supervision_contract_json(path: str) -> Dict[str, Any]:
    """Load and normalize an epic_goal_supervision.v1 contract for persistence."""
    with open(path, "r", encoding="utf-8") as f:
        return normalize_epic_goal_supervision(json.load(f))


def format_epic_goal_resume_readback(readback: Dict[str, Any]) -> str:
    """Render deterministic, sanitized one-screen readback for /goal resume."""
    decision = sanitize_evidence_text(readback.get("decision"), limit=80)
    active = sanitize_evidence_text(readback.get("active_story_issue"), limit=300)
    state = sanitize_evidence_text(readback.get("active_story_state"), limit=80)
    missing_gates = [sanitize_evidence_text(g, limit=120) for g in (readback.get("missing_gates") or [])]
    missing_evidence = [
        sanitize_evidence_text(g, limit=120) for g in (readback.get("missing_required_evidence") or [])
    ]
    lines = [
        "Epic goal resume readback:",
        f"- decision: {decision}",
        f"- active_story_issue: {active}",
        f"- active_story_state: {state}",
        f"- parent_terminal_candidate: {bool(readback.get('parent_terminal_candidate'))}",
    ]
    if missing_gates:
        lines.append(f"- missing_gates: {', '.join(missing_gates)}")
    if missing_evidence:
        lines.append(f"- missing_required_evidence: {', '.join(missing_evidence)}")
    refs = readback.get("evidence_refs") or []
    if refs:
        lines.append("- evidence_refs:")
        for ref in refs[:8]:
            if not isinstance(ref, dict):
                continue
            ref_type = sanitize_evidence_text(ref.get("type"), limit=80)
            status = sanitize_evidence_text(ref.get("status"), limit=40)
            uri = sanitize_evidence_text(ref.get("uri"), limit=300)
            lines.append(f"  - {ref_type} [{status}] {uri}")
        if len(refs) > 8:
            lines.append(f"  - … {len(refs) - 8} more refs")
    lines.append("Authority boundary: this is readback/candidate state only; GCW validator and completion guard remain closeout authorities.")
    return "\n".join(lines)


def _main(argv: Optional[List[str]] = None) -> int:
    """Small agent-native CLI for GCW readback probes.

    Usage: python -m hermes_cli.goals epic-readback /path/to/contract.json
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m hermes_cli.goals")
    sub = parser.add_subparsers(dest="command", required=True)
    epic = sub.add_parser("epic-readback", help="emit epic_goal_resume_readback.v1 JSON")
    epic.add_argument("contract_json", help="path to an epic_goal_supervision.v1 JSON contract")
    args = parser.parse_args(argv)
    if args.command == "epic-readback":
        try:
            payload = read_epic_goal_resume_readback_json(args.contract_json)
        except (EpicGoalContractError, OSError, json.JSONDecodeError) as exc:
            import sys

            error_payload = {
                "schema_version": "epic_goal_resume_readback_error.v1",
                "error": exc.__class__.__name__,
                "message": sanitize_evidence_text(str(exc), limit=1000),
            }
            print(json.dumps(error_payload, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            return 1
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())


# ──────────────────────────────────────────────────────────────────────
# Judge
# ──────────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "… [truncated]"


_GCW_GOAL_RE = re.compile(
    r"(\bgcw\b|/gcw|galeharness|status\.json|ledger-updates\.jsonl|"
    r"phase report|handoff|completion guard|github\.com/.+/issues/\d+)",
    re.IGNORECASE,
)


def _is_gcw_bound_goal(goal: str, subgoals: Optional[List[str]] = None) -> bool:
    """Heuristic: should /goal apply the GCW evidence-aware contract?"""
    text = "\n".join([goal or "", *[s or "" for s in (subgoals or [])]])
    return bool(_GCW_GOAL_RE.search(text))


_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _goal_judge_max_tokens() -> int:
    """Resolve auxiliary.goal_judge.max_tokens, falling back to the default.

    ``load_config()`` is cached on the config file's (mtime, size), so calling
    this once per judge turn is cheap. A non-positive or non-int value falls
    back to the default rather than crashing the goal loop.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        value = (
            (cfg.get("auxiliary") or {})
            .get("goal_judge", {})
            .get("max_tokens", DEFAULT_JUDGE_MAX_TOKENS)
        )
        value = int(value)
        if value > 0:
            return value
    except Exception:
        pass
    return DEFAULT_JUDGE_MAX_TOKENS


def _parse_judge_response(raw: str) -> Tuple[bool, str, bool]:
    """Parse the judge's reply. Fail-open to ``(False, "<reason>", parse_failed)``.

    Returns ``(done, reason, parse_failed)``. ``parse_failed`` is True when the
    judge returned output that couldn't be interpreted as the expected JSON
    verdict (empty body, prose, malformed JSON). Callers use that flag to
    auto-pause after N consecutive parse failures so a weak judge model
    doesn't silently burn the turn budget.
    """
    if not raw:
        return False, "judge returned empty response", True

    text = raw.strip()

    # Strip markdown code fences the model may wrap JSON in.
    if text.startswith("```"):
        text = text.strip("`")
        # Peel off leading json/JSON/etc tag
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1:]

    # First try: parse the whole blob.
    data: Optional[Dict[str, Any]] = None
    try:
        data = json.loads(text)
    except Exception:
        # Second try: pull the first JSON object out.
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                data = json.loads(match.group(0))
            except Exception:
                data = None

    if not isinstance(data, dict):
        return False, f"judge reply was not JSON: {_truncate(raw, 200)!r}", True

    done_val = data.get("done")
    if isinstance(done_val, str):
        done = done_val.strip().lower() in {"true", "yes", "1", "done"}
    else:
        done = bool(done_val)
    reason = str(data.get("reason") or "").strip()
    if not reason:
        reason = "no reason provided"
    return done, reason, False


def judge_goal(
    goal: str,
    last_response: str,
    *,
    timeout: float = DEFAULT_JUDGE_TIMEOUT,
    subgoals: Optional[List[str]] = None,
) -> Tuple[str, str, bool]:
    """Ask the auxiliary model whether the goal is satisfied.

    Returns ``(verdict, reason, parse_failed)`` where verdict is ``"done"``,
    ``"continue"``, or ``"skipped"`` (when the judge couldn't be reached).

    ``parse_failed`` is True only when the judge call succeeded but its output
    was unusable (empty or non-JSON). API/transport errors return False — they
    are transient and should fail-open silently. Callers use this flag to
    auto-pause after N consecutive parse failures (see
    ``DEFAULT_MAX_CONSECUTIVE_PARSE_FAILURES``).

    ``subgoals`` is an optional list of user-added criteria (from
    ``/subgoal``) that the judge must also factor into its DONE/CONTINUE
    decision. When non-empty the prompt switches to the with-subgoals
    template; otherwise behavior is identical to the original judge.

    This is deliberately fail-open: any error returns ``("continue", "...", False)``
    so a broken judge doesn't wedge progress — the turn budget and the
    consecutive-parse-failures auto-pause are the backstops.
    """
    if not goal.strip():
        return "skipped", "empty goal", False
    if not last_response.strip():
        # No substantive reply this turn — almost certainly not done yet.
        return "continue", "empty response (nothing to evaluate)", False

    try:
        from agent.auxiliary_client import get_auxiliary_extra_body, get_text_auxiliary_client
    except Exception as exc:
        logger.debug("goal judge: auxiliary client import failed: %s", exc)
        return "continue", "auxiliary client unavailable", False

    try:
        client, model = get_text_auxiliary_client("goal_judge")
    except Exception as exc:
        logger.debug("goal judge: get_text_auxiliary_client failed: %s", exc)
        return "continue", "auxiliary client unavailable", False

    if client is None or not model:
        return "continue", "no auxiliary client configured", False

    # Build the prompt — pick the with-subgoals variant when applicable.
    clean_subgoals = [s.strip() for s in (subgoals or []) if s and s.strip()]
    current_time = datetime.now(tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    if clean_subgoals:
        subgoals_block = "\n".join(
            f"- {i}. {text}" for i, text in enumerate(clean_subgoals, start=1)
        )
        prompt = JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            subgoals_block=_truncate(subgoals_block, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            current_time=current_time,
        )
    else:
        prompt = JUDGE_USER_PROMPT_TEMPLATE.format(
            goal=_truncate(goal, 2000),
            response=_truncate(last_response, _JUDGE_RESPONSE_SNIPPET_CHARS),
            current_time=current_time,
        )
    if _is_gcw_bound_goal(goal, clean_subgoals):
        prompt += GCW_JUDGE_EVIDENCE_BLOCK

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=_goal_judge_max_tokens(),
            timeout=timeout,
            extra_body=get_auxiliary_extra_body() or None,
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
    logger.info("goal judge: verdict=%s reason=%s", verdict, _truncate(reason, 120))
    return verdict, reason, parse_failed


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
    - ``evaluate_after_turn(last_response)`` — call the judge, update state,
      and return a decision dict the caller uses to drive the next turn.
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
        return self._state is not None and self._state.status in {"active", "paused"}

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status in {"cleared",}:
            return "No active goal. Set one with /goal <text>."
        turns = f"{s.turns_used}/{s.max_turns} turns"
        sub = f", {len(s.subgoals)} subgoal{'s' if len(s.subgoals) != 1 else ''}" if s.subgoals else ""
        if s.status == "active":
            return f"⊙ Goal (active, {turns}{sub}): {s.goal}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Goal (paused, {turns}{sub}{extra}): {s.goal}"
        if s.status == "done":
            return f"✓ Goal done ({turns}{sub}): {s.goal}"
        return f"Goal ({s.status}, {turns}{sub}): {s.goal}"

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

    def resume(
        self,
        *,
        reset_budget: bool = True,
        epic_goal_supervision: Optional[Dict[str, Any]] = None,
    ) -> Optional[GoalState]:
        if not self._state:
            return None
        if epic_goal_supervision is not None:
            self._state.epic_goal_supervision = normalize_epic_goal_supervision(epic_goal_supervision)
        self._state.status = "active"
        self._state.paused_reason = None
        if reset_budget:
            self._state.turns_used = 0
        save_goal(self.session_id, self._state)
        return self._state

    def attach_epic_goal_supervision(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        """Persist a normalized Epic supervision contract on the current goal."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        self._state.epic_goal_supervision = normalize_epic_goal_supervision(contract)
        save_goal(self.session_id, self._state)
        return self._state.epic_goal_supervision

    def epic_goal_resume_readback(self) -> Optional[Dict[str, Any]]:
        """Return deterministic Epic readback for the persisted contract, if any."""
        if not self._state or not self._state.epic_goal_supervision:
            return None
        return evaluate_epic_goal_resume_readback(self._state.epic_goal_supervision)

    def formatted_epic_goal_resume_readback(self) -> Optional[str]:
        readback = self.epic_goal_resume_readback()
        if readback is None:
            return None
        return format_epic_goal_resume_readback(readback)

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

    def add_subgoal(self, text: str) -> str:
        """Append a user-added criterion to the active goal. Requires
        ``has_goal()``; raises ``RuntimeError`` otherwise.

        Returns the cleaned text so the caller can show it back to the user.
        """
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        text = (text or "").strip()
        if not text:
            raise ValueError("subgoal text is empty")
        self._state.subgoals.append(text)
        save_goal(self.session_id, self._state)
        return text

    def remove_subgoal(self, index_1based: int) -> str:
        """Remove a subgoal by 1-based index. Returns the removed text."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        idx = int(index_1based) - 1
        if idx < 0 or idx >= len(self._state.subgoals):
            raise IndexError(
                f"index out of range (1..{len(self._state.subgoals)})"
            )
        removed = self._state.subgoals.pop(idx)
        save_goal(self.session_id, self._state)
        return removed

    def clear_subgoals(self) -> int:
        """Wipe all subgoals. Returns the previous count."""
        if self._state is None or not self.has_goal():
            raise RuntimeError("no active goal")
        prev = len(self._state.subgoals)
        self._state.subgoals = []
        save_goal(self.session_id, self._state)
        return prev

    def render_subgoals(self) -> str:
        """Public helper for the /subgoal slash command."""
        if self._state is None:
            return "(no active goal)"
        if not self._state.subgoals:
            return "(no subgoals — use /subgoal <text> to add criteria)"
        return self._state.render_subgoals_block()

    # --- the main entry point called after every turn -----------------

    def evaluate_after_turn(
        self,
        last_response: str,
        *,
        user_initiated: bool = True,
    ) -> Dict[str, Any]:
        """Run the judge and update state. Return a decision dict.

        ``user_initiated`` distinguishes a real user prompt (True) from a
        continuation prompt we fed ourselves (False). Both increment
        ``turns_used`` because both consume model budget.

        Decision keys:
          - ``status``: current goal status after update
          - ``should_continue``: bool — caller should fire another turn
          - ``continuation_prompt``: str or None
          - ``verdict``: "done" | "continue" | "skipped" | "inactive"
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

        verdict, reason, parse_failed = judge_goal(
            state.goal, last_response, subgoals=state.subgoals or None
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
        # verdict N turns in a row. Points the user at the goal_judge config
        # so they can route this side task to a model that follows the
        # contract (e.g. google/gemini-3-flash-preview). Without this guard,
        # weak judge models burn the entire turn budget returning prose or
        # empty strings.
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
        return {
            "status": "active",
            "should_continue": True,
            "continuation_prompt": self.next_continuation_prompt(),
            "verdict": "continue",
            "reason": reason,
            "message": (
                f"↻ Continuing toward goal ({state.turns_used}/{state.max_turns}): {reason}"
            ),
        }

    def next_continuation_prompt(self) -> Optional[str]:
        if not self._state or self._state.status != "active":
            return None
        if self._state.subgoals:
            prompt = CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE.format(
                goal=self._state.goal,
                subgoals_block=self._state.render_subgoals_block(),
            )
        else:
            prompt = CONTINUATION_PROMPT_TEMPLATE.format(goal=self._state.goal)
        if _is_gcw_bound_goal(self._state.goal, self._state.subgoals):
            prompt += GCW_CONTINUATION_EVIDENCE_BLOCK
        return prompt


__all__ = [
    "GoalState",
    "GoalManager",
    "CONTINUATION_PROMPT_TEMPLATE",
    "CONTINUATION_PROMPT_WITH_SUBGOALS_TEMPLATE",
    "JUDGE_USER_PROMPT_TEMPLATE",
    "JUDGE_USER_PROMPT_WITH_SUBGOALS_TEMPLATE",
    "GCW_CONTINUATION_EVIDENCE_BLOCK",
    "GCW_JUDGE_EVIDENCE_BLOCK",
    "DEFAULT_MAX_TURNS",
    "load_goal",
    "save_goal",
    "clear_goal",
    "judge_goal",
    "EpicGoalContractError",
    "normalize_evidence_ref",
    "normalize_epic_goal_supervision",
    "evaluate_epic_goal_resume_readback",
    "read_epic_goal_resume_readback_json",
    "read_epic_goal_supervision_contract_json",
    "format_epic_goal_resume_readback",
    "sanitize_evidence_text",
]
