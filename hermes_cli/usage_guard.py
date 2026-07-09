"""Usage-warning compact-mode helpers.

This module is deliberately small and dependency-light so tool paths can ask a
single question: "is compact mode active for this task/session?"  The agent loop
activates it when context/usage pressure appears; tools then enforce narrower
reads/searches and compact terminal output without needing to know why.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional


USAGE_GUARD_ENV = "HERMES_USAGE_GUARD_ACTIVE"
USAGE_GUARD_REASON_ENV = "HERMES_USAGE_GUARD_REASON"
MAX_READ_LINES_AFTER_WARNING = 200
MAX_SEARCH_RESULTS_AFTER_WARNING = 25
MAX_TERMINAL_CHARS_AFTER_WARNING = 12_000

REQUIRED_COMPACT_HANDOFF_FIELDS = (
    "task_id",
    "phase",
    "touched_files",
    "recent_diff_summary",
    "failing_tests",
    "missing_symbols",
    "blocked_commands",
    "declared_artifacts",
    "next_small_step",
    "must_not_repeat",
)

REVIEWER_VERDICT_CATEGORIES = (
    "actionable",
    "trade-off",
    "contract-misread",
    "noise",
)

SYNTHESIZER_EVIDENCE_SOURCES = (
    "curated parent summaries",
    "parent summaries",
    "explicit artifacts",
    "artifacts",
    "capped worker logs",
    "task comments",
    "verifier gate",
    "blackboard",
)

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_TASK_IDS: set[str] = set()
_ACTIVE_SESSION_IDS: set[str] = set()
_ACTIVE_GLOBAL = False

_SECRET_PATTERNS = (
    re.compile(r"sk-proj-[A-Za-z0-9_-]{12,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{20,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"(?i)\b(password|token|secret|api[_-]?key)\s*[:=]\s*"
        r"['\"]?[A-Za-z0-9_./+=-]{12,}"
    ),
)


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "compact"}


def _clean_text(value: Any, *, max_chars: int = 2_000) -> str:
    text = "" if value is None else str(value)
    text = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 15].rstrip() + " [truncated]"
    return _redact_obvious_secrets(text)


def _clean_list(values: Any, *, max_items: int = 20, max_chars: int = 500) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        values = [values]
    if not isinstance(values, Iterable):
        values = [values]
    cleaned: list[str] = []
    for value in values:
        text = _clean_text(value, max_chars=max_chars)
        if text:
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _redact_obvious_secrets(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def _contains_obvious_secret(value: Any) -> bool:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(value)
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def activate_usage_guard_after_warning(
    *,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    reason: str = "usage_warning",
) -> None:
    """Activate compact mode for this process, optionally scoped by id."""
    global _ACTIVE_GLOBAL
    with _ACTIVE_LOCK:
        if task_id:
            _ACTIVE_TASK_IDS.add(str(task_id))
        if session_id:
            _ACTIVE_SESSION_IDS.add(str(session_id))
        if not task_id and not session_id:
            _ACTIVE_GLOBAL = True
    os.environ[USAGE_GUARD_ENV] = "1"
    if reason:
        os.environ[USAGE_GUARD_REASON_ENV] = _clean_text(reason, max_chars=160)


def reset_usage_guard_for_tests() -> None:
    global _ACTIVE_GLOBAL
    with _ACTIVE_LOCK:
        _ACTIVE_GLOBAL = False
        _ACTIVE_TASK_IDS.clear()
        _ACTIVE_SESSION_IDS.clear()
    os.environ.pop(USAGE_GUARD_ENV, None)
    os.environ.pop(USAGE_GUARD_REASON_ENV, None)


def usage_guard_active(
    *,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    active: Optional[bool] = None,
) -> bool:
    if active is not None:
        return bool(active)
    if _truthy(os.environ.get(USAGE_GUARD_ENV)):
        return True
    with _ACTIVE_LOCK:
        if _ACTIVE_GLOBAL:
            return True
        if task_id and str(task_id) in _ACTIVE_TASK_IDS:
            return True
        if session_id and str(session_id) in _ACTIVE_SESSION_IDS:
            return True
    return False


def _mode_policy(
    *,
    mode: Optional[str],
    task_id: Optional[str],
    session_id: Optional[str],
    active: Optional[bool],
):
    guard_active = usage_guard_active(
        task_id=task_id,
        session_id=session_id,
        active=active,
    )
    try:
        from tools.tool_output_limits import get_mode_output_policy

        return get_mode_output_policy(mode, usage_guard_active=guard_active), guard_active
    except Exception:
        return None, guard_active


def build_compact_handoff_packet(**kwargs: Any) -> dict[str, Any]:
    """Return a value-free compact handoff packet with all required fields."""
    packet = {
        "task_id": _clean_text(kwargs.get("task_id") or "unknown", max_chars=200),
        "phase": _clean_text(kwargs.get("phase") or "unknown", max_chars=200),
        "touched_files": _clean_list(kwargs.get("touched_files")),
        "recent_diff_summary": _clean_text(kwargs.get("recent_diff_summary")),
        "failing_tests": _clean_list(kwargs.get("failing_tests")),
        "missing_symbols": _clean_list(kwargs.get("missing_symbols")),
        "blocked_commands": _clean_list(kwargs.get("blocked_commands")),
        "declared_artifacts": _clean_list(kwargs.get("declared_artifacts")),
        "next_small_step": _clean_text(kwargs.get("next_small_step")),
        "must_not_repeat": _clean_list(kwargs.get("must_not_repeat")),
    }
    return packet


def validate_compact_handoff_packet(packet: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(packet, Mapping):
        return ["packet must be a mapping"]
    for field in REQUIRED_COMPACT_HANDOFF_FIELDS:
        if field not in packet:
            errors.append(f"missing required field: {field}")
    for field in (
        "touched_files",
        "failing_tests",
        "missing_symbols",
        "blocked_commands",
        "declared_artifacts",
        "must_not_repeat",
    ):
        if field in packet and not isinstance(packet[field], list):
            errors.append(f"{field} must be a list")
    if _contains_obvious_secret(packet):
        errors.append("packet appears to contain an unredacted secret")
    return errors


_SESSION_STOP_MARKERS = (
    "session_stop",
    "session stop",
    "context_estimate >=",
    "last_input >=",
)
_CLOSEOUT_ONLY_MARKERS = (
    "final",
    "closeout",
    "report",
    "summary",
    "cleanup",
    "verification",
    "verify",
    "smoke",
    "ci",
    "deploy gate",
    "post-main",
)
_IMPLEMENTATION_WORK_MARKERS = (
    "implement",
    "write code",
    "patch",
    "fix failing",
    "add feature",
    "migration",
    "schema",
)


def _model_policy_note(*, required_model: str, fixed_model_policy: bool) -> str:
    model = _clean_text(required_model or "gpt-5.5", max_chars=80)
    if fixed_model_policy:
        return (
            f"Fixed model policy requires {model}; model switch requires explicit "
            "user approval."
        )
    return f"Keep recovery on {model} unless the operator explicitly changes policy."


def classify_usage_guard_closeout_state(
    *,
    usage_guard_reason: str,
    repo_clean: bool = False,
    known_dirty_changes_recorded: bool = False,
    tests_run: Any = None,
    verification_run: Any = None,
    build_run: bool = False,
    lint_run: bool = False,
    smoke_run: bool = False,
    remaining_work: Any = None,
    implementation_commands_needed: bool = False,
    required_model: str = "gpt-5.5",
    fixed_model_policy: bool = False,
) -> dict[str, Any]:
    """Classify whether usage-guard ``session_stop`` should force closeout.

    This is deliberately value-free and packet-shaped.  It lets callers turn a
    high-context, post-green state into compact finalization instead of asking
    the model for another broad history replay just to produce a final answer.
    """

    reason = _clean_text(usage_guard_reason, max_chars=300).lower()
    remaining = _clean_list(remaining_work, max_items=12, max_chars=240)
    remaining_text = " ".join(item.lower() for item in remaining)
    tests = _clean_list(tests_run, max_items=12, max_chars=240)
    verification = _clean_list(verification_run, max_items=12, max_chars=240)

    reasons: list[str] = []
    if not any(marker in reason for marker in _SESSION_STOP_MARKERS):
        reasons.append("usage_guard_not_session_stop")
    if not (repo_clean or known_dirty_changes_recorded):
        reasons.append("repo_state_not_recorded")
    if not (tests or verification or build_run or lint_run or smoke_run):
        reasons.append("verification_evidence_missing")
    if implementation_commands_needed or any(
        marker in remaining_text for marker in _IMPLEMENTATION_WORK_MARKERS
    ):
        reasons.append("implementation_work_remains")
    if remaining and not any(marker in remaining_text for marker in _CLOSEOUT_ONLY_MARKERS):
        reasons.append("remaining_work_not_closeout_only")

    closeout_required = not reasons
    return {
        "status": (
            "compact_finalization_required"
            if closeout_required
            else "continue_narrow_work"
        ),
        "recommended_action": (
            "compact_finalization_prompt"
            if closeout_required
            else "narrow_next_step"
        ),
        "reasons": reasons,
        "remaining_work": remaining,
        "tests_run": tests,
        "verification_run": verification,
        "required_model": _clean_text(required_model or "gpt-5.5", max_chars=80),
        "model_policy_note": _model_policy_note(
            required_model=required_model,
            fixed_model_policy=fixed_model_policy,
        ),
        "repo_state_recorded": bool(repo_clean or known_dirty_changes_recorded),
    }


def reviewer_verdict_instruction(
    *,
    task_id: Optional[str] = None,
    review_attempt: Optional[int] = None,
) -> str:
    task = _clean_text(task_id or "unknown", max_chars=200)
    attempt = f"review attempt {int(review_attempt)}" if review_attempt else "review attempt"
    categories = ", ".join(REVIEWER_VERDICT_CATEGORIES)
    return (
        f"Reviewer verdict-first mode for task {task} ({attempt}). "
        "Do not start another broad read/search/delegation. Produce a verdict "
        f"comment first with categories: {categories}. "
        "After the verdict, request at most one narrow follow-up if one exact "
        "missing evidence check is still needed. full-suite unrelated failures "
        "are advisory under scoped acceptance; touched/scope/safety/contract/"
        "migration/unknown failures remain blockers."
    )


def validate_reviewer_verdict_packet(packet: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(packet, Mapping):
        return ["packet must be a mapping"]
    categories = packet.get("categories")
    if not isinstance(categories, Mapping):
        errors.append("categories must be a mapping")
        categories = {}
    for category in REVIEWER_VERDICT_CATEGORIES:
        if category not in categories:
            errors.append(f"missing verdict category: {category}")
        elif not isinstance(categories[category], list):
            errors.append(f"{category} must be a list")
    follow_up = packet.get("follow_up")
    if isinstance(follow_up, list):
        if len(follow_up) > 1:
            errors.append("at most one narrow follow-up")
        follow_up_item = follow_up[0] if follow_up else None
    else:
        follow_up_item = follow_up
    if follow_up_item:
        if not isinstance(follow_up_item, Mapping):
            errors.append("follow_up must be a mapping")
        else:
            scope = _clean_text(follow_up_item.get("scope"), max_chars=200)
            question = _clean_text(follow_up_item.get("question"), max_chars=500)
            if not scope or not question:
                errors.append("follow_up requires scope and question")
    return errors


def synthesizer_mode_instruction(*, task_id: Optional[str] = None) -> str:
    task = _clean_text(task_id or "unknown", max_chars=200)
    sources = ", ".join(SYNTHESIZER_EVIDENCE_SOURCES[:5])
    return (
        f"Synthesizer evidence-first mode for task {task}. Read only curated "
        "parent summaries, explicit artifacts, task comments/blackboard, and "
        "capped worker logs by default. If required evidence is absent, emit a "
        "missing-evidence request naming the exact missing item and checked "
        f"sources ({sources}). Do not run broad find. Do not run unbounded rg. "
        "Do not run huge terminal commands. Use short terminal excerpts that "
        "include command, exit code, artifact paths, and the relevant output."
    )


def synthesizer_missing_evidence_request(
    *,
    task_id: Optional[str] = None,
    missing: Any = None,
    checked_sources: Any = None,
) -> dict[str, Any]:
    return {
        "request_type": "missing-evidence",
        "task_id": _clean_text(task_id or "unknown", max_chars=200),
        "missing_evidence": _clean_list(missing, max_items=20, max_chars=500),
        "checked_sources": _clean_list(checked_sources, max_items=20, max_chars=200),
        "next_step": (
            "Provide the named curated artifact/source or unblock a narrow "
            "read of that exact path; do not broaden repository discovery."
        ),
    }


def validate_synthesizer_evidence_packet(packet: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(packet, Mapping):
        return ["packet must be a mapping"]
    if packet.get("request_type") != "missing-evidence":
        errors.append("request_type must be missing-evidence")
    missing = packet.get("missing_evidence")
    if not isinstance(missing, list) or not missing:
        errors.append("missing_evidence must be a non-empty list")
    checked = packet.get("checked_sources")
    if not isinstance(checked, list):
        errors.append("checked_sources must be a list")
        checked = []
    allowed = {source.lower() for source in SYNTHESIZER_EVIDENCE_SOURCES}
    for source in checked:
        cleaned = _clean_text(source, max_chars=200).lower()
        if cleaned and cleaned not in allowed:
            errors.append(f"unsupported synthesizer evidence source: {source}")
    if _contains_obvious_secret(packet):
        errors.append("packet appears to contain an unredacted secret")
    return errors


def _reviewer_verdict_suffix(mode_name: str, guard_active: bool) -> str:
    if mode_name != "reviewer" or not guard_active:
        return ""
    return (
        " Reviewer mode: write a verdict comment first with categories "
        "actionable, trade-off, contract-misread, noise; then request at most "
        "one narrow follow-up."
    )


def _synthesizer_evidence_suffix(mode_name: str) -> str:
    if mode_name != "synthesizer":
        return ""
    return (
        " Synthesizer mode: use curated parent summaries, explicit artifacts, "
        "task comments/blackboard, and capped worker logs first; if evidence "
        "is absent, emit a missing-evidence request instead of broad discovery."
    )


def compact_instruction_after_code_patch(
    *,
    task_id: Optional[str] = None,
    touched_files: Optional[Iterable[str]] = None,
) -> str:
    fields = ", ".join(REQUIRED_COMPACT_HANDOFF_FIELDS)
    files = ", ".join(_clean_list(touched_files, max_items=8)) or "unknown"
    task = _clean_text(task_id or "unknown", max_chars=200)
    return (
        "USAGE GUARD ACTIVE after code patches. On the next response, either "
        "patch one narrow issue using the already-known context, or emit a "
        "compact handoff packet and stop broad exploration. "
        f"task_id={task}; touched_files={files}. "
        f"The compact handoff packet must include: {fields}. "
        "Do not run broad file reads, full-repo dumps, find *, or giant rg output."
    )


def read_request_denial_after_warning(
    *,
    path: str,
    offset: int,
    limit: int,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    active: Optional[bool] = None,
    mode: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    policy, guard_active = _mode_policy(
        mode=mode,
        task_id=task_id,
        session_id=session_id,
        active=active,
    )
    policy_requires_narrow = bool(
        policy is not None and getattr(policy, "require_narrow_reads", False)
    )
    if not guard_active and not policy_requires_narrow:
        return None
    max_lines = (
        int(getattr(policy, "read_max_lines", MAX_READ_LINES_AFTER_WARNING))
        if policy is not None else MAX_READ_LINES_AFTER_WARNING
    )
    if int(limit or 0) <= max_lines:
        return None
    mode_name = getattr(policy, "mode", mode or "builder") if policy is not None else mode or "builder"
    reason = "usage/context warning" if guard_active else f"audit-first mode {mode_name}"
    reviewer_suffix = _reviewer_verdict_suffix(mode_name, guard_active)
    synthesizer_suffix = _synthesizer_evidence_suffix(mode_name)
    return {
        "error": (
            f"{reason}: broad read_file requests are blocked. Use a narrow "
            f"line range, exact symbol, or limit <= {max_lines}."
            f"{reviewer_suffix}"
            f"{synthesizer_suffix}"
        ),
        "usage_guard": "active" if guard_active else "mode_policy",
        "mode": mode_name,
        "path": path,
        "offset": offset,
        "requested_limit": limit,
        "max_limit": max_lines,
    }


def search_request_denial_after_warning(
    *,
    pattern: str,
    target: str,
    path: str,
    file_glob: Optional[str],
    limit: int,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    active: Optional[bool] = None,
    mode: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    policy, guard_active = _mode_policy(
        mode=mode,
        task_id=task_id,
        session_id=session_id,
        active=active,
    )
    policy_bounded = bool(
        policy is not None and (
            getattr(policy, "compact_terminal_output", False)
            or getattr(policy, "require_narrow_reads", False)
        )
    )
    if not guard_active and not policy_bounded:
        return None
    max_results = (
        int(getattr(policy, "search_max_results", MAX_SEARCH_RESULTS_AFTER_WARNING))
        if policy is not None else MAX_SEARCH_RESULTS_AFTER_WARNING
    )
    mode_name = getattr(policy, "mode", mode or "builder") if policy is not None else mode or "builder"
    reviewer_suffix = _reviewer_verdict_suffix(mode_name, guard_active)
    synthesizer_suffix = _synthesizer_evidence_suffix(mode_name)
    broad_pattern = str(pattern or "").strip() in {"", "*", ".*", "."}
    broad_path = str(path or ".").strip() in {"", ".", "./"}
    if broad_pattern and broad_path and not file_glob:
        return {
            "error": (
                "Usage guard active: broad repository searches are blocked. "
                "Use a specific pattern plus file_glob/path."
                f"{reviewer_suffix}"
                f"{synthesizer_suffix}"
            ),
            "usage_guard": "active" if guard_active else "mode_policy",
            "mode": mode_name,
            "pattern": pattern,
            "path": path,
        }
    if int(limit or 0) > max_results:
        return {
            "error": (
                "Usage guard active: search result caps are reduced after a "
                f"usage/context warning. Use limit <= {max_results} "
                "or narrow pattern/file_glob."
                f"{reviewer_suffix}"
                f"{synthesizer_suffix}"
            ),
            "usage_guard": "active" if guard_active else "mode_policy",
            "mode": mode_name,
            "pattern": pattern,
            "requested_limit": limit,
            "max_limit": max_results,
        }
    return None


def terminal_command_denial_after_warning(
    command: str,
    *,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    active: Optional[bool] = None,
    mode: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    policy, guard_active = _mode_policy(
        mode=mode,
        task_id=task_id,
        session_id=session_id,
        active=active,
    )
    policy_bounded = bool(
        policy is not None and getattr(policy, "compact_terminal_output", False)
    )
    if not guard_active and not policy_bounded:
        return None
    mode_name = getattr(policy, "mode", mode or "builder") if policy is not None else mode or "builder"
    normalized = " ".join(str(command or "").strip().lower().split())
    blocked = (
        re.search(r"(^|[;&|])\s*find\s+(\.|\*)(\s|$)", normalized)
        or re.search(r"(^|[;&|])\s*rg\s+--files(\s*$|\s+[;&|])", normalized)
        or ("get-childitem" in normalized and "-recurse" in normalized
            and "-filter" not in normalized and "-include" not in normalized)
        or re.search(r"\b(cat|type)\s+\*", normalized)
    )
    if not blocked:
        return None
    reviewer_suffix = _reviewer_verdict_suffix(mode_name, guard_active)
    synthesizer_suffix = _synthesizer_evidence_suffix(mode_name)
    return {
        "output": "",
        "exit_code": -1,
        "error": (
            "Usage guard active: broad terminal enumeration is blocked after "
            "a usage/context warning. Use rg with a specific pattern, file_glob, "
            "or an exact path/range."
            f"{reviewer_suffix}"
            f"{synthesizer_suffix}"
        ),
        "status": "error",
        "usage_guard": "active" if guard_active else "mode_policy",
        "mode": mode_name,
    }


def _extract_artifact_paths_from_text(text: str, *, max_items: int = 6) -> list[str]:
    candidates = re.findall(
        r"(?:(?:[A-Za-z]:[\\/])|/)[^\s'\"<>|]{1,220}\."
        r"(?:md|markdown|txt|json|csv|html|htm|zip|tar|gz|tgz|pdf|png|jpg|jpeg|webp|log)",
        text or "",
    )
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        item = _clean_text(candidate.rstrip(".,;:)"), max_chars=240)
        key = item.lower()
        if item and key not in seen:
            cleaned.append(item)
            seen.add(key)
        if len(cleaned) >= max_items:
            break
    return cleaned


def compact_terminal_output_after_warning(
    output: str,
    *,
    exit_code: int,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    active: Optional[bool] = None,
    max_chars: Optional[int] = None,
    mode: Optional[str] = None,
    command: Optional[str] = None,
    artifact_paths: Optional[Iterable[str]] = None,
) -> str:
    policy, guard_active = _mode_policy(
        mode=mode,
        task_id=task_id,
        session_id=session_id,
        active=active,
    )
    policy_compacts = bool(
        policy is not None and getattr(policy, "compact_terminal_output", False)
    )
    if not guard_active and not policy_compacts:
        return output
    cap = int(
        max_chars
        or (getattr(policy, "terminal_max_chars", MAX_TERMINAL_CHARS_AFTER_WARNING)
            if policy is not None else MAX_TERMINAL_CHARS_AFTER_WARNING)
    )
    if len(output or "") <= cap:
        return output
    mode_name = getattr(policy, "mode", mode or "builder") if policy is not None else mode or "builder"
    head_chars = max(80, int(cap * 0.4))
    tail_chars = max(80, cap - head_chars)
    omitted = max(0, len(output) - head_chars - tail_chars)
    if mode_name == "synthesizer":
        artifacts = _clean_list(artifact_paths, max_items=6, max_chars=240)
        if not artifacts:
            artifacts = _extract_artifact_paths_from_text(
                "\n".join(filter(None, [command or "", output or ""]))
            )
        artifact_text = ", ".join(artifacts) if artifacts else "(none detected)"
        notice = (
            "\n\n[OUTPUT COMPACTED; SYNTHESIZER TERMINAL SUMMARY: "
            f"command={_clean_text(command or '(unknown)', max_chars=500)}; "
            f"exit_code={exit_code}; artifacts={artifact_text}; "
            f"omitted_chars={omitted}; short excerpt preserved]\n\n"
        )
    else:
        notice = (
            "\n\n[OUTPUT COMPACTED AFTER USAGE WARNING: "
            f"exit_code={exit_code}; omitted_chars={omitted}; "
            "first and last output segments preserved]\n\n"
        )
    return output[:head_chars].rstrip() + notice + output[-tail_chars:].lstrip()


@dataclass(frozen=True)
class NoToolSpinEvidence:
    api_call_count: int
    db_tool_call_count: int
    runtime_tool_activity_seen: bool
    same_session_resume_attempted: bool
    runtime_log_checked: bool = True


@dataclass(frozen=True)
class NoToolSpinClassification:
    is_no_tool_spin: bool
    reason: str


def classify_no_tool_spin(evidence: NoToolSpinEvidence) -> NoToolSpinClassification:
    if evidence.api_call_count <= 0:
        return NoToolSpinClassification(False, "no API-call drift observed")
    if not evidence.same_session_resume_attempted:
        return NoToolSpinClassification(
            False,
            "same-session resume/steer has not been attempted yet",
        )
    if evidence.db_tool_call_count > 0:
        return NoToolSpinClassification(False, "DB shows tool activity")
    if not evidence.runtime_log_checked:
        return NoToolSpinClassification(False, "runtime logs were not checked")
    if evidence.runtime_tool_activity_seen:
        return NoToolSpinClassification(
            False,
            "runtime logs show tool activity despite DB tool_call_count=0",
        )
    return NoToolSpinClassification(
        True,
        "DB and runtime logs both show no tool activity after same-session resume",
    )
