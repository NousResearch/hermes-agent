"""Control-plane side of the unified Hermes agent harness.

This module owns the seven-case ``harness-core`` regression suite and the
sidecar evidence it uses. It records small, structured facts about turns and
learning mutations without storing message content or becoming part of the
model-visible prompt.  Use ``agent.harness.HermesHarness`` when callers need the
single top-level harness facade.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from collections import Counter, deque
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
_RECENT_LIMIT = 50
CORE_HARNESS_NAME = "harness-core"
TRACE_SCHEMA = {
    "name": "hermes.turn_trace",
    "version": SCHEMA_VERSION,
    "content_policy": "metadata_only",
}
_ACTIVE_TRACE_ID: ContextVar[Optional[str]] = ContextVar(
    "hermes_harness_trace_id",
    default=None,
)
_ACTIVE_SESSION_ID: ContextVar[Optional[str]] = ContextVar(
    "hermes_harness_session_id",
    default=None,
)

_SECRET_KEY_RE = re.compile(
    r"(api[_-]?key|token|password|secret|credential|authorization|cookie)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"("
    r"sk-[A-Za-z0-9_-]{16,}|"
    r"xox[baprs]-[A-Za-z0-9-]{16,}|"
    r"gh[pousr]_[A-Za-z0-9_]{16,}|"
    r"AIza[0-9A-Za-z_-]{16,}|"
    r"Bearer\s+[A-Za-z0-9._~+/=-]{16,}"
    r")",
    re.IGNORECASE,
)
_LONG_VALUE_LIMIT = 240
_PAYLOAD_DEPTH_LIMIT = 6
_PAYLOAD_LIST_LIMIT = 12

_DEFAULT_HARNESS_CASES = [
    {
        "id": "goal-skill-expansion",
        "description": "/goal wrapping a slash skill expands to skill payload before model submission",
        "checks": ["tests/tui_gateway/test_goal_command.py::test_goal_set_with_skill_command_sends_expanded_skill_payload"],
    },
    {
        "id": "codex-trace-projection",
        "description": "Codex app-server turns are projected into Hermes traces",
        "checks": ["tests/agent/test_harness_control_plane.py::test_record_turn_result_emits_codex_trace"],
    },
    {
        "id": "harness-event-safety",
        "description": "Harness events redact secrets and summarize large payloads",
        "checks": [
            "tests/agent/test_harness_control_plane.py::test_harness_event_redacts_and_summarizes_payload",
            "tests/agent/test_tool_executor_harness.py::test_tool_start_records_argument_shape_without_raw_content",
        ],
    },
    {
        "id": "dashboard-learning-health",
        "description": "Dashboard harness health endpoint returns a complete degraded-safe summary",
        "checks": ["tests/hermes_cli/test_web_server.py::TestWebServerEndpoints::test_harness_learning_health_endpoint"],
    },
    {
        "id": "memory-admission-routing",
        "description": "Task progress is routed away from durable prompt memory",
        "checks": ["tests/agent/test_harness_control_plane.py::test_memory_admission_classifies_task_progress_as_session_search"],
    },
    {
        "id": "skill-mutation-contract",
        "description": "Skill mutations create draft contracts that require verification before promotion",
        "checks": ["tests/agent/test_harness_control_plane.py::test_skill_registry_tracks_draft_and_promotion"],
    },
    {
        "id": "mutation-contract-summary",
        "description": "Mutation contracts are summarized by component and status",
        "checks": ["tests/agent/test_harness_control_plane.py::test_mutation_contract_summary_tracks_component_status"],
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _expires_iso(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def _harness_dir() -> Path:
    return get_hermes_home() / "harness"


def _jsonl_path(name: str) -> Path:
    return _harness_dir() / name


def _json_path(name: str) -> Path:
    return _harness_dir() / name


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _text_digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _text_fingerprint_fields(prefix: str, text: Optional[str]) -> Dict[str, Any]:
    """Return content-free metadata for a possibly sensitive free-text value."""
    value = text or ""
    return {
        f"{prefix}_present": bool(value),
        f"{prefix}_chars": len(value),
        f"{prefix}_sha256": _text_digest(value) if value else None,
    }


def _list_fingerprint_fields(prefix: str, values: Optional[Iterable[Any]]) -> Dict[str, Any]:
    """Return count/hash metadata for a list without persisting raw entries."""
    items = [str(item) for item in (values or [])]
    encoded = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return {
        f"{prefix}_count": len(items),
        f"{prefix}_sha256": _text_digest(encoded) if items else None,
    }


def _safe_turn_exit_reason(value: Any) -> Optional[str]:
    """Return a bounded reason code without persisting raw error text."""
    text = str(value or "").strip()
    if not text:
        return None

    head = text.split("(", 1)[0].strip()
    if re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", head or ""):
        candidate = head.lower()
        if not _SECRET_KEY_RE.search(candidate) and not _SECRET_VALUE_RE.search(candidate):
            return candidate

    lower = text.lower()
    if "route" in lower or "contract" in lower:
        return "route_contract"
    if "tool" in lower:
        return "tool_error"
    if "context" in lower or "token" in lower:
        return "context_limit"
    if "budget" in lower or "iteration" in lower or "max" in lower:
        return "iteration_budget"
    if "guardrail" in lower:
        return "guardrail_halt"
    if "interrupt" in lower:
        return "interrupted"
    if "error" in lower or "exception" in lower or "traceback" in lower:
        return "runtime_error"
    return "other"


def classify_turn_failure(record: Dict[str, Any]) -> str:
    """Classify a normalized turn trace into a stable failure taxonomy."""
    route_proof = record.get("route_proof") if isinstance(record, dict) else None
    if isinstance(route_proof, dict):
        contract = route_proof.get("contract") if isinstance(route_proof.get("contract"), dict) else {}
        if contract.get("status") == "blocked":
            return "route_contract"

    if record.get("interrupted"):
        return "interrupted"
    if record.get("completed") is True and not record.get("error_present"):
        return "none"

    reason = str(record.get("turn_exit_reason") or "").lower()
    if "route" in reason or "contract" in reason:
        return "route_contract"
    if "tool" in reason:
        return "tool_error"
    if "context" in reason or "token" in reason:
        return "context_limit"
    if "budget" in reason or "iteration" in reason or "max" in reason:
        return "iteration_budget"
    if record.get("error_present"):
        return "runtime_error"
    if record.get("completed") is False:
        return "incomplete"
    return "none"


def _compact_text(value: Any, *, limit: int = _LONG_VALUE_LIMIT) -> str:
    text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return f"{text[:limit - 28]}... [sha256:{_text_digest(text)}]"


def _sanitize_payload(value: Any, *, depth: int = 0) -> Any:
    """Return a trace-safe payload with secrets and huge text removed."""
    if depth >= _PAYLOAD_DEPTH_LIMIT:
        return "[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _SECRET_VALUE_RE.search(value):
            return "[REDACTED]"
        if _SECRET_KEY_RE.search(value) and len(value) > 16:
            return "[REDACTED]"
        return _compact_text(value)
    if isinstance(value, (list, tuple)):
        items = [
            _sanitize_payload(item, depth=depth + 1)
            for item in list(value)[:_PAYLOAD_LIST_LIMIT]
        ]
        if len(value) > _PAYLOAD_LIST_LIMIT:
            items.append(f"[+{len(value) - _PAYLOAD_LIST_LIMIT} more]")
        return items
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in list(value.items())[:_PAYLOAD_LIST_LIMIT * 2]:
            key_text = str(key)
            if _SECRET_KEY_RE.search(key_text):
                cleaned[key_text] = "[REDACTED]"
            else:
                cleaned[key_text] = _sanitize_payload(item, depth=depth + 1)
        if len(value) > _PAYLOAD_LIST_LIMIT * 2:
            cleaned["_truncated_keys"] = len(value) - (_PAYLOAD_LIST_LIMIT * 2)
        return cleaned
    return _compact_text(repr(value))


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: deque[Dict[str, Any]] | List[Dict[str, Any]]
    rows = deque(maxlen=limit) if limit else []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except OSError:
        return []
    return list(rows)


def _read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def current_trace_id() -> Optional[str]:
    return _ACTIVE_TRACE_ID.get()


def current_session_id() -> Optional[str]:
    return _ACTIVE_SESSION_ID.get()


def record_harness_event(
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    component: Optional[str] = None,
    runtime: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a trace-safe harness event.

    Events intentionally carry metadata, not model-visible content. The goal is
    observability for routing, state, tools, verification, and UI behavior.
    """
    record = {
        "schema_version": SCHEMA_VERSION,
        "event_id": _new_id("evt"),
        "trace_id": trace_id or current_trace_id(),
        "session_id": session_id or current_session_id(),
        "recorded_at": _now_iso(),
        "profile": current_profile_name(),
        "event_type": event_type,
        "component": component,
        "runtime": runtime,
        "payload": _sanitize_payload(payload or {}),
    }
    _append_jsonl(_jsonl_path("harness-events.jsonl"), record)
    return record


def start_turn_trace(
    agent: Any,
    *,
    user_message: Optional[str] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and bind a turn trace for the current execution context."""
    trace_id = _new_id("turn")
    session_id = getattr(agent, "session_id", None)
    _ACTIVE_TRACE_ID.set(trace_id)
    _ACTIVE_SESSION_ID.set(session_id)
    try:
        setattr(agent, "_harness_trace_id", trace_id)
        setattr(agent, "_harness_trace_started_at", time.time())
    except Exception:
        pass

    api_mode = getattr(agent, "api_mode", None) or ""
    runtime = "codex_app_server" if api_mode == "codex_app_server" else "hermes"
    payload = {
        "task_id": task_id,
        "platform": getattr(agent, "platform", None),
        "provider": getattr(agent, "provider", None),
        "model": getattr(agent, "model", None),
        "api_mode": api_mode,
        "route_proof": getattr(agent, "_route_proof", None),
        "cwd": os.environ.get("TERMINAL_CWD") or os.getcwd(),
        "user_message_chars": len(user_message or ""),
        "user_message_sha256": _text_digest(user_message or "") if user_message else None,
    }
    return record_harness_event(
        "turn.start",
        payload,
        trace_id=trace_id,
        session_id=session_id,
        component="conversation_loop",
        runtime=runtime,
    )


def record_goal_decision(
    *,
    session_id: Optional[str],
    action: str,
    goal: Optional[str] = None,
    should_continue: Optional[bool] = None,
    message: Optional[str] = None,
    turn_count: Optional[int] = None,
) -> Dict[str, Any]:
    return record_harness_event(
        "goal.judge" if action == "judge" else f"goal.{action}",
        {
            "goal_chars": len(goal or ""),
            "goal_sha256": _text_digest(goal or "") if goal else None,
            "should_continue": should_continue,
            **_text_fingerprint_fields("message", message),
            "turn_count": turn_count,
        },
        session_id=session_id,
        component="goal",
    )


def record_skill_load(
    *,
    session_id: Optional[str],
    name: str,
    command: Optional[str] = None,
    arg: Optional[str] = None,
    source: str = "slash",
) -> Dict[str, Any]:
    return record_harness_event(
        "skill.loaded",
        {
            "name": name,
            "command": command,
            "arg_chars": len(arg or ""),
            "arg_sha256": _text_digest(arg or "") if arg else None,
            "source": source,
        },
        session_id=session_id,
        component="skills",
    )


def record_approval_decision(
    *,
    session_id: Optional[str],
    choice: str,
    resolved: Optional[int] = None,
    resolve_all: bool = False,
) -> Dict[str, Any]:
    return record_harness_event(
        "approval.decided",
        {"choice": choice, "resolved": resolved, "resolve_all": resolve_all},
        session_id=session_id,
        component="approval",
    )


def record_verification_result(
    *,
    name: str,
    status: str,
    command: Optional[str] = None,
    result: Optional[str] = None,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    return record_harness_event(
        "verification.ran",
        {
            "name": name,
            "status": status,
            "command": command,
            "result": result,
        },
        trace_id=trace_id,
        session_id=session_id,
        component="verification",
    )


def current_profile_name() -> str:
    explicit = os.environ.get("HERMES_PROFILE") or os.environ.get("HERMES_ACTIVE_PROFILE")
    if explicit:
        return explicit
    home = get_hermes_home()
    if home.parent.name == "profiles" and home.name:
        return home.name
    return "default"


def _message_tool_names(messages: Iterable[Dict[str, Any]]) -> List[str]:
    names: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            if isinstance(fn, dict) and fn.get("name"):
                names.append(str(fn["name"]))
        tool_name = msg.get("tool_name") or msg.get("name")
        if msg.get("role") == "tool" and tool_name:
            names.append(str(tool_name))
    return names


def record_turn_result(agent: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    """Append one normalized turn trace from a completed AIAgent result."""
    messages = result.get("messages") or []
    tool_names = _message_tool_names(messages if isinstance(messages, list) else [])
    api_mode = getattr(agent, "api_mode", None) or result.get("api_mode") or ""
    runtime = "codex_app_server" if api_mode == "codex_app_server" else "hermes"
    trace_id = (
        result.get("harness_trace_id")
        or getattr(agent, "_harness_trace_id", None)
        or current_trace_id()
        or _new_id("turn")
    )
    started_at = getattr(agent, "_harness_trace_started_at", None)
    duration_s = (time.time() - started_at) if isinstance(started_at, (int, float)) else None
    raw_turn_exit_reason = result.get("turn_exit_reason")
    record = {
        "schema_version": SCHEMA_VERSION,
        "trace_schema": dict(TRACE_SCHEMA),
        "trace_id": trace_id,
        "recorded_at": _now_iso(),
        "profile": current_profile_name(),
        "session_id": getattr(agent, "session_id", None),
        "platform": getattr(agent, "platform", None),
        "provider": result.get("provider") or getattr(agent, "provider", None),
        "model": result.get("model") or getattr(agent, "model", None),
        "api_mode": api_mode,
        "route_proof": getattr(agent, "_route_proof", None),
        "runtime": runtime,
        "cwd": os.environ.get("TERMINAL_CWD") or os.getcwd(),
        "completed": bool(result.get("completed")),
        "partial": bool(result.get("partial")),
        "interrupted": bool(result.get("interrupted")),
        "turn_exit_reason": _safe_turn_exit_reason(raw_turn_exit_reason),
        **_text_fingerprint_fields("turn_exit_reason_raw", raw_turn_exit_reason),
        **_text_fingerprint_fields("error", result.get("error")),
        "api_calls": int(result.get("api_calls") or 0),
        "tool_call_count": len(tool_names),
        "tool_names": sorted(set(tool_names)),
        "input_tokens": int(result.get("input_tokens") or 0),
        "output_tokens": int(result.get("output_tokens") or 0),
        "reasoning_tokens": int(result.get("reasoning_tokens") or 0),
        "estimated_cost_usd": result.get("estimated_cost_usd"),
    }
    record["failure_kind"] = classify_turn_failure(record)
    if duration_s is not None:
        record["duration_s"] = duration_s
    if result.get("codex_thread_id"):
        record["codex_thread_id"] = result.get("codex_thread_id")
    if result.get("codex_turn_id"):
        record["codex_turn_id"] = result.get("codex_turn_id")
    _append_jsonl(_jsonl_path("turn-traces.jsonl"), record)
    record_harness_event(
        "turn.finish",
        {
            "completed": record["completed"],
            "partial": record["partial"],
            "interrupted": record["interrupted"],
            "turn_exit_reason": record.get("turn_exit_reason"),
            "error_present": record.get("error_present"),
            "error_chars": record.get("error_chars"),
            "error_sha256": record.get("error_sha256"),
            "failure_kind": record.get("failure_kind"),
            "tool_call_count": record["tool_call_count"],
            "tool_names": record["tool_names"],
            "duration_s": duration_s,
        },
        trace_id=trace_id,
        session_id=record.get("session_id"),
        component="conversation_loop",
        runtime=runtime,
    )
    return record


_SECRET_PATTERNS = (
    re.compile(r"\b(api[_-]?key|token|password|secret|credential)\b", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_]{20,}\.[A-Za-z0-9_.-]{20,}\b"),
)
_TASK_PROGRESS_PATTERNS = (
    re.compile(r"\b(commit|sha|pr|pull request|issue)\s*#?[0-9a-f]{4,}\b", re.IGNORECASE),
    re.compile(r"\b(fixed|implemented|completed|merged|deployed|released|submitted)\b", re.IGNORECASE),
    re.compile(r"\b(phase|task|todo|blocker|status|checklist)\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+tests?\s+(passed|failed|skipped)\b", re.IGNORECASE),
)
_PROCEDURE_PATTERNS = (
    re.compile(r"\b(workflow|procedure|steps|run this|command|debug|install|setup|verify)\b", re.IGNORECASE),
    re.compile(r"`[^`]+`"),
)
_USER_PREF_PATTERNS = (
    re.compile(r"\b(user|carson)\s+(prefers|wants|expects|likes|hates|requires)\b", re.IGNORECASE),
    re.compile(r"\b(preference|communication style|voice|tone|pet peeve)\b", re.IGNORECASE),
)
_ENV_FACT_PATTERNS = (
    re.compile(r"\b(project|repo|workspace|environment|tool|service|runtime)\s+(uses|is|has|runs)\b", re.IGNORECASE),
    re.compile(r"(/home/|/mnt/|~/.hermes|\.venv|python|rust|node|wsl|ubuntu)", re.IGNORECASE),
)


def classify_memory_admission(
    *,
    action: str,
    target: str,
    content: Optional[str] = None,
    old_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify whether a memory write belongs in durable prompt memory."""
    text = (content or old_text or "").strip()
    lowered = text.lower()
    reasons: List[str] = []
    kind = "unknown"
    decision = "review"
    route = "human_or_future_turn_review"
    expires_at: Optional[str] = _expires_iso(30)

    if action == "remove":
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "admit",
            "kind": "removal",
            "durability": "mutation",
            "route": "memory",
            "expires_at": None,
            "reasons": ["removing memory does not add prompt surface"],
        }

    if any(p.search(text) for p in _SECRET_PATTERNS):
        kind = "secret_or_sensitive"
        decision = "reject"
        route = "do_not_store"
        expires_at = None
        reasons.append("looks like secret or credential material")
    elif any(p.search(lowered) for p in _TASK_PROGRESS_PATTERNS):
        kind = "task_progress_or_session_outcome"
        decision = "reject"
        route = "session_search"
        expires_at = _expires_iso(7)
        reasons.append("task progress belongs in transcripts, not durable memory")
    elif any(p.search(text) for p in _PROCEDURE_PATTERNS):
        kind = "procedure_candidate"
        decision = "review"
        route = "skill_manage"
        expires_at = _expires_iso(14)
        reasons.append("procedure should become a skill if reusable")
    elif any(p.search(text) for p in _USER_PREF_PATTERNS):
        kind = "user_preference"
        decision = "admit"
        route = "memory"
        expires_at = None
        reasons.append("stable user preference")
    elif target == "memory" and any(p.search(text) for p in _ENV_FACT_PATTERNS):
        kind = "environment_or_project_fact"
        decision = "admit"
        route = "memory"
        expires_at = None
        reasons.append("stable environment or project fact")
    elif not text:
        kind = "empty"
        decision = "reject"
        route = "do_not_store"
        expires_at = None
        reasons.append("empty content")
    else:
        reasons.append("not clearly durable; review before relying on it")

    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "kind": kind,
        "durability": "durable" if decision == "admit" else "ephemeral",
        "route": route,
        "expires_at": expires_at,
        "reasons": reasons,
    }


def record_memory_admission(
    *,
    action: str,
    target: str,
    result: Dict[str, Any],
    content: Optional[str] = None,
    old_text: Optional[str] = None,
    admission: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    admission = admission or classify_memory_admission(
        action=action,
        target=target,
        content=content,
        old_text=old_text,
    )
    record = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": _now_iso(),
        "profile": current_profile_name(),
        "action": action,
        "target": target,
        "success": bool(result.get("success")),
        "decision": admission.get("decision"),
        "kind": admission.get("kind"),
        "route": admission.get("route"),
        "expires_at": admission.get("expires_at"),
        "content_chars": len(content or ""),
        "old_text_chars": len(old_text or ""),
    }
    _append_jsonl(_jsonl_path("memory-admissions.jsonl"), record)
    return record


def _skill_registry_path() -> Path:
    return _json_path("skill-registry.json")


def record_skill_mutation(
    *,
    action: str,
    name: str,
    result: Dict[str, Any],
    category: Optional[str] = None,
    file_path: Optional[str] = None,
    origin: str = "foreground",
) -> Dict[str, Any]:
    success = bool(result.get("success"))
    status = "removed" if action == "delete" and success else "draft"
    promotion_status = "not_applicable" if status == "removed" else "needs_verification"
    record = {
        "schema_version": SCHEMA_VERSION,
        "recorded_at": _now_iso(),
        "profile": current_profile_name(),
        "action": action,
        "name": name,
        "category": category,
        "file_path": file_path,
        "origin": origin,
        "success": success,
        "status": status if success else "failed",
        "promotion_status": promotion_status if success else "blocked",
    }
    _append_jsonl(_jsonl_path("skill-drafts.jsonl"), record)

    if success:
        registry = _read_json(_skill_registry_path(), {"schema_version": SCHEMA_VERSION, "skills": {}})
        skills = registry.setdefault("skills", {})
        skill = skills.setdefault(name, {"name": name, "created_at": record["recorded_at"], "events": 0})
        skill.update({
            "updated_at": record["recorded_at"],
            "profile": record["profile"],
            "latest_action": action,
            "status": record["status"],
            "promotion_status": record["promotion_status"],
            "origin": origin,
            "category": category,
        })
        skill["events"] = int(skill.get("events") or 0) + 1
        _write_json(_skill_registry_path(), registry)
        if status != "removed":
            record_mutation_contract(
                component="skill",
                action=action,
                target=name,
                evidence=[record.get("recorded_at")],
                prediction=(
                    "Skill guidance should improve future runs for this task class; "
                    "promotion requires targeted replay or test evidence."
                ),
                rollback=file_path or name,
                verification=[
                    "pytest tests/agent/test_harness_control_plane.py -q",
                ],
                status="draft",
                origin=origin,
            )
    return record


def _mutation_contracts_path() -> Path:
    return _jsonl_path("mutation-contracts.jsonl")


def record_mutation_contract(
    *,
    component: str,
    action: str,
    target: str,
    evidence: Optional[List[str]] = None,
    prediction: Optional[str] = None,
    rollback: Optional[str] = None,
    verification: Optional[List[str]] = None,
    status: str = "draft",
    origin: str = "foreground",
) -> Dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "mutation_id": _new_id("mut"),
        "recorded_at": _now_iso(),
        "profile": current_profile_name(),
        "component": component,
        "action": action,
        **_text_fingerprint_fields("target", target),
        **_list_fingerprint_fields("evidence", evidence),
        **_text_fingerprint_fields("prediction", prediction),
        **_text_fingerprint_fields("rollback", rollback),
        **_list_fingerprint_fields("verification", verification),
        "status": status,
        "origin": origin,
    }
    _append_jsonl(_mutation_contracts_path(), record)
    return record


def promote_skill(name: str, *, evidence: Optional[str] = None) -> Dict[str, Any]:
    registry = _read_json(_skill_registry_path(), {"schema_version": SCHEMA_VERSION, "skills": {}})
    skills = registry.setdefault("skills", {})
    skill = skills.setdefault(name, {"name": name, "created_at": _now_iso(), "events": 0})
    gate = evaluate_promotion_gate(component="skill", target=name, required_suites=[CORE_HARNESS_NAME])
    if gate.get("status") != "passed":
        skill.update({
            "updated_at": _now_iso(),
            "status": "draft",
            "promotion_status": "blocked",
            "promotion_gate_status": gate.get("status"),
            **_text_fingerprint_fields("promotion_evidence", evidence),
        })
        _write_json(_skill_registry_path(), registry)
        return skill

    skill.update({
        "updated_at": _now_iso(),
        "status": "promoted",
        "promotion_status": "verified",
        "promotion_gate_status": gate.get("status"),
        **_text_fingerprint_fields("promotion_evidence", evidence),
    })
    _write_json(_skill_registry_path(), registry)
    return skill


def _replay_corpus_path() -> Path:
    return _jsonl_path("replay-corpus.jsonl")


def record_replay_case(
    *,
    source_trace_id: str,
    failure_kind: str,
    checks: Optional[List[str]] = None,
    status: str = "candidate",
    note: Optional[str] = None,
    source: str = "historical_failure",
) -> Dict[str, Any]:
    """Record a content-free replay-corpus candidate for a historical failure."""
    record = {
        "schema_version": SCHEMA_VERSION,
        "replay_id": _new_id("replay"),
        "recorded_at": _now_iso(),
        "profile": current_profile_name(),
        "source": source,
        "source_trace_id": source_trace_id,
        "failure_kind": failure_kind or "unknown",
        **_list_fingerprint_fields("checks", checks),
        "status": status,
        **_text_fingerprint_fields("note", note),
    }
    _append_jsonl(_replay_corpus_path(), record)
    return record


def replay_corpus_summary() -> Dict[str, Any]:
    corpus = _read_jsonl(_replay_corpus_path())
    recent = corpus[-_RECENT_LIMIT:]
    return {
        "schema_version": SCHEMA_VERSION,
        "total": len(corpus),
        "recent": len(recent),
        "by_status": _count_by(corpus, "status"),
        "by_failure_kind": _count_by(corpus, "failure_kind"),
        "last_recorded_at": corpus[-1].get("recorded_at") if corpus else None,
        "candidates": [
            {
                "replay_id": item.get("replay_id"),
                "source_trace_id": item.get("source_trace_id"),
                "failure_kind": item.get("failure_kind"),
                "status": item.get("status"),
                "checks_count": int(item.get("checks_count") or 0),
                "recorded_at": item.get("recorded_at"),
            }
            for item in recent
        ],
    }


def _promotion_gates_path() -> Path:
    return _jsonl_path("promotion-gates.jsonl")


def evaluate_promotion_gate(
    *,
    component: str,
    target: str,
    required_suites: Optional[List[str]] = None,
    origin: str = "offline_eval",
) -> Dict[str, Any]:
    """Persist and return whether a mutation target can be promoted."""
    profile = current_profile_name()
    required = [str(item) for item in (required_suites or [CORE_HARNESS_NAME]) if str(item).strip()]
    eval_state = ensure_profile_eval_suite(profile)
    suites = (
        eval_state.get("profiles", {})
        .get(profile, {})
        .get("suites", {})
    )
    passed_suites: List[str] = []
    missing_suites: List[str] = []
    for name in required:
        suite = suites.get(name) if isinstance(suites, dict) else None
        if isinstance(suite, dict) and suite.get("status") == "passed":
            passed_suites.append(name)
        else:
            missing_suites.append(name)
    status = "passed" if not missing_suites else "blocked"
    record = {
        "schema_version": SCHEMA_VERSION,
        "gate_id": _new_id("gate"),
        "recorded_at": _now_iso(),
        "profile": profile,
        "component": component,
        **_text_fingerprint_fields("target", target),
        "origin": origin,
        "status": status,
        "required_suites": required,
        "passed_suites": passed_suites,
        "missing_suites": missing_suites,
    }
    _append_jsonl(_promotion_gates_path(), record)
    return record


def promotion_gate_summary() -> Dict[str, Any]:
    gates = _read_jsonl(_promotion_gates_path())
    recent = gates[-_RECENT_LIMIT:]
    return {
        "schema_version": SCHEMA_VERSION,
        "total": len(gates),
        "recent": len(recent),
        "passed": sum(1 for item in gates if item.get("status") == "passed"),
        "blocked": sum(1 for item in gates if item.get("status") == "blocked"),
        "by_component": _count_by(gates, "component"),
        "last_recorded_at": gates[-1].get("recorded_at") if gates else None,
        "recent_gates": [
            {
                "gate_id": item.get("gate_id"),
                "component": item.get("component"),
                "target_present": item.get("target_present"),
                "target_sha256": item.get("target_sha256"),
                "status": item.get("status"),
                "missing_suites": list(item.get("missing_suites") or []),
                "recorded_at": item.get("recorded_at"),
            }
            for item in recent
        ],
    }


def _eval_suites_path() -> Path:
    return _json_path("eval-suites.json")


def ensure_profile_eval_suite(profile: Optional[str] = None) -> Dict[str, Any]:
    profile = profile or current_profile_name()
    state = _read_json(_eval_suites_path(), {"schema_version": SCHEMA_VERSION, "profiles": {}})
    profiles = state.setdefault("profiles", {})
    profile_state = profiles.setdefault(profile, {"profile": profile, "suites": {}})
    suites = profile_state.setdefault("suites", {})
    suite = suites.setdefault(CORE_HARNESS_NAME, {
        "name": CORE_HARNESS_NAME,
        "status": "defined",
        "checks": [
            "turn trace emitted",
            "harness event emitted",
            "memory admission classified",
            "skill mutation registered",
            "mutation contract registered",
            "goal skill expansion covered",
            "dashboard health endpoint responds",
        ],
        "cases": _DEFAULT_HARNESS_CASES,
        "created_at": _now_iso(),
        "last_run_at": None,
        "last_result": None,
    })
    _write_json(_eval_suites_path(), state)
    return state


def core_harness_suite(profile: Optional[str] = None) -> Dict[str, Any]:
    """Return the profile-scoped seven-case Hermes core harness suite."""
    profile_name = profile or current_profile_name()
    state = ensure_profile_eval_suite(profile_name)
    return (
        state.setdefault("profiles", {})
        .setdefault(profile_name, {"profile": profile_name, "suites": {}})
        .setdefault("suites", {})
        .setdefault(CORE_HARNESS_NAME, {})
    )


def core_harness_status(profile: Optional[str] = None) -> Dict[str, Any]:
    """Return the first-class status for the Hermes core harness."""
    profile_name = profile or current_profile_name()
    suite = core_harness_suite(profile_name)
    cases = suite.get("cases") if isinstance(suite.get("cases"), list) else []
    last_case_results = (
        suite.get("last_case_results")
        if isinstance(suite.get("last_case_results"), list)
        else []
    )
    result_by_id = {
        str(item.get("id")): item
        for item in last_case_results
        if isinstance(item, dict) and item.get("id")
    }
    case_rows = []
    for case in cases:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("id") or "")
        last_result = result_by_id.get(case_id, {})
        case_rows.append({
            "id": case_id,
            "description": case.get("description"),
            "checks": list(case.get("checks") or []),
            "last_status": last_result.get("status"),
            "last_duration_s": last_result.get("duration_s"),
            "last_run_at": last_result.get("ran_at"),
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "name": CORE_HARNESS_NAME,
        "profile": profile_name,
        "status": suite.get("status", "defined"),
        "case_count": len(case_rows),
        "checks": list(suite.get("checks") or []),
        "cases": case_rows,
        "last_run_at": suite.get("last_run_at"),
        "last_result": suite.get("last_result"),
    }


def _normalize_case_ids(case_ids: Optional[Iterable[str]]) -> Optional[set[str]]:
    if case_ids is None:
        return None
    normalized = {
        str(case_id).strip()
        for case_id in case_ids
        if str(case_id).strip()
    }
    return normalized or None


def _core_harness_command(checks: List[str]) -> List[str]:
    runner = _repo_root() / "scripts" / "run_tests.sh"
    if runner.exists():
        return [str(runner), *checks, "-q"]
    return [shutil.which("pytest") or "pytest", *checks, "-q"]


def _command_text(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _tail_text(text: str, *, limit: int = 4000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[-limit:]}".lstrip()


def _run_core_harness_case(
    case: Dict[str, Any],
    *,
    timeout_s: float,
) -> Dict[str, Any]:
    checks = [str(check) for check in case.get("checks") or []]
    command = _core_harness_command(checks)
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    duration_s = time.time() - started
    output = "\n".join(
        part for part in [completed.stdout, completed.stderr] if part
    ).strip()
    return {
        "id": case.get("id"),
        "description": case.get("description"),
        "checks": checks,
        "command": _command_text(command),
        "status": "passed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "duration_s": duration_s,
        "ran_at": _now_iso(),
        "output_tail": _tail_text(output),
    }


def _normalize_core_case_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Strip raw subprocess/test output from persisted core-harness results."""
    normalized = dict(result)
    raw_command = normalized.pop("command", None)
    raw_output = normalized.pop("output_tail", None)
    normalized.update(_text_fingerprint_fields("command", raw_command))
    normalized.update(_text_fingerprint_fields("output_tail", raw_output))
    return normalized


def run_core_harness(
    *,
    profile: Optional[str] = None,
    case_ids: Optional[Iterable[str]] = None,
    timeout_s: float = 600.0,
    runner: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the seven-case Hermes core harness and persist its result.

    ``runner`` is an injection seam for tests. It receives a case dict and
    returns a case-result dict with at least ``status``.
    """
    profile_name = profile or current_profile_name()
    suite = core_harness_suite(profile_name)
    cases = [case for case in suite.get("cases") or [] if isinstance(case, dict)]
    wanted = _normalize_case_ids(case_ids)
    if wanted is not None:
        cases = [case for case in cases if str(case.get("id") or "") in wanted]
        found = {str(case.get("id") or "") for case in cases}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(f"unknown core harness case(s): {', '.join(missing)}")
    if not cases:
        raise ValueError("no core harness cases selected")

    run_started = time.time()
    case_results: List[Dict[str, Any]] = []
    for case in cases:
        try:
            if runner is not None:
                raw = runner(dict(case))
                result = raw if isinstance(raw, dict) else {"status": str(raw)}
                result.setdefault("id", case.get("id"))
                result.setdefault("description", case.get("description"))
                result.setdefault("checks", list(case.get("checks") or []))
                result.setdefault("ran_at", _now_iso())
                result.setdefault("duration_s", 0.0)
            else:
                result = _run_core_harness_case(case, timeout_s=timeout_s)
        except subprocess.TimeoutExpired as exc:
            result = {
                "id": case.get("id"),
                "description": case.get("description"),
                "checks": list(case.get("checks") or []),
                "command": _command_text(exc.cmd if isinstance(exc.cmd, list) else [str(exc.cmd)]),
                "status": "failed",
                "returncode": None,
                "duration_s": timeout_s,
                "ran_at": _now_iso(),
                "output_tail": f"timed out after {timeout_s:.0f}s",
            }
        except Exception as exc:
            result = {
                "id": case.get("id"),
                "description": case.get("description"),
                "checks": list(case.get("checks") or []),
                "status": "failed",
                "returncode": None,
                "duration_s": 0.0,
                "ran_at": _now_iso(),
                "output_tail": str(exc),
            }
        result["status"] = "passed" if result.get("status") == "passed" else "failed"
        case_results.append(_normalize_core_case_result(result))

    passed = sum(1 for item in case_results if item.get("status") == "passed")
    failed = len(case_results) - passed
    status = "passed" if failed == 0 else "failed"
    duration_s = time.time() - run_started
    check_labels = [
        str(check)
        for case in cases
        for check in (case.get("checks") or [])
    ]
    result_summary = (
        f"{CORE_HARNESS_NAME}: {passed}/{len(case_results)} case(s) passed"
    )

    state = ensure_profile_eval_suite(profile_name)
    profile_state = state.setdefault("profiles", {}).setdefault(
        profile_name,
        {"profile": profile_name, "suites": {}},
    )
    suites = profile_state.setdefault("suites", {})
    suite_record = suites.setdefault(CORE_HARNESS_NAME, {"name": CORE_HARNESS_NAME})
    suite_record.update({
        "status": status,
        "checks": check_labels,
        "last_run_at": _now_iso(),
        "last_result": result_summary,
        "last_duration_s": duration_s,
        "last_case_results": case_results,
    })
    _write_json(_eval_suites_path(), state)

    record_harness_event(
        "harness.core.ran",
        {
            "name": CORE_HARNESS_NAME,
            "status": status,
            "case_count": len(case_results),
            "passed": passed,
            "failed": failed,
            "duration_s": duration_s,
            "case_ids": [item.get("id") for item in case_results],
        },
        component="harness",
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "name": CORE_HARNESS_NAME,
        "profile": profile_name,
        "status": status,
        "passed": passed,
        "failed": failed,
        "case_count": len(case_results),
        "duration_s": duration_s,
        "cases": case_results,
        "last_run_at": suite_record["last_run_at"],
        "result": result_summary,
    }


def record_eval_suite(
    *,
    profile: Optional[str],
    name: str,
    status: str,
    checks: Optional[List[str]] = None,
    result: Optional[str] = None,
) -> Dict[str, Any]:
    state = ensure_profile_eval_suite(profile)
    profile_name = profile or current_profile_name()
    suites = state.setdefault("profiles", {}).setdefault(profile_name, {"profile": profile_name, "suites": {}}).setdefault("suites", {})
    suite = suites.setdefault(name, {"name": name, "created_at": _now_iso()})
    suite.update({
        "status": status,
        "checks": checks if checks is not None else suite.get("checks", []),
        "last_run_at": _now_iso(),
        "last_result": "recorded" if result else None,
        **_text_fingerprint_fields("last_result", result),
    })
    _write_json(_eval_suites_path(), state)
    return suite


def _count_by(items: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        value = item.get(key)
        if value is not None:
            counter[str(value)] += 1
    return dict(counter)


def learning_health_unavailable_summary(error: Optional[str] = None) -> Dict[str, Any]:
    """Return a non-throwing fallback for optional learning-loop telemetry."""
    profile = current_profile_name()

    summary: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "profile": profile,
        "hermes_home": str(get_hermes_home()),
        "degraded": True,
        "error": error or "learning health unavailable",
        "traces": {
            "total": 0,
            "recent": 0,
            "failure_count": 0,
            "interrupted_count": 0,
            "codex_turns": 0,
            "last_turn_at": None,
        },
        "events": {
            "total": 0,
            "recent": 0,
            "by_type": {},
            "last_event_at": None,
            "goal_events": 0,
            "approval_events": 0,
            "skill_loads": 0,
            "tool_errors": 0,
        },
        "memory": {
            "total": 0,
            "decisions": {},
            "kinds": {},
            "review_or_reject_count": 0,
            "last_recorded_at": None,
        },
        "skills": {
            "total_events": 0,
            "registered": 0,
            "needs_verification": 0,
            "promoted": 0,
            "removed": 0,
            "agent_created_count": 0,
            "last_recorded_at": None,
        },
        "mutations": {
            "total": 0,
            "draft": 0,
            "promoted": 0,
            "rejected": 0,
            "by_component": {},
            "last_recorded_at": None,
        },
        "replay_corpus": {
            "schema_version": SCHEMA_VERSION,
            "total": 0,
            "recent": 0,
            "by_status": {},
            "by_failure_kind": {},
            "last_recorded_at": None,
            "candidates": [],
        },
        "promotion_gates": {
            "schema_version": SCHEMA_VERSION,
            "total": 0,
            "recent": 0,
            "passed": 0,
            "blocked": 0,
            "by_component": {},
            "last_recorded_at": None,
            "recent_gates": [],
        },
        "failure_taxonomy": {},
        "evals": {
            "profiles": 0,
            "current_profile": profile,
            "suite_count": 0,
            "missing_run_count": 0,
        },
        "core_harness": {
            "name": CORE_HARNESS_NAME,
            "status": "unavailable",
            "case_count": 0,
            "last_run_at": None,
            "last_result": None,
        },
    }
    return summary


def learning_health_summary() -> Dict[str, Any]:
    """Return compact dashboard/API summary of the harness learning loop."""
    profile = current_profile_name()
    degraded_error: Optional[str] = None
    try:
        ensure_profile_eval_suite(profile)
    except OSError as exc:
        degraded_error = f"eval suite unavailable: {exc}"

    traces = _read_jsonl(_jsonl_path("turn-traces.jsonl"))
    recent_traces = traces[-_RECENT_LIMIT:]
    memory = _read_jsonl(_jsonl_path("memory-admissions.jsonl"))
    events = _read_jsonl(_jsonl_path("harness-events.jsonl"))
    recent_events = events[-_RECENT_LIMIT:]
    mutations = _read_jsonl(_mutation_contracts_path())
    skills = _read_jsonl(_jsonl_path("skill-drafts.jsonl"))
    registry = _read_json(_skill_registry_path(), {"schema_version": SCHEMA_VERSION, "skills": {}})
    eval_state = _read_json(_eval_suites_path(), {"schema_version": SCHEMA_VERSION, "profiles": {}})
    profile_suites = (
        eval_state.get("profiles", {})
        .get(profile, {})
        .get("suites", {})
    )
    core_suite = profile_suites.get(CORE_HARNESS_NAME, {}) if isinstance(profile_suites, dict) else {}
    core_cases = core_suite.get("cases") if isinstance(core_suite.get("cases"), list) else []

    skill_records = registry.get("skills", {}) if isinstance(registry.get("skills"), dict) else {}
    needs_verification = sum(
        1 for rec in skill_records.values()
        if isinstance(rec, dict) and rec.get("promotion_status") == "needs_verification"
    )
    promoted = sum(
        1 for rec in skill_records.values()
        if isinstance(rec, dict) and rec.get("promotion_status") == "verified"
    )
    failure_taxonomy = _count_by(recent_traces, "failure_kind")

    agent_created_count = 0
    try:
        from tools.skill_usage import agent_created_report
        agent_created_count = int(agent_created_report().get("agent_created_count") or 0)
    except Exception:
        agent_created_count = 0

    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "profile": profile,
        "hermes_home": str(get_hermes_home()),
        "traces": {
            "total": len(traces),
            "recent": len(recent_traces),
            "failure_count": sum(1 for item in recent_traces if item.get("error_present") or item.get("completed") is False),
            "interrupted_count": sum(1 for item in recent_traces if item.get("interrupted")),
            "codex_turns": sum(1 for item in recent_traces if item.get("runtime") == "codex_app_server"),
            "by_failure_kind": failure_taxonomy,
            "last_turn_at": recent_traces[-1].get("recorded_at") if recent_traces else None,
        },
        "events": {
            "total": len(events),
            "recent": len(recent_events),
            "by_type": _count_by(recent_events, "event_type"),
            "last_event_at": recent_events[-1].get("recorded_at") if recent_events else None,
            "goal_events": sum(1 for item in recent_events if str(item.get("event_type") or "").startswith("goal.")),
            "approval_events": sum(1 for item in recent_events if str(item.get("event_type") or "").startswith("approval.")),
            "skill_loads": sum(1 for item in recent_events if item.get("event_type") == "skill.loaded"),
            "tool_errors": sum(
                1
                for item in recent_events
                if item.get("event_type") in {"tool.complete", "tool.error"}
                and isinstance(item.get("payload"), dict)
                and item["payload"].get("error")
            ),
        },
        "memory": {
            "total": len(memory),
            "decisions": _count_by(memory, "decision"),
            "kinds": _count_by(memory, "kind"),
            "review_or_reject_count": sum(
                1 for item in memory if item.get("decision") in {"review", "reject"}
            ),
            "last_recorded_at": memory[-1].get("recorded_at") if memory else None,
        },
        "skills": {
            "total_events": len(skills),
            "registered": len(skill_records),
            "needs_verification": needs_verification,
            "promoted": promoted,
            "removed": sum(
                1 for rec in skill_records.values()
                if isinstance(rec, dict) and rec.get("status") == "removed"
            ),
            "agent_created_count": agent_created_count,
            "last_recorded_at": skills[-1].get("recorded_at") if skills else None,
        },
        "mutations": {
            "total": len(mutations),
            "draft": sum(1 for item in mutations if item.get("status") == "draft"),
            "promoted": sum(1 for item in mutations if item.get("status") == "promoted"),
            "rejected": sum(1 for item in mutations if item.get("status") == "rejected"),
            "by_component": _count_by(mutations, "component"),
            "last_recorded_at": mutations[-1].get("recorded_at") if mutations else None,
        },
        "replay_corpus": replay_corpus_summary(),
        "promotion_gates": promotion_gate_summary(),
        "failure_taxonomy": failure_taxonomy,
        "trace_schema": dict(TRACE_SCHEMA),
        "evals": {
            "profiles": len(eval_state.get("profiles", {}) or {}),
            "current_profile": profile,
            "suite_count": len(profile_suites) if isinstance(profile_suites, dict) else 0,
            "missing_run_count": sum(
                1 for suite in profile_suites.values()
                if isinstance(suite, dict) and not suite.get("last_run_at")
            ) if isinstance(profile_suites, dict) else 0,
        },
        "core_harness": {
            "name": CORE_HARNESS_NAME,
            "status": core_suite.get("status", "defined"),
            "case_count": len(core_cases),
            "last_run_at": core_suite.get("last_run_at"),
            "last_result": core_suite.get("last_result"),
            "last_duration_s": core_suite.get("last_duration_s"),
            "failed_cases": [
                item.get("id")
                for item in core_suite.get("last_case_results", []) or []
                if isinstance(item, dict) and item.get("status") == "failed"
            ],
        },
    }
    if degraded_error:
        summary["degraded"] = True
        summary["error"] = degraded_error
    return summary


def learning_snapshot_summary() -> Dict[str, Any]:
    """Return a content-free learning-loop snapshot for trace/replay gates."""
    health = dict(learning_health_summary())
    # The snapshot is intended for model/control-plane inspection. Keep it
    # metadata-only: no user/home paths, no raw mutation prose, no raw errors.
    health.pop("hermes_home", None)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "profile": health.get("profile", current_profile_name()),
        "content_policy": "metadata_only",
        "trace_schema": dict(TRACE_SCHEMA),
        "traces": health.get("traces", {}),
        "events": health.get("events", {}),
        "memory": health.get("memory", {}),
        "skills": health.get("skills", {}),
        "mutations": health.get("mutations", {}),
        "evals": health.get("evals", {}),
        "core_harness": health.get("core_harness", {}),
        "replay_corpus": health.get("replay_corpus", replay_corpus_summary()),
        "promotion_gates": health.get("promotion_gates", promotion_gate_summary()),
        "failure_taxonomy": health.get("failure_taxonomy", {}),
    }
