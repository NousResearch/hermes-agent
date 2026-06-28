"""Read-only backend for the Olympus Hermes dashboard plugin."""
from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter
except Exception:
    class APIRouter:  # type: ignore
        def get(self, *_args, **_kwargs):
            return lambda fn: fn

try:
    from hermes_constants import get_hermes_home
except Exception:
    def get_hermes_home() -> Path:  # type: ignore[misc]
        val = (os.environ.get("HERMES_HOME") or "").strip()
        return Path(val) if val else Path.home() / ".hermes"

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

router = APIRouter()
logger = logging.getLogger("olympus.dashboard")

STALE_SECONDS = 60 * 60 * 24
RECENT_SECONDS = 60 * 60
ACTIVE_SECONDS = 60 * 10
KANBAN_READY_STALE_SECONDS = 60 * 60 * 24
KANBAN_HEARTBEAT_STALE_SECONDS = 60 * 60
TOOL_HEAVY_THRESHOLD = 20
LONG_THREAD_THRESHOLD = 40
OVERLOADED_OPEN_THRESHOLD = 5
OVERLOADED_RUNNING_THRESHOLD = 2
CONTEXT_PRESSURE_TOKENS = 50000
RUNAWAY_TOOLS_THRESHOLD = 40
EXPENSIVE_RUN_USD = 1.0
MAX_TURNS_REVIEW_THRESHOLD = 80
API_RESPONSE_BUDGET_MS = 750.0
PAYLOAD_BUDGET_BYTES = 250_000
CLIENT_RENDER_BUDGET_MS = 150.0
OPEN_KANBAN_STATUSES = {"triage", "todo", "scheduled", "ready", "running", "blocked", "review"}
KANBAN_COLUMNS = ["triage", "todo", "scheduled", "ready", "running", "blocked", "review", "done", "archived"]
FAILED_RUN_STATUSES = {"crashed", "timed_out", "failed"}
FAILED_RUN_OUTCOMES = {"crashed", "timed_out", "spawn_failed", "gave_up"}
SKILLS_SH_AUDIT_CHECKS = ("agent-trust-hub", "socket", "snyk")
SKILLS_SH_AUDIT_OK = {"pass", "passed", "ok", "clean"}
SKILLS_SH_AUDIT_WARN = {"warn", "warning", "review"}
SKILLS_SH_AUDIT_FAIL = {"fail", "failed", "error", "critical", "block"}
EXPOSE_LOCAL_LABELS = os.environ.get("OLYMPUS_EXPOSE_LOCAL_LABELS", "").strip().lower() in {"1", "true", "yes", "on"}
REDACTION_PATTERNS = [
    re.compile(r"(?i)['\"]?(?:api[_-]?key|token|secret|password|passwd)['\"]?\s*[:=]\s*['\"]?[^'\"\s,}]+['\"]?"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd)\s*[:=]\s*['\"]?[^'\"\s]+"),
    re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"ghp_[A-Za-z0-9_]+"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]+"),
]
# Free-text error/log fields routinely embed absolute filesystem paths and
# delivery URLs (worker tracebacks, handoff failures, webhook errors). The
# privacy contract says local paths and base URLs are hidden by default, so
# scrub them alongside secret-like strings before any error text is returned.
URL_PATTERN = re.compile(r"(?i)\b(?:https?|ftp|wss?)://[^\s'\"<>]+")
PATH_PATTERNS = [
    re.compile(r"(?i)[a-z]:\\[^\s'\"|<>]+"),         # Windows: C:\Users\alice\...
    re.compile(r"(?<![\w.])~?(?:/[\w.\-]+){2,}/?"),  # POSIX / home: /home/alice/.hermes, ~/x/y
]
# Precise log failure signals (avoids false positives like "0 failed" / "failed: false").
LOG_ERROR_PATTERNS = [
    ("traceback", re.compile(r"\btraceback\b")),
    ("exception", re.compile(r"\bexception\b")),
    ("unauthorized", re.compile(r"\bunauthorized\b|\b401\b")),
    ("telegram conflict", re.compile(r"terminated by other getupdates")),
    ("rate limit", re.compile(r"\brate limit(?:ed|ing)?\b|\b429\b")),
    ("failure", re.compile(r"\bfailed to\b|\bfailure\b")),
    ("critical", re.compile(r"\bcritical\b")),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact_log_value(value: Any, limit: int = 180) -> str:
    # Same redaction the API payloads use, so log warnings can't leak a path
    # or secret that the response fields scrub.
    return redact_text(value, limit)


def log_read_warning(context: str, exc: BaseException) -> None:
    logger.warning("%s: %s", context, redact_log_value(exc))


def ts_to_iso(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, str):
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def age_state(iso_or_ts: Any) -> str:
    try:
        if isinstance(iso_or_ts, str):
            dt = datetime.fromisoformat(iso_or_ts.replace("Z", "+00:00"))
            ts = dt.timestamp()
        else:
            ts = float(iso_or_ts)
        age = time.time() - ts
    except Exception:
        return "unknown"
    if age <= ACTIVE_SECONDS:
        return "active"
    if age <= RECENT_SECONDS:
        return "recent"
    if age <= STALE_SECONDS:
        return "idle"
    return "stale"


def read_text(path: Path, limit: int = 12000) -> str:
    try:
        data = path.read_text(errors="replace")
        return data[-limit:]
    except Exception as exc:
        if path.exists():
            log_read_warning(f"read_text failed for {path.name}", exc)
        return ""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(errors="replace"))
    except Exception as exc:
        if path.exists():
            log_read_warning(f"read_json failed for {path.name}", exc)
        return None


def env_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return keys
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            keys.add(key.upper())
    return keys


def redact_text(value: Any, limit: int = 180) -> str:
    text = "" if value is None else str(value)
    for pattern in REDACTION_PATTERNS:
        text = pattern.sub(lambda m: m.group(1) + "[redacted]" if m.groups() else "[redacted]", text)
    text = URL_PATTERN.sub("[redacted-url]", text)
    for pattern in PATH_PATTERNS:
        text = pattern.sub("[redacted-path]", text)
    return text[:limit]


def safe_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def safe_id(value: Any, prefix: str = "item") -> str:
    raw = str(value or "")
    if not raw:
        return prefix
    digest = hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{prefix}:{digest}"


def public_ref(value: Any, prefix: str = "item") -> Optional[str]:
    if value in (None, ""):
        return None
    return redact_text(value, 120) if EXPOSE_LOCAL_LABELS else safe_id(value, prefix)


def public_label(raw: Any, fallback: str, prefix: str = "item") -> str:
    if EXPOSE_LOCAL_LABELS and raw:
        return redact_text(raw, 120)
    return fallback or safe_id(raw, prefix)


def public_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in profile.items() if k not in {"_path", "path", "_public_name"}}
    if "_public_name" in profile:
        out["name"] = profile["_public_name"]
    return out


def strip_internal(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: strip_internal(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [strip_internal(v) for v in value]
    return value


def payload_size_bytes(value: Dict[str, Any]) -> int:
    try:
        return len(json.dumps(value, separators=(",", ":"), default=str).encode("utf-8"))
    except Exception:
        return 0


def runtime_diagnostics(route: str, started_at: float, payload: Dict[str, Any], profiles: List[Dict[str, Any]], sessions: List[Dict[str, Any]], kanban: Dict[str, Any]) -> Dict[str, Any]:
    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    payload_bytes = payload_size_bytes(payload)
    boards = as_list(kanban.get("boards"))
    board_read_failures = sum(1 for board in boards if isinstance(board, dict) and board.get("error"))
    state = "warning" if elapsed_ms > API_RESPONSE_BUDGET_MS or payload_bytes > PAYLOAD_BUDGET_BYTES or board_read_failures else "ok"
    return {
        "route": route,
        "state": state,
        "generated_ms": elapsed_ms,
        "payload_bytes": payload_bytes,
        "budgets": {
            "api_response_ms": API_RESPONSE_BUDGET_MS,
            "payload_bytes": PAYLOAD_BUDGET_BYTES,
            "client_render_ms": CLIENT_RENDER_BUDGET_MS,
        },
        "budget_status": {
            "api_response": "warning" if elapsed_ms > API_RESPONSE_BUDGET_MS else "ok",
            "payload": "warning" if payload_bytes > PAYLOAD_BUDGET_BYTES else "ok",
            "client_render": "reported_by_browser",
        },
        "counts": {
            "profiles_scanned": len(profiles),
            "sessions_scanned": len(sessions),
            "kanban_boards_scanned": len(boards),
            "kanban_board_read_failures": board_read_failures,
            "kanban_attention_items": len(as_list(kanban.get("attention"))),
        },
        "hermes": {
            "version": os.environ.get("HERMES_VERSION") or "unknown",
            "home_detected": bool(get_hermes_home()),
        },
    }


def _source_state(present: bool, read_failures: int = 0) -> str:
    if not present:
        return "missing"
    return "warning" if read_failures else "ok"


def _sqlite_count(path: Path, table: str) -> Optional[int]:
    if not path.exists():
        return None
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=1.5)
        con.row_factory = sqlite3.Row
        row = con.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"] or 0) if row is not None else 0
    except Exception as exc:
        log_read_warning(f"sqlite count failed for {table}", exc)
        return None
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass


def _json_object_count(path: Path, key: Optional[str] = None) -> Optional[int]:
    data = read_json(path)
    if key and isinstance(data, dict):
        data = data.get(key)
    if isinstance(data, dict):
        return len(data)
    if isinstance(data, list):
        return len(data)
    return None


def build_evidence_sources(profiles: List[Dict[str, Any]], sessions: List[Dict[str, Any]], cron: List[Dict[str, Any]], gateways: List[Dict[str, Any]], kanban: Dict[str, Any]) -> Dict[str, Any]:
    hermes_home = get_hermes_home()
    state_store = hermes_home / "state.db"
    config_file = hermes_home / "config.yaml"
    skill_usage = hermes_home / "skills" / ".usage.json"
    hub_lock = hermes_home / "skills" / ".hub" / "lock.json"
    boards = as_list(kanban.get("boards"))
    board_failures = sum(1 for board in boards if isinstance(board, dict) and board.get("error"))

    session_count = _sqlite_count(state_store, "sessions")
    message_count = _sqlite_count(state_store, "messages")
    state_failures = int(state_store.exists() and (session_count is None or message_count is None))
    skill_usage_count = _json_object_count(skill_usage)
    hub_installed_count = _json_object_count(hub_lock, "installed")

    items = [
        {
            "id": "hermes_state",
            "label": "Hermes state store",
            "type": "sqlite",
            "state": _source_state(state_store.exists(), state_failures),
            "present": state_store.exists(),
            "counts": {
                "sessions_returned": len(sessions),
                "sessions_recorded": session_count if session_count is not None else 0,
                "messages_recorded": message_count if message_count is not None else 0,
            },
            "fields": [
                "sessions.started_at",
                "sessions.ended_at",
                "sessions.message_count",
                "sessions.tool_call_count",
                "sessions.input_tokens",
                "sessions.output_tokens",
                "sessions.api_call_count",
                "sessions.handoff_error",
            ],
            "redaction": "Session IDs are hashed and titles are hidden unless local labels are enabled.",
            "read_failures": state_failures,
            "recommended_view": "/sessions",
        },
        {
            "id": "hermes_kanban",
            "label": "Hermes Kanban store",
            "type": "sqlite",
            "state": _source_state(bool(boards), board_failures),
            "present": bool(boards),
            "counts": {
                "boards_scanned": len(boards),
                "open_tasks": int(kanban.get("open") or 0),
                "attention_items": len(as_list(kanban.get("attention"))),
                "recent_runs": len(as_list(kanban.get("recent_runs"))),
            },
            "fields": [
                "tasks.status",
                "tasks.assignee",
                "tasks.skills",
                "tasks.session_id",
                "tasks.model_override",
                "task_runs.status",
                "task_runs.outcome",
                "task_runs.last_heartbeat_at",
            ],
            "redaction": "Task IDs and board labels are hashed unless local labels are enabled.",
            "read_failures": board_failures,
            "recommended_view": "/kanban",
        },
        {
            "id": "hermes_config",
            "label": "Hermes config",
            "type": "yaml",
            "state": _source_state(config_file.exists(), 0),
            "present": config_file.exists(),
            "counts": {
                "profiles_scanned": len(profiles),
                "cron_jobs": len(cron),
                "gateways": len(gateways),
            },
            "fields": [
                "model.provider_presence",
                "model.default_presence",
                "fallback_providers.presence",
                "toolsets.presence",
                "agent.max_turns",
                "agent.gateway_timeout",
                "tool_loop_guardrails.presence",
                "compression.presence",
                "browser.privacy_flags",
                "auxiliary_provider.presence",
                "profile.gateway_state",
                "profile.skill_count",
                "cron.enabled",
                "cron.last_status",
            ],
            "redaction": "Secrets, prompt text, local paths, and exact model labels are not returned by default.",
            "read_failures": 0,
            "recommended_view": "/config",
        },
        {
            "id": "skill_usage",
            "label": "Skill usage metadata",
            "type": "json",
            "state": _source_state(skill_usage.exists(), int(skill_usage.exists() and skill_usage_count is None)),
            "present": skill_usage.exists(),
            "counts": {
                "skills_recorded": skill_usage_count if skill_usage_count is not None else 0,
            },
            "fields": [
                "created_at",
                "last_used_at",
                "last_viewed_at",
                "last_patched_at",
                "use_count",
                "state",
                "pinned",
            ],
            "redaction": "Skill names are summarized and can be hashed when labels are hidden.",
            "read_failures": int(skill_usage.exists() and skill_usage_count is None),
            "recommended_view": "/skills",
        },
        {
            "id": "skill_hub_lock",
            "label": "Skill hub lock metadata",
            "type": "json",
            "state": _source_state(hub_lock.exists(), int(hub_lock.exists() and hub_installed_count is None)),
            "present": hub_lock.exists(),
            "counts": {
                "hub_installed": hub_installed_count if hub_installed_count is not None else 0,
            },
            "fields": [
                "version",
                "installed",
                "optional_trust_metadata",
                "optional_scan_metadata",
            ],
            "redaction": "Only local hub metadata presence and counts are shown in Phase 0.",
            "read_failures": int(hub_lock.exists() and hub_installed_count is None),
            "recommended_view": "/skills",
        },
    ]
    missing = sum(1 for item in items if item["state"] == "missing")
    warnings = sum(1 for item in items if item["state"] == "warning")
    return {
        "summary": {
            "sources": len(items),
            "available": len(items) - missing,
            "missing": missing,
            "warnings": warnings,
            "privacy": "local labels hidden" if not EXPOSE_LOCAL_LABELS else "local labels visible",
        },
        "items": items,
    }


def _epoch_seconds(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        if isinstance(value, (int, float)):
            raw = float(value)
            return raw / 1000 if raw > 10_000_000_000 else raw
        text = str(value).strip()
        if not text:
            return None
        if re.match(r"^\d+(\.\d+)?$", text):
            raw = float(text)
            return raw / 1000 if raw > 10_000_000_000 else raw
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception as exc:
        log_read_warning("timestamp parse failed", exc)
        return None


def _public_skill_label(name: Any) -> str:
    raw = str(name or "")
    if EXPOSE_LOCAL_LABELS:
        return redact_text(raw, 120)
    return safe_id(raw, "skill")


def _skill_audit_verdict(value: Any) -> Optional[str]:
    raw: Any = value
    if isinstance(value, dict):
        raw = value.get("verdict") or value.get("status") or value.get("result")
    text = str(raw or "").strip().lower()
    if not text:
        return None
    if text in SKILLS_SH_AUDIT_OK:
        return "pass"
    if text in SKILLS_SH_AUDIT_WARN:
        return "warn"
    if text in SKILLS_SH_AUDIT_FAIL:
        return "fail"
    return None


def _skill_audit_state(verdict: str) -> str:
    if verdict == "fail":
        return "critical"
    if verdict == "warn":
        return "warning"
    return "ok"


def _stored_skill_audits(meta: Dict[str, Any]) -> List[Dict[str, str]]:
    source = meta.get("skills_sh_audit")
    if not isinstance(source, dict):
        source = meta.get("security_audit")
    if not isinstance(source, dict):
        source = meta
    audits: List[Dict[str, str]] = []
    for check in SKILLS_SH_AUDIT_CHECKS:
        verdict = _skill_audit_verdict(source.get(check) if isinstance(source, dict) else None)
        if verdict:
            audits.append({
                "check": check,
                "verdict": verdict,
                "state": _skill_audit_state(verdict),
            })
    return audits


def _profile_skill_usage_sources(profiles: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Return read-only skill usage files grounded in configured Hermes profiles."""
    base = get_hermes_home()
    sources: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(label: str, public_name: str, home: Path) -> None:
        key = str(home.resolve()) if home.exists() else str(home)
        if key in seen:
            return
        seen.add(key)
        sources.append({
            "profile": public_name,
            "label": label,
            "path": home / "skills" / ".usage.json",
        })

    if profiles:
        for idx, profile in enumerate(profiles):
            raw_path = profile.get("_path")
            if not raw_path:
                continue
            label = str(profile.get("label") or f"Profile {idx}")
            public_name = str(profile.get("_public_name") or profile.get("id") or f"profile_{idx}")
            add(label, public_name, Path(str(raw_path)))
    else:
        add("Default", "default", base)

    return sources


def collect_skill_metadata(profiles: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    skills_dir = get_hermes_home() / "skills"
    hub_lock_path = skills_dir / ".hub" / "lock.json"
    hub_data = read_json(hub_lock_path)
    installed = hub_data.get("installed") if isinstance(hub_data, dict) and isinstance(hub_data.get("installed"), dict) else {}
    now_ts = time.time()

    usage_items: List[Dict[str, Any]] = []
    profile_usage: List[Dict[str, Any]] = []
    usage_sources = _profile_skill_usage_sources(profiles)
    usage_present = False
    usage_read_failures = 0

    for source in usage_sources:
        usage_path = Path(source["path"])
        usage_data = read_json(usage_path)
        if usage_path.exists():
            usage_present = True
        if usage_path.exists() and not isinstance(usage_data, dict):
            usage_read_failures += 1
        usage = usage_data if isinstance(usage_data, dict) else {}
        source_items: List[Dict[str, Any]] = []
        for name, meta in sorted(usage.items()):
            if not isinstance(meta, dict):
                continue
            use_count = safe_int(meta.get("use_count"))
            view_count = safe_int(meta.get("view_count"))
            patch_count = safe_int(meta.get("patch_count"))
            last_used_ts = _epoch_seconds(meta.get("last_used_at"))
            last_patched_ts = _epoch_seconds(meta.get("last_patched_at"))
            archived = bool(meta.get("archived_at")) or str(meta.get("state") or "").lower() == "archived"
            stale = not last_used_ts or now_ts - last_used_ts > 90 * 24 * 60 * 60
            recently_patched = bool(last_patched_ts and now_ts - last_patched_ts <= 14 * 24 * 60 * 60)
            created_ts = _epoch_seconds(meta.get("created_at"))
            item = {
                "_name": str(name),
                "_profile": source.get("profile"),
                "id": _public_skill_label(f"{source.get('profile')}:{name}"),
                "label": _public_skill_label(name),
                "profile": source.get("profile"),
                "profile_label": source.get("label"),
                "state": "archived" if archived else (str(meta.get("state") or "active")),
                "created_by": "agent" if meta.get("created_by") == "agent" or meta.get("agent_created") is True else None,
                "use_count": use_count,
                "view_count": view_count,
                "patch_count": patch_count,
                "pinned": bool(meta.get("pinned")),
                "archived": archived,
                "stale": stale,
                "recently_patched": recently_patched,
                "created_recently": bool(created_ts and now_ts - created_ts <= 30 * 24 * 60 * 60),
                "created_at": ts_to_iso(meta.get("created_at")),
                "last_used_at": ts_to_iso(meta.get("last_used_at")),
                "last_viewed_at": ts_to_iso(meta.get("last_viewed_at")),
                "last_patched_at": ts_to_iso(meta.get("last_patched_at")),
            }
            usage_items.append(item)
            source_items.append(item)
        profile_usage.append({
            "profile": source.get("profile"),
            "label": source.get("label"),
            "usage_present": usage_path.exists(),
            "skills": len(source_items),
            "created_30d": sum(1 for item in source_items if item.get("created_recently")),
            "used": sum(1 for item in source_items if int(item.get("use_count") or 0) > 0),
            "never_used": sum(1 for item in source_items if int(item.get("use_count") or 0) == 0),
            "stale": sum(1 for item in source_items if item.get("stale")),
            "archived": sum(1 for item in source_items if item.get("archived")),
            "recently_patched": sum(1 for item in source_items if item.get("recently_patched")),
        })

    hub_items: List[Dict[str, Any]] = []
    for name, meta in sorted(installed.items()):
        if not isinstance(meta, dict):
            continue
        trust_level = meta.get("trust_level")
        scan_verdict = meta.get("scan_verdict")
        audits = _stored_skill_audits(meta)
        audit_states = {item["state"] for item in audits}
        hub_items.append({
            "_name": str(name),
            "id": _public_skill_label(name),
            "label": _public_skill_label(name),
            "trust_level": str(trust_level) if trust_level not in (None, "") else None,
            "scan_verdict": str(scan_verdict) if scan_verdict not in (None, "") else None,
            "security_audits": audits,
            "audit_summary": " / ".join(f"{item['check']}: {item['verdict']}" for item in audits) if audits else None,
            "state": "critical" if "critical" in audit_states else ("warning" if "warning" in audit_states or not trust_level or not scan_verdict else "ok"),
            "installed_at": ts_to_iso(meta.get("installed_at")),
            "updated_at": ts_to_iso(meta.get("updated_at")),
        })

    archived = [item for item in usage_items if item.get("archived")]
    stale = [item for item in usage_items if item.get("stale")]
    never_used = [item for item in usage_items if int(item.get("use_count") or 0) == 0]
    recently_patched = [item for item in usage_items if item.get("recently_patched")]
    hub_missing_trust = [item for item in hub_items if not item.get("trust_level")]
    hub_missing_scan = [item for item in hub_items if not item.get("scan_verdict")]
    audit_records = [audit for item in hub_items for audit in as_list(item.get("security_audits")) if isinstance(audit, dict)]
    return {
        "summary": {
            "usage_present": usage_present,
            "usage_sources": len(usage_sources),
            "hub_lock_present": hub_lock_path.exists(),
            "total_skills": len(usage_items),
            "created_30d": sum(1 for item in usage_items if item.get("created_recently")),
            "agent_created": sum(1 for item in usage_items if item.get("created_by") == "agent"),
            "archived": len(archived),
            "pinned": sum(1 for item in usage_items if item.get("pinned")),
            "never_used": len(never_used),
            "used": sum(1 for item in usage_items if int(item.get("use_count") or 0) > 0),
            "stale": len(stale),
            "recently_patched": len(recently_patched),
            "hub_installed": len(hub_items),
            "hub_missing_trust": len(hub_missing_trust),
            "hub_missing_scan": len(hub_missing_scan),
            "hub_audit_pass": sum(1 for item in audit_records if item.get("verdict") == "pass"),
            "hub_audit_warn": sum(1 for item in audit_records if item.get("verdict") == "warn"),
            "hub_audit_fail": sum(1 for item in audit_records if item.get("verdict") == "fail"),
            "read_failures": usage_read_failures + int(hub_lock_path.exists() and not isinstance(hub_data, dict)),
        },
        "profile_usage": profile_usage,
        "usage_items": usage_items,
        "hub_items": hub_items,
    }


def build_skill_hygiene(skill_metadata: Dict[str, Any], skill_coverage: Dict[str, Any], kanban: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    kanban = kanban or {}
    summary = skill_metadata.get("summary") if isinstance(skill_metadata.get("summary"), dict) else {}
    usage_items = [item for item in as_list(skill_metadata.get("usage_items")) if isinstance(item, dict)]
    hub_items = [item for item in as_list(skill_metadata.get("hub_items")) if isinstance(item, dict)]
    usage_names = {str(item.get("_name")) for item in usage_items if item.get("_name")}
    hub_names = {str(item.get("_name")) for item in hub_items if item.get("_name")}
    findings: List[Dict[str, Any]] = []

    def add_signal(severity: str, title: str, detail: str, evidence: str, view: str = "/skills", basis: str = "Hermes skill metadata") -> None:
        findings.append(_tuning_item(
            "skill",
            severity,
            title,
            detail,
            evidence,
            view,
            _action_label_for_view(view),
            "Apollo",
            basis,
        ))

    if not summary.get("usage_present"):
        add_signal("info", "Skill usage metadata is not present", "Hermes has not recorded skill usage metadata on this machine yet.", "usage metadata missing")
    if not summary.get("hub_lock_present"):
        add_signal("info", "Skill hub lock metadata is not present", "Hub install provenance is unavailable until Hermes records a skill hub lock.", "hub lock missing")

    heavily_used_unpinned = sorted(
        [item for item in usage_items if int(item.get("use_count") or 0) >= 10 and not item.get("pinned")],
        key=lambda item: int(item.get("use_count") or 0),
        reverse=True,
    )
    if heavily_used_unpinned:
        add_signal(
            "warning",
            "Pin heavily used skills",
            "Frequently used skills are part of the operating surface. Pinning or reviewing them makes drift easier to spot.",
            ", ".join(f"{item.get('label')}: {item.get('use_count')} uses" for item in heavily_used_unpinned[:3]),
        )

    archived_recent = [
        item for item in usage_items
        if item.get("archived") and item.get("last_used_at") and not item.get("stale")
    ]
    if archived_recent:
        add_signal(
            "warning",
            "Archived skills still show recent use metadata",
            "An archived skill with use evidence may indicate stale provenance or a workflow depending on an archived procedure.",
            ", ".join(str(item.get("label")) for item in archived_recent[:3]),
        )

    frequently_patched = sorted(
        [item for item in usage_items if int(item.get("patch_count") or 0) >= 3 or item.get("recently_patched")],
        key=lambda item: (int(item.get("patch_count") or 0), str(item.get("last_patched_at") or "")),
        reverse=True,
    )
    if frequently_patched:
        add_signal(
            "info",
            "Recently changed skills need review",
            "Review frequently patched skills before treating their recommendations as stable procedure.",
            ", ".join(f"{item.get('label')}: {item.get('patch_count')} patches" for item in frequently_patched[:3]),
        )

    hub_missing = [item for item in hub_items if not item.get("trust_level") or not item.get("scan_verdict")]
    if hub_missing:
        add_signal(
            "warning",
            "Hub skills are missing trust or scan metadata",
            "Hub-installed skills need trust and scan evidence when Hermes has recorded it locally.",
            ", ".join(str(item.get("label")) for item in hub_missing[:3]),
        )

    audit_findings = []
    for item in hub_items:
        for audit in as_list(item.get("security_audits")):
            if not isinstance(audit, dict) or audit.get("verdict") == "pass":
                continue
            audit_findings.append(f"{item.get('label')}: {audit.get('check')} {audit.get('verdict')}")
    if audit_findings:
        add_signal(
            "warning",
            "Stored skill audit needs review",
            "Hermes has stored a warn or fail audit result for installed hub skills. Review the Skills page before using those skills in forced workflows.",
            ", ".join(audit_findings[:4]),
            "/skills",
            "Hermes skill hub lock stored skills.sh audit metadata",
        )

    forced_skill_names: set[str] = set()
    for task in kanban_items(kanban, "recent_tasks"):
        for name in as_list(task.get("_skills")):
            if name:
                forced_skill_names.add(str(name))
    missing_forced = sorted(name for name in forced_skill_names if name not in usage_names and name not in hub_names)
    if missing_forced:
        add_signal(
            "warning",
            "Forced-skill Kanban work lacks local metadata",
            "Kanban references skills that do not appear in local usage or hub metadata. Review the task or skill install.",
            ", ".join(_public_skill_label(name) for name in missing_forced[:4]),
            "/kanban",
            "Hermes Kanban tasks.skills plus local skill metadata",
        )

    coverage_summary = skill_coverage.get("summary") if isinstance(skill_coverage.get("summary"), dict) else {}
    state = "warning" if any(item.get("severity") == "warning" for item in findings) else ("active" if usage_items or hub_items else "unknown")
    return {
        "summary": {
            "state": state,
            "issues": len(findings),
            "total_skills": summary.get("total_skills") or coverage_summary.get("total_skills") or 0,
            "archived": summary.get("archived") or 0,
            "stale": summary.get("stale") or 0,
            "never_used": summary.get("never_used") or 0,
            "recently_patched": summary.get("recently_patched") or 0,
            "hub_installed": summary.get("hub_installed") or 0,
            "hub_missing_trust": summary.get("hub_missing_trust") or 0,
            "hub_missing_scan": summary.get("hub_missing_scan") or 0,
            "hub_audit_pass": summary.get("hub_audit_pass") or 0,
            "hub_audit_warn": summary.get("hub_audit_warn") or 0,
            "hub_audit_fail": summary.get("hub_audit_fail") or 0,
            "forced_skill_metadata_gaps": len(missing_forced),
        },
        "signals": findings[:8],
        "usage": strip_internal(sorted(usage_items, key=lambda item: int(item.get("use_count") or 0), reverse=True)[:8]),
        "hub": strip_internal(hub_items[:8]),
    }


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Small fallback for Hermes config.yaml when PyYAML is unavailable."""
    data: Dict[str, Any] = {}
    current: Optional[str] = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not raw.startswith((" ", "\t")) and line.endswith(":"):
            current = line[:-1].strip()
            data[current] = {}
            continue
        if not raw.startswith((" ", "\t")) and ":" in line:
            key, val = line.split(":", 1)
            data[key.strip()] = val.strip().strip('"\'')
            current = None
            continue
        if current and ":" in line:
            key, val = line.strip().split(":", 1)
            val = val.strip().strip('"\'')
            if val.lower() == "true":
                parsed: Any = True
            elif val.lower() == "false":
                parsed = False
            elif val in ("null", "None", "~"):
                parsed = None
            elif val in ("{}", ""):
                parsed = {}
            elif val == "[]":
                parsed = []
            else:
                parsed = val
            if isinstance(data.get(current), dict):
                data[current][key.strip()] = parsed
    return data


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return {}
    if yaml is None:
        return _parse_simple_yaml(text)
    try:
        data = yaml.safe_load(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return _parse_simple_yaml(text)


def as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def row_value(row: sqlite3.Row, key: str, default: Any = None) -> Any:
    try:
        if key in row.keys():
            return row[key]
    except Exception:
        pass
    return default


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(r["name"]) for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def kanban_root() -> Path:
    override = os.environ.get("HERMES_KANBAN_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    base = get_hermes_home()
    try:
        if base.parent.name == "profiles":
            return base.parent.parent
    except Exception:
        pass
    return base


def kanban_board_db(slug: str) -> Path:
    root = kanban_root()
    if slug == "default":
        return root / "kanban.db"
    return root / "kanban" / "boards" / slug / "kanban.db"


def read_kanban_current() -> str:
    current_path = kanban_root() / "kanban" / "current"
    current = current_path.read_text(errors="replace").strip() if current_path.exists() else ""
    return current or "default"


def discover_kanban_boards() -> List[Dict[str, Any]]:
    root = kanban_root()
    boards: List[Dict[str, Any]] = [{
        "slug": "default",
        "_slug": "default",
        "name": "Default",
        "description": "Default Kanban board",
        "db_exists": (root / "kanban.db").exists(),
    }]
    boards_root = root / "kanban" / "boards"
    if boards_root.exists():
        try:
            for p in sorted(boards_root.iterdir()):
                if not p.is_dir() or p.name.startswith("_") or p.name == "default":
                    continue
                meta = read_json(p / "board.json") or {}
                public_slug = p.name if EXPOSE_LOCAL_LABELS else safe_id(p.name, "board")
                boards.append({
                    "slug": public_slug,
                    "_slug": p.name,
                    "name": public_label(meta.get("name") or p.name, public_slug, "board"),
                    "description": meta.get("description") if EXPOSE_LOCAL_LABELS else None,
                    "db_exists": (p / "kanban.db").exists(),
                })
        except Exception:
            pass
    current = read_kanban_current()
    for board in boards:
        board["is_current"] = str(board.get("_slug") or board.get("slug")) == current
    return boards


def _kanban_conn(path: Path) -> Optional[sqlite3.Connection]:
    if not path.exists():
        return None
    # Some Kanban SQLite files can be opened in `mode=ro` but fail on the
    # first read when SQLite tries to resolve journal/WAL sidecars. Verify the
    # connection before returning it and fall back to immutable read-only mode.
    for uri in (f"file:{path}?mode=ro", f"file:{path}?mode=ro&immutable=1"):
        con: Optional[sqlite3.Connection] = None
        try:
            con = sqlite3.connect(uri, uri=True, timeout=1.5)
            con.row_factory = sqlite3.Row
            con.execute("SELECT 1 FROM sqlite_master LIMIT 1").fetchone()
            return con
        except Exception:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass
    return None


def profile_config(profile_home: Path) -> Dict[str, Any]:
    return read_yaml(profile_home / "config.yaml")


def summarize_model(config: Dict[str, Any]) -> Dict[str, Optional[str]]:
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    if not isinstance(model, dict):
        model = {}
    return {
        "provider": model.get("provider"),
        "model": model.get("default") or model.get("model"),
    }


def _cfg_value(config: Dict[str, Any], *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _cfg_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled", "allow", "record"}
    return False


def _cfg_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _configured_count(value: Any) -> int:
    if isinstance(value, dict):
        return len([k for k, v in value.items() if v not in (None, "", False, [], {})])
    if isinstance(value, list):
        return len([item for item in value if item not in (None, "", False, [], {})])
    return 1 if value not in (None, "", False, [], {}) else 0


def _cfg_any_count(config: Dict[str, Any], paths: List[List[str]]) -> int:
    total = 0
    for path in paths:
        total += _configured_count(_cfg_value(config, *path))
    return total


def _cfg_any_bool(config: Dict[str, Any], paths: List[List[str]]) -> bool:
    return any(_cfg_bool(_cfg_value(config, *path)) for path in paths)


def _profile_config_paths(profiles: List[Dict[str, Any]]) -> List[Path]:
    paths: List[Path] = []
    for profile in profiles:
        raw = profile.get("_path") if isinstance(profile, dict) else None
        if raw:
            paths.append(Path(str(raw)) / "config.yaml")
    return paths


def _config_max_turns(configs: List[Dict[str, Any]], kanban: Dict[str, Any]) -> List[int]:
    values: List[int] = []
    for cfg in configs:
        for path in (["agent", "max_turns"], ["agents", "max_turns"], ["max_turns"]):
            val = _cfg_int(_cfg_value(cfg, *path))
            if val is not None:
                values.append(val)
    for task in kanban_items(kanban, "recent_tasks"):
        val = _cfg_int(task.get("goal_max_turns"))
        if val is not None:
            values.append(val)
    return values


def _loop_guardrail_enabled(config: Dict[str, Any]) -> bool:
    for path in (["tool_loop_guardrails"], ["tool_loop_guardrail"], ["agent", "tool_loop_guardrails"], ["agent", "loop_guardrails"]):
        section = _cfg_value(config, *path)
        if isinstance(section, dict):
            if any(_cfg_bool(section.get(key)) for key in ("enabled", "hard_stop", "hard_stop_enabled", "stop_enabled")):
                return True
            if any(section.get(key) not in (None, "", 0, False) for key in ("max_repeats", "max_tool_repeats", "max_iterations", "max_tool_calls")):
                return True
        elif _cfg_bool(section):
            return True
    return False


def _auxiliary_route_count(config: Dict[str, Any], env_names: set[str]) -> int:
    config_count = _cfg_any_count(config, [
        ["auxiliary_provider"],
        ["auxiliary_providers"],
        ["auxiliary"],
        ["aux"],
        ["model", "auxiliary_provider"],
        ["model", "auxiliary_providers"],
        ["providers", "auxiliary"],
        ["providers", "background"],
        ["providers", "summarizer"],
    ])
    env_count = 1 if any("AUX" in key or "AUXILIARY" in key for key in env_names) else 0
    return config_count + env_count


def collect_config_policy(profiles: List[Dict[str, Any]], sessions: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    kanban = kanban or {}
    hermes_home = get_hermes_home()
    root_config_path = hermes_home / "config.yaml"
    root_config = read_yaml(root_config_path)
    profile_paths = _profile_config_paths(profiles)
    profile_configs = [read_yaml(path) for path in profile_paths if path.exists()]
    configs = [root_config, *profile_configs]
    env_names = set(env_keys(hermes_home / ".env"))
    for profile in profiles:
        raw = profile.get("_path") if isinstance(profile, dict) else None
        if raw:
            env_names.update(env_keys(Path(str(raw)) / ".env"))

    max_turns_values = _config_max_turns(configs, kanban)
    max_turns = max(max_turns_values) if max_turns_values else 0
    hard_loop_stop = any(_loop_guardrail_enabled(cfg) for cfg in configs)
    fallback_count = _cfg_any_count(root_config, [["fallback_providers"], ["model", "fallback_providers"], ["providers", "fallback"], ["providers", "fallback_providers"]])
    toolset_count = sum(_cfg_any_count(cfg, [["toolsets"], ["tools", "toolsets"], ["agent", "toolsets"]]) for cfg in configs)
    compression_enabled = any(_cfg_any_bool(cfg, [["compression", "enabled"], ["agent", "compression"], ["memory", "compression"]]) for cfg in configs)
    browser_private_urls = _cfg_any_bool(root_config, [["browser", "allow_private_urls"], ["browser", "allow_local_urls"], ["browser", "private_urls"], ["browser", "allow_private_network"]])
    browser_recording = _cfg_any_bool(root_config, [["browser", "record_sessions"], ["browser", "recording"], ["browser", "session_recording"]])
    auxiliary_routes = _auxiliary_route_count(root_config, env_names)
    route_audit_profiles = sum(1 for profile in profiles if profile.get("model") or profile.get("provider"))
    model_override_tasks = int(((build_orchestration(kanban).get("summary") or {}).get("model_override_tasks")) or 0)
    sessions_with_cost = sum(1 for session in sessions if _session_cost(session) > 0)
    total_cost = round(sum(_session_cost(session) for session in sessions), 4)
    cost_signal_available = bool(sessions_with_cost or total_cost > 0)

    findings: List[Dict[str, Any]] = []

    def add(kind: str, severity: str, title: str, detail: str, evidence: str, view: str, basis: str, threshold: str = "") -> None:
        action = "Open Config" if view == "/config" else ("Open Sessions" if view == "/sessions" else "Open Analytics")
        findings.append(_tuning_item(kind, severity, title, detail, evidence, view, action, "Olympus", basis, threshold))

    if root_config_path.exists() is False and not profile_configs:
        add(
            "config",
            "info",
            "Hermes config policy evidence is missing",
            "Olympus did not find a readable root or profile config. Runtime signals still render, but policy checks cannot explain agent limits.",
            "config files missing",
            "/config",
            "Hermes config.yaml presence",
        )
    if max_turns >= MAX_TURNS_REVIEW_THRESHOLD and not hard_loop_stop:
        add(
            "policy",
            "warning",
            "High turn limit lacks a visible loop stop",
            "High agent turn limits need a visible hard loop guardrail so tool thrash stops before it burns time and tokens.",
            f"max_turns {max_turns} / hard stop not visible",
            "/config",
            "Hermes agent.max_turns plus tool_loop_guardrails",
            f"max_turns >= {MAX_TURNS_REVIEW_THRESHOLD} and hard loop stop disabled or missing",
        )
    if browser_private_urls:
        add(
            "browser",
            "warning",
            "Browser private URL access is enabled",
            "Private or local browser targets widen the data boundary. Keep this enabled only when local automation needs it.",
            "browser private URL flag enabled",
            "/config",
            "Hermes browser privacy flags",
        )
    if browser_recording:
        add(
            "browser",
            "warning",
            "Browser session recording is enabled",
            "Recorded browser sessions can capture sensitive local workflow context. Review retention and access before long-running agents use it.",
            "browser recording flag enabled",
            "/config",
            "Hermes browser privacy flags",
        )
    if auxiliary_routes and not cost_signal_available:
        add(
            "cost",
            "warning",
            "Auxiliary route cost is not visible",
            "Auxiliary or background provider routes are configured, but recent sessions do not expose cost. Background work can hide spend without a usage signal.",
            f"{auxiliary_routes} auxiliary route signal(s) / 0 costed sessions",
            "/analytics",
            "Hermes auxiliary route presence plus per-session cost fields",
        )
    if fallback_count and not route_audit_profiles:
        add(
            "model",
            "warning",
            "Fallback providers lack route audit evidence",
            "Fallback providers are configured, but profile/session route evidence is not visible. Make routing explicit before tuning model behavior.",
            f"{fallback_count} fallback provider(s) / 0 visible profile routes",
            "/config",
            "Hermes fallback provider config plus profile route metadata",
        )
    if fallback_count and route_audit_profiles:
        add(
            "model",
            "info",
            "Fallback providers have visible route evidence",
            "Fallback routes are present and profile route metadata is visible, so route changes can be audited from Olympus.",
            f"{fallback_count} fallback provider(s) / {route_audit_profiles} routed profile(s)",
            "/config",
            "Hermes fallback provider config plus profile route metadata",
        )

    settings = [
        {
            "id": "max_turns",
            "label": "Max Turns",
            "value": max_turns,
            "state": "warning" if max_turns >= MAX_TURNS_REVIEW_THRESHOLD and not hard_loop_stop else ("active" if max_turns else "unknown"),
            "detail": "Highest configured agent or goal turn limit.",
            "source": "agent.max_turns and Kanban goal_max_turns",
            "recommended_view": "/config",
        },
        {
            "id": "loop_guard",
            "label": "Loop Stop",
            "value": "visible" if hard_loop_stop else "not visible",
            "state": "ok" if hard_loop_stop else ("warning" if max_turns >= MAX_TURNS_REVIEW_THRESHOLD else "unknown"),
            "detail": "Hard loop guardrail or repeated-tool stop evidence.",
            "source": "tool_loop_guardrails",
            "recommended_view": "/config",
        },
        {
            "id": "fallbacks",
            "label": "Fallbacks",
            "value": fallback_count,
            "state": "active" if fallback_count else "idle",
            "detail": "Fallback provider routes configured in safe config structure.",
            "source": "fallback_providers",
            "recommended_view": "/config",
        },
        {
            "id": "toolsets",
            "label": "Toolsets",
            "value": toolset_count,
            "state": "active" if toolset_count else "unknown",
            "detail": "Configured toolset collections across root and profile config.",
            "source": "toolsets",
            "recommended_view": "/config",
        },
        {
            "id": "browser",
            "label": "Browser Privacy",
            "value": (1 if browser_private_urls else 0) + (1 if browser_recording else 0),
            "state": "warning" if browser_private_urls or browser_recording else "ok",
            "detail": "Private URL access and browser recording flags.",
            "source": "browser privacy flags",
            "recommended_view": "/config",
        },
        {
            "id": "aux_cost",
            "label": "Aux Cost",
            "value": total_cost,
            "unit": "usd",
            "state": "warning" if auxiliary_routes and not cost_signal_available else ("active" if cost_signal_available else "unknown"),
            "detail": f"{auxiliary_routes} auxiliary route signal(s), {sessions_with_cost} costed session(s).",
            "source": "auxiliary route presence and estimated/actual_cost_usd",
            "recommended_view": "/analytics",
        },
    ]
    findings.sort(key=lambda item: _severity_rank(str(item.get("severity"))))
    state = "warning" if any(item.get("severity") in ("critical", "warning") for item in findings) else ("info" if findings else ("ok" if root_config_path.exists() or profile_configs else "unknown"))
    return {
        "summary": {
            "state": state,
            "findings": len(findings),
            "root_config_present": root_config_path.exists(),
            "profile_configs": len(profile_configs),
            "max_turns": max_turns,
            "hard_loop_stop": hard_loop_stop,
            "fallback_providers": fallback_count,
            "toolsets": toolset_count,
            "compression_enabled": compression_enabled,
            "browser_private_flags": (1 if browser_private_urls else 0) + (1 if browser_recording else 0),
            "auxiliary_routes": auxiliary_routes,
            "costed_sessions": sessions_with_cost,
            "total_cost_usd": total_cost,
            "route_audit_profiles": route_audit_profiles,
            "model_override_tasks": model_override_tasks,
        },
        "settings": settings,
        "findings": findings[:8],
    }


def _pid_alive(pid: int) -> Optional[bool]:
    """True/False when liveness is known, None when it cannot be determined."""
    try:
        import psutil  # type: ignore

        return bool(psutil.pid_exists(pid))
    except Exception:
        pass
    if os.name == "posix":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return None
    return None


def _process_cmdline(pid: int) -> Optional[str]:
    """Lowercased command line of ``pid`` via psutil, or None if unavailable."""
    try:
        import psutil  # type: ignore

        return " ".join(psutil.Process(pid).cmdline()).lower()
    except Exception:
        return None


def gateway_process_state(profile_home: Path) -> str:
    """Gateway liveness for a profile using Hermes' `gateway.pid` contract."""
    pid_path = profile_home / "gateway.pid"
    if not pid_path.exists():
        return "stopped"
    try:
        from gateway.status import get_running_pid  # type: ignore

        return "running" if get_running_pid(pid_path, cleanup_stale=False) else "stopped"
    except Exception:
        pass

    record = read_json(pid_path)
    pid: Optional[int] = None
    kind: Optional[str] = None
    if isinstance(record, dict):
        kind = record.get("kind")
        try:
            pid = int(record.get("pid"))
        except (TypeError, ValueError):
            pid = None
    elif isinstance(record, (int, float)):
        pid = int(record)
    if not pid:
        return "unknown"
    alive = _pid_alive(pid)
    if alive is False:
        return "stopped"
    if alive is None:
        return "unknown"
    # Avoid treating a recycled PID as a gateway.
    cmdline = _process_cmdline(pid)
    if cmdline is not None:
        return "running" if "gateway" in cmdline else "stopped"
    return "running" if kind == "hermes-gateway" else "unknown"


def collect_profiles() -> List[Dict[str, Any]]:
    base = get_hermes_home()
    profiles: List[Dict[str, Any]] = []

    candidates = [("default", base)]
    prof_dir = base / "profiles"
    if not prof_dir.exists() and base.parent.name == "profiles":
        prof_dir = base.parent
    if prof_dir.exists():
        for p in sorted(prof_dir.iterdir()):
            if p.is_dir() and p.resolve() != base.resolve():
                candidates.append((p.name, p))

    for name, home in candidates:
        cfg = profile_config(home)
        model = summarize_model(cfg)
        model_name = model.get("model")
        provider_name = model.get("provider")
        env_exists = (home / ".env").exists()
        soul_exists = (home / "SOUL.md").exists()
        skills_dir = home / "skills"
        skill_count = 0
        if skills_dir.exists():
            try:
                skill_count = sum(1 for p in skills_dir.rglob("SKILL.md") if p.is_file())
            except Exception:
                skill_count = 0
        gstate = gateway_process_state(home)
        trust = "primary" if name == "default" else "isolated"
        profile_number = len(profiles)
        public_name = name if (EXPOSE_LOCAL_LABELS or name == "default") else f"profile_{profile_number}"
        label = "Default" if name == "default" else (name.capitalize() if EXPOSE_LOCAL_LABELS else f"Profile {profile_number}")
        profiles.append({
            "id": f"profile:{public_name}",
            "kind": "profile",
            "name": name,
            "_public_name": public_name,
            "label": label,
            "state": "active" if gstate == "running" else "idle",
            "trust_boundary": trust,
            "_path": str(home),
            "model": model_name if EXPOSE_LOCAL_LABELS else ("configured" if model_name else None),
            "provider": provider_name if EXPOSE_LOCAL_LABELS else ("configured" if provider_name else None),
            "has_env": env_exists,
            "has_soul": soul_exists,
            "skill_count": skill_count,
            "gateway_state": gstate,
        })
    return profiles


def cron_schedule_display(job: Dict[str, Any]) -> Optional[str]:
    """Human-readable schedule for a cron job, tolerant of legacy shapes.

    ``schedule`` may be a dict (current shape, carrying a ``display`` field), a
    bare string (older / hand-edited ``jobs.json``), or absent. Mirrors Hermes'
    own ``_schedule_display_for_job`` normalization so a string schedule doesn't
    raise ``AttributeError`` on ``.get("display")`` and 500 the whole scan.
    """
    display = job.get("schedule_display")
    if display:
        return str(display)
    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        return schedule.get("display")
    if schedule is not None:
        return str(schedule)
    return None


def collect_cron(profiles: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    base = get_hermes_home()
    data = read_json(base / "cron" / "jobs.json") or {}
    jobs = data.get("jobs") if isinstance(data, dict) else []
    if not isinstance(jobs, list):
        return []
    profile_public_names = profile_public_map(as_list(profiles))
    out: List[Dict[str, Any]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        last_status = job.get("last_status")
        enabled = bool(job.get("enabled", True))
        error = job.get("last_error") or job.get("last_delivery_error")
        state = "error" if error or last_status == "error" else ("scheduled" if enabled else "paused")
        raw_profile = str(job.get("profile") or "default")
        public_name = public_profile_label(raw_profile, profile_public_names) or "default"
        job_ref = public_ref(job.get("id"), "cron") or "cron"
        out.append({
            "id": job_ref,
            "kind": "cron",
            "job_id": job_ref,
            "label": public_label(job.get("name"), job_ref, "cron"),
            "state": state,
            "enabled": enabled,
            "schedule": cron_schedule_display(job),
            "next_run_at": ts_to_iso(job.get("next_run_at")),
            "last_run_at": ts_to_iso(job.get("last_run_at")),
            "last_status": last_status,
            "last_error": redact_text(error) if error else None,
            "profile": public_name,
            "_profile": raw_profile,
            "no_agent": bool(job.get("no_agent")),
        })
    return out


def collect_sessions(limit: int = 12) -> List[Dict[str, Any]]:
    db = get_hermes_home() / "state.db"
    if not db.exists():
        return []
    con: Optional[sqlite3.Connection] = None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.5)
        con.row_factory = sqlite3.Row
        cols = table_columns(con, "sessions")
        wanted = [
            "id", "source", "started_at", "ended_at", "end_reason", "message_count",
            "tool_call_count", "model", "title", "handoff_platform", "handoff_error",
            "input_tokens", "output_tokens", "reasoning_tokens", "api_call_count",
            "estimated_cost_usd", "actual_cost_usd",
        ]
        select = [c for c in wanted if c in cols]
        if "id" not in select:
            return []
        if "ended_at" in cols and "started_at" in cols:
            order = "COALESCE(ended_at, started_at)"
        elif "started_at" in cols:
            order = "started_at"
        else:
            order = "id"
        rows = con.execute(
            f"SELECT {', '.join(select)} FROM sessions ORDER BY {order} DESC LIMIT ?",
            (limit,),
        ).fetchall()
    except Exception as exc:
        log_read_warning("session scan failed", exc)
        return []
    finally:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
    out: List[Dict[str, Any]] = []
    for r in rows:
        ended = row_value(r, "ended_at")
        started = row_value(r, "started_at")
        last_ts = ended or started
        if ended is None:
            state = "active" if age_state(started) in ("active", "recent") else "stale"
        else:
            state = "recent" if age_state(last_ts) in ("active", "recent") else "completed"
        handoff_error = row_value(r, "handoff_error")
        if handoff_error:
            state = "error"
        duration_seconds: Optional[float] = None
        try:
            if started and ended:
                duration_seconds = max(0.0, float(ended) - float(started))
        except Exception:
            duration_seconds = None
        session_id = row_value(r, "id")
        model = row_value(r, "model")
        message_count = safe_int(row_value(r, "message_count", 0))
        tool_call_count = safe_int(row_value(r, "tool_call_count", 0))
        input_tokens = safe_int(row_value(r, "input_tokens", 0))
        output_tokens = safe_int(row_value(r, "output_tokens", 0))
        reasoning_tokens = safe_int(row_value(r, "reasoning_tokens", 0))
        api_calls = safe_int(row_value(r, "api_call_count", 0))
        # Average per API call is the context-pressure signal; cumulative tokens
        # over-count chatty sessions.
        avg_input_tokens = int(input_tokens / api_calls) if api_calls > 0 else 0
        actual_cost = row_value(r, "actual_cost_usd")
        estimated_cost = row_value(r, "estimated_cost_usd")
        cost_source = actual_cost if actual_cost not in (None, "") else estimated_cost
        try:
            cost_usd = float(cost_source) if cost_source not in (None, "") else 0.0
        except (TypeError, ValueError):
            cost_usd = 0.0
        cost_estimated = actual_cost in (None, "") and estimated_cost not in (None, "")
        public_session_ref = str(session_id) if EXPOSE_LOCAL_LABELS else safe_id(session_id, "session")
        out.append({
            "id": f"session:{public_session_ref}" if EXPOSE_LOCAL_LABELS else public_session_ref,
            "kind": "session",
            "session_ref": public_session_ref,
            "label": public_label(row_value(r, "title"), str(row_value(r, "source") or safe_id(session_id, "session")), "session"),
            "state": state,
            "source": row_value(r, "source"),
            "started_at": ts_to_iso(started),
            "ended_at": ts_to_iso(ended),
            "message_count": message_count,
            "tool_call_count": tool_call_count,
            "duration_seconds": duration_seconds,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens + reasoning_tokens,
            "avg_input_tokens": avg_input_tokens,
            "api_call_count": api_calls,
            "cost_usd": round(cost_usd, 4),
            "cost_estimated": cost_estimated,
            "model": model if EXPOSE_LOCAL_LABELS else ("configured" if model else None),
            "handoff_platform": row_value(r, "handoff_platform"),
            "handoff_error": redact_text(handoff_error) if handoff_error else None,
        })
    return out


def collect_gateways(profiles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Match Hermes gateway env markers.
    platform_markers = {
        "telegram": ["TELEGRAM_BOT_TOKEN"],
        "discord": ["DISCORD_BOT_TOKEN"],
        "slack": ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"],
        "whatsapp": ["WHATSAPP_ENABLED"],
        "signal": ["SIGNAL_HTTP_URL", "SIGNAL_ACCOUNT"],
        "matrix": ["MATRIX_ACCESS_TOKEN", "MATRIX_HOMESERVER"],
        "email": ["EMAIL_ADDRESS", "EMAIL_IMAP_HOST", "EMAIL_SMTP_HOST"],
        "sms": ["TWILIO_ACCOUNT_SID"],
        "mattermost": ["MATTERMOST_TOKEN", "MATTERMOST_URL"],
        "homeassistant": ["HASS_TOKEN", "HASS_URL"],
        "webhook": ["WEBHOOK_ENABLED"],
        "api_server": ["API_SERVER_KEY", "API_SERVER_ENABLED"],
    }
    out: List[Dict[str, Any]] = []
    for prof in profiles:
        name = prof["name"]
        public_name = prof.get("_public_name") or name
        home = Path(prof["_path"])
        cfg = profile_config(home)
        platforms_block = cfg.get("platforms") if isinstance(cfg.get("platforms"), dict) else {}
        configured_keys = env_keys(home / ".env")
        gateway_state = prof.get("gateway_state") or "unknown"
        for platform, markers in platform_markers.items():
            # Hermes accepts top-level and `platforms.<platform>` config.
            pdata = cfg.get(platform)
            if not isinstance(pdata, dict):
                pdata = platforms_block.get(platform)
            explicitly_enabled = bool(pdata.get("enabled", False)) if isinstance(pdata, dict) else False
            configured = any(marker in configured_keys for marker in markers)
            enabled = explicitly_enabled or configured
            if not enabled:
                continue
            out.append({
                "id": f"gateway:{public_name}:{platform}",
                "kind": "gateway",
                "label": platform.replace("_", " ").title(),
                "platform": platform,
                "profile": public_name,
                "state": gateway_state if enabled else "disabled",
                "enabled": enabled,
                "configured": configured,
                "trust_boundary": prof.get("trust_boundary"),
            })
    return out


def collect_kanban(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    profile_states = {str(p.get("name")): p.get("state") for p in profiles}
    profile_gateway = {str(p.get("name")): p.get("gateway_state") for p in profiles}
    profile_public_names = profile_public_map(profiles)
    now = int(time.time())
    boards = discover_kanban_boards()
    totals = {status: 0 for status in KANBAN_COLUMNS}
    attention: List[Dict[str, Any]] = []
    assignee_load: Dict[str, Dict[str, Any]] = {}
    recent_tasks: List[Dict[str, Any]] = []
    recent_runs: List[Dict[str, Any]] = []
    active_workers: List[Dict[str, Any]] = []

    def add_attention(severity: str, label: str, detail: str, board: str, task_id: Optional[str] = None) -> None:
        attention.append({
            "severity": severity,
            "label": label,
            "detail": detail,
            "board": board,
            "task_id": task_id,
            "recommended_view": "/kanban",
        })

    for board in boards:
        slug = str(board["slug"])
        raw_slug = str(board.get("_slug") or slug)
        db_path = kanban_board_db(raw_slug)
        con = _kanban_conn(db_path)
        board_counts = {status: 0 for status in KANBAN_COLUMNS}
        board.update({
            "counts": board_counts,
            "total": 0,
            "open": 0,
            "diagnostic_count": 0,
            "active_workers": 0,
        })
        if con is None:
            continue
        try:
            task_cols = table_columns(con, "tasks")
            run_cols = table_columns(con, "task_runs")
            for r in con.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall():
                status = str(r["status"] or "unknown")
                n = int(r["n"] or 0)
                board_counts[status] = n
                totals[status] = totals.get(status, 0) + n
            board["total"] = sum(board_counts.values())
            board["open"] = sum(board_counts.get(s, 0) for s in OPEN_KANBAN_STATUSES)

            task_select = [
                "id", "title", "status", "assignee", "priority", "created_at", "started_at", "completed_at",
                "consecutive_failures", "last_failure_error", "worker_pid", "claim_expires",
                "last_heartbeat_at", "current_run_id", "max_runtime_seconds", "session_id",
                "goal_mode", "goal_max_turns", "workspace_kind", "tenant", "skills", "model_override",
            ]
            task_select = [c for c in task_select if c in task_cols]
            if task_select:
                task_rows = con.execute(
                    f"SELECT {', '.join(task_select)} FROM tasks WHERE status != 'archived' "
                    "ORDER BY COALESCE(completed_at, started_at, created_at) DESC LIMIT 400"
                ).fetchall()
            else:
                task_rows = []

            for t in task_rows:
                status = str(row_value(t, "status", "unknown"))
                assignee = str(row_value(t, "assignee") or "unassigned")
                public_assignee = public_profile_label(assignee, profile_public_names) or "unassigned"
                if status in OPEN_KANBAN_STATUSES:
                    load = assignee_load.setdefault(assignee, {
                        "_assignee": assignee,
                        "assignee": public_assignee,
                        "open": 0,
                        "running": 0,
                        "ready": 0,
                        "blocked": 0,
                        "review": 0,
                        "boards": set(),
                        "profile_state": profile_states.get(assignee, "unknown"),
                        "gateway_state": profile_gateway.get(assignee, "unknown"),
                    })
                    load["open"] += 1
                    load["boards"].add(slug)
                    if status in ("running", "ready", "blocked", "review"):
                        load[status] += 1

                raw_title = str(row_value(t, "title", row_value(t, "id", "task")))
                task_id = str(row_value(t, "id", ""))
                public_task_id = task_id if EXPOSE_LOCAL_LABELS else safe_id(task_id, "task")
                title = public_label(raw_title, public_task_id, "task")
                created_at = row_value(t, "created_at")
                started_at = row_value(t, "started_at")
                heartbeat = row_value(t, "last_heartbeat_at")
                claim_expires = row_value(t, "claim_expires")
                failures = int(row_value(t, "consecutive_failures", 0) or 0)

                if status == "running":
                    stale_by_heartbeat = heartbeat and now - int(heartbeat) > KANBAN_HEARTBEAT_STALE_SECONDS
                    expired_claim = claim_expires and int(claim_expires) < now
                    if stale_by_heartbeat or expired_claim:
                        add_attention(
                            "critical",
                            f"Stale running task: {title}",
                            f"{public_assignee} has a running task with stale heartbeat or expired claim.",
                            slug,
                            public_task_id,
                        )
                if status == "ready" and assignee == "unassigned":
                    add_attention("warning", f"Ready task is unassigned: {title}", "Dispatcher cannot claim ready work without an assignee.", slug, public_task_id)
                if status == "ready" and assignee != "unassigned" and profile_gateway.get(assignee) not in ("running", None):
                    add_attention("warning", f"Ready task may not dispatch: {title}", f"Assignee {public_assignee} does not have a visible running gateway.", slug, public_task_id)
                if status == "blocked":
                    add_attention("warning", f"Blocked task: {title}", "Blocked Kanban work needs operator review or clearer acceptance criteria.", slug, public_task_id)
                if failures > 0:
                    add_attention("warning", f"Retry pressure: {title}", f"{failures} consecutive failure(s). {redact_text(row_value(t, 'last_failure_error'), 120)}", slug, public_task_id)
                if status in ("triage", "todo", "ready") and created_at:
                    try:
                        if now - int(created_at) > KANBAN_READY_STALE_SECONDS:
                            add_attention("info", f"Aging Kanban work: {title}", f"Task has been waiting in {status} for more than 24h.", slug, public_task_id)
                    except Exception:
                        pass

                if len(recent_tasks) < 16:
                    forced_skills: List[str] = []
                    raw_skills = row_value(t, "skills")
                    if raw_skills:
                        try:
                            parsed_skills = json.loads(raw_skills)
                            if isinstance(parsed_skills, list):
                                forced_skills = [str(x) for x in parsed_skills if x]
                        except Exception:
                            forced_skills = []
                    recent_tasks.append({
                        "id": public_task_id,
                        "board": slug,
                        "title": title,
                        "status": status,
                        "assignee": public_assignee,
                        "_assignee": assignee,
                        "priority": row_value(t, "priority", 0),
                        "created_at": ts_to_iso(created_at),
                        "started_at": ts_to_iso(started_at),
                        "completed_at": ts_to_iso(row_value(t, "completed_at")),
                        "last_heartbeat_at": ts_to_iso(heartbeat),
                        "claim_expires_at": ts_to_iso(claim_expires),
                        "worker_ref": public_ref(row_value(t, "worker_pid"), "pid"),
                        "current_run_ref": public_ref(row_value(t, "current_run_id"), "run"),
                        "max_runtime_seconds": row_value(t, "max_runtime_seconds"),
                        "consecutive_failures": failures,
                        "goal_mode": bool(row_value(t, "goal_mode", 0)),
                        "goal_max_turns": row_value(t, "goal_max_turns"),
                        "_skills": forced_skills,
                        "forced_skill_count": len(forced_skills),
                        "model_override": row_value(t, "model_override") if EXPOSE_LOCAL_LABELS else ("override" if row_value(t, "model_override") else None),
                        "session_ref": row_value(t, "session_id") if EXPOSE_LOCAL_LABELS else (safe_id(row_value(t, "session_id"), "session") if row_value(t, "session_id") else None),
                    })

            if run_cols:
                run_select = [
                    "id", "task_id", "profile", "step_key", "status", "outcome", "worker_pid",
                    "last_heartbeat_at", "started_at", "ended_at", "error", "summary",
                ]
                run_select = [c for c in run_select if c in run_cols]
                run_rows = con.execute(
                    f"SELECT {', '.join(run_select)} FROM task_runs ORDER BY id DESC LIMIT 100"
                ).fetchall() if run_select else []
                for r in run_rows:
                    run_status = str(row_value(r, "status", "unknown"))
                    outcome = row_value(r, "outcome")
                    ended_at = row_value(r, "ended_at")
                    heartbeat = row_value(r, "last_heartbeat_at")
                    task_id = str(row_value(r, "task_id", ""))
                    public_task_id = task_id if EXPOSE_LOCAL_LABELS else safe_id(task_id, "task")
                    if ended_at is None and run_status == "running":
                        active_workers.append({
                            "board": slug,
                            "run_ref": public_ref(row_value(r, "id"), "run"),
                            "task_id": public_task_id,
                            "profile": public_profile_label(row_value(r, "profile"), profile_public_names),
                            "worker_ref": public_ref(row_value(r, "worker_pid"), "pid"),
                            "started_at": ts_to_iso(row_value(r, "started_at")),
                            "last_heartbeat_at": ts_to_iso(heartbeat),
                        })
                    if is_failed_run({"status": run_status, "outcome": outcome}):
                        add_attention("critical", f"Kanban worker {outcome or run_status}: {public_task_id}", redact_text(row_value(r, "error") or "Worker run failed", 200), slug, public_task_id)
                    if len(recent_runs) < 16:
                        recent_runs.append({
                            "id": public_ref(row_value(r, "id"), "run"),
                            "board": slug,
                            "task_id": public_task_id,
                            "profile": public_profile_label(row_value(r, "profile"), profile_public_names),
                            "status": run_status,
                            "outcome": outcome,
                            "started_at": ts_to_iso(row_value(r, "started_at")),
                            "ended_at": ts_to_iso(ended_at),
                            "error": redact_text(row_value(r, "error"), 180),
                        })

            event_cols = table_columns(con, "task_events")
            recent_events: List[Dict[str, Any]] = board.setdefault("_recent_events", [])
            if {"id", "task_id", "kind", "created_at"}.issubset(event_cols):
                event_select = ["id", "task_id", "run_id", "kind", "created_at"]
                event_select = [c for c in event_select if c in event_cols]
                event_rows = con.execute(
                    f"SELECT {', '.join('e.' + c for c in event_select)}, t.title, t.assignee, t.status "
                    "FROM task_events e LEFT JOIN tasks t ON t.id = e.task_id "
                    "ORDER BY e.created_at DESC LIMIT 80"
                ).fetchall()
                for e in event_rows:
                    task_id = str(row_value(e, "task_id", ""))
                    public_task_id = task_id if EXPOSE_LOCAL_LABELS else safe_id(task_id, "task")
                    raw_title = str(row_value(e, "title") or task_id or "task")
                    assignee = str(row_value(e, "assignee") or "unassigned")
                    public_assignee = public_profile_label(assignee, profile_public_names) or "unassigned"
                    event_ref = public_ref(row_value(e, "id"), "event") or "event"
                    recent_events.append({
                        "id": f"{slug}:{event_ref}",
                        "board": slug,
                        "task_id": public_task_id,
                        "run_ref": public_ref(row_value(e, "run_id"), "run"),
                        "kind": row_value(e, "kind"),
                        "created_at": ts_to_iso(row_value(e, "created_at")),
                        "task_title": public_label(raw_title, public_task_id, "task"),
                        "task_status": row_value(e, "status"),
                        "assignee": public_assignee,
                        "_assignee": assignee,
                    })

            board["active_workers"] = sum(1 for w in active_workers if w.get("board") == slug)
        except Exception as exc:
            log_read_warning(f"kanban board scan failed for {slug}", exc)
            board["error"] = redact_text(exc, 200)
            add_attention("warning", f"Could not read Kanban board {slug}", redact_text(exc), slug)
        finally:
            con.close()

    for load in assignee_load.values():
        load["boards"] = sorted(load["boards"])

    all_events: List[Dict[str, Any]] = []
    for board in boards:
        all_events.extend(as_list(board.pop("_recent_events", [])))
    all_events.sort(key=lambda x: str(x.get("created_at") or ""), reverse=True)
    attention.sort(key=lambda x: _severity_rank(str(x.get("severity"))))
    current_board = read_kanban_current()
    public_current_board = current_board if (EXPOSE_LOCAL_LABELS or current_board == "default") else safe_id(current_board, "board")
    return {
        "current": public_current_board,
        "boards": boards,
        "totals": totals,
        "open": sum(totals.get(s, 0) for s in OPEN_KANBAN_STATUSES),
        "active_workers": active_workers[:12],
        "assignee_load": sorted(assignee_load.values(), key=lambda x: (-int(x.get("open") or 0), str(x.get("assignee")))),
        "attention": attention[:16],
        "recent_tasks": recent_tasks,
        "recent_runs": recent_runs,
        "recent_events": all_events[:40],
    }


def collect_health(profiles: List[Dict[str, Any]], cron: List[Dict[str, Any]]) -> Dict[str, Any]:
    base = get_hermes_home()
    logs_dir = base / "logs"
    agent_log = logs_dir / "agent.log"
    gateway_log = logs_dir / "gateway.log"
    error_log = logs_dir / "errors.log"
    # agent.log always exists; gateway.log appears after gateway startup.
    log_tail = "\n".join(
        read_text(p, 8000) for p in (agent_log, gateway_log, error_log) if p.exists()
    ).lower()
    recent_errors = [label for label, pattern in LOG_ERROR_PATTERNS if pattern.search(log_tail)]
    failed_cron = [j for j in cron if j.get("state") == "error"]
    gateway_running = any(p.get("gateway_state") == "running" for p in profiles)
    status = "error" if failed_cron else ("warning" if recent_errors or not gateway_running else "ok")
    if failed_cron:
        status_label = "Needs action"
        summary = f"{len(failed_cron)} scheduled job(s) report errors."
    elif recent_errors:
        status_label = "Needs review"
        summary = "Log tail contains failure terms. Inspect logs."
    elif not gateway_running:
        status_label = "Check gateway"
        summary = "No visible gateway process was detected."
    else:
        status_label = "Healthy"
        summary = "No cron failures or log failure terms detected in the current scan."
    return {
        "status": status,
        "status_label": status_label,
        "summary": summary,
        "gateway_running": gateway_running,
        "recent_error_terms": recent_errors[:5],
        "log_scan_window": "last 8KB per log file",
        "failed_cron_count": len(failed_cron),
        "agent_log_mtime": ts_to_iso(agent_log.stat().st_mtime) if agent_log.exists() else None,
        "gateway_log_mtime": ts_to_iso(gateway_log.stat().st_mtime) if gateway_log.exists() else None,
        "errors_log_mtime": ts_to_iso(error_log.stat().st_mtime) if error_log.exists() else None,
    }


def build_attention(profiles: List[Dict[str, Any]], gateways: List[Dict[str, Any]], cron: List[Dict[str, Any]], sessions: List[Dict[str, Any]], health: Dict[str, Any], kanban: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if health.get("status") == "error":
        items.append({"severity": "critical", "label": "Health check reports errors", "detail": ", ".join(health.get("recent_error_terms") or []) or "Cron/log error detected"})
    elif health.get("status") == "warning":
        if not health.get("gateway_running"):
            items.append({"severity": "warning", "label": "No running gateway process detected", "detail": "Gateway may be intentionally stopped or not visible to Olympus."})
        else:
            terms = ", ".join(health.get("recent_error_terms") or [])
            items.append({"severity": "warning", "label": "Log-tail warnings detected", "detail": f"Hermes log tail contains failure terms: {terms}." if terms else (health.get("summary") or "Log tail contains failure terms.")})
    for job in cron:
        if job.get("state") == "error":
            items.append({"severity": "critical", "label": f"Cron failed: {job.get('label')}", "detail": str(job.get("last_error") or "last_status=error")[:240]})
    for sess in sessions:
        if sess.get("state") == "error":
            items.append({"severity": "warning", "label": f"Session handoff error: {sess.get('label')}", "detail": str(sess.get("handoff_error"))[:240]})
    for item in as_list((kanban or {}).get("attention")):
        if isinstance(item, dict):
            items.append({
                "severity": item.get("severity") or "warning",
                "label": item.get("label") or "Kanban attention item",
                "detail": item.get("detail") or "",
                "recommended_view": item.get("recommended_view") or "/kanban",
            })
    items.sort(key=lambda item: _severity_rank(str(item.get("severity"))))
    return items[:12]


def _severity_rank(value: str) -> int:
    return {"critical": 0, "error": 1, "warning": 2, "info": 3, "ok": 4}.get(value, 3)


def _action_label_for_view(view: str) -> str:
    return {
        "/analytics": "Open Analytics",
        "/config": "Open Config",
        "/cron": "Open Cron",
        "/kanban": "Open Kanban",
        "/logs": "Open Logs",
        "/profiles": "Open Profiles",
        "/sessions": "Open Sessions",
        "/skills": "Open Skills",
    }.get(view, "Open Route")


def _recommendation(severity: str, title: str, detail: str, evidence: str, view: str, owner: str, action: str, basis: str = "") -> Dict[str, Any]:
    return {
        "severity": severity,
        "title": title,
        "detail": detail,
        "evidence": evidence,
        "recommended_view": view,
        "action_label": action,
        "owner": owner,
        "basis": basis,
    }


def _tuning_item(kind: str, severity: str, title: str, detail: str, evidence: str, view: str, action: str, owner: str = "Olympus", basis: str = "", threshold: str = "") -> Dict[str, Any]:
    return {
        "kind": kind,
        "severity": severity,
        "title": title,
        "detail": detail,
        "evidence": evidence,
        "recommended_view": view,
        "action_label": action,
        "owner": owner,
        "basis": basis,
        "threshold": threshold,
    }


def _session_cost(s: Dict[str, Any]) -> float:
    try:
        return float(s.get("cost_usd") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def profile_public_map(profiles: List[Dict[str, Any]]) -> Dict[str, str]:
    return {
        str(p.get("name")): str(p.get("_public_name") or p.get("name"))
        for p in profiles
        if isinstance(p, dict) and p.get("name")
    }


def public_profile_label(raw: Any, profile_names: Dict[str, str]) -> Optional[str]:
    if raw in (None, ""):
        return None
    name = str(raw)
    return profile_names.get(name, name if EXPOSE_LOCAL_LABELS or name in {"default", "unassigned"} else safe_id(name, "profile"))


def cron_profile(job: Dict[str, Any]) -> str:
    return str(job.get("_profile") or job.get("profile") or "default")


def group_cron_by_profile(cron: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for job in cron:
        grouped.setdefault(cron_profile(job), []).append(job)
    return grouped


def group_gateways_by_profile(gateways: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for gateway in gateways:
        grouped.setdefault(str(gateway.get("profile") or ""), []).append(gateway)
    return grouped


def kanban_items(kanban: Optional[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    return [x for x in as_list((kanban or {}).get(key)) if isinstance(x, dict)]


def session_matches_profile(session: Dict[str, Any], raw_name: str, public_name: Any) -> bool:
    labels = {str(raw_name).lower(), str(public_name or "").lower()}
    labels.discard("")
    if not labels:
        return False
    for key in ("profile", "source", "label"):
        value = str(session.get(key) or "").lower()
        if value in labels or any(label and label in value for label in labels):
            return True
    return False


def is_failed_run(run: Dict[str, Any]) -> bool:
    return run.get("status") in FAILED_RUN_STATUSES or run.get("outcome") in FAILED_RUN_OUTCOMES


def task_stale_reason(task: Dict[str, Any], now: Optional[float] = None) -> Optional[str]:
    if task.get("status") != "running":
        return None
    now = time.time() if now is None else now
    try:
        heartbeat = task.get("last_heartbeat_at")
        if heartbeat:
            ts = datetime.fromisoformat(str(heartbeat).replace("Z", "+00:00")).timestamp()
            if now - ts > KANBAN_HEARTBEAT_STALE_SECONDS:
                return "heartbeat stale"
    except Exception:
        pass
    try:
        claim = task.get("claim_expires_at")
        if claim:
            ts = datetime.fromisoformat(str(claim).replace("Z", "+00:00")).timestamp()
            if ts < now:
                return "claim expired"
    except Exception:
        pass
    return None


def score_state(score: int) -> str:
    if score < 55:
        return "critical"
    if score < 75:
        return "warning"
    if score < 90:
        return "info"
    return "ok"


def collect_usage_rollup(profiles: Optional[List[Dict[str, Any]]] = None, days: int = 30) -> Dict[str, Any]:
    """Aggregate Hermes usage counters from configured profile state stores.

    Olympus uses this only as operational evidence. Hermes Analytics owns the
    usage ledger, daily bars, model leaderboards, skill leaderboards, and raw
    cost totals.
    """
    cutoff = time.time() - max(1, int(days)) * 86400
    sources = profiles or [{"label": "Default", "_public_name": "default", "_path": str(get_hermes_home())}]
    totals = {
        "sessions": 0,
        "messages": 0,
        "tool_calls": 0,
        "api_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "actual_cost_usd": 0.0,
        "costed_sessions": 0,
        "actual_costed_sessions": 0,
        "estimated_costed_sessions": 0,
        "zero_usage_suspect_sessions": 0,
        "zero_cost_token_sessions": 0,
        "completed_sessions": 0,
        "errored_sessions": 0,
        "stale_or_open_sessions": 0,
    }
    profile_rollups: List[Dict[str, Any]] = []
    read_failures = 0
    stores_seen = 0

    def sum_expr(col: str, alias: Optional[str] = None) -> str:
        alias = alias or col
        return f"COALESCE(SUM({col}), 0) AS {alias}"

    for idx, profile in enumerate(sources):
        raw_path = profile.get("_path")
        if not raw_path:
            continue
        db = Path(str(raw_path)) / "state.db"
        label = str(profile.get("label") or f"Profile {idx}")
        public_name = str(profile.get("_public_name") or profile.get("id") or f"profile_{idx}")
        roll = {
            "profile": public_name,
            "label": label,
            "state_store": "missing",
            "sessions": 0,
            "tool_calls": 0,
            "api_calls": 0,
            "total_tokens": 0,
            "costed_sessions": 0,
            "zero_usage_suspect_sessions": 0,
            "zero_cost_token_sessions": 0,
        }
        if not db.exists():
            profile_rollups.append(roll)
            continue
        con: Optional[sqlite3.Connection] = None
        try:
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=1.5)
            con.row_factory = sqlite3.Row
            cols = table_columns(con, "sessions")
            if not cols:
                raise RuntimeError("sessions table missing")
            stores_seen += 1
            terms = ["COUNT(*) AS sessions"]
            for col, alias in [
                ("message_count", "messages"),
                ("tool_call_count", "tool_calls"),
                ("api_call_count", "api_calls"),
                ("input_tokens", "input_tokens"),
                ("output_tokens", "output_tokens"),
                ("reasoning_tokens", "reasoning_tokens"),
                ("cache_read_tokens", "cache_read_tokens"),
                ("cache_write_tokens", "cache_write_tokens"),
                ("estimated_cost_usd", "estimated_cost_usd"),
                ("actual_cost_usd", "actual_cost_usd"),
            ]:
                terms.append(sum_expr(col, alias) if col in cols else f"0 AS {alias}")
            if "actual_cost_usd" in cols:
                terms.append("SUM(CASE WHEN COALESCE(actual_cost_usd, 0) > 0 THEN 1 ELSE 0 END) AS actual_costed_sessions")
            else:
                terms.append("0 AS actual_costed_sessions")
            if "estimated_cost_usd" in cols:
                terms.append("SUM(CASE WHEN COALESCE(estimated_cost_usd, 0) > 0 THEN 1 ELSE 0 END) AS estimated_costed_sessions")
            else:
                terms.append("0 AS estimated_costed_sessions")
            token_expr = " + ".join(f"COALESCE({c}, 0)" for c in ("input_tokens", "output_tokens", "reasoning_tokens") if c in cols) or "0"
            work_expr = " + ".join(f"COALESCE({c}, 0)" for c in ("message_count", "tool_call_count", "api_call_count") if c in cols) or "0"
            terms.append(f"SUM(CASE WHEN ({work_expr}) > 0 AND ({token_expr}) = 0 THEN 1 ELSE 0 END) AS zero_usage_suspect_sessions")
            cost_expr = " + ".join(f"COALESCE({c}, 0)" for c in ("actual_cost_usd", "estimated_cost_usd") if c in cols) or "0"
            terms.append(f"SUM(CASE WHEN ({token_expr}) > 0 AND ({cost_expr}) = 0 THEN 1 ELSE 0 END) AS zero_cost_token_sessions")
            if "end_reason" in cols:
                terms.append("SUM(CASE WHEN end_reason IN ('completed', 'end', 'user_exit') THEN 1 ELSE 0 END) AS completed_sessions")
                terms.append("SUM(CASE WHEN end_reason IS NOT NULL AND end_reason NOT IN ('completed', 'end', 'user_exit') THEN 1 ELSE 0 END) AS errored_sessions")
            else:
                terms.append("0 AS completed_sessions")
                terms.append("0 AS errored_sessions")
            if "ended_at" in cols:
                terms.append("SUM(CASE WHEN ended_at IS NULL THEN 1 ELSE 0 END) AS stale_or_open_sessions")
            else:
                terms.append("0 AS stale_or_open_sessions")
            where = ""
            params: tuple[Any, ...] = ()
            if "started_at" in cols:
                where = " WHERE started_at >= ?"
                params = (cutoff,)
            row = con.execute(f"SELECT {', '.join(terms)} FROM sessions{where}", params).fetchone()
            if row is None:
                profile_rollups.append(roll)
                continue
            token_total = safe_int(row_value(row, "input_tokens")) + safe_int(row_value(row, "output_tokens")) + safe_int(row_value(row, "reasoning_tokens"))
            actual_costed = safe_int(row_value(row, "actual_costed_sessions"))
            estimated_costed = safe_int(row_value(row, "estimated_costed_sessions"))
            roll.update({
                "state_store": "ok",
                "sessions": safe_int(row_value(row, "sessions")),
                "tool_calls": safe_int(row_value(row, "tool_calls")),
                "api_calls": safe_int(row_value(row, "api_calls")),
                "total_tokens": token_total,
                "costed_sessions": actual_costed + estimated_costed,
                "zero_usage_suspect_sessions": safe_int(row_value(row, "zero_usage_suspect_sessions")),
                "zero_cost_token_sessions": safe_int(row_value(row, "zero_cost_token_sessions")),
            })
            for key in totals:
                if key in ("estimated_cost_usd", "actual_cost_usd"):
                    totals[key] += float(row_value(row, key, 0) or 0)
                elif key == "total_tokens":
                    totals[key] += token_total
                elif key == "costed_sessions":
                    totals[key] += actual_costed + estimated_costed
                elif key in row.keys():
                    totals[key] += safe_int(row_value(row, key))
            profile_rollups.append(roll)
        except Exception as exc:
            read_failures += 1
            log_read_warning("usage rollup failed", exc)
            roll["state_store"] = "warning"
            profile_rollups.append(roll)
        finally:
            if con is not None:
                try:
                    con.close()
                except Exception:
                    pass

    cost_confidence = "unknown"
    if totals["actual_costed_sessions"]:
        cost_confidence = "actual"
    elif totals["estimated_costed_sessions"] and totals["zero_cost_token_sessions"]:
        cost_confidence = "partial"
    elif totals["estimated_costed_sessions"]:
        cost_confidence = "estimated"
    elif totals["total_tokens"]:
        cost_confidence = "missing"

    return {
        "window_days": max(1, int(days)),
        "state": "warning" if read_failures or totals["zero_usage_suspect_sessions"] or cost_confidence in {"partial", "missing"} else ("active" if totals["sessions"] else "unknown"),
        "stores_seen": stores_seen,
        "read_failures": read_failures,
        "cost_confidence": cost_confidence,
        **totals,
        "estimated_cost_usd": round(float(totals["estimated_cost_usd"]), 4),
        "actual_cost_usd": round(float(totals["actual_cost_usd"]), 4),
        "profiles": profile_rollups[:12],
        "recommended_view": "/analytics",
    }


def build_metrics(sessions: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Window rollup of the performance signals Hermes already records."""
    kanban = kanban or {}
    durations = sorted(float(s["duration_seconds"]) for s in sessions if isinstance(s.get("duration_seconds"), (int, float)))
    median_duration = durations[len(durations) // 2] if durations else None
    p90_duration = durations[min(len(durations) - 1, int(len(durations) * 0.9))] if durations else None
    total_tools = sum(int(s.get("tool_call_count") or 0) for s in sessions)
    total_tokens = sum(int(s.get("total_tokens") or 0) for s in sessions)
    total_api_calls = sum(int(s.get("api_call_count") or 0) for s in sessions)
    failed_runs = [r for r in kanban_items(kanban, "recent_runs") if is_failed_run(r)]
    return {
        "window_sessions": len(sessions),
        "total_cost_usd": round(sum(_session_cost(s) for s in sessions), 4),
        "total_tokens": total_tokens,
        "total_tool_calls": total_tools,
        "total_api_calls": total_api_calls,
        "median_duration_seconds": median_duration,
        "p90_duration_seconds": p90_duration,
        "avg_tools_per_session": round(total_tools / len(sessions), 2) if sessions else 0,
        "avg_tokens_per_session": int(total_tokens / len(sessions)) if sessions else 0,
        "avg_tokens_per_call": int(total_tokens / total_api_calls) if total_api_calls else 0,
        "looping_sessions": sum(1 for s in sessions if int(s.get("tool_call_count") or 0) >= RUNAWAY_TOOLS_THRESHOLD),
        "expensive_sessions": sum(1 for s in sessions if _session_cost(s) >= EXPENSIVE_RUN_USD),
        "context_pressure_sessions": sum(1 for s in sessions if int(s.get("avg_input_tokens") or 0) >= CONTEXT_PRESSURE_TOKENS),
        "completed_sessions": sum(1 for s in sessions if s.get("state") == "completed"),
        "stale_sessions": sum(1 for s in sessions if s.get("state") == "stale"),
        "errored_sessions": sum(1 for s in sessions if s.get("state") == "error"),
        "failed_kanban_runs": len(failed_runs),
    }


def build_performance_tracking(sessions: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Operator-facing performance lanes from first-party Hermes evidence."""
    kanban = kanban or {}
    metrics = build_metrics(sessions, kanban)
    looping = [s for s in sessions if int(s.get("tool_call_count") or 0) >= RUNAWAY_TOOLS_THRESHOLD]
    tool_heavy = [
        s for s in sessions
        if TOOL_HEAVY_THRESHOLD <= int(s.get("tool_call_count") or 0) < RUNAWAY_TOOLS_THRESHOLD
    ]
    context_pressure = [s for s in sessions if int(s.get("avg_input_tokens") or 0) >= CONTEXT_PRESSURE_TOKENS]
    stale_sessions = [s for s in sessions if s.get("state") == "stale"]
    errored_sessions = [s for s in sessions if s.get("state") == "error"]
    failed_runs = [r for r in kanban_items(kanban, "recent_runs") if is_failed_run(r)]
    total_cost = float(metrics.get("total_cost_usd") or 0)
    expensive_sessions = int(metrics.get("expensive_sessions") or 0)
    median_duration = metrics.get("median_duration_seconds")
    p90_duration = metrics.get("p90_duration_seconds")

    speed_state = "idle"
    if sessions:
        speed_state = "warning" if (p90_duration and p90_duration > 1800) or (median_duration and median_duration > 900) else "active"
    tool_state = "warning" if looping or tool_heavy else ("active" if metrics.get("total_tool_calls") else "idle")
    context_state = "warning" if context_pressure else ("active" if metrics.get("avg_tokens_per_call") else "idle")
    reliability_state = "warning" if failed_runs or stale_sessions or errored_sessions else ("ok" if sessions or kanban.get("open") else "unknown")
    cost_state = "warning" if expensive_sessions else ("ok" if sessions else "unknown")

    lanes = [
        {
            "id": "speed",
            "label": "Speed",
            "value": p90_duration,
            "unit": "seconds",
            "state": speed_state,
            "detail": f"median {int(median_duration or 0)}s / p90 {int(p90_duration or 0)}s" if sessions else "no session duration data",
            "source": "Hermes sessions duration_seconds",
            "recommended_view": "/sessions",
        },
        {
            "id": "tools",
            "label": "Tool Pressure",
            "value": metrics.get("total_tool_calls"),
            "state": tool_state,
            "detail": f"{len(looping)} looping / {len(tool_heavy)} tool-heavy",
            "source": "Hermes sessions tool_call_count",
            "recommended_view": "/sessions",
        },
        {
            "id": "context",
            "label": "Context",
            "value": metrics.get("avg_tokens_per_call"),
            "state": context_state,
            "detail": f"{len(context_pressure)} pressure session(s)",
            "source": "Hermes sessions input_tokens/api_call_count",
            "recommended_view": "/sessions",
        },
        {
            "id": "reliability",
            "label": "Reliability",
            "value": len(failed_runs) + len(stale_sessions) + len(errored_sessions),
            "state": reliability_state,
            "detail": f"{len(failed_runs)} failed runs / {len(stale_sessions)} stale / {len(errored_sessions)} errored",
            "source": "Hermes task_runs and sessions",
            "recommended_view": "/kanban" if failed_runs else "/sessions",
        },
        {
            "id": "cost_risk",
            "label": "Cost Risk",
            "value": expensive_sessions,
            "state": cost_state,
            "detail": f"{expensive_sessions} session(s) over ${EXPENSIVE_RUN_USD:.2f}" if sessions else "no session cost evidence",
            "source": "Hermes per-session cost fields; Usage/Analytics owns totals",
            "recommended_view": "/analytics",
        },
    ]

    signals: List[Dict[str, Any]] = []
    if looping:
        signals.append({
            "severity": "warning",
            "label": "Looping sessions",
            "detail": ", ".join(f"{s.get('label')}: {s.get('tool_call_count')} tools" for s in looping[:3]),
            "recommended_view": "/sessions",
        })
    if tool_heavy:
        signals.append({
            "severity": "info",
            "label": "Tool-heavy sessions",
            "detail": ", ".join(f"{s.get('label')}: {s.get('tool_call_count')} tools" for s in tool_heavy[:3]),
            "recommended_view": "/skills",
        })
    if context_pressure:
        signals.append({
            "severity": "info",
            "label": "Context pressure",
            "detail": ", ".join(f"{s.get('label')}: {int(s.get('avg_input_tokens') or 0):,} avg in-tokens/call" for s in context_pressure[:3]),
            "recommended_view": "/sessions",
        })
    if failed_runs:
        signals.append({
            "severity": "warning",
            "label": "Kanban worker failures",
            "detail": f"{len(failed_runs)} recent failed run(s)",
            "recommended_view": "/kanban",
        })
    if stale_sessions:
        signals.append({
            "severity": "warning",
            "label": "Stale sessions",
            "detail": f"{len(stale_sessions)} stale session(s)",
            "recommended_view": "/sessions",
        })

    return {
        "summary": {
            "state": "warning" if any(lane["state"] == "warning" for lane in lanes) else ("active" if sessions else "unknown"),
            "window_sessions": metrics.get("window_sessions"),
            "completed_sessions": metrics.get("completed_sessions"),
            "total_tokens": metrics.get("total_tokens"),
            "total_tool_calls": metrics.get("total_tool_calls"),
            "avg_tools_per_session": metrics.get("avg_tools_per_session"),
            "avg_tokens_per_call": metrics.get("avg_tokens_per_call"),
            "total_cost_usd": total_cost,
        },
        "lanes": lanes,
        "signals": signals[:8],
        "metrics": metrics,
    }


def detect_friction(sessions: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Tuning signals for loops, runaway cost, and context pressure."""
    findings: List[Dict[str, Any]] = []
    looping = [s for s in sessions if int(s.get("tool_call_count") or 0) >= RUNAWAY_TOOLS_THRESHOLD]
    expensive = sorted((s for s in sessions if _session_cost(s) >= EXPENSIVE_RUN_USD), key=_session_cost, reverse=True)
    context_pressure = [s for s in sessions if int(s.get("avg_input_tokens") or 0) >= CONTEXT_PRESSURE_TOKENS]

    if looping:
        findings.append(_tuning_item(
            "skill",
            "warning",
            "An agent is looping or tool-thrashing",
            "One or more runs spent an unusually high number of tool calls. Step-repetition/looping is the most common agent failure mode and silently burns tokens. Add a recap or checklist skill, lower max turns, or split the task.",
            ", ".join(f"{s.get('label')}: {s.get('tool_call_count')} tools" for s in looping[:3]),
            "/sessions",
            "Open Sessions",
            "Olympus",
            "Hermes records tool_call_count per session; runaway tool use without termination is a known failure signature.",
            f">= {RUNAWAY_TOOLS_THRESHOLD} tool calls in one run",
        ))
    if expensive:
        total = sum(_session_cost(s) for s in expensive)
        findings.append(_tuning_item(
            "model",
            "warning",
            "Expensive runs are driving cost",
            f"{len(expensive)} recent run(s) each cost over ${EXPENSIVE_RUN_USD:.2f} (~${total:.2f} total). Consider a cheaper model/route for routine work, tighter prompts, or task decomposition.",
            ", ".join(f"{s.get('label')}: ${_session_cost(s):.2f}" for s in expensive[:3]),
            "/analytics",
            "Open Analytics",
            "Olympus",
            "Hermes records per-session token cost (estimated/actual_cost_usd); cost concentration is a routing signal.",
            f"single-run cost >= ${EXPENSIVE_RUN_USD:.2f}",
        ))
    if context_pressure:
        findings.append(_tuning_item(
            "memory",
            "info",
            "Runs are averaging a very large context per call",
            "Some runs sent a large prompt on every model call (full history re-sent each turn). Agent quality, latency, and cost degrade as the per-call context grows; add summarization/recap, prune stale memory, or split the task.",
            ", ".join(f"{s.get('label')}: {int(s.get('avg_input_tokens') or 0):,} avg in-tokens/call" for s in context_pressure[:3]),
            "/sessions",
            "Open Sessions",
            "Mnemosyne",
            "Hermes records cumulative input_tokens and api_call_count per session; their ratio estimates the per-call context size.",
            f">= {CONTEXT_PRESSURE_TOKENS:,} average input tokens per API call",
        ))
    return findings


def build_agent_hq(profiles: List[Dict[str, Any]], gateways: List[Dict[str, Any]], cron: List[Dict[str, Any]], sessions: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    kanban = kanban or {}
    assignee_load = {
        str(a.get("_assignee") or a.get("assignee")): a for a in as_list(kanban.get("assignee_load"))
        if isinstance(a, dict) and (a.get("_assignee") or a.get("assignee"))
    }
    cron_by_profile = group_cron_by_profile(cron)

    tool_heavy = [s for s in sessions if int(s.get("tool_call_count") or 0) >= TOOL_HEAVY_THRESHOLD]
    # Runaway sessions are handled by detect_friction to avoid duplicate cards.
    _looping_ids = {s.get("id") for s in sessions if int(s.get("tool_call_count") or 0) >= RUNAWAY_TOOLS_THRESHOLD}
    tool_heavy = [s for s in tool_heavy if s.get("id") not in _looping_ids]
    message_heavy = [s for s in sessions if int(s.get("message_count") or 0) >= LONG_THREAD_THRESHOLD]
    active_sessions = [s for s in sessions if s.get("state") in ("active", "recent")]
    stale_sessions = [s for s in sessions if s.get("state") == "stale"]
    failed_runs = [r for r in kanban_items(kanban, "recent_runs") if is_failed_run(r)]
    ready_unassigned = [
        t for t in kanban_items(kanban, "recent_tasks")
        if t.get("status") == "ready" and (not t.get("assignee") or t.get("assignee") == "unassigned")
    ]
    blocked_tasks = [
        t for t in kanban_items(kanban, "recent_tasks")
        if t.get("status") == "blocked"
    ]

    agents: List[Dict[str, Any]] = []
    for profile in profiles:
        name = str(profile.get("name") or profile.get("label") or "default")
        load = assignee_load.get(name, {})
        open_work = int(load.get("open") or 0)
        blocked = int(load.get("blocked") or 0)
        running = int(load.get("running") or 0)
        ready = int(load.get("ready") or 0)
        skill_count = int(profile.get("skill_count") or 0)
        flags: List[str] = []
        if not profile.get("model"):
            flags.append("route metadata")
        if skill_count == 0:
            flags.append("skill coverage")
        if blocked:
            flags.append("blocked work")
        if open_work >= OVERLOADED_OPEN_THRESHOLD or running >= OVERLOADED_RUNNING_THRESHOLD:
            flags.append("load balance")
        if profile.get("gateway_state") != "running" and (ready or running):
            flags.append("gateway route")
        cron_jobs = len(cron_by_profile.get(name, []))
        if cron_jobs:
            flags.append("scheduled work")
        agents.append({
            "id": profile.get("id") or name,
            "label": profile.get("label") or name,
            "state": "warning" if flags else (profile.get("state") or "unknown"),
            "model": profile.get("model"),
            "provider": profile.get("provider"),
            "skill_count": skill_count,
            "gateway_state": profile.get("gateway_state"),
            "cron_jobs": cron_jobs,
            "kanban": {
                "open": open_work,
                "ready": ready,
                "running": running,
                "blocked": blocked,
                "review": int(load.get("review") or 0),
            },
            "flags": flags or ["stable"],
        })

    tuning_items: List[Dict[str, Any]] = []
    overloaded = [a for a in agents if int(a["kanban"]["open"]) >= OVERLOADED_OPEN_THRESHOLD or int(a["kanban"]["running"]) >= OVERLOADED_RUNNING_THRESHOLD]
    missing_models = [a for a in agents if not a.get("model")]
    zero_skill_profiles = [a for a in agents if int(a.get("skill_count") or 0) == 0]

    if tool_heavy:
        tuning_items.append(_tuning_item(
            "skill",
            "warning",
            "Create or preload a skill for repeated tool-heavy work",
            "Recent sessions spent many turns in tools. Add a narrower skill, checklist, or reusable procedure.",
            ", ".join(f"{s.get('label')}: {s.get('tool_call_count')} tools" for s in tool_heavy[:3]),
            "/skills",
            "Open Skills",
            "Olympus",
            "Hermes records tool_call_count. Use traces, tool counts, outcomes, and repeatable evidence before changing routes.",
            f">= {TOOL_HEAVY_THRESHOLD} tool calls in a session",
        ))
    if message_heavy:
        tuning_items.append(_tuning_item(
            "memory",
            "info",
            "Add recap or memory discipline for long-running agents",
            "Long threads need summary checkpoints, memory rules, or task splits before context pressure hurts quality.",
            ", ".join(f"{s.get('label')}: {s.get('message_count')} messages" for s in message_heavy[:3]),
            "/sessions",
            "Open Sessions",
            "Mnemosyne",
            "Hermes records message_count. Long threads can hide context drift.",
            f">= {LONG_THREAD_THRESHOLD} messages in a session",
        ))
    if overloaded:
        tuning_items.append(_tuning_item(
            "agent",
            "warning",
            "Consider a specialist agent or route split",
            "One profile owns enough open or running work to need route review.",
            ", ".join(f"{a.get('label')}: {a['kanban']['open']} open" for a in overloaded[:4]),
            "/profiles",
            "Open Profiles",
            "Hermes",
            "Kanban assignee load shows route concentration.",
            f">= {OVERLOADED_OPEN_THRESHOLD} open or >= {OVERLOADED_RUNNING_THRESHOLD} running tasks",
        ))
    if blocked_tasks:
        tuning_items.append(_tuning_item(
            "kanban",
            "warning",
            "Clarify blocked work before tuning autonomy upward",
            "Blocked tasks usually need clearer acceptance criteria, a dependency, or a human decision.",
            ", ".join(str(t.get("title") or t.get("id")) for t in blocked_tasks[:4]),
            "/kanban",
            "Open Kanban",
            "Olympus",
            "Kanban blocked status is first-party orchestration evidence.",
            "task status = blocked",
        ))
    if ready_unassigned:
        tuning_items.append(_tuning_item(
            "agent",
            "warning",
            "Assign ready work to an agent profile",
            "Ready tasks need an assignee before performance evidence is useful.",
            ", ".join(str(t.get("title") or t.get("id")) for t in ready_unassigned[:4]),
            "/kanban",
            "Open Kanban",
            "Hermes",
            "Kanban ready tasks need an assignee before worker routing can run.",
            "task status = ready and assignee is empty/unassigned",
        ))
    if failed_runs:
        tuning_items.append(_tuning_item(
            "tool",
            "critical",
            "Inspect failed worker runs before adding more automation",
            "Worker failures point to a tool, runtime, approval, or instruction boundary.",
            ", ".join(f"{r.get('task_id')}: {r.get('outcome') or r.get('status')}" for r in failed_runs[:4]),
            "/kanban",
            "Open Kanban",
            "Olympus",
            "Kanban task_runs record worker failures, crashes, and timeouts.",
            "task_run status/outcome indicates failure",
        ))
    if missing_models:
        tuning_items.append(_tuning_item(
            "model",
            "warning",
            "Make routing explicit per profile",
            "Agent tuning is easier when each profile has an intentional route.",
            ", ".join(str(a.get("label")) for a in missing_models[:4]),
            "/profiles",
            "Open Profiles",
            "Olympus",
            "Profile route metadata makes behavior changes auditable.",
            "profile route metadata is unset",
        ))
    if zero_skill_profiles:
        tuning_items.append(_tuning_item(
            "skill",
            "info",
            "Review skill coverage for bare profiles",
            "Profiles with no local skills lack an explicit operating procedure.",
            ", ".join(str(a.get("label")) for a in zero_skill_profiles[:4]),
            "/skills",
            "Open Skills",
            "Apollo",
            "Hermes skills are reusable operating instructions.",
            "profile skill_count = 0",
        ))
    if stale_sessions:
        tuning_items.append(_tuning_item(
            "operations",
            "warning",
            "Resolve stale sessions so the monitor reflects reality",
            "Stale work makes tuning noisy. Close, resume, or annotate it.",
            ", ".join(str(s.get("label") or s.get("session_ref")) for s in stale_sessions[:3]),
            "/sessions",
            "Open Sessions",
            "Chronos",
            "Hermes sessions without an end time and recent activity are stale.",
            "session has no end time and is older than the freshness window",
        ))
    metrics = build_metrics(sessions, kanban)
    tuning_items = detect_friction(sessions, kanban) + tuning_items
    if not tuning_items:
        tuning_items.append(_tuning_item(
            "review",
            "ok",
            "No current agent tuning gap",
            "Olympus did not find overloaded assignees, failed worker runs, blocked tasks, missing routes, looping, runaway cost, or context pressure.",
            "Current cross-surface scan is clean.",
            "/analytics",
            "Open Analytics",
            "Olympus",
        ))

    tuning_items.sort(key=lambda x: _severity_rank(str(x.get("severity"))))
    return {
        "summary": {
            "agents": len(agents),
            "recommendations": len(tuning_items),
            "active_sessions": len(active_sessions),
            "tool_heavy_sessions": len(tool_heavy),
            "long_threads": len(message_heavy),
            "kanban_open": int(kanban.get("open") or 0),
            "total_cost_usd": metrics["total_cost_usd"],
            "total_tokens": metrics["total_tokens"],
            "looping_sessions": metrics["looping_sessions"],
            "expensive_sessions": metrics["expensive_sessions"],
            "context_pressure_sessions": metrics["context_pressure_sessions"],
        },
        "agents": agents,
        "recommendations": tuning_items[:8],
        "metrics": metrics,
    }


def build_skill_coverage(profiles: List[Dict[str, Any]], sessions: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    kanban = kanban or {}
    tasks = kanban_items(kanban, "recent_tasks")
    forced_skill_tasks = [t for t in tasks if int(t.get("forced_skill_count") or 0) > 0]
    forced_skill_total = sum(int(t.get("forced_skill_count") or 0) for t in forced_skill_tasks)
    looping = [s for s in sessions if int(s.get("tool_call_count") or 0) >= RUNAWAY_TOOLS_THRESHOLD]
    tool_heavy = [
        s for s in sessions
        if TOOL_HEAVY_THRESHOLD <= int(s.get("tool_call_count") or 0) < RUNAWAY_TOOLS_THRESHOLD
    ]
    long_threads = [s for s in sessions if int(s.get("message_count") or 0) >= LONG_THREAD_THRESHOLD]
    context_pressure = [s for s in sessions if int(s.get("avg_input_tokens") or 0) >= CONTEXT_PRESSURE_TOKENS]
    zero_skill_profiles = [p for p in profiles if int(p.get("skill_count") or 0) == 0]

    profile_rows: List[Dict[str, Any]] = []
    for profile in profiles:
        raw_name = str(profile.get("name") or "default")
        profile_tasks = [t for t in tasks if str(t.get("_assignee") or t.get("assignee")) == raw_name]
        profile_forced = [t for t in profile_tasks if int(t.get("forced_skill_count") or 0) > 0]
        profile_sessions = [s for s in sessions if session_matches_profile(s, raw_name, profile.get("label"))]
        profile_looping = [s for s in profile_sessions if int(s.get("tool_call_count") or 0) >= RUNAWAY_TOOLS_THRESHOLD]
        profile_tool_heavy = [
            s for s in profile_sessions
            if TOOL_HEAVY_THRESHOLD <= int(s.get("tool_call_count") or 0) < RUNAWAY_TOOLS_THRESHOLD
        ]
        skill_count = int(profile.get("skill_count") or 0)
        open_work = sum(1 for t in profile_tasks if t.get("status") in OPEN_KANBAN_STATUSES)
        blocked = sum(1 for t in profile_tasks if t.get("status") == "blocked")
        if skill_count == 0 and open_work:
            state = "warning"
            issue = "Profile has assigned work but no local skills."
            link = "/skills"
        elif profile_looping or profile_tool_heavy:
            state = "warning"
            issue = "Recent sessions show tool pressure. Review whether this profile needs a checklist or bundle."
            link = "/sessions"
        elif profile_forced:
            state = "active"
            issue = "Kanban is already forcing skills for some assigned work."
            link = "/kanban"
        elif skill_count == 0:
            state = "idle"
            issue = "No local skills detected."
            link = "/skills"
        else:
            state = "ok" if blocked == 0 else "warning"
            issue = "Skill coverage has no current warning signal." if blocked == 0 else "Blocked work may need a clearer skill or acceptance checklist."
            link = "/skills" if blocked == 0 else "/kanban"
        profile_rows.append({
            "id": profile.get("id"),
            "label": profile.get("label"),
            "state": state,
            "skill_count": skill_count,
            "open_work": open_work,
            "forced_skill_tasks": len(profile_forced),
            "blocked_work": blocked,
            "top_issue": issue,
            "recommended_view": link,
        })

    suggestions: List[Dict[str, Any]] = []
    if looping:
        suggestions.append(_tuning_item(
            "skill",
            "warning",
            "Add a loop-stop checklist skill",
            "One or more sessions crossed the runaway tool threshold. A short checklist or stop condition skill can prevent silent loops.",
            ", ".join(f"{s.get('label')}: {s.get('tool_call_count')} tools" for s in looping[:3]),
            "/skills",
            "Open Skills",
            "Olympus",
            "Hermes session tool_call_count is first-party trace evidence.",
            f">= {RUNAWAY_TOOLS_THRESHOLD} tool calls in one session",
        ))
    if tool_heavy:
        suggestions.append(_tuning_item(
            "skill",
            "info",
            "Turn repeated tool-heavy work into a reusable skill",
            "Moderate tool pressure often means the agent is rediscovering a procedure worth writing once.",
            ", ".join(f"{s.get('label')}: {s.get('tool_call_count')} tools" for s in tool_heavy[:3]),
            "/skills",
            "Open Skills",
            "Olympus",
            "Hermes records tool calls per session.",
            f"{TOOL_HEAVY_THRESHOLD}-{RUNAWAY_TOOLS_THRESHOLD - 1} tool calls in one session",
        ))
    if long_threads or context_pressure:
        suggestions.append(_tuning_item(
            "skill",
            "info",
            "Add recap and handoff discipline",
            "Long or high-context sessions need a recap habit, memory rules, or task splits before quality drifts.",
            ", ".join(str(s.get("label")) for s in (context_pressure or long_threads)[:3]),
            "/skills",
            "Open Skills",
            "Mnemosyne",
            "Hermes records message_count and average input tokens per API call.",
            f">= {LONG_THREAD_THRESHOLD} messages or >= {CONTEXT_PRESSURE_TOKENS:,} avg input tokens/call",
        ))
    if zero_skill_profiles:
        suggestions.append(_tuning_item(
            "skill",
            "info",
            "Review bare profiles",
            "Profiles with no local skills may still work, but their operating procedure is implicit and harder to tune.",
            ", ".join(str(p.get("label")) for p in zero_skill_profiles[:4]),
            "/skills",
            "Open Skills",
            "Apollo",
            "Hermes profiles expose local skill counts.",
            "profile skill_count = 0",
        ))
    if forced_skill_tasks:
        suggestions.append(_tuning_item(
            "skill",
            "ok",
            "Kanban is using explicit skills",
            "Some tasks already declare forced skills. Keep this pattern for work that needs repeatable procedure.",
            ", ".join(str(t.get("title") or t.get("id")) for t in forced_skill_tasks[:4]),
            "/kanban",
            "Open Kanban",
            "Olympus",
            "Kanban task metadata exposes forced skill counts.",
            "task forced_skill_count > 0",
        ))

    suggestions.sort(key=lambda x: _severity_rank(str(x.get("severity"))))
    if not suggestions:
        suggestions.append(_tuning_item(
            "skill",
            "ok",
            "No skill coverage gap found",
            "Olympus did not find bare active profiles, forced-skill pressure, looping, tool-heavy sessions, long threads, or high per-call context in this scan.",
            "Current skill coverage scan is clean.",
            "/skills",
            "Open Skills",
            "Olympus",
        ))

    summary_state = "warning" if looping or zero_skill_profiles else ("info" if tool_heavy or long_threads or context_pressure else "ok")
    return {
        "summary": {
            "state": summary_state,
            "profiles": len(profiles),
            "total_skills": sum(int(p.get("skill_count") or 0) for p in profiles),
            "zero_skill_profiles": len(zero_skill_profiles),
            "forced_skill_tasks": len(forced_skill_tasks),
            "forced_skill_total": forced_skill_total,
            "looping_sessions": len(looping),
            "tool_heavy_sessions": len(tool_heavy),
            "long_threads": len(long_threads),
            "context_pressure_sessions": len(context_pressure),
        },
        "profiles": profile_rows,
        "suggestions": suggestions[:5],
    }


def build_profile_fitness(profiles: List[Dict[str, Any]], gateways: List[Dict[str, Any]], cron: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    kanban = kanban or {}
    assignee_load = {
        str(a.get("_assignee") or a.get("assignee")): a
        for a in as_list(kanban.get("assignee_load"))
        if isinstance(a, dict) and (a.get("_assignee") or a.get("assignee"))
    }
    tasks = kanban_items(kanban, "recent_tasks")
    runs = kanban_items(kanban, "recent_runs")
    cron_by_profile = group_cron_by_profile(cron)
    gateways_by_profile = group_gateways_by_profile(gateways)

    rows: List[Dict[str, Any]] = []
    now = time.time()
    for profile in profiles:
        raw_name = str(profile.get("name") or "default")
        public_name = str(profile.get("_public_name") or raw_name)
        load = assignee_load.get(raw_name, {})
        profile_tasks = [t for t in tasks if str(t.get("_assignee") or t.get("assignee")) == raw_name]
        profile_runs = [r for r in runs if str(r.get("profile") or "") == public_name]
        failed_runs = [r for r in profile_runs if is_failed_run(r)]
        stale_tasks = [t for t in profile_tasks if task_stale_reason(t, now)]
        profile_cron = cron_by_profile.get(raw_name, [])
        profile_gateways = gateways_by_profile.get(public_name, [])
        open_work = int(load.get("open") or 0)
        running_work = int(load.get("running") or 0)
        ready_work = int(load.get("ready") or 0)
        blocked_work = int(load.get("blocked") or 0)
        skill_count = int(profile.get("skill_count") or 0)
        score = 100
        reasons: List[Dict[str, Any]] = []

        def deduct(points: int, label: str, detail: str, view: str) -> None:
            nonlocal score
            score -= points
            reasons.append({
                "label": label,
                "points": points,
                "detail": detail,
                "recommended_view": view,
            })

        if not profile.get("model"):
            deduct(12, "Route metadata unset", "Explicit profile routes make behavior changes auditable.", "/profiles")
        if skill_count == 0:
            deduct(18, "No local skills", "A bare profile has no reusable operating procedure to tune.", "/skills")
        if profile.get("gateway_state") != "running" and (ready_work or running_work or profile_gateways):
            deduct(16, "Gateway not running", "Assigned or gateway-backed work may not dispatch reliably.", "/profiles")
        if open_work >= OVERLOADED_OPEN_THRESHOLD:
            deduct(14, "Open work concentrated", f"{open_work} open Kanban task(s) are assigned here.", "/kanban")
        if running_work >= OVERLOADED_RUNNING_THRESHOLD:
            deduct(16, "Concurrent worker pressure", f"{running_work} running task(s) are assigned here.", "/kanban")
        if blocked_work:
            deduct(min(18, blocked_work * 6), "Blocked work", f"{blocked_work} blocked task(s) need review.", "/kanban")
        if failed_runs:
            deduct(min(24, len(failed_runs) * 12), "Worker failures", f"{len(failed_runs)} recent failed run(s) for this profile.", "/kanban")
        if stale_tasks:
            deduct(min(20, len(stale_tasks) * 10), "Possible stale worker", "Running work has heartbeat or claim data that needs review.", "/kanban")
        if profile_cron and skill_count < 3:
            deduct(8, "Scheduled work has thin skill coverage", "Cron-backed agents benefit from explicit runbooks.", "/skills")

        score = max(0, score)
        state = score_state(score)
        if reasons:
            top = reasons[0]
            top_issue = top.get("label")
            recommended_view = top.get("recommended_view")
        else:
            top_issue = "No profile fitness issue found."
            recommended_view = "/profiles"
        rows.append({
            "id": profile.get("id") or f"profile:{public_name}",
            "label": profile.get("label") or public_name,
            "state": state,
            "score": score,
            "top_issue": top_issue,
            "recommended_view": recommended_view,
            "reasons": reasons[:4],
            "metrics": {
                "skills": skill_count,
                "cron": len(profile_cron),
                "gateways": len(profile_gateways),
                "open": open_work,
                "ready": ready_work,
                "running": running_work,
                "blocked": blocked_work,
                "failed_runs": len(failed_runs),
            },
        })

    rows.sort(key=lambda row: (int(row.get("score") or 0), str(row.get("label") or "")))
    needs_review = [row for row in rows if row.get("state") in ("critical", "warning")]
    return {
        "summary": {
            "profiles": len(rows),
            "needs_review": len(needs_review),
            "average_score": int(sum(int(r.get("score") or 0) for r in rows) / len(rows)) if rows else 0,
            "lowest_score": int(rows[0].get("score") or 0) if rows else 0,
        },
        "profiles": rows,
    }


def build_orchestration(kanban: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    kanban = kanban or {}
    totals = kanban.get("totals") if isinstance(kanban.get("totals"), dict) else {}
    tasks = kanban_items(kanban, "recent_tasks")
    runs = kanban_items(kanban, "recent_runs")
    workers = kanban_items(kanban, "active_workers")
    now = time.time()

    failed_runs = [r for r in runs if is_failed_run(r)]
    stale_workers: List[Dict[str, Any]] = []
    for t in tasks:
        reason = task_stale_reason(t, now)
        if reason:
            stale_workers.append({
                "task_id": t.get("id"),
                "title": t.get("title"),
                "assignee": t.get("assignee"),
                "detail": reason,
            })

    return {
        "summary": {
            "boards": len(as_list(kanban.get("boards"))),
            "open": int(kanban.get("open") or 0),
            "ready": int(totals.get("ready") or 0),
            "running": int(totals.get("running") or 0),
            "blocked": int(totals.get("blocked") or 0),
            "review": int(totals.get("review") or 0),
            "active_workers": len(workers),
            "failed_runs": len(failed_runs),
            "stale_workers": len(stale_workers),
            "goal_mode_tasks": sum(1 for t in tasks if t.get("goal_mode")),
            "forced_skill_tasks": sum(1 for t in tasks if int(t.get("forced_skill_count") or 0) > 0),
            "model_override_tasks": sum(1 for t in tasks if t.get("model_override")),
            "ready_unassigned": sum(1 for t in tasks if t.get("status") == "ready" and (not t.get("assignee") or t.get("assignee") == "unassigned")),
        },
        "workers": workers[:12],
        "pressure": as_list(kanban.get("assignee_load"))[:12],
        "failed_runs": failed_runs[:8],
        "stale_workers": stale_workers[:8],
        "attention": as_list(kanban.get("attention"))[:12],
    }


def build_party(profiles: List[Dict[str, Any]], gateways: List[Dict[str, Any]], cron: List[Dict[str, Any]], sessions: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]], orchestration: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    kanban = kanban or {}
    orchestration = orchestration or {}
    assignee_load = {
        str(a.get("_assignee") or a.get("assignee")): a
        for a in as_list(kanban.get("assignee_load"))
        if isinstance(a, dict) and (a.get("_assignee") or a.get("assignee"))
    }
    tasks = kanban_items(kanban, "recent_tasks")
    runs = kanban_items(kanban, "recent_runs")
    cron_by_profile = group_cron_by_profile(cron)
    gateways_by_profile = group_gateways_by_profile(gateways)

    members: List[Dict[str, Any]] = []
    for idx, profile in enumerate(profiles):
        raw_name = str(profile.get("name") or "default")
        public_name = str(profile.get("_public_name") or raw_name)
        load = assignee_load.get(raw_name, {})
        profile_tasks = [t for t in tasks if str(t.get("_assignee") or t.get("assignee")) == raw_name]
        active_tasks = [t for t in profile_tasks if t.get("status") == "running"]
        queued_tasks = [t for t in profile_tasks if t.get("status") in ("triage", "todo", "scheduled", "ready", "review")]
        blocked_tasks = [t for t in profile_tasks if t.get("status") == "blocked"]
        profile_runs = [r for r in runs if str(r.get("profile") or "") == public_name]
        failed_runs = [r for r in profile_runs if is_failed_run(r)]
        profile_cron = cron_by_profile.get(raw_name, [])
        profile_gateways = gateways_by_profile.get(public_name, [])
        flags: List[str] = []
        if active_tasks:
            flags.append("working")
        if queued_tasks:
            flags.append("queued")
        if blocked_tasks:
            flags.append("blocked")
        if failed_runs:
            flags.append("worker failures")
        if profile.get("gateway_state") != "running" and (active_tasks or queued_tasks):
            flags.append("gateway route")
        if not profile.get("model"):
            flags.append("route metadata")
        if int(profile.get("skill_count") or 0) == 0:
            flags.append("skill coverage")
        if profile_cron:
            flags.append("scheduled")

        if failed_runs or blocked_tasks:
            state = "warning"
        elif active_tasks:
            state = "running"
        elif queued_tasks:
            state = "active"
        elif profile.get("gateway_state") == "running":
            state = "ready"
        else:
            state = profile.get("state") or "idle"

        current_task = active_tasks[0] if active_tasks else (queued_tasks[0] if queued_tasks else None)
        last_event_at = None
        for candidate in [current_task, *(profile_runs[:3]), *(profile_cron[:2])]:
            if not isinstance(candidate, dict):
                continue
            for key in ("last_heartbeat_at", "started_at", "ended_at", "last_run_at", "created_at"):
                if candidate.get(key):
                    last_event_at = candidate.get(key)
                    break
            if last_event_at:
                break

        members.append({
            "id": profile.get("id") or f"profile:{public_name}",
            "label": profile.get("label") or public_name,
            "state": state,
            "trust_boundary": profile.get("trust_boundary"),
            "model": profile.get("model"),
            "provider": profile.get("provider"),
            "skill_count": int(profile.get("skill_count") or 0),
            "gateway_state": profile.get("gateway_state"),
            "gateway_count": len(profile_gateways),
            "cron_jobs": len(profile_cron),
            "open_work": int(load.get("open") or 0),
            "ready_work": int(load.get("ready") or 0),
            "running_work": int(load.get("running") or 0),
            "blocked_work": int(load.get("blocked") or 0),
            "active_tasks": active_tasks[:4],
            "queued_tasks": queued_tasks[:4],
            "failed_runs": failed_runs[:4],
            "current_task": current_task,
            "last_event_at": last_event_at,
            "flags": flags or ["stable"],
            "position": {
                "x": 22 + (idx % 3) * 28,
                "y": 34 + (idx // 3) * 24,
            },
        })

    summary = {
        "members": len(members),
        "working": sum(1 for m in members if m.get("state") == "running"),
        "queued": sum(1 for m in members if int(m.get("ready_work") or 0) > 0 or int(m.get("open_work") or 0) > 0),
        "blocked": sum(int(m.get("blocked_work") or 0) for m in members),
        "workers": int((orchestration.get("summary") or {}).get("active_workers") or 0),
        "warnings": sum(1 for m in members if m.get("state") == "warning"),
    }
    return {
        "summary": summary,
        "members": members,
    }


def _event(kind: str, label: str, detail: str, state: str, at: Any, source: str, profile: Optional[str] = None, link: Optional[str] = None) -> Dict[str, Any]:
    return {
        "kind": kind,
        "label": label,
        "detail": detail,
        "state": state,
        "at": at,
        "source": source,
        "profile": profile,
        "link": link,
    }


def build_activity_events(sessions: List[Dict[str, Any]], cron: List[Dict[str, Any]], gateways: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for item in as_list((kanban or {}).get("recent_events")):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "kanban")
        profile = item.get("assignee")
        state = "warning" if kind in ("blocked", "reclaimed", "timed_out", "crashed", "gave_up", "spawn_failed") else ("running" if kind in ("claimed", "heartbeat") else "active")
        detail = " / ".join(str(x) for x in [item.get("task_title"), item.get("task_status"), item.get("board")] if x)
        events.append(_event(
            f"kanban:{kind}",
            kind.replace("_", " ").title(),
            detail,
            state,
            item.get("created_at"),
            "Kanban",
            str(profile) if profile else None,
            "/kanban",
        ))
    for session in sessions[:12]:
        if not isinstance(session, dict):
            continue
        events.append(_event(
            "session",
            f"Session {session.get('state') or 'updated'}",
            " / ".join(str(x) for x in [session.get("label"), f"{session.get('tool_call_count') or 0} tools", f"{session.get('message_count') or 0} msgs"] if x),
            str(session.get("state") or "unknown"),
            session.get("ended_at") or session.get("started_at"),
            "Sessions",
            None,
            "/sessions",
        ))
    for job in cron[:8]:
        if not isinstance(job, dict) or not job.get("last_run_at"):
            continue
        events.append(_event(
            "cron",
            "Cron fired",
            " / ".join(str(x) for x in [job.get("label"), job.get("schedule")] if x),
            str(job.get("state") or "scheduled"),
            job.get("last_run_at"),
            "Cron",
            str(job.get("profile") or "default"),
            "/cron",
        ))
    for gateway in gateways:
        if not isinstance(gateway, dict):
            continue
        events.append(_event(
            "gateway",
            f"{gateway.get('label') or 'Gateway'} {gateway.get('state') or 'unknown'}",
            str(gateway.get("platform") or ""),
            str(gateway.get("state") or "unknown"),
            None,
            "Gateway",
            str(gateway.get("profile") or ""),
            "/logs",
        ))
    events.sort(key=lambda x: str(x.get("at") or ""), reverse=True)
    return events[:24]


def build_trace_spine(sessions: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    kanban = kanban or {}
    session_by_ref = {
        str(session.get("session_ref") or session.get("id")): session
        for session in sessions
        if isinstance(session, dict) and (session.get("session_ref") or session.get("id"))
    }
    tasks = kanban_items(kanban, "recent_tasks")
    runs_by_task: Dict[str, List[Dict[str, Any]]] = {}
    events_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for run in kanban_items(kanban, "recent_runs"):
        task_ref = str(run.get("task_id") or "")
        if task_ref:
            runs_by_task.setdefault(task_ref, []).append(run)
    for event in kanban_items(kanban, "recent_events"):
        task_ref = str(event.get("task_id") or "")
        if task_ref:
            events_by_task.setdefault(task_ref, []).append(event)

    def trace_signal(task: Dict[str, Any], session: Optional[Dict[str, Any]], runs: List[Dict[str, Any]], events: List[Dict[str, Any]]) -> Dict[str, Any]:
        signals: List[str] = []
        failed_runs = [run for run in runs if is_failed_run(run) or str(run.get("status") or "").lower() in {"failed", "failure"}]
        failed_events = [
            event for event in events
            if str(event.get("kind") or "").lower() in {"failed", "failure", "crashed", "timed_out", "spawn_failed", "gave_up"}
        ]
        if task.get("status") == "blocked":
            signals.append("blocked")
        if task.get("status") == "ready" and str(task.get("assignee") or "") == "unassigned":
            signals.append("unassigned")
        if int(task.get("consecutive_failures") or 0) > 0:
            signals.append("retry")
        if task_stale_reason(task):
            signals.append("stale")
        if failed_runs:
            signals.append("failed_run")
        if failed_events:
            signals.append("failed_event")
        if session and session.get("handoff_error"):
            signals.append("handoff_failure")
        tool_count = int((session or {}).get("tool_call_count") or 0)
        if tool_count >= RUNAWAY_TOOLS_THRESHOLD:
            signals.append("tool_pressure")

        if "blocked" in signals:
            return {
                "severity": "warning",
                "label": "Clarify blocked task",
                "detail": "Blocked work needs a decision, dependency, or acceptance criteria before more worker runs add noise.",
                "signals": signals,
                "recommended_view": "/kanban",
            }
        if "failed_run" in signals or "failed_event" in signals or "retry" in signals:
            return {
                "severity": "warning",
                "label": "Review failed task run",
                "detail": "Task run or event history shows failure pressure. Inspect the Kanban item before rerouting.",
                "signals": signals,
                "recommended_view": "/kanban",
            }
        if "stale" in signals:
            return {
                "severity": "warning",
                "label": "Recover stale running work",
                "detail": "The running task has stale worker evidence. Reclaim, close, or resume it in Kanban.",
                "signals": signals,
                "recommended_view": "/kanban",
            }
        if "unassigned" in signals:
            return {
                "severity": "warning",
                "label": "Assign ready work",
                "detail": "Ready work without an assignee cannot be claimed by a profile-specific worker.",
                "signals": signals,
                "recommended_view": "/kanban",
            }
        if "handoff_failure" in signals:
            return {
                "severity": "warning",
                "label": "Review session handoff",
                "detail": "The linked session reports handoff trouble. Check delivery or cancellation behavior.",
                "signals": signals,
                "recommended_view": "/sessions",
            }
        if "tool_pressure" in signals:
            return {
                "severity": "info",
                "label": "Reduce trace tool pressure",
                "detail": "The linked session crossed the runaway tool threshold. Add a checklist, split the task, or tighten the route.",
                "signals": signals,
                "recommended_view": "/sessions",
            }
        return {
            "severity": "ok",
            "label": "Trace linked",
            "detail": "Task, run, event, and session evidence are linked without an urgent threshold.",
            "signals": signals,
            "recommended_view": "/kanban",
        }

    items: List[Dict[str, Any]] = []
    failure_points = 0
    for task in tasks[:24]:
        task_ref = str(task.get("id") or "")
        if not task_ref:
            continue
        session_ref = task.get("session_ref")
        session = session_by_ref.get(str(session_ref or "")) if session_ref else None
        runs = runs_by_task.get(task_ref, [])
        events = events_by_task.get(task_ref, [])
        signal = trace_signal(task, session, runs, events)
        failure_points += sum(1 for name in signal["signals"] if name in {"blocked", "retry", "stale", "failed_run", "failed_event", "handoff_failure", "unassigned"})
        item = {
            "task_ref": task_ref,
            "board": task.get("board"),
            "title": task.get("title") or task_ref,
            "status": task.get("status") or "unknown",
            "assignee": task.get("assignee") or "unassigned",
            "session_ref": session_ref,
            "session_state": session.get("state") if session else None,
            "session_tools": int(session.get("tool_call_count") or 0) if session else 0,
            "session_messages": int(session.get("message_count") or 0) if session else 0,
            "run_refs": [run.get("id") for run in runs[:4] if run.get("id")],
            "event_refs": [event.get("id") for event in events[:6] if event.get("id")],
            "run_count": len(runs),
            "event_count": len(events),
            "signals": signal["signals"],
            "severity": signal["severity"],
            "recommendation": signal["label"],
            "detail": signal["detail"],
            "recommended_view": signal["recommended_view"],
            "action_label": _action_label_for_view(str(signal["recommended_view"])),
            "basis": "Hermes sessions, Kanban tasks, task_runs, and task_events",
        }
        items.append(item)

    items.sort(key=lambda item: (_severity_rank(str(item.get("severity"))), -len(as_list(item.get("signals"))), str(item.get("task_ref"))))
    correlated = sum(1 for item in items if item.get("session_ref") or item.get("run_count") or item.get("event_count"))
    state = "warning" if any(item.get("severity") == "warning" for item in items) else ("active" if items else "unknown")
    return {
        "summary": {
            "state": state,
            "tasks": len(tasks),
            "correlated_tasks": correlated,
            "sessions": len(session_by_ref),
            "runs": sum(len(value) for value in runs_by_task.values()),
            "events": sum(len(value) for value in events_by_task.values()),
            "failure_points": failure_points,
        },
        "items": items[:8],
    }


def build_ops_evals(
    sessions: List[Dict[str, Any]],
    kanban: Optional[Dict[str, Any]],
    skill_coverage: Dict[str, Any],
    skill_hygiene: Dict[str, Any],
    config_policy: Dict[str, Any],
) -> Dict[str, Any]:
    kanban = kanban or {}
    kanban_totals = kanban.get("totals") if isinstance(kanban.get("totals"), dict) else {}
    coverage_summary = skill_coverage.get("summary") if isinstance(skill_coverage.get("summary"), dict) else {}
    hygiene_summary = skill_hygiene.get("summary") if isinstance(skill_hygiene.get("summary"), dict) else {}
    policy_summary = config_policy.get("summary") if isinstance(config_policy.get("summary"), dict) else {}
    has_evidence = bool(
        sessions or
        kanban.get("open") or
        kanban.get("totals") or
        kanban_items(kanban, "recent_runs") or
        kanban_items(kanban, "recent_tasks") or
        coverage_summary or
        hygiene_summary or
        policy_summary
    )
    if not has_evidence:
        return {
            "summary": {"state": "unknown", "score": 0, "checks": 0, "passed": 0, "warnings": 0, "failures": 0},
            "items": [],
        }

    failed_runs = [item for item in kanban_items(kanban, "recent_runs") if is_failed_run(item)]
    stale_sessions = [item for item in sessions if item.get("state") == "stale"]
    errored_sessions = [item for item in sessions if item.get("state") == "error"]
    blocked_tasks = int(kanban_totals.get("blocked") or 0)
    ready_unassigned = [
        item for item in kanban_items(kanban, "recent_tasks")
        if item.get("status") == "ready" and (not item.get("assignee") or item.get("assignee") == "unassigned")
    ]
    overloaded_assignees = [
        item for item in as_list(kanban.get("assignee_load"))
        if isinstance(item, dict) and (
            int(item.get("open") or 0) >= OVERLOADED_OPEN_THRESHOLD or
            int(item.get("running") or 0) >= OVERLOADED_RUNNING_THRESHOLD or
            int(item.get("blocked") or 0) >= 2
        )
    ]
    looping_sessions = [item for item in sessions if int(item.get("tool_call_count") or 0) >= RUNAWAY_TOOLS_THRESHOLD]
    tool_heavy = [
        item for item in sessions
        if TOOL_HEAVY_THRESHOLD <= int(item.get("tool_call_count") or 0) < RUNAWAY_TOOLS_THRESHOLD
    ]
    context_pressure = [item for item in sessions if int(item.get("avg_input_tokens") or 0) >= CONTEXT_PRESSURE_TOKENS]
    long_threads = [item for item in sessions if int(item.get("message_count") or 0) >= LONG_THREAD_THRESHOLD]
    high_turn_limit = int(policy_summary.get("max_turns") or 0) >= MAX_TURNS_REVIEW_THRESHOLD and not policy_summary.get("hard_loop_stop")

    def item_state(critical: int, warning: int) -> str:
        if critical:
            return "critical"
        if warning:
            return "warning"
        return "ok"

    reliability_state = item_state(len(failed_runs), len(stale_sessions) + len(errored_sessions))
    routing_state = item_state(0, len(overloaded_assignees) + len(ready_unassigned) + blocked_tasks)
    skill_state = item_state(
        int(hygiene_summary.get("hub_audit_fail") or 0),
        int(coverage_summary.get("zero_skill_profiles") or 0) +
        int(coverage_summary.get("forced_skill_tasks") or 0) +
        int(hygiene_summary.get("forced_skill_metadata_gaps") or 0) +
        int(hygiene_summary.get("hub_audit_warn") or 0),
    )
    efficiency_state = item_state(len(looping_sessions), len(tool_heavy) + len(context_pressure) + len(long_threads) + int(high_turn_limit))

    reliability_view = "/kanban" if failed_runs else "/sessions"
    items = [
        {
            "id": "reliability",
            "label": "Reliability",
            "state": reliability_state,
            "score": 100 - min(60, len(failed_runs) * 30 + len(stale_sessions) * 10 + len(errored_sessions) * 12),
            "detail": "Worker and session health from recent Hermes evidence.",
            "evidence": f"{len(failed_runs)} failed run(s), {len(stale_sessions)} stale session(s), {len(errored_sessions)} errored session(s)",
            "basis": "Hermes task_runs and session state",
            "signals": [
                f"failed runs: {len(failed_runs)}",
                f"stale sessions: {len(stale_sessions)}",
                f"errored sessions: {len(errored_sessions)}",
            ],
            "recommended_view": reliability_view,
            "action_label": _action_label_for_view(reliability_view),
        },
        {
            "id": "routing",
            "label": "Routing",
            "state": routing_state,
            "score": 100 - min(55, len(overloaded_assignees) * 20 + len(ready_unassigned) * 15 + blocked_tasks * 5),
            "detail": "Kanban ownership, blocked work, and unassigned ready work.",
            "evidence": f"{len(overloaded_assignees)} loaded assignee(s), {len(ready_unassigned)} unassigned ready task(s), {blocked_tasks} blocked task(s)",
            "basis": "Hermes Kanban assignee load and task status",
            "signals": [
                f"loaded assignees: {len(overloaded_assignees)}",
                f"unassigned ready: {len(ready_unassigned)}",
                f"blocked tasks: {blocked_tasks}",
            ],
            "recommended_view": "/kanban",
            "action_label": _action_label_for_view("/kanban"),
        },
        {
            "id": "skill_use",
            "label": "Skill Use",
            "state": skill_state,
            "score": 100 - min(
                55,
                int(coverage_summary.get("zero_skill_profiles") or 0) * 18 +
                int(hygiene_summary.get("forced_skill_metadata_gaps") or 0) * 16 +
                int(hygiene_summary.get("hub_audit_fail") or 0) * 25 +
                int(hygiene_summary.get("hub_audit_warn") or 0) * 12 +
                int(coverage_summary.get("forced_skill_tasks") or 0) * 3,
            ),
            "detail": "Coverage, forced-skill metadata, and stored audit results.",
            "evidence": (
                f"{int(coverage_summary.get('zero_skill_profiles') or 0)} bare profile(s), "
                f"{int(hygiene_summary.get('forced_skill_metadata_gaps') or 0)} forced-skill gap(s), "
                f"{int(hygiene_summary.get('hub_audit_warn') or 0)} audit warning(s), "
                f"{int(hygiene_summary.get('hub_audit_fail') or 0)} audit fail(s)"
            ),
            "basis": "Hermes skill coverage, skill usage metadata, and hub lock metadata",
            "signals": [
                f"bare profiles: {int(coverage_summary.get('zero_skill_profiles') or 0)}",
                f"forced-skill tasks: {int(coverage_summary.get('forced_skill_tasks') or 0)}",
                f"metadata gaps: {int(hygiene_summary.get('forced_skill_metadata_gaps') or 0)}",
            ],
            "recommended_view": "/skills",
            "action_label": _action_label_for_view("/skills"),
        },
        {
            "id": "efficiency",
            "label": "Efficiency",
            "state": efficiency_state,
            "score": 100 - min(60, len(looping_sessions) * 25 + len(tool_heavy) * 10 + len(context_pressure) * 14 + len(long_threads) * 6 + int(high_turn_limit) * 10),
            "detail": "Tool pressure, context pressure, long threads, and visible loop guardrails.",
            "evidence": (
                f"{len(looping_sessions)} looping session(s), {len(tool_heavy)} tool-heavy session(s), "
                f"{len(context_pressure)} context-pressure session(s), {len(long_threads)} long thread(s)"
            ),
            "basis": "Hermes session counters plus safe config policy",
            "signals": [
                f"looping: {len(looping_sessions)}",
                f"tool-heavy: {len(tool_heavy)}",
                f"context pressure: {len(context_pressure)}",
                f"loop stop visible: {'no' if high_turn_limit else 'yes'}",
            ],
            "recommended_view": "/sessions",
            "action_label": _action_label_for_view("/sessions"),
        },
    ]

    warnings = sum(1 for item in items if item["state"] == "warning")
    failures = sum(1 for item in items if item["state"] == "critical")
    score = max(0, int(sum(int(item["score"]) for item in items) / len(items))) if items else 100
    state = "critical" if failures else ("warning" if warnings else "ok")
    return {
        "summary": {
            "state": state,
            "score": score,
            "checks": len(items),
            "passed": sum(1 for item in items if item["state"] == "ok"),
            "warnings": warnings,
            "failures": failures,
        },
        "items": items,
    }


def build_metrics_spine(
    profiles: List[Dict[str, Any]],
    sessions: List[Dict[str, Any]],
    kanban: Optional[Dict[str, Any]],
    skill_metadata: Dict[str, Any],
    skill_coverage: Dict[str, Any],
    skill_hygiene: Dict[str, Any],
    profile_fitness: Dict[str, Any],
    performance: Dict[str, Any],
    ops_evals: Dict[str, Any],
    config_policy: Dict[str, Any],
    usage_rollup: Optional[Dict[str, Any]] = None,
    orchestration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Hermes-grounded operational metrics for tuning, not a usage ledger."""
    kanban = kanban or {}
    # ``stale_workers`` is produced by build_orchestration().summary, never by
    # the collect_kanban dict — read it from orchestration so the work metric
    # isn't stuck at 0. Fall back to deriving it if a caller didn't pass one.
    orchestration = orchestration or build_orchestration(kanban)
    orch_summary = orchestration.get("summary") if isinstance(orchestration.get("summary"), dict) else {}
    rollup = usage_rollup or collect_usage_rollup(profiles)
    skill_summary = skill_metadata.get("summary") if isinstance(skill_metadata.get("summary"), dict) else {}
    coverage_summary = skill_coverage.get("summary") if isinstance(skill_coverage.get("summary"), dict) else {}
    hygiene_summary = skill_hygiene.get("summary") if isinstance(skill_hygiene.get("summary"), dict) else {}
    fitness_summary = profile_fitness.get("summary") if isinstance(profile_fitness.get("summary"), dict) else {}
    performance_summary = performance.get("summary") if isinstance(performance.get("summary"), dict) else {}
    ops_summary = ops_evals.get("summary") if isinstance(ops_evals.get("summary"), dict) else {}
    policy_summary = config_policy.get("summary") if isinstance(config_policy.get("summary"), dict) else {}
    kanban_totals = kanban.get("totals") if isinstance(kanban.get("totals"), dict) else {}
    active_profiles = sum(1 for profile in profiles if profile.get("state") == "active")
    idle_profiles = sum(1 for profile in profiles if profile.get("state") == "idle")
    profiles_with_skills = sum(1 for profile in profiles if int(profile.get("skill_count") or 0) > 0)
    profile_rows = as_list(skill_metadata.get("profile_usage"))[:8]
    usage_state = rollup.get("state") or "unknown"
    cost_confidence = rollup.get("cost_confidence") or "unknown"

    signals: List[Dict[str, Any]] = []
    def add_signal(kind: str, severity: str, title: str, detail: str, evidence: str, view: str) -> None:
        signals.append({
            "kind": kind,
            "severity": severity,
            "title": title,
            "detail": detail,
            "evidence": evidence,
            "recommended_view": view,
            "action_label": _action_label_for_view(view),
        })

    if int(rollup.get("zero_usage_suspect_sessions") or 0):
        add_signal(
            "usage",
            "warning",
            "Token accounting has gaps",
            "Some Hermes sessions have work evidence but zero recorded token usage. Treat usage and cost totals as lower-bound evidence.",
            f"{rollup.get('zero_usage_suspect_sessions')} zero-usage suspect session(s)",
            "/analytics",
        )
    if cost_confidence in {"partial", "missing"}:
        add_signal(
            "cost",
            "warning",
            "Cost visibility is incomplete",
            "Hermes recorded tokens without reliable cost on some routes. Olympus should use cost as a risk signal, not a spend ledger.",
            f"cost confidence: {cost_confidence}",
            "/analytics",
        )
    if int(hygiene_summary.get("never_used") or 0):
        add_signal(
            "skill",
            "info",
            "Unused skills need review",
            "Skills with no use evidence may be stale, over-specific, or created before the workflow repeated.",
            f"{hygiene_summary.get('never_used')} never-used skill record(s)",
            "/skills",
        )
    if int(performance_summary.get("avg_tools_per_session") or 0) >= TOOL_HEAVY_THRESHOLD:
        add_signal(
            "efficiency",
            "warning",
            "Tool pressure is high",
            "High tool calls per session usually means missing procedure, weak routing, or too-broad tasks.",
            f"avg {performance_summary.get('avg_tools_per_session')} tools/session",
            "/sessions",
        )
    if int(kanban_totals.get("blocked") or 0) or int(orch_summary.get("stale_workers") or 0):
        add_signal(
            "work",
            "warning",
            "Work reliability needs review",
            "Blocked work or stale workers reduce agent usability even when usage volume looks healthy.",
            f"{kanban_totals.get('blocked') or 0} blocked / {orch_summary.get('stale_workers') or 0} stale worker(s)",
            "/kanban",
        )

    return {
        "schema_version": "olympus.metrics_spine.v1",
        "generated_at": now_iso(),
        "window": {
            "days": int(rollup.get("window_days") or 30),
            "session_sample_limit": len(sessions),
            "ledger_owner": "/analytics",
        },
        "ownership": {
            "olympus": "operational metrics, risk, readiness, usability signals, and handoff evidence",
            "hermes_analytics": "usage ledger, daily token bars, top models, top skills, and raw cost totals",
        },
        "coverage": {
            "state_store": usage_state,
            "state_stores_seen": rollup.get("stores_seen") or 0,
            "skill_usage": "warning" if skill_summary.get("read_failures") else ("ok" if skill_summary.get("usage_present") else "missing"),
            "skill_usage_sources": skill_summary.get("usage_sources") or 0,
            "kanban": "warning" if any(isinstance(board, dict) and board.get("error") for board in as_list(kanban.get("boards"))) else ("ok" if kanban.get("boards") else "missing"),
            "cost_visibility": cost_confidence,
            "config_grounding": {
                "toolsets": policy_summary.get("toolsets") or 0,
                "max_turns": policy_summary.get("max_turns") or 0,
                "auxiliary_routes": policy_summary.get("auxiliary_routes") or 0,
            },
        },
        "usage": {
            "sessions": rollup.get("sessions") or 0,
            "api_calls": rollup.get("api_calls") or 0,
            "total_tokens": rollup.get("total_tokens") or 0,
            "tool_calls": rollup.get("tool_calls") or 0,
            "costed_sessions": rollup.get("costed_sessions") or 0,
            "estimated_cost_usd": rollup.get("estimated_cost_usd") or 0,
            "actual_cost_usd": rollup.get("actual_cost_usd") or 0,
            "cost_confidence": cost_confidence,
            "zero_usage_suspect_sessions": rollup.get("zero_usage_suspect_sessions") or 0,
            "zero_cost_token_sessions": rollup.get("zero_cost_token_sessions") or 0,
            "recommended_view": "/analytics",
            "note": "Operational evidence only. Hermes Analytics owns raw usage and cost ledgers.",
        },
        "agents": {
            "profiles": len(profiles),
            "active_profiles": active_profiles,
            "idle_profiles": idle_profiles,
            "profiles_with_skills": profiles_with_skills,
            "needs_review": fitness_summary.get("needs_review") or 0,
            "average_score": fitness_summary.get("average_score") or 0,
            "cohorts": strip_internal(rollup.get("profiles") or []),
            "recommended_view": "/profiles",
        },
        "skills": {
            "recorded": skill_summary.get("total_skills") or 0,
            "created_30d": skill_summary.get("created_30d") or 0,
            "agent_created": skill_summary.get("agent_created") or 0,
            "used": skill_summary.get("used") or 0,
            "never_used": hygiene_summary.get("never_used") or skill_summary.get("never_used") or 0,
            "stale": hygiene_summary.get("stale") or skill_summary.get("stale") or 0,
            "archived": hygiene_summary.get("archived") or skill_summary.get("archived") or 0,
            "recently_patched": hygiene_summary.get("recently_patched") or skill_summary.get("recently_patched") or 0,
            "forced_skill_tasks": coverage_summary.get("forced_skill_tasks") or 0,
            "metadata_gaps": hygiene_summary.get("forced_skill_metadata_gaps") or 0,
            "by_profile": strip_internal(profile_rows),
            "recommended_view": "/skills",
        },
        "work": {
            "open": kanban.get("open") or 0,
            "ready": kanban_totals.get("ready") or 0,
            "running": kanban_totals.get("running") or 0,
            "blocked": kanban_totals.get("blocked") or 0,
            "failed_runs": performance.get("metrics", {}).get("failed_kanban_runs") if isinstance(performance.get("metrics"), dict) else 0,
            "stale_workers": orch_summary.get("stale_workers") or 0,
            "recommended_view": "/kanban",
        },
        "evals": {
            "state": ops_summary.get("state") or "unknown",
            "score": ops_summary.get("score") or 0,
            "reliability": next((item.get("state") for item in as_list(ops_evals.get("items")) if isinstance(item, dict) and item.get("id") == "reliability"), "unknown"),
            "routing": next((item.get("state") for item in as_list(ops_evals.get("items")) if isinstance(item, dict) and item.get("id") == "routing"), "unknown"),
            "skill_use": next((item.get("state") for item in as_list(ops_evals.get("items")) if isinstance(item, dict) and item.get("id") == "skill_use"), "unknown"),
            "efficiency": next((item.get("state") for item in as_list(ops_evals.get("items")) if isinstance(item, dict) and item.get("id") == "efficiency"), "unknown"),
            "recommended_view": "/sessions",
        },
        "signals": signals[:8],
    }


def build_tuning(profiles: List[Dict[str, Any]], gateways: List[Dict[str, Any]], cron: List[Dict[str, Any]], sessions: List[Dict[str, Any]], health: Dict[str, Any], attention: List[Dict[str, Any]], kanban: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    failed_cron = [j for j in cron if j.get("state") == "error"]
    paused_cron = [j for j in cron if j.get("state") == "paused"]
    stale_sessions = [s for s in sessions if s.get("state") == "stale"]
    errored_sessions = [s for s in sessions if s.get("state") == "error"]
    active_sessions = [s for s in sessions if s.get("state") in ("active", "recent")]
    looping_sessions = [s for s in sessions if int(s.get("tool_call_count") or 0) >= RUNAWAY_TOOLS_THRESHOLD]
    tool_heavy = [
        s for s in sessions
        if TOOL_HEAVY_THRESHOLD <= int(s.get("tool_call_count") or 0) < RUNAWAY_TOOLS_THRESHOLD
    ]
    message_heavy = [s for s in sessions if int(s.get("message_count") or 0) >= LONG_THREAD_THRESHOLD]
    context_pressure = [s for s in sessions if int(s.get("avg_input_tokens") or 0) >= CONTEXT_PRESSURE_TOKENS]
    missing_models = [p for p in profiles if not p.get("model")]
    idle_profiles = [p for p in profiles if p.get("state") == "idle"]
    running_gateways = [g for g in gateways if g.get("state") == "running"]
    kanban = kanban or {}
    kanban_totals = kanban.get("totals") if isinstance(kanban.get("totals"), dict) else {}
    kanban_attention = [x for x in as_list(kanban.get("attention")) if isinstance(x, dict)]
    kanban_open = int(kanban.get("open") or 0)
    kanban_blocked = int(kanban_totals.get("blocked") or 0)
    active_workers = as_list(kanban.get("active_workers"))
    overloaded_assignees = [
        a for a in as_list(kanban.get("assignee_load"))
        if isinstance(a, dict) and (int(a.get("open") or 0) >= 5 or int(a.get("blocked") or 0) >= 2 or int(a.get("running") or 0) >= 2)
    ]

    metrics = build_metrics(sessions, kanban)
    recommendations: List[Dict[str, Any]] = []
    if failed_cron:
        recommendations.append(_recommendation(
            "critical",
            "Tune failing scheduled work",
            f"{len(failed_cron)} cron job(s) report errors. Review cadence, prompt, profile, approvals, and delivery target.",
            ", ".join(str(j.get("label") or j.get("job_id")) for j in failed_cron[:3]),
            "/cron",
            "Apollo",
            "Open Cron",
            "Cron failures are direct runtime evidence. Fix scheduled work first.",
        ))
    if errored_sessions:
        recommendations.append(_recommendation(
            "warning",
            "Review handoff failures",
            f"{len(errored_sessions)} recent session(s) contain handoff errors. Check delivery and cancellation behavior.",
            ", ".join(str(s.get("label") or s.get("session_ref")) for s in errored_sessions[:3]),
            "/sessions",
            "Hermes",
            "Open Sessions",
            "Session handoff_error comes from Hermes session metadata and indicates delivery or cancellation trouble.",
        ))
    if stale_sessions:
        recommendations.append(_recommendation(
            "warning",
            "Inspect stale work",
            f"{len(stale_sessions)} recent session(s) are stale. Close, resume, or annotate them.",
            ", ".join(str(s.get("label") or s.get("session_ref")) for s in stale_sessions[:3]),
            "/sessions",
            "Chronos",
            "Open Sessions",
            "Stale sessions are open-ended records with no recent activity; they make agent performance review noisy.",
        ))
    if looping_sessions:
        recommendations.append(_recommendation(
            "warning",
            "Stop looping or tool-thrashing runs",
            f"{len(looping_sessions)} session(s) crossed {RUNAWAY_TOOLS_THRESHOLD} tool calls. Add a checklist skill, reduce max turns, or split the task.",
            ", ".join(f"{s.get('label')}: {s.get('tool_call_count')} tools" for s in looping_sessions[:3]),
            "/sessions",
            "Olympus",
            "Open Sessions",
            "Runaway tool_call_count is a trace-level signal for loops, unclear plans, or missing procedures.",
        ))
    if tool_heavy:
        recommendations.append(_recommendation(
            "info",
            "Tune tool-heavy routes",
            f"{len(tool_heavy)} session(s) used 20 or more tool calls. Add skills, narrower prompts, or decomposition.",
            ", ".join(f"{s.get('label')}: {s.get('tool_call_count')} tools" for s in tool_heavy[:3]),
            "/analytics",
            "Olympus",
            "Open Analytics",
            "High tool_call_count is a trace-level signal for workflow friction, missing skills, or poor decomposition.",
        ))
    if context_pressure:
        recommendations.append(_recommendation(
            "info",
            "Reduce per-call context pressure",
            f"{len(context_pressure)} session(s) averaged {CONTEXT_PRESSURE_TOKENS:,}+ input tokens per API call. Add recap, memory pruning, or task splits.",
            ", ".join(f"{s.get('label')}: {int(s.get('avg_input_tokens') or 0):,} avg in-tokens/call" for s in context_pressure[:3]),
            "/sessions",
            "Mnemosyne",
            "Open Sessions",
            "Large repeated prompts increase latency, cost, and drift risk.",
        ))
    if message_heavy:
        recommendations.append(_recommendation(
            "info",
            "Watch long conversations",
            f"{len(message_heavy)} session(s) crossed 40 messages. Add recap, memory, or task splits.",
            ", ".join(f"{s.get('label')}: {s.get('message_count')} messages" for s in message_heavy[:3]),
            "/sessions",
            "Mnemosyne",
            "Open Sessions",
            "High message_count is a context-pressure signal; summaries, memory, or task splitting can improve follow-up quality.",
        ))
    if missing_models:
        recommendations.append(_recommendation(
            "warning",
            "Set explicit profile routes",
            f"{len(missing_models)} profile(s) do not expose a default route in config.",
            ", ".join(str(p.get("label") or p.get("name")) for p in missing_models[:4]),
            "/profiles",
            "Olympus",
            "Open Profiles",
            "Explicit route metadata makes changes auditable and keeps profile behavior intentional.",
        ))
    if not running_gateways and gateways:
        recommendations.append(_recommendation(
            "warning",
            "Check gateway process state",
            "Gateway markers exist, but no running gateway was detected.",
            ", ".join(str(g.get("label")) for g in gateways[:4]),
            "/logs",
            "Olympus",
            "Open Logs",
            "Gateway process state determines whether platform-delivered work can be routed.",
        ))
    if health.get("recent_error_terms"):
        recommendations.append(_recommendation(
            "critical",
            "Investigate log-tail errors",
            "Gateway or error log tails contain failure terms.",
            ", ".join(str(x) for x in health.get("recent_error_terms", [])[:5]),
            "/logs",
            "Olympus",
            "Open Logs",
            "Olympus scans the last 8KB of each Hermes log file; timestamp recency is not inferred.",
        ))
    if kanban_attention:
        recommendations.append(_recommendation(
            str(kanban_attention[0].get("severity") or "warning"),
            "Review Kanban attention",
            f"{len(kanban_attention)} Kanban issue(s) need review.",
            ", ".join(str(x.get("label") or "Kanban issue") for x in kanban_attention[:3]),
            "/kanban",
            "Olympus",
            "Open Kanban",
            "Kanban attention is derived from task status, heartbeats, retries, assignees, and task_runs.",
        ))
    if overloaded_assignees:
        recommendations.append(_recommendation(
            "warning",
            "Balance agent workload",
            f"{len(overloaded_assignees)} assignee(s) show Kanban pressure. Split work, clarify blockers, or reroute specialist tasks.",
            ", ".join(f"{a.get('assignee')}: {a.get('open')} open" for a in overloaded_assignees[:4]),
            "/kanban",
            "Hermes",
            "Open Kanban",
            "Assignee load comes from Kanban task ownership and is the right signal for route balancing.",
        ))
    if not recommendations:
        recommendations.append(_recommendation(
            "ok",
            "No urgent tuning pressure",
            "Olympus did not detect failing cron jobs, stale sessions, handoff errors, or missing profile routes.",
            "Current metadata scan is clean.",
            "/analytics",
            "Olympus",
            "Open Analytics",
            "No current heuristic crossed its threshold.",
        ))

    score = 100
    deductions: List[Dict[str, Any]] = []

    def deduct(label: str, amount: int, reason: str, evidence: str, source: str) -> None:
        nonlocal score
        if amount <= 0:
            return
        score -= amount
        deductions.append({
            "label": label,
            "points": amount,
            "reason": reason,
            "evidence": evidence,
            "source": source,
        })

    deduct(
        "Runtime health",
        25 if health.get("status") == "error" else 10 if health.get("status") == "warning" else 0,
        "Gateway, cron, or log evidence needs operator review.",
        health.get("summary") or "",
        "Hermes logs, cron metadata, and gateway process state",
    )
    deduct("Cron failures", min(25, len(failed_cron) * 8), "Scheduled agent work is failing.", f"{len(failed_cron)} failed job(s)", "Hermes cron jobs")
    deduct("Handoff failures", min(20, len(errored_sessions) * 6), "Recent sessions contain handoff errors.", f"{len(errored_sessions)} session(s)", "Hermes sessions")
    deduct("Stale sessions", min(15, len(stale_sessions) * 3), "Open-ended stale sessions make performance review noisy.", f"{len(stale_sessions)} stale session(s)", "Hermes sessions")
    deduct("Looping/tool thrash", min(20, len(looping_sessions) * 3), "Runaway tool use wastes time and tokens.", f"{len(looping_sessions)} session(s)", "Hermes sessions tool_call_count")
    deduct("Context pressure", min(12, len(context_pressure) * 6), "Large repeated context increases latency, cost, and drift risk.", f"{len(context_pressure)} session(s)", "Hermes sessions input_tokens/api_call_count")
    deduct("Tool-heavy work", min(10, len(tool_heavy) * 2), "Repeated tool work may need a skill, narrower prompt, or task split.", f"{len(tool_heavy)} session(s)", "Hermes sessions tool_call_count")
    deduct("Long threads", min(8, len(message_heavy)), "Long conversations need recap, memory, or task split discipline.", f"{len(message_heavy)} session(s)", "Hermes sessions message_count")
    deduct("Unset profile routes", min(15, len(missing_models) * 4), "Implicit routing makes tuning harder to audit.", f"{len(missing_models)} profile(s)", "Hermes profile config")
    deduct("Kanban attention", min(20, len(kanban_attention) * 4), "Kanban has blocked, stale, failed, or unassigned work.", f"{len(kanban_attention)} attention item(s)", "Hermes Kanban tasks/task_runs")
    deduct("Blocked Kanban work", min(12, kanban_blocked * 3), "Blocked work usually needs a human decision, dependency, or clearer acceptance criteria.", f"{kanban_blocked} blocked task(s)", "Hermes Kanban tasks")
    deduct("Worker failures", min(20, int(metrics.get("failed_kanban_runs") or 0) * 6), "Kanban worker failures show reliability risk.", f"{metrics.get('failed_kanban_runs') or 0} failed run(s)", "Hermes Kanban task_runs")
    if health.get("status") == "error":
        score = min(score, 54)
    elif health.get("status") == "warning":
        score = min(score, 84)
    score = max(0, score)

    if score >= 85:
        score_label = "Stable"
    elif score >= 70:
        score_label = "Watch"
    elif score >= 55:
        score_label = "Needs review"
    else:
        score_label = "Needs action"

    signals = [
        {"id": "attention", "label": "Attention Items", "value": len(attention), "state": "warning" if attention else "ok", "hint": "Cross-system findings ranked before inventory."},
        {"id": "kanban_open", "label": "Open Kanban Work", "value": kanban_open, "state": "active" if kanban_open else "idle", "hint": "Open task load across Kanban boards."},
        {"id": "kanban_blocked", "label": "Blocked Tasks", "value": kanban_blocked, "state": "warning" if kanban_blocked else "ok", "hint": "Kanban cards needing decisions or clearer specs."},
        {"id": "kanban_workers", "label": "Active Workers", "value": len(active_workers), "state": "running" if active_workers else "idle", "hint": "Kanban task runs currently in flight."},
        {"id": "active_sessions", "label": "Active/Recent Work", "value": len(active_sessions), "state": "active" if active_sessions else "idle", "hint": "Sessions that still look fresh."},
        {"id": "failed_cron", "label": "Cron Failures", "value": len(failed_cron), "state": "error" if failed_cron else "ok", "hint": "Scheduled work that needs review."},
        {"id": "tool_heavy", "label": "Tool-Heavy Runs", "value": len(tool_heavy), "state": "warning" if tool_heavy else "ok", "hint": "Sessions with 20+ tool calls."},
        {"id": "looping", "label": "Looping Runs", "value": len(looping_sessions), "state": "warning" if looping_sessions else "ok", "hint": "Sessions with 40+ tool calls."},
        {"id": "context_pressure", "label": "Context Pressure", "value": len(context_pressure), "state": "warning" if context_pressure else "ok", "hint": "Sessions averaging 50k+ input tokens per API call."},
        {"id": "message_heavy", "label": "Long Threads", "value": len(message_heavy), "state": "warning" if message_heavy else "ok", "hint": "Sessions with 40+ messages."},
        {"id": "missing_routes", "label": "Unset Routes", "value": len(missing_models), "state": "warning" if missing_models else "ok", "hint": "Profiles without explicit route metadata."},
        {"id": "paused_cron", "label": "Paused Cron", "value": len(paused_cron), "state": "idle" if paused_cron else "ok", "hint": "Paused scheduled jobs."},
        {"id": "idle_profiles", "label": "Idle Profiles", "value": len(idle_profiles), "state": "idle" if idle_profiles else "active", "hint": "Profiles without visible gateway activity."},
    ]

    agent_hq = build_agent_hq(profiles, gateways, cron, sessions, kanban)
    recommendations.sort(key=lambda x: _severity_rank(str(x.get("severity"))))
    return {
        "score": score,
        "score_breakdown": {
            "base": 100,
            "score": score,
            "label": score_label,
            "deductions": deductions,
            "explanation": "A transparent heuristic readiness score. It prioritizes current operational risks, not absolute agent intelligence.",
        },
        "methodology": {
            "thresholds": [
                {"signal": "Looping session", "threshold": f">= {RUNAWAY_TOOLS_THRESHOLD} tool calls", "why": "Runaway tool use is a direct signal for loops, unclear plans, or missing procedures."},
                {"signal": "Tool-heavy session", "threshold": f"{TOOL_HEAVY_THRESHOLD}-{RUNAWAY_TOOLS_THRESHOLD - 1} tool calls", "why": "Repeated tool use can indicate missing skills, unclear prompts, or poor decomposition."},
                {"signal": "Context pressure", "threshold": f">= {CONTEXT_PRESSURE_TOKENS:,} average input tokens per API call", "why": "Large repeated prompts increase latency, cost, and drift risk."},
                {"signal": "Long thread", "threshold": f">= {LONG_THREAD_THRESHOLD} messages", "why": "Long conversations can create context pressure and benefit from summaries, memory rules, or task splits."},
                {"signal": "Overloaded assignee", "threshold": f">= {OVERLOADED_OPEN_THRESHOLD} open or >= {OVERLOADED_RUNNING_THRESHOLD} running Kanban tasks", "why": "Load concentration suggests route balancing or a specialist profile may help."},
                {"signal": "Stale session", "threshold": "open session older than the freshness window", "why": "Unresolved stale work makes performance evidence unreliable."},
            ],
            "sources": [
                {"label": "Hermes session store", "detail": "Uses local runtime metadata, message_count, tool_call_count, handoff_error, and timestamps from the Hermes state store."},
                {"label": "Hermes Kanban", "detail": "Uses first-party task status, assignee load, worker heartbeats, retries, and task_runs."},
                {"label": "Hermes profiles and skills", "detail": "Uses profile runtime-route metadata, gateway state, and local SKILL.md counts."},
                {"label": "Provider-neutral agent evaluation", "detail": "Uses traces, tool calls, outcomes, and repeatable signals before changing agents or routes."},
            ],
        },
        "summary": "Read-only tuning scan generated from Hermes profile, gateway, cron, session, and log metadata.",
        "recommendations": recommendations[:8],
        "signals": signals,
        "agent_hq": agent_hq,
        "metrics": agent_hq.get("metrics", {}),
    }


@router.get("/health")
async def health() -> Dict[str, Any]:
    profiles = collect_profiles()
    cron = collect_cron(profiles)
    h = collect_health(profiles, cron)
    return {"ok": h.get("status") != "error", "generated_at": now_iso(), **h}


@router.get("/overview")
async def overview() -> Dict[str, Any]:
    started_at = time.perf_counter()
    profiles = collect_profiles()
    cron = collect_cron(profiles)
    sessions = collect_sessions(60)
    gateways = collect_gateways(profiles)
    kanban = collect_kanban(profiles)
    orchestration = build_orchestration(kanban)
    party = build_party(profiles, gateways, cron, sessions, kanban, orchestration)
    activity_events = build_activity_events(sessions, cron, gateways, kanban)
    skill_coverage = build_skill_coverage(profiles, sessions, kanban)
    skill_metadata = collect_skill_metadata(profiles)
    skill_hygiene = build_skill_hygiene(skill_metadata, skill_coverage, kanban)
    profile_fitness = build_profile_fitness(profiles, gateways, cron, kanban)
    performance = build_performance_tracking(sessions, kanban)
    trace_spine = build_trace_spine(sessions, kanban)
    evidence_sources = build_evidence_sources(profiles, sessions, cron, gateways, kanban)
    config_policy = collect_config_policy(profiles, sessions, kanban)
    ops_evals = build_ops_evals(sessions, kanban, skill_coverage, skill_hygiene, config_policy)
    usage_rollup = collect_usage_rollup(profiles)
    metrics_spine = build_metrics_spine(
        profiles,
        sessions,
        kanban,
        skill_metadata,
        skill_coverage,
        skill_hygiene,
        profile_fitness,
        performance,
        ops_evals,
        config_policy,
        usage_rollup,
        orchestration=orchestration,
    )
    health = collect_health(profiles, cron)
    attention = build_attention(profiles, gateways, cron, sessions, health, kanban)
    tuning = build_tuning(profiles, gateways, cron, sessions, health, attention, kanban)
    payload = {
        "generated_at": now_iso(),
        "health": health,
        "attention": attention,
        "tuning": strip_internal(tuning),
        "profiles": [public_profile(p) for p in profiles],
        "gateways": gateways,
        "cron": strip_internal(cron),
        "sessions": sessions,
        "kanban": strip_internal(kanban),
        "orchestration": strip_internal(orchestration),
        "party": strip_internal(party),
        "activity_events": strip_internal(activity_events),
        "skill_coverage": strip_internal(skill_coverage),
        "skill_hygiene": strip_internal(skill_hygiene),
        "profile_fitness": strip_internal(profile_fitness),
        "performance": strip_internal(performance),
        "trace_spine": strip_internal(trace_spine),
        "ops_evals": strip_internal(ops_evals),
        "metrics_spine": strip_internal(metrics_spine),
        "evidence_sources": strip_internal(evidence_sources),
        "config_policy": strip_internal(config_policy),
    }
    payload["diagnostics"] = runtime_diagnostics("/overview", started_at, payload, profiles, sessions, kanban)
    return payload


@router.get("/tuning")
async def tuning() -> Dict[str, Any]:
    started_at = time.perf_counter()
    profiles = collect_profiles()
    cron = collect_cron(profiles)
    sessions = collect_sessions(60)
    gateways = collect_gateways(profiles)
    kanban = collect_kanban(profiles)
    orchestration = build_orchestration(kanban)
    party = build_party(profiles, gateways, cron, sessions, kanban, orchestration)
    activity_events = build_activity_events(sessions, cron, gateways, kanban)
    skill_coverage = build_skill_coverage(profiles, sessions, kanban)
    skill_metadata = collect_skill_metadata(profiles)
    skill_hygiene = build_skill_hygiene(skill_metadata, skill_coverage, kanban)
    profile_fitness = build_profile_fitness(profiles, gateways, cron, kanban)
    performance = build_performance_tracking(sessions, kanban)
    trace_spine = build_trace_spine(sessions, kanban)
    evidence_sources = build_evidence_sources(profiles, sessions, cron, gateways, kanban)
    config_policy = collect_config_policy(profiles, sessions, kanban)
    ops_evals = build_ops_evals(sessions, kanban, skill_coverage, skill_hygiene, config_policy)
    usage_rollup = collect_usage_rollup(profiles)
    metrics_spine = build_metrics_spine(
        profiles,
        sessions,
        kanban,
        skill_metadata,
        skill_coverage,
        skill_hygiene,
        profile_fitness,
        performance,
        ops_evals,
        config_policy,
        usage_rollup,
        orchestration=orchestration,
    )
    health = collect_health(profiles, cron)
    attention = build_attention(profiles, gateways, cron, sessions, health, kanban)
    payload = {
        "generated_at": now_iso(),
        "health": health,
        "attention": attention,
        "kanban": strip_internal(kanban),
        "orchestration": strip_internal(orchestration),
        "party": strip_internal(party),
        "activity_events": strip_internal(activity_events),
        "skill_coverage": strip_internal(skill_coverage),
        "skill_hygiene": strip_internal(skill_hygiene),
        "profile_fitness": strip_internal(profile_fitness),
        "performance": strip_internal(performance),
        "trace_spine": strip_internal(trace_spine),
        "ops_evals": strip_internal(ops_evals),
        "metrics_spine": strip_internal(metrics_spine),
        "evidence_sources": strip_internal(evidence_sources),
        "config_policy": strip_internal(config_policy),
        "tuning": strip_internal(build_tuning(profiles, gateways, cron, sessions, health, attention, kanban)),
    }
    payload["diagnostics"] = runtime_diagnostics("/tuning", started_at, payload, profiles, sessions, kanban)
    return payload
