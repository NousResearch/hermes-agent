#!/usr/bin/env python3
"""Read-only Quinn operational awareness MCP server.

This module intentionally exposes eyes, not hands: structured local status only,
no writes, no platform history, no config mutation, and no service control.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import yaml
except Exception:  # pragma: no cover - dependency may be absent on minimal hosts
    yaml = None  # type: ignore[assignment]

ROOT = Path("/home/quinn/.hermes/hermes-agent")
HERMES_HOME = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
CONFIG_PATH = HERMES_HOME / "config.yaml"
AUTH_PATH = HERMES_HOME / "auth.json"
SESSIONS_DIR = HERMES_HOME / "sessions"
SNAPSHOT_SCHEMA_VERSION = 1
SNAPSHOT_PATH = HERMES_HOME / "mcp" / "quinn_ops_state" / "overview_snapshot.json"
LOG_PATHS = [
    HERMES_HOME / "logs" / "gateway.log",
    HERMES_HOME / "logs" / "agent.log",
    HERMES_HOME / "logs" / "errors.log",
]
RUNTIME_DIR = Path("/home/quinn/quinn/runtime")
KNOWN_RUNTIME_FILES = [
    RUNTIME_DIR / "quinn_loader_order.json",
    Path("/home/quinn/quinn/docs/QUINN_V1_V35_INSTALL_REPORT.md"),
    Path("/home/quinn/docs/quinn-hermes-server.md"),
]
DEFAULT_TIMEOUT = 10
MAX_LOG_LINES = 200

SENSITIVE_EXACT_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "client_secret",
    "cookie",
    "env",
    "headers",
    "password",
    "refresh_token",
    "secret",
    "token",
}
SENSITIVE_SUFFIXES = ("_api_key", "_token", "_password", "_secret", "_authorization", "_cookie")
KNOWN_PLATFORMS = ("discord", "telegram", "slack", "matrix", "whatsapp", "signal", "email", "sms", "teams", "feishu", "dingtalk", "wecom", "irc", "line")
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"ghp_[A-Za-z0-9_]{8,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{8,}"),
    re.compile(r"(?i)(Authorization\s*:\s*Bearer\s+)[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(DISCORD(?:_BOT)?_TOKEN\s*=\s*)[^\s]+"),
    re.compile(r"(?i)(TELEGRAM(?:_BOT)?_TOKEN\s*=\s*)[^\s]+"),
    re.compile(r"\b\d{6,}:[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{20,}\b"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def response(data: dict[str, Any] | None = None, errors: list[dict[str, Any]] | None = None, warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "ok": not bool(errors),
        "data": sanitize(data or {}),
        "errors": sanitize(errors or []),
        "warnings": sanitize(warnings or []),
    }


def empty_change_groups() -> dict[str, list[dict[str, Any]]]:
    return {
        "gateway": [],
        "platforms": [],
        "mcp": [],
        "cron": [],
        "sessions": [],
        "repo": [],
        "recent_errors": [],
        "runtime_files": [],
        "toolsets": [],
        "version": [],
    }


def redact_string(value: str) -> str:
    redacted = value
    for pattern in SECRET_PATTERNS:
        def repl(match: re.Match[str]) -> str:
            if match.lastindex:
                return f"{match.group(1)}[REDACTED]"
            return "[REDACTED]"

        redacted = pattern.sub(repl, redacted)
    return redacted


def sanitize(value: Any, key: str | None = None) -> Any:
    if key and is_sensitive_key(key):
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, (int, float)):
            return "[REDACTED]"
        if isinstance(value, str):
            return {"present": bool(value), "redacted": True}
        if isinstance(value, (dict, list)):
            return {"present": True, "redacted": True}
    if isinstance(value, str):
        return redact_string(value)
    if isinstance(value, dict):
        return {str(k): sanitize(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v, key) for v in value]
    return value


def is_sensitive_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in SENSITIVE_EXACT_KEYS or any(normalized.endswith(suffix) for suffix in SENSITIVE_SUFFIXES)


def command_error(source: str, message: str, kind: str = "command_error") -> dict[str, str]:
    return {"source": source, "message": redact_string(message), "kind": kind}


def run_cmd(argv: list[str], source: str, timeout: int = DEFAULT_TIMEOUT, cwd: Path | None = None) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd or ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
            check=False,
        )
        return {
            "argv": [Path(argv[0]).name, *argv[1:]],
            "returncode": completed.returncode,
            "stdout": redact_string((completed.stdout or "").rstrip("\n")),
            "stderr": redact_string((completed.stderr or "").rstrip("\n")),
        }, None
    except subprocess.TimeoutExpired:
        return None, command_error(source, "command timed out", "timeout")
    except FileNotFoundError:
        return None, command_error(source, f"command not found: {Path(argv[0]).name}", "not_found")
    except Exception as exc:
        return None, command_error(source, type(exc).__name__, "exception")


def hermes_cmd(*args: str) -> list[str]:
    hermes = shutil.which("hermes")
    if hermes:
        return [hermes, *args]
    local = ROOT / "hermes"
    return [str(local), *args] if local.exists() else ["hermes", *args]


def parse_yaml_config() -> tuple[dict[str, Any], list[str]]:
    if not CONFIG_PATH.exists():
        return {}, ["config.yaml not found"]
    if yaml is None:
        return {}, ["PyYAML unavailable; config summary limited"]
    try:
        raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        return raw if isinstance(raw, dict) else {}, []
    except Exception as exc:
        return {}, [f"config parse failed: {type(exc).__name__}"]


def file_meta(path: Path) -> dict[str, Any]:
    try:
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
    except FileNotFoundError:
        return {"path": str(path), "exists": False}
    except Exception as exc:
        return {"path": str(path), "exists": False, "error": type(exc).__name__}


def summarize_platforms(config: dict[str, Any]) -> dict[str, Any]:
    gateway = config.get("gateway") if isinstance(config.get("gateway"), dict) else {}
    platforms = config.get("platforms") if isinstance(config.get("platforms"), dict) else {}
    configured_sections = set()
    for container in (gateway, platforms):
        for platform in KNOWN_PLATFORMS:
            if isinstance(container.get(platform), dict):
                configured_sections.add(platform)
    return {
        "configured_sections": sorted(configured_sections),
        "status": "config_section_presence_only",
        "authoritative_source": "hermes status --all",
    }


def platform_delivery_probe(configured: Any, connected: Any, status: str) -> dict[str, Any]:
    """Passive, side-effect-free delivery capability summary.

    This intentionally does not read channel history or send probe messages. It
    only classifies the parsed local status output into delivery-capable,
    not-capable, or unknown.
    """
    if connected is True:
        delivery_capable = True
        delivery_status = "delivery_capable"
    elif configured is False or status == "not_configured":
        delivery_capable = False
        delivery_status = "not_configured"
    elif connected is False:
        delivery_capable = False
        delivery_status = "not_connected"
    elif configured is True:
        delivery_capable = None
        delivery_status = "configured_delivery_unknown"
    else:
        delivery_capable = None
        delivery_status = "unknown"
    return {
        "delivery_capable": delivery_capable,
        "delivery_status": delivery_status,
        "probe_method": "passive_status_parse",
        "history_read": False,
        "delivery_attempted": False,
    }


def parse_platform_status_text(text: str) -> dict[str, dict[str, Any]]:
    statuses = {
        platform: {
            "mentioned": False,
            "configured": None,
            "connected": None,
            "status": "unknown",
            **platform_delivery_probe(None, None, "unknown"),
        }
        for platform in KNOWN_PLATFORMS
    }
    lines = text.splitlines()
    in_messaging = False
    section_lines: list[str] = []
    for line in lines:
        lowered = line.lower()
        if "messaging platforms" in lowered:
            in_messaging = True
            continue
        if in_messaging and line.strip() and not line.startswith((" ", "\t", "-", "*")) and lowered.endswith(":"):
            break
        if in_messaging:
            section_lines.append(line)
    scan_lines = section_lines or lines
    for line in scan_lines:
        lowered = line.lower()
        for platform in KNOWN_PLATFORMS:
            if platform not in lowered:
                continue
            connection_unknown = any(phrase in lowered for phrase in ("connection unknown", "unknown connection", "status unknown", "unknown"))
            not_configured = any(phrase in lowered for phrase in ("not configured", "unconfigured", "missing", "disabled", "not set"))
            explicit_connected = any(word in lowered for word in ("connected", "running", "active", "online"))
            explicit_disconnected = "not connected" in lowered or "offline" in lowered
            if not_configured:
                configured = False
                connected = False
                status = "not_configured"
            elif explicit_disconnected:
                configured = True if "configured" in lowered else None
                connected = False
                status = "configured" if configured is True else "not_connected"
            elif explicit_connected:
                configured = True
                connected = True
                status = "connected"
            elif "configured" in lowered or "enabled" in lowered:
                configured = True
                connected = None
                status = "configured"
            elif connection_unknown:
                configured = None
                connected = None
                status = "mentioned"
            else:
                configured = None
                connected = None
                status = "mentioned"
            statuses[platform] = {
                "mentioned": True,
                "configured": configured,
                "connected": connected,
                "status": status,
                **platform_delivery_probe(configured, connected, status),
            }
    return statuses


def parse_count(text: str, active_words: tuple[str, ...] = ("active", "enabled", "running")) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    active = sum(1 for line in lines if any(word in line.lower() for word in active_words))
    return {"total": len(lines), "active_like": active, "lines_returned": min(len(lines), 20)}


def get_config_summary() -> dict[str, Any]:
    config, warnings = parse_yaml_config()
    mcp = config.get("mcp_servers") if isinstance(config.get("mcp_servers"), dict) else {}
    tools = config.get("tools") if isinstance(config.get("tools"), dict) else {}
    memory = config.get("memory") if isinstance(config.get("memory"), dict) else {}
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    data = {
        "config_exists": CONFIG_PATH.exists(),
        "auth_file": {"exists": AUTH_PATH.exists(), "metadata_only": True},
        "model": {
            "section_present": bool(model),
            "provider_present": bool(model.get("provider")) if isinstance(model, dict) else False,
            "default_present": bool(model.get("default") or model.get("model")) if isinstance(model, dict) else False,
        },
        "memory": {
            "section_present": bool(memory),
            "enabled": memory.get("enabled") if isinstance(memory.get("enabled"), bool) else None,
            "provider_present": bool(memory.get("provider")) if isinstance(memory, dict) else False,
        },
        "platforms": summarize_platforms(config),
        "mcp_servers": sorted(str(name) for name in mcp.keys()),
        "tool_sections": sorted(str(name) for name in tools.keys()),
        "privacy_security_sections_present": {
            "privacy": "privacy" in config,
            "security": "security" in config,
            "permissions": "permissions" in config,
            "approvals": "approvals" in config,
        },
    }
    return response(data, warnings=warnings)


def get_gateway_status() -> dict[str, Any]:
    data: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    cmd, err = run_cmd(["systemctl", "--user", "is-active", "hermes-gateway"], "systemctl is-active", timeout=5)
    if err:
        errors.append(err)
    else:
        data["systemd_active"] = cmd["stdout"]
    cmd, err = run_cmd(["systemctl", "--user", "show", "hermes-gateway", "--property=MainPID,ActiveState,SubState,LoadState,UnitFileState"], "systemctl show", timeout=5)
    if err:
        errors.append(err)
    else:
        props = {}
        for line in cmd["stdout"].splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                props[k] = v
        data["systemd_properties"] = {k: props.get(k) for k in ("MainPID", "ActiveState", "SubState", "LoadState", "UnitFileState")}
        data["pid"] = props.get("MainPID")
        data["manager"] = "systemd_user"
        data["booleans"] = {
            "active": props.get("ActiveState") == "active",
            "running": props.get("SubState") == "running",
            "loaded": props.get("LoadState") == "loaded",
            "enabled": props.get("UnitFileState") == "enabled",
        }
    return response(data, errors)


def get_platform_status() -> dict[str, Any]:
    cfg = get_config_summary()
    data = {"config_summary": cfg["data"].get("platforms", {})}
    errors: list[dict[str, str]] = []
    cmd, err = run_cmd(hermes_cmd("status", "--all"), "hermes status --all", timeout=10)
    if err:
        errors.append(err)
    else:
        data["platform_status"] = parse_platform_status_text(cmd["stdout"])
        data["source"] = "hermes status --all"
    return response(data, errors + cfg["errors"], cfg["warnings"])


def get_mcp_status() -> dict[str, Any]:
    cfg = get_config_summary()
    data = {"configured_servers": cfg["data"].get("mcp_servers", [])}
    errors: list[dict[str, str]] = []
    cmd, err = run_cmd(hermes_cmd("mcp", "list"), "hermes mcp list", timeout=10)
    if err:
        errors.append(err)
    else:
        data["mcp_list"] = {"returncode": cmd["returncode"], "summary": cmd["stdout"][:2000], **parse_count(cmd["stdout"])}
    return response(data, errors + cfg["errors"], cfg["warnings"])


def get_toolsets_status() -> dict[str, Any]:
    cmd, err = run_cmd(hermes_cmd("tools", "list"), "hermes tools list", timeout=10)
    if err:
        return response({"toolsets": []}, [err])
    lines = [line.strip() for line in cmd["stdout"].splitlines() if line.strip()]
    return response({"toolsets": lines[:100], "summary": parse_count(cmd["stdout"], ("enabled", "on", "active"))})


def get_cron_status() -> dict[str, Any]:
    cmd, err = run_cmd(hermes_cmd("cron", "list"), "hermes cron list", timeout=10)
    if err:
        return response({"active": None, "total": None}, [err])
    lines = [line.strip() for line in cmd["stdout"].splitlines() if line.strip()]
    if not lines or any("no scheduled jobs" in line.lower() for line in lines):
        return response({"active": 0, "total": 0, "jobs": []})
    active = sum(1 for line in lines if "active" in line.lower() or "enabled" in line.lower())
    safe_jobs = []
    for line in lines[:50]:
        parts = line.split()
        safe_jobs.append({"id_or_name": parts[0] if parts else "", "summary": redact_string(line[:160])})
    return response({"active": active, "total": len(lines), "jobs": safe_jobs})


def get_sessions_summary() -> dict[str, Any]:
    data: dict[str, Any] = {"sessions_dir_exists": SESSIONS_DIR.exists(), "count": 0, "metadata_only": True}
    if not SESSIONS_DIR.exists():
        return response(data)
    try:
        session_files = list(SESSIONS_DIR.glob("**/*"))
        json_files = [p for p in session_files if p.is_file() and p.suffix.lower() == ".json"]
        data["file_count"] = sum(1 for p in session_files if p.is_file())
        data["json_file_count"] = len(json_files)
        index = SESSIONS_DIR / "sessions.json"
        if index.exists():
            obj = json.loads(index.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                data["count"] = len(obj)
                data["recent_metadata"] = [
                    {
                        "platform": sanitize(v.get("platform") or (v.get("origin") or {}).get("platform")),
                        "chat_type": sanitize(v.get("chat_type") or (v.get("origin") or {}).get("chat_type")),
                        "updated_at": sanitize(v.get("updated_at")),
                    }
                    for v in list(obj.values())[:10]
                    if isinstance(v, dict)
                ]
        else:
            data["count"] = len(json_files)
        return response(data)
    except Exception as exc:
        return response(data, [command_error("sessions summary", type(exc).__name__, "exception")])


def categorize_log_line(line: str) -> str | None:
    lowered = line.lower()
    if "traceback" in lowered:
        return "traceback"
    if "exception" in lowered:
        return "exception"
    if "failed" in lowered or "failure" in lowered:
        return "failed"
    if "error" in lowered:
        return "error"
    if "warn" in lowered:
        return "warning"
    return None


def parse_line_timestamp(line: str) -> str | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}[T ][0-9:.+-]{8,}(?:Z)?", line)
    return match.group(0) if match else None


def safe_log_snippet(line: str) -> str | None:
    lowered = line.lower()
    if any(marker in lowered for marker in (" user:", "assistant:", "content=", "content:", "message=", "message:", "prompt=", "cmdline", "command line", "argv=")):
        return None
    return redact_string(line[-240:])


def get_recent_errors(limit: int = 50, include_snippets: bool = False) -> dict[str, Any]:
    limit = max(1, min(int(limit or 50), MAX_LOG_LINES))
    snippets_allowed = bool(include_snippets and os.environ.get("QUINN_OPS_ALLOW_LOG_SNIPPETS") == "1")
    grouped: dict[str, dict[str, Any]] = {}
    snippets = []
    errors = []
    for path in LOG_PATHS:
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-2000:]:
                category = categorize_log_line(line)
                if not category:
                    continue
                source = grouped.setdefault(path.name, {"total": 0, "categories": {}, "last_seen": None})
                source["total"] += 1
                source["categories"][category] = source["categories"].get(category, 0) + 1
                source["last_seen"] = parse_line_timestamp(line) or source["last_seen"]
                if snippets_allowed and len(snippets) < limit:
                    snippet = safe_log_snippet(line)
                    if snippet:
                        snippets.append({"source": path.name, "category": category, "snippet": snippet})
        except Exception as exc:
            errors.append(command_error(path.name, type(exc).__name__, "exception"))
    matched_total = sum(item["total"] for item in grouped.values())
    return response({"limit": limit, "grouped": grouped, "matched_total": matched_total, "snippets": snippets if snippets_allowed else []}, errors)


def get_repo_status() -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    data: dict[str, Any] = {"repo": str(ROOT)}
    for key, argv in {
        "status_short": ["git", "status", "--porcelain=v1"],
        "head": ["git", "rev-parse", "HEAD"],
        "describe": ["git", "describe", "--tags", "--always", "--dirty"],
        "branch": ["git", "branch", "--show-current"],
    }.items():
        cmd, err = run_cmd(argv, f"git {key}", timeout=10, cwd=ROOT)
        if err:
            errors.append(err)
            continue
        if key == "status_short":
            files = []
            for line in cmd["stdout"].splitlines():
                if not line:
                    continue
                path = line[3:] if len(line) > 3 else ""
                if " -> " in path:
                    path = path.split(" -> ", 1)[1]
                if path:
                    files.append(path)
            data[key] = {"dirty_count": len(files), "files": files[:100]}
        elif key == "head":
            data[key] = cmd["stdout"][:12]
        else:
            data[key] = cmd["stdout"]
    return response(data, errors)


def get_runtime_files_status() -> dict[str, Any]:
    return response({"files": [file_meta(path) for path in KNOWN_RUNTIME_FILES]})


def healthcheck() -> dict[str, Any]:
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
        mcp_sdk = True
    except Exception:
        mcp_sdk = False
    tools = sorted(TOOL_FUNCTIONS)
    return response({
        "server": "quinn_ops",
        "timestamp_utc": utc_now(),
        "read_only": True,
        "mcp_sdk_available": mcp_sdk,
        "commands_available": {
            "hermes": bool(shutil.which("hermes") or (ROOT / "hermes").exists()),
            "git": bool(shutil.which("git")),
            "systemctl": bool(shutil.which("systemctl")),
        },
        "tools": tools,
    }, warnings=[] if mcp_sdk else ["Python MCP SDK not installed; install it before running stdio server"])


def get_overview() -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []

    def collect(name: str, fn: Callable[..., dict[str, Any]], *args: Any) -> Any:
        result = fn(*args)
        errors.extend(result.get("errors", []))
        warnings.extend(result.get("warnings", []))
        return result.get("data", {})

    recent_error_data = collect("recent_errors", get_recent_errors, 50)
    data = {
        "timestamp_utc": utc_now(),
        "hermes_version": None,
        "gateway": collect("gateway", get_gateway_status),
        "platforms": collect("platforms", get_platform_status),
        "mcp": collect("mcp", get_mcp_status),
        "cron": collect("cron", get_cron_status),
        "sessions": collect("sessions", get_sessions_summary),
        "toolsets": collect("toolsets", get_toolsets_status),
        "repo": collect("repo", get_repo_status),
        "recent_errors": {"count": recent_error_data.get("matched_total", 0), "grouped": recent_error_data.get("grouped", {})},
        "runtime_files": collect("runtime_files", get_runtime_files_status),
    }
    cmd, err = run_cmd(hermes_cmd("--version"), "hermes --version", timeout=10)
    if err:
        errors.append(err)
    else:
        data["hermes_version"] = cmd["stdout"][:200]
    return response(data, errors, warnings)


def snapshot_path() -> Path:
    return SNAPSHOT_PATH


def build_overview_snapshot() -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    overview = get_overview()
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "created_by": "quinn_ops",
        "updated_at_utc": utc_now(),
        "overview": sanitize(overview.get("data", {})),
        "overview_errors": sanitize(overview.get("errors", [])),
        "overview_warnings": sanitize(overview.get("warnings", [])),
    }
    return sanitize(snapshot), overview.get("errors", []), overview.get("warnings", [])


def write_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    path = snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = json.dumps(sanitize(snapshot), indent=2, sort_keys=True)
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload + "\n")
    try:
        os.chmod(tmp_path, 0o600)
    except Exception:
        pass
    tmp_path.replace(path)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "schema_version": snapshot.get("schema_version"),
        "snapshot_timestamp_utc": snapshot.get("updated_at_utc"),
    }


def read_snapshot() -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = snapshot_path()
    if not path.exists():
        return None, None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None, command_error("overview snapshot", "snapshot file is not a JSON object", "invalid_snapshot")
        return sanitize(obj), None
    except Exception as exc:
        return None, command_error("overview snapshot", type(exc).__name__, "exception")


def snapshot_identity(snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not snapshot:
        return {}
    overview = snapshot.get("overview") if isinstance(snapshot.get("overview"), dict) else {}
    gateway = overview.get("gateway") if isinstance(overview.get("gateway"), dict) else {}
    repo = overview.get("repo") if isinstance(overview.get("repo"), dict) else {}
    return {
        "gateway_active": gateway.get("systemd_active"),
        "gateway_pid": gateway.get("pid"),
        "repo_head": repo.get("head"),
        "repo_branch": repo.get("branch"),
        "hermes_version": overview.get("hermes_version"),
    }


def get_snapshot_status() -> dict[str, Any]:
    path = snapshot_path()
    data: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "schema_version": None,
        "snapshot_timestamp_utc": None,
        "baseline_identity": {},
    }
    if not path.exists():
        return response(data)
    try:
        stat = path.stat()
        snapshot, err = read_snapshot()
        if err:
            return response(data | {
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            }, [err])
        data.update({
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "schema_version": snapshot.get("schema_version") if snapshot else None,
            "snapshot_timestamp_utc": snapshot.get("updated_at_utc") if snapshot else None,
            "baseline_identity": snapshot_identity(snapshot),
        })
        return response(data)
    except Exception as exc:
        return response(data, [command_error("snapshot status", type(exc).__name__, "exception")])


def save_overview_snapshot() -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    snapshot, overview_errors, overview_warnings = build_overview_snapshot()
    errors.extend(overview_errors)
    warnings.extend(overview_warnings)
    try:
        meta = write_snapshot(snapshot)
    except Exception as exc:
        errors.append(command_error("overview snapshot write", type(exc).__name__, "exception"))
        return response({"saved": False, "path": str(snapshot_path())}, errors, warnings)
    overview = snapshot.get("overview") if isinstance(snapshot.get("overview"), dict) else {}
    summary = {
        "gateway_active": (overview.get("gateway") or {}).get("systemd_active") if isinstance(overview.get("gateway"), dict) else None,
        "platforms_tracked": len(((overview.get("platforms") or {}).get("platform_status") or {})) if isinstance(overview.get("platforms"), dict) else 0,
        "recent_error_count": (overview.get("recent_errors") or {}).get("count") if isinstance(overview.get("recent_errors"), dict) else None,
    }
    return response({"saved": True, "snapshot": meta, "summary": summary}, errors, warnings)


def as_map(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def add_change(changes: dict[str, list[dict[str, Any]]], group: str, path: str, change_type: str, before: Any, after: Any, severity: str = "info") -> None:
    if before == after:
        return
    changes.setdefault(group, []).append({
        "path": path,
        "type": change_type,
        "before": sanitize(before),
        "after": sanitize(after),
        "severity": severity,
    })


def severity_rank(severity: str) -> int:
    return {"info": 0, "warning": 1, "critical": 2}.get(severity, 0)


def count_by_metadata(rows: list[Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get(key)
        if value is None:
            value = "unknown"
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def file_set_from_repo(repo: dict[str, Any]) -> set[str]:
    return {str(item) for item in as_list(as_map(repo.get("status_short")).get("files"))}


def runtime_file_map(runtime_files: dict[str, Any]) -> dict[str, dict[str, Any]]:
    files = {}
    for item in as_list(runtime_files.get("files")):
        if isinstance(item, dict) and item.get("path"):
            files[str(item["path"])] = item
    return files


def job_id_set(cron: dict[str, Any]) -> set[str]:
    ids = set()
    for item in as_list(cron.get("jobs")):
        if isinstance(item, dict):
            ids.add(str(item.get("id_or_name") or item.get("id") or item.get("name") or item))
        else:
            ids.add(str(item))
    return ids


def line_set(toolsets: dict[str, Any]) -> set[str]:
    return {str(line) for line in as_list(toolsets.get("toolsets"))}


def build_error_delta_summary(prev_errors: dict[str, Any], curr_errors: dict[str, Any]) -> dict[str, Any]:
    prev_count = prev_errors.get("count") if isinstance(prev_errors.get("count"), int) else 0
    curr_count = curr_errors.get("count") if isinstance(curr_errors.get("count"), int) else 0
    prev_grouped = as_map(prev_errors.get("grouped"))
    curr_grouped = as_map(curr_errors.get("grouped"))
    summary: dict[str, Any] = {
        "total_before": prev_count,
        "total_after": curr_count,
        "total_delta": curr_count - prev_count,
        "sources": {},
        "new_categories": [],
        "repeated_categories": [],
        "last_seen_moved": [],
    }
    for source in sorted(set(prev_grouped) | set(curr_grouped)):
        prev_source = as_map(prev_grouped.get(source))
        curr_source = as_map(curr_grouped.get(source))
        prev_total = prev_source.get("total") if isinstance(prev_source.get("total"), int) else 0
        curr_total = curr_source.get("total") if isinstance(curr_source.get("total"), int) else 0
        prev_last_seen = prev_source.get("last_seen")
        curr_last_seen = curr_source.get("last_seen")
        source_summary: dict[str, Any] = {
            "total_before": prev_total,
            "total_after": curr_total,
            "total_delta": curr_total - prev_total,
            "is_new_source": source not in prev_grouped and source in curr_grouped,
            "last_seen_before": prev_last_seen,
            "last_seen_after": curr_last_seen,
            "last_seen_moved": bool(prev_last_seen and curr_last_seen and prev_last_seen != curr_last_seen),
            "categories": {},
        }
        if source_summary["last_seen_moved"]:
            summary["last_seen_moved"].append({"source": source, "before": prev_last_seen, "after": curr_last_seen})
        prev_categories = as_map(prev_source.get("categories"))
        curr_categories = as_map(curr_source.get("categories"))
        for category in sorted(set(prev_categories) | set(curr_categories)):
            before = prev_categories.get(category, 0)
            after = curr_categories.get(category, 0)
            if not isinstance(before, int):
                before = 0
            if not isinstance(after, int):
                after = 0
            category_summary = {
                "before": before,
                "after": after,
                "delta": after - before,
                "is_new": before == 0 and after > 0,
                "repeated": before > 0 and after > 0,
            }
            source_summary["categories"][category] = category_summary
            if category_summary["is_new"]:
                summary["new_categories"].append({"source": source, "category": category, "after": after})
            if category_summary["repeated"]:
                summary["repeated_categories"].append({"source": source, "category": category, "before": before, "after": after, "delta": after - before})
        summary["sources"][source] = source_summary
    return sanitize(summary)


def compare_overviews(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    changes = empty_change_groups()

    prev_gateway = as_map(previous.get("gateway"))
    curr_gateway = as_map(current.get("gateway"))
    gateway_fields = [
        ("systemd_active", "gateway.systemd_active"),
        ("pid", "gateway.pid"),
    ]
    for key, path in gateway_fields:
        severity = "info"
        if key == "systemd_active" and curr_gateway.get(key) not in (None, "active"):
            severity = "critical"
        add_change(changes, "gateway", path, "changed", prev_gateway.get(key), curr_gateway.get(key), severity)
    prev_props = as_map(prev_gateway.get("systemd_properties"))
    curr_props = as_map(curr_gateway.get("systemd_properties"))
    for key in ("ActiveState", "SubState", "UnitFileState"):
        severity = "info"
        if key in {"ActiveState", "SubState"} and curr_props.get(key) not in (None, "active", "running"):
            severity = "critical"
        if key == "UnitFileState" and curr_props.get(key) in {"disabled", "masked"}:
            severity = "warning"
        add_change(changes, "gateway", f"gateway.systemd_properties.{key}", "changed", prev_props.get(key), curr_props.get(key), severity)

    prev_platforms = as_map(as_map(previous.get("platforms")).get("platform_status"))
    curr_platforms = as_map(as_map(current.get("platforms")).get("platform_status"))
    for platform in sorted(set(prev_platforms) | set(curr_platforms)):
        prev = as_map(prev_platforms.get(platform))
        curr = as_map(curr_platforms.get(platform))
        for key in ("configured", "connected", "status"):
            severity = "info"
            if key == "configured" and prev.get(key) is True and curr.get(key) in (False, None):
                severity = "warning"
            if key == "connected" and prev.get(key) is True and curr.get(key) is False:
                severity = "critical"
            if key == "connected" and prev.get(key) is True and curr.get(key) is None:
                severity = "warning"
            if key == "status" and prev.get(key) == "configured" and curr.get(key) in (None, "unknown", "mentioned"):
                severity = "warning"
            add_change(changes, "platforms", f"platforms.{platform}.{key}", "changed", prev.get(key), curr.get(key), severity)

    prev_mcp = as_map(previous.get("mcp"))
    curr_mcp = as_map(current.get("mcp"))
    prev_servers = set(str(x) for x in as_list(prev_mcp.get("configured_servers")))
    curr_servers = set(str(x) for x in as_list(curr_mcp.get("configured_servers")))
    if prev_servers != curr_servers:
        severity = "critical" if "quinn_ops" in prev_servers and "quinn_ops" not in curr_servers else "info"
        add_change(changes, "mcp", "mcp.configured_servers", "changed", sorted(prev_servers), sorted(curr_servers), severity)
    prev_mcp_list = as_map(prev_mcp.get("mcp_list"))
    curr_mcp_list = as_map(curr_mcp.get("mcp_list"))
    active_severity = "critical" if "quinn_ops" in curr_servers and (curr_mcp_list.get("active_like") or 0) <= 0 else "info"
    add_change(changes, "mcp", "mcp.mcp_list.active_like", "changed", prev_mcp_list.get("active_like"), curr_mcp_list.get("active_like"), active_severity)

    prev_cron = as_map(previous.get("cron"))
    curr_cron = as_map(current.get("cron"))
    for key in ("total", "active"):
        add_change(changes, "cron", f"cron.{key}", "changed", prev_cron.get(key), curr_cron.get(key), "info")
    prev_jobs = job_id_set(prev_cron)
    curr_jobs = job_id_set(curr_cron)
    if prev_jobs != curr_jobs:
        add_change(changes, "cron", "cron.jobs", "changed", sorted(prev_jobs), sorted(curr_jobs), "info")

    prev_sessions = as_map(previous.get("sessions"))
    curr_sessions = as_map(current.get("sessions"))
    for key in ("count", "file_count", "json_file_count"):
        add_change(changes, "sessions", f"sessions.{key}", "changed", prev_sessions.get(key), curr_sessions.get(key), "info")
    for key in ("platform", "chat_type"):
        prev_counts = count_by_metadata(as_list(prev_sessions.get("recent_metadata")), key)
        curr_counts = count_by_metadata(as_list(curr_sessions.get("recent_metadata")), key)
        add_change(changes, "sessions", f"sessions.recent_metadata.{key}_counts", "changed", prev_counts, curr_counts, "info")

    prev_repo = as_map(previous.get("repo"))
    curr_repo = as_map(current.get("repo"))
    for key in ("head", "branch", "describe"):
        severity = "warning" if key == "branch" else "info"
        add_change(changes, "repo", f"repo.{key}", "changed", prev_repo.get(key), curr_repo.get(key), severity)
    prev_dirty = as_map(prev_repo.get("status_short")).get("dirty_count")
    curr_dirty = as_map(curr_repo.get("status_short")).get("dirty_count")
    if prev_dirty != curr_dirty:
        change_type = "increased" if isinstance(prev_dirty, int) and isinstance(curr_dirty, int) and curr_dirty > prev_dirty else "decreased"
        add_change(changes, "repo", "repo.status_short.dirty_count", change_type, prev_dirty, curr_dirty, "info")
    prev_files = file_set_from_repo(prev_repo)
    curr_files = file_set_from_repo(curr_repo)
    if prev_files != curr_files:
        added_files = sorted(curr_files - prev_files)
        removed_files = sorted(prev_files - curr_files)
        if added_files:
            add_change(changes, "repo", "repo.status_short.files.added", "added", [], added_files, "info")
        if removed_files:
            add_change(changes, "repo", "repo.status_short.files.removed", "removed", removed_files, [], "info")

    prev_errors = as_map(previous.get("recent_errors"))
    curr_errors = as_map(current.get("recent_errors"))
    prev_count = prev_errors.get("count")
    curr_count = curr_errors.get("count")
    if prev_count != curr_count:
        change_type = "increased" if isinstance(prev_count, int) and isinstance(curr_count, int) and curr_count > prev_count else "decreased"
        severity = "warning" if change_type == "increased" else "info"
        add_change(changes, "recent_errors", "recent_errors.count", change_type, prev_count, curr_count, severity)
    prev_grouped = as_map(prev_errors.get("grouped"))
    curr_grouped = as_map(curr_errors.get("grouped"))
    for source in sorted(set(prev_grouped) | set(curr_grouped)):
        prev_source = as_map(prev_grouped.get(source))
        curr_source = as_map(curr_grouped.get(source))
        add_change(changes, "recent_errors", f"recent_errors.grouped.{source}.total", "changed", prev_source.get("total"), curr_source.get("total"), "warning" if (curr_source.get("total") or 0) > (prev_source.get("total") or 0) else "info")
        prev_categories = as_map(prev_source.get("categories"))
        curr_categories = as_map(curr_source.get("categories"))
        for category in sorted(set(prev_categories) | set(curr_categories)):
            before = prev_categories.get(category, 0)
            after = curr_categories.get(category, 0)
            severity = "warning" if after > before else "info"
            if category in {"traceback", "exception"} and after > before + 5:
                severity = "critical"
            add_change(changes, "recent_errors", f"recent_errors.grouped.{source}.categories.{category}", "increased" if after > before else "decreased", before, after, severity)
        add_change(changes, "recent_errors", f"recent_errors.grouped.{source}.last_seen", "changed", prev_source.get("last_seen"), curr_source.get("last_seen"), "info")

    prev_runtime = runtime_file_map(as_map(previous.get("runtime_files")))
    curr_runtime = runtime_file_map(as_map(current.get("runtime_files")))
    for path in sorted(set(prev_runtime) | set(curr_runtime)):
        prev_file = as_map(prev_runtime.get(path))
        curr_file = as_map(curr_runtime.get(path))
        for key in ("exists", "size_bytes"):
            add_change(changes, "runtime_files", f"runtime_files.{path}.{key}", "changed", prev_file.get(key), curr_file.get(key), "info")

    prev_toolsets = as_map(previous.get("toolsets"))
    curr_toolsets = as_map(current.get("toolsets"))
    prev_summary = as_map(prev_toolsets.get("summary"))
    curr_summary = as_map(curr_toolsets.get("summary"))
    for key in ("total", "active_like"):
        add_change(changes, "toolsets", f"toolsets.summary.{key}", "changed", prev_summary.get(key), curr_summary.get(key), "info")
    prev_lines = line_set(prev_toolsets)
    curr_lines = line_set(curr_toolsets)
    if prev_lines != curr_lines:
        add_change(changes, "toolsets", "toolsets.lines", "changed", sorted(prev_lines), sorted(curr_lines), "info")

    add_change(changes, "version", "hermes_version", "changed", previous.get("hermes_version"), current.get("hermes_version"), "info")

    flattened = [item for group in changes.values() for item in group]
    severity = "info"
    if flattened:
        severity = max((item.get("severity", "info") for item in flattened), key=severity_rank)
    headlines = build_diff_headlines(changes)
    return {
        "summary": {
            "changed_count": len(flattened),
            "severity": severity,
            "headlines": headlines[:12],
            "error_delta": build_error_delta_summary(prev_errors, curr_errors),
        },
        "changes": changes,
    }


def build_diff_headlines(changes: dict[str, list[dict[str, Any]]]) -> list[str]:
    headlines: list[str] = []
    for group, items in changes.items():
        for item in items:
            path = item.get("path", group)
            before = item.get("before")
            after = item.get("after")
            severity = item.get("severity", "info")
            headlines.append(f"{severity}: {path} changed from {before!r} to {after!r}")
    return headlines


def get_overview_diff(update_baseline: bool = False) -> dict[str, Any]:
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    current_overview = get_overview()
    errors.extend(current_overview.get("errors", []))
    warnings.extend(current_overview.get("warnings", []))
    current_data = sanitize(current_overview.get("data", {}))
    previous_snapshot, read_error = read_snapshot()
    if read_error:
        warnings.append("Previous overview snapshot could not be read; treating this as the first snapshot.")
    has_previous = bool(previous_snapshot and isinstance(previous_snapshot.get("overview"), dict) and not read_error)
    data: dict[str, Any] = {
        "has_previous": has_previous,
        "baseline_timestamp_utc": previous_snapshot.get("updated_at_utc") if has_previous else None,
        "current_timestamp_utc": current_data.get("timestamp_utc") or utc_now(),
        "updated_baseline": False,
        "summary": {
            "changed_count": 0,
            "severity": "info",
            "headlines": [],
        },
        "changes": empty_change_groups(),
    }
    if not has_previous:
        data["summary"]["headlines"] = ["No previous Quinn Ops overview snapshot exists yet; save a snapshot to establish the baseline."]
        warnings.append("No previous Quinn Ops overview snapshot exists yet.")
    else:
        diff = compare_overviews(as_map(previous_snapshot.get("overview")), current_data)
        data.update(diff)
    if update_baseline:
        snapshot = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "created_by": "quinn_ops",
            "updated_at_utc": utc_now(),
            "overview": current_data,
            "overview_errors": sanitize(current_overview.get("errors", [])),
            "overview_warnings": sanitize(current_overview.get("warnings", [])),
        }
        try:
            data["updated_baseline"] = True
            data["snapshot"] = write_snapshot(snapshot)
        except Exception as exc:
            data["updated_baseline"] = False
            errors.append(command_error("overview snapshot write", type(exc).__name__, "exception"))
    return response(data, errors, warnings)


TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_overview": get_overview,
    "get_snapshot_status": get_snapshot_status,
    "get_overview_diff": get_overview_diff,
    "save_overview_snapshot": save_overview_snapshot,
    "get_gateway_status": get_gateway_status,
    "get_platform_status": get_platform_status,
    "get_mcp_status": get_mcp_status,
    "get_toolsets_status": get_toolsets_status,
    "get_cron_status": get_cron_status,
    "get_sessions_summary": get_sessions_summary,
    "get_recent_errors": get_recent_errors,
    "get_config_summary": get_config_summary,
    "get_repo_status": get_repo_status,
    "get_runtime_files_status": get_runtime_files_status,
    "healthcheck": healthcheck,
}


def build_mcp():
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError(
            "Python MCP SDK is not installed. Install the official MCP SDK before "
            "running quinn_ops over stdio; collection functions remain importable."
        ) from exc
    mcp = FastMCP("quinn_ops")
    for fn in TOOL_FUNCTIONS.values():
        mcp.tool()(fn)
    return mcp


def main() -> int:
    try:
        build_mcp().run()
        return 0
    except RuntimeError as exc:
        print(json.dumps(response({"server": "quinn_ops"}, [{"source": "mcp import", "message": str(exc), "kind": "missing_dependency"}])), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
