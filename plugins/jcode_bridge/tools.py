"""Tool handlers for running jcode as a Hermes sidecar."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

from plugins.jcode_bridge.contracts import (
    BRIDGE_CONTRACT_VERSION,
    BRIDGE_SCHEMA_FILENAMES,
    bridge_schema_dir,
    make_debug_command_request,
    validate_debug_command_request,
    validate_debug_response_payload,
    validate_run_json_payload,
    validate_run_ndjson_events,
)
from plugins.jcode_bridge.safety import evaluate_jcode_bridge_safety


MAX_OUTPUT_CHARS = 64_000
DEFAULT_TIMEOUT_SECONDS = 600
MAX_TIMEOUT_SECONDS = 3600


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True)


def _truncate(value: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n...[truncated {len(value) - limit} chars]"


def _resolve_jcode_bin(args: dict[str, Any] | None = None) -> str | None:
    args = args or {}
    explicit = args.get("jcode_bin")
    if isinstance(explicit, str) and explicit.strip():
        return str(Path(explicit).expanduser())

    env_bin = os.getenv("JCODE_BIN")
    if env_bin:
        return str(Path(env_bin).expanduser())

    return shutil.which("jcode")


def check_jcode_available() -> bool:
    """Return True when a jcode executable is available."""
    resolved = _resolve_jcode_bin()
    if not resolved:
        return False
    if os.sep in resolved or Path(resolved).is_absolute():
        path = Path(resolved)
        return path.exists() and os.access(path, os.X_OK)
    return shutil.which(resolved) is not None


def _timeout_seconds(args: dict[str, Any]) -> int:
    raw = args.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    try:
        timeout = int(raw)
    except (TypeError, ValueError):
        timeout = DEFAULT_TIMEOUT_SECONDS
    return max(1, min(timeout, MAX_TIMEOUT_SECONDS))


def _base_command(args: dict[str, Any], jcode_bin: str, *, include_resume: bool = True) -> list[str]:
    cmd = [jcode_bin, "--quiet", "--no-update", "--no-selfdev"]

    provider = args.get("provider")
    if isinstance(provider, str) and provider.strip():
        cmd.extend(["--provider", provider.strip()])

    model = args.get("model")
    if isinstance(model, str) and model.strip():
        cmd.extend(["--model", model.strip()])

    provider_profile = args.get("provider_profile")
    if isinstance(provider_profile, str) and provider_profile.strip():
        cmd.extend(["--provider-profile", provider_profile.strip()])

    session = args.get("session")
    if include_resume and isinstance(session, str) and session.strip():
        cmd.extend(["--resume", session.strip()])

    cwd = args.get("cwd")
    if isinstance(cwd, str) and cwd.strip():
        cmd.extend(["-C", str(Path(cwd).expanduser())])

    return cmd


def _server_debug_command(args: dict[str, Any], jcode_bin: str, message: str) -> list[str]:
    cmd = _base_command(args, jcode_bin, include_resume=False)
    cmd.append("debug")

    socket = args.get("socket")
    if isinstance(socket, str) and socket.strip():
        cmd.extend(["--socket", str(Path(socket).expanduser())])

    session = args.get("session")
    if isinstance(session, str) and session.strip():
        cmd.extend(["--session", session.strip()])

    cmd.extend(["--wait", "message", message])
    return cmd


def _debug_start_command(args: dict[str, Any], jcode_bin: str) -> list[str]:
    cmd = _base_command(args, jcode_bin, include_resume=False)
    cmd.append("debug")

    socket = args.get("socket")
    if isinstance(socket, str) and socket.strip():
        cmd.extend(["--socket", str(Path(socket).expanduser())])

    cmd.append("start")
    return cmd


def _runtime_user_discriminator() -> str:
    if hasattr(os, "geteuid"):
        return str(os.geteuid())
    raw = os.getenv("USERNAME") or os.getenv("USER") or "user"
    sanitized = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")[:64]
    return sanitized or "user"


def _jcode_runtime_dir() -> Path:
    if os.getenv("JCODE_RUNTIME_DIR"):
        return Path(os.environ["JCODE_RUNTIME_DIR"])
    if os.getenv("XDG_RUNTIME_DIR"):
        return Path(os.environ["XDG_RUNTIME_DIR"])
    if sys.platform == "darwin" and os.getenv("TMPDIR"):
        return Path(os.environ["TMPDIR"])
    return Path(tempfile.gettempdir()) / f"jcode-{_runtime_user_discriminator()}"


def _candidate_runtime_dirs() -> list[Path]:
    candidates: list[Path] = []
    for env_name in ("JCODE_RUNTIME_DIR", "XDG_RUNTIME_DIR"):
        value = os.getenv(env_name)
        if value:
            candidates.append(Path(value).expanduser())
    if sys.platform == "darwin" and os.getenv("TMPDIR"):
        candidates.append(Path(os.environ["TMPDIR"]).expanduser())
    temp_dir = Path(tempfile.gettempdir())
    candidates.extend([
        temp_dir / f"jcode-{_runtime_user_discriminator()}",
        temp_dir,
    ])

    seen: set[str] = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _debug_socket_from_main_socket(path: str | Path) -> str:
    main = Path(path).expanduser()
    name = main.name
    if name.endswith("-debug.sock"):
        return str(main)
    if name.endswith(".sock"):
        return str(main.with_name(name[:-5] + "-debug.sock"))
    return str(main.with_name(name + "-debug.sock"))


def _resolve_debug_socket(args: dict[str, Any]) -> str:
    return _debug_socket_candidates(args)[0]


def _discover_debug_sockets(limit: int = 20) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for directory in _candidate_runtime_dirs():
        try:
            matches = sorted(directory.glob("jcode*-debug.sock"))
        except OSError:
            continue
        for path in matches:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            found.append(key)
            if len(found) >= limit:
                return found
    return found


def _debug_socket_candidates(args: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    debug_socket = args.get("debug_socket")
    if isinstance(debug_socket, str) and debug_socket.strip():
        candidates.append(str(Path(debug_socket).expanduser()))

    socket_path = args.get("socket") or os.getenv("JCODE_SOCKET")
    if isinstance(socket_path, str) and socket_path.strip():
        candidates.append(_debug_socket_from_main_socket(socket_path))

    candidates.append(str(_jcode_runtime_dir() / "jcode-debug.sock"))
    candidates.extend(_discover_debug_sockets())

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return unique


def _read_socket_line(sock: socket.socket) -> str:
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    return b"".join(chunks).split(b"\n", 1)[0].decode("utf-8", errors="replace")


def _display_debug_socket_command(debug_socket: str, command: str) -> list[str]:
    display_command = command
    if command.startswith("message:"):
        display_command = "message:<message>"
    return ["debug_socket", debug_socket, display_command]


def _run_single_debug_socket_command(
    args: dict[str, Any],
    command: str,
    debug_socket: str,
    *,
    redact_command: bool = False,
) -> dict[str, Any]:
    timeout = _timeout_seconds(args)
    request_id = int(time.time() * 1000) % 1_000_000_000
    session = args.get("session")
    session_id = session.strip() if isinstance(session, str) and session.strip() else None
    request = make_debug_command_request(command, request_id=request_id, session_id=session_id)

    display_command = _display_debug_socket_command(debug_socket, command) if redact_command else [
        "debug_socket",
        debug_socket,
        command,
    ]

    if not hasattr(socket, "AF_UNIX"):
        return {
            "success": False,
            "error": "direct debug socket mode requires Unix sockets",
            "debug_socket": debug_socket,
            "command": display_command,
        }

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(debug_socket)
            sock.sendall((json.dumps(request, ensure_ascii=True) + "\n").encode("utf-8"))
            line = _read_socket_line(sock)
    except OSError as exc:
        return {
            "success": False,
            "error": f"debug socket unavailable: {exc}",
            "debug_socket": debug_socket,
            "command": display_command,
        }

    if not line:
        return {
            "success": False,
            "error": "debug socket closed without a response",
            "debug_socket": debug_socket,
            "command": display_command,
        }

    try:
        response = json.loads(line)
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "error": f"failed to parse debug socket response: {exc}",
            "debug_socket": debug_socket,
            "command": display_command,
            "stdout": _truncate(line),
        }

    contract = validate_debug_response_payload(response)
    if not contract.ok:
        return {
            "success": False,
            "error": "debug socket response violated jcode bridge contract",
            "contract_version": BRIDGE_CONTRACT_VERSION,
            "contract_errors": contract.errors,
            "debug_socket": debug_socket,
            "command": display_command,
            "response": response,
        }

    if response.get("type") == "error":
        return {
            "success": False,
            "error": response.get("message") or "jcode debug socket returned an error",
            "contract_version": BRIDGE_CONTRACT_VERSION,
            "debug_socket": debug_socket,
            "command": display_command,
            "response": response,
        }

    ok = bool(response.get("ok"))
    output = response.get("output")
    if not isinstance(output, str):
        output = response.get("message") if isinstance(response.get("message"), str) else ""

    return {
        "success": ok,
        "contract_version": BRIDGE_CONTRACT_VERSION,
        "debug_socket": debug_socket,
        "command": display_command,
        "stdout": _truncate(output or ""),
        "response": response,
    }


def _run_debug_socket_command(args: dict[str, Any], command: str, *, redact_command: bool = False) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    candidates = _debug_socket_candidates(args)
    for debug_socket in candidates:
        result = _run_single_debug_socket_command(
            args,
            command,
            debug_socket,
            redact_command=redact_command,
        )
        if result.get("success"):
            result["attempted_debug_sockets"] = [item["debug_socket"] for item in attempts] + [debug_socket]
            return result
        attempts.append({
            "debug_socket": debug_socket,
            "error": result.get("error", "unknown error"),
        })

    fallback_socket = candidates[0] if candidates else _resolve_debug_socket(args)
    return {
        "success": False,
        "error": "no reachable jcode debug socket",
        "debug_socket": fallback_socket,
        "command": _display_debug_socket_command(fallback_socket, command)
        if redact_command
        else ["debug_socket", fallback_socket, command],
        "attempts": attempts,
    }


def _debug_socket_status(args: dict[str, Any]) -> dict[str, Any]:
    candidates = _debug_socket_candidates(args)
    probes: list[dict[str, Any]] = []
    for debug_socket in candidates:
        probe = _run_single_debug_socket_command(args, "sessions", debug_socket)
        probes.append({
            "debug_socket": debug_socket,
            "success": bool(probe.get("success")),
            "error": probe.get("error"),
            "stdout": _truncate(str(probe.get("stdout") or ""), 4000),
        })
    return {
        "success": any(item["success"] for item in probes),
        "candidates": candidates,
        "probes": probes,
    }


def _display_command(cmd: Iterable[str], redact_last: bool = False) -> list[str]:
    display = list(cmd)
    if redact_last and display:
        display[-1] = "<message>"
    return display


def _run_subprocess(cmd: list[str], args: dict[str, Any], *, redact_last: bool = False) -> dict[str, Any]:
    cwd_arg = args.get("cwd")
    run_cwd = str(Path(cwd_arg).expanduser()) if isinstance(cwd_arg, str) and cwd_arg.strip() else None
    timeout = _timeout_seconds(args)
    if run_cwd and not Path(run_cwd).is_dir():
        return {
            "success": False,
            "error": "cwd does not exist or is not a directory",
            "cwd": run_cwd,
            "command": _display_command(cmd, redact_last=redact_last),
        }

    try:
        completed = subprocess.run(
            cmd,
            cwd=run_cwd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return {
            "success": False,
            "error": "jcode executable not found",
            "command": _display_command(cmd, redact_last=redact_last),
        }
    except OSError as exc:
        return {
            "success": False,
            "error": f"failed to launch jcode: {exc}",
            "command": _display_command(cmd, redact_last=redact_last),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "timed_out": True,
            "timeout_seconds": timeout,
            "command": _display_command(cmd, redact_last=redact_last),
            "stdout": _truncate(exc.stdout or ""),
            "stderr": _truncate(exc.stderr or ""),
        }

    return {
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": _display_command(cmd, redact_last=redact_last),
        "stdout": _truncate(completed.stdout or ""),
        "stderr": _truncate(completed.stderr or ""),
    }


def _parse_json_stdout(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return None
    return json.loads(raw)


def _parse_ndjson_stdout(raw: str) -> list[Any]:
    events: list[Any] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        events.append(json.loads(line))
    return events


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _safety_audit(args: dict[str, Any], details: dict[str, Any]) -> dict[str, Any] | None:
    reason = args.get("safety_override_reason")
    confirmed_fields: list[str] = []
    for field in ("confirm_outbound_human_contact", "confirm_sensitive_person_data"):
        if _truthy(args.get(field)):
            confirmed_fields.append(field)

    if not details and not confirmed_fields and not reason:
        return None

    audit: dict[str, Any] = {
        "matched_risk_categories": sorted(details.keys()),
    }
    if confirmed_fields:
        audit["confirmed_fields"] = confirmed_fields
    if isinstance(reason, str) and reason.strip():
        audit["override_reason"] = reason.strip()
    return audit


def _attach_safety_audit(result: dict[str, Any], audit: dict[str, Any] | None) -> dict[str, Any]:
    if audit:
        result["safety"] = audit
    return result


def _repo_root() -> Path:
    override = os.getenv("JCODE_BRIDGE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _fixture_dir() -> Path:
    return _repo_root() / "tests" / "fixtures" / "jcode_bridge"


def _schema_dir() -> Path:
    return bridge_schema_dir(_repo_root())


def _load_fixture_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fixture_ndjson(path: Path) -> list[Any]:
    events: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _contract_check(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "ok": bool(ok)}
    result.update(details)
    return result


def _validation_check(name: str, validation: Any) -> dict[str, Any]:
    return _contract_check(name, bool(validation.ok), errors=validation.errors)


def _safe_validation_check(name: str, loader, validator) -> dict[str, Any]:
    try:
        validation = validator(loader())
    except Exception as exc:
        return _contract_check(name, False, errors=[str(exc)])
    return _validation_check(name, validation)


def _contract_fixture_checks(fixture_dir: Path) -> list[dict[str, Any]]:
    request = make_debug_command_request(
        "message:hello",
        request_id=1,
        session_id="session_test",
    )
    return [
        _safe_validation_check(
            "fixture:run_json_success",
            lambda: _load_fixture_json(fixture_dir / "run_json_success.json"),
            validate_run_json_payload,
        ),
        _safe_validation_check(
            "fixture:run_ndjson_success",
            lambda: _load_fixture_ndjson(fixture_dir / "run_ndjson_success.ndjson"),
            validate_run_ndjson_events,
        ),
        _safe_validation_check(
            "fixture:debug_response_success",
            lambda: _load_fixture_json(fixture_dir / "debug_response_success.json"),
            validate_debug_response_payload,
        ),
        _safe_validation_check(
            "fixture:debug_response_error",
            lambda: _load_fixture_json(fixture_dir / "debug_response_error.json"),
            validate_debug_response_payload,
        ),
        _validation_check(
            "generated:debug_command_request",
            validate_debug_command_request(request),
        ),
    ]


def _contract_schema_checks(schema_dir: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for filename in BRIDGE_SCHEMA_FILENAMES:
        path = schema_dir / filename
        if not path.exists():
            checks.append(_contract_check(
                f"schema:{filename}",
                False,
                errors=["schema file is missing"],
            ))
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            checks.append(_contract_check(
                f"schema:{filename}",
                False,
                errors=[str(exc)],
            ))
            continue

        errors: list[str] = []
        if payload.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append("schema must declare JSON Schema draft 2020-12")
        if payload.get("x-bridge-contract-version") != BRIDGE_CONTRACT_VERSION:
            errors.append(
                f"schema x-bridge-contract-version must be {BRIDGE_CONTRACT_VERSION}"
            )
        if not isinstance(payload.get("$id"), str) or not payload.get("$id"):
            errors.append("schema must declare a non-empty $id")
        checks.append(_contract_check(
            f"schema:{filename}",
            not errors,
            errors=errors,
        ))
    return checks


def _contract_live_args(args: dict[str, Any]) -> dict[str, Any]:
    live_args: dict[str, Any] = {
        "timeout_seconds": _timeout_seconds(args),
    }
    for key in ("jcode_bin", "cwd", "provider", "model", "provider_profile"):
        value = args.get(key)
        if isinstance(value, str) and value.strip():
            live_args[key] = value
    return live_args


JCODE_RUN_SCHEMA = {
    "name": "jcode_run",
    "description": (
        "Run one prompt through a local jcode sidecar using jcode's wrapper CLI. "
        "Use for low-latency local execution, jcode browser/profile workflows, "
        "or jcode swarm-aware work while keeping Hermes as the orchestrator."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Prompt to send to jcode.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory for the jcode run.",
            },
            "session": {
                "type": "string",
                "description": "Optional jcode session ID/name to resume.",
            },
            "provider": {
                "type": "string",
                "description": "Optional jcode provider override, e.g. openai or claude.",
            },
            "model": {
                "type": "string",
                "description": "Optional jcode model override.",
            },
            "provider_profile": {
                "type": "string",
                "description": "Optional jcode provider profile name.",
            },
            "output_mode": {
                "type": "string",
                "enum": ["json", "ndjson", "text"],
                "description": "How to ask jcode CLI mode to emit results. Defaults to json.",
            },
            "execution_mode": {
                "type": "string",
                "enum": ["cli", "server_debug", "debug_socket", "auto"],
                "description": (
                    "cli runs one jcode process; server_debug shells through jcode debug; "
                    "debug_socket talks directly to a running jcode debug socket; auto tries "
                    "debug_socket, then server_debug, then cli."
                ),
            },
            "ensure_server": {
                "type": "boolean",
                "description": (
                    "When true, try `jcode debug start` before falling back from "
                    "server-backed modes. This favors a hot Rust sidecar over "
                    "one-shot CLI startup."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Subprocess timeout, clamped to 1..3600 seconds.",
            },
            "socket": {
                "type": "string",
                "description": "Optional jcode server socket path for execution_mode=server_debug.",
            },
            "debug_socket": {
                "type": "string",
                "description": "Optional jcode debug socket path for execution_mode=debug_socket.",
            },
            "jcode_bin": {
                "type": "string",
                "description": "Optional path to the jcode executable. Defaults to JCODE_BIN or PATH.",
            },
            "confirm_outbound_human_contact": {
                "type": "boolean",
                "description": (
                    "Required true before routing prompts that appear to send, post, "
                    "reply, DM, text, call, or otherwise contact a person/account."
                ),
            },
            "confirm_sensitive_person_data": {
                "type": "boolean",
                "description": (
                    "Required true before routing prompts that appear to find private "
                    "personal contact or identity data such as phone numbers, home "
                    "addresses, personal email, SSN, or date of birth."
                ),
            },
            "safety_override_reason": {
                "type": "string",
                "description": "Optional audit note explaining why a confirmation flag was set.",
            },
        },
        "required": ["message"],
    },
}


JCODE_STATUS_SCHEMA = {
    "name": "jcode_status",
    "description": "Check whether the local jcode sidecar is reachable and return wrapper-friendly JSON diagnostics.",
    "parameters": {
        "type": "object",
        "properties": {
            "checks": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "version",
                        "auth_status",
                        "provider_current",
                        "browser_status",
                        "server_list",
                        "debug_sockets",
                    ],
                },
                "description": "Status checks to run. Defaults to version only.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory for jcode status commands.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Subprocess timeout per check, clamped to 1..3600 seconds.",
            },
            "jcode_bin": {
                "type": "string",
                "description": "Optional path to the jcode executable. Defaults to JCODE_BIN or PATH.",
            },
        },
    },
}


JCODE_CONTRACT_CHECK_SCHEMA = {
    "name": "jcode_contract_check",
    "description": (
        "Validate the Hermes/jcode bridge contract fixtures and optionally run "
        "lightweight live compatibility checks against a local jcode binary."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "live": {
                "type": "boolean",
                "description": "Also check a local jcode binary with `jcode version --json`.",
            },
            "live_run": {
                "type": "boolean",
                "description": "With live=true, run one harmless prompt through `jcode run --json`.",
            },
            "live_run_message": {
                "type": "string",
                "description": "Prompt for live_run. Defaults to a short OK response request.",
            },
            "cwd": {
                "type": "string",
                "description": "Optional working directory for live jcode checks.",
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Timeout per live check, clamped to 1..3600 seconds.",
            },
            "jcode_bin": {
                "type": "string",
                "description": "Optional path to the jcode executable. Defaults to JCODE_BIN or PATH.",
            },
            "provider": {
                "type": "string",
                "description": "Optional jcode provider override for live_run.",
            },
            "model": {
                "type": "string",
                "description": "Optional jcode model override for live_run.",
            },
            "provider_profile": {
                "type": "string",
                "description": "Optional jcode provider profile for live_run.",
            },
        },
    },
}


def handle_jcode_run(args: dict[str, Any], **_: Any) -> str:
    """Run `jcode run` and return a JSON string for Hermes."""
    if not isinstance(args, dict):
        args = {}

    message = args.get("message")
    if not isinstance(message, str) or not message.strip():
        return _json({"success": False, "error": "message is required"})

    output_mode = args.get("output_mode") or "json"
    if output_mode not in {"json", "ndjson", "text"}:
        return _json({"success": False, "error": "output_mode must be json, ndjson, or text"})

    execution_mode = args.get("execution_mode") or "cli"
    if execution_mode not in {"cli", "server_debug", "debug_socket", "auto"}:
        return _json({"success": False, "error": "execution_mode must be cli, server_debug, debug_socket, or auto"})

    safety = evaluate_jcode_bridge_safety(message, args)
    if not safety.allowed:
        return _json({
            "success": False,
            "error": "jcode bridge safety confirmation required",
            "requires_confirmation": True,
            "risk_types": safety.risk_types,
            "confirmation_fields": safety.confirmation_fields,
            "safety_details": safety.details,
            "message_chars": len(message),
        })
    safety_audit = _safety_audit(args, safety.details)
    ensure_server = _truthy(args.get("ensure_server"))
    server_start_attempt = None
    jcode_bin = None

    debug_socket_attempt = None
    if execution_mode in {"debug_socket", "auto"}:
        debug_socket_attempt = _run_debug_socket_command(
            args,
            f"message:{message}",
            redact_command=True,
        )
        debug_socket_attempt["execution_mode"] = "debug_socket"
        debug_socket_attempt["output_mode"] = "text"
        debug_socket_attempt["message_chars"] = len(message)
        if debug_socket_attempt.get("success"):
            debug_socket_attempt["parsed"] = {
                "text": str(debug_socket_attempt.get("stdout") or "").rstrip("\n")
            }
            return _json(_attach_safety_audit(debug_socket_attempt, safety_audit))

    if ensure_server and execution_mode in {"server_debug", "debug_socket", "auto"}:
        jcode_bin = _resolve_jcode_bin(args)
        if jcode_bin:
            server_start_attempt = _run_subprocess(_debug_start_command(args, jcode_bin), args)
            server_start_attempt["jcode_bin"] = jcode_bin
            server_start_attempt["execution_mode"] = "server_start"
        else:
            server_start_attempt = {
                "success": False,
                "error": "jcode executable not found; cannot start jcode server",
            }

        if execution_mode in {"debug_socket", "auto"}:
            retry_attempt = _run_debug_socket_command(
                args,
                f"message:{message}",
                redact_command=True,
            )
            retry_attempt["execution_mode"] = "debug_socket"
            retry_attempt["output_mode"] = "text"
            retry_attempt["message_chars"] = len(message)
            retry_attempt["server_start_attempt"] = server_start_attempt
            if debug_socket_attempt is not None:
                retry_attempt["debug_socket_attempt"] = debug_socket_attempt
            debug_socket_attempt = retry_attempt
            if debug_socket_attempt.get("success"):
                debug_socket_attempt["parsed"] = {
                    "text": str(debug_socket_attempt.get("stdout") or "").rstrip("\n")
                }
                return _json(_attach_safety_audit(debug_socket_attempt, safety_audit))

    if execution_mode == "debug_socket":
        if debug_socket_attempt is not None and server_start_attempt is not None:
            debug_socket_attempt["server_start_attempt"] = server_start_attempt
        return _json(_attach_safety_audit(debug_socket_attempt or {
            "success": False,
            "error": "no reachable jcode debug socket",
        }, safety_audit))

    if not jcode_bin:
        jcode_bin = _resolve_jcode_bin(args)
    if not jcode_bin:
        result = {
            "success": False,
            "error": "jcode executable not found; install jcode or set JCODE_BIN",
        }
        if debug_socket_attempt is not None:
            result["debug_socket_attempt"] = debug_socket_attempt
        if server_start_attempt is not None:
            result["server_start_attempt"] = server_start_attempt
        return _json(_attach_safety_audit(result, safety_audit))

    if execution_mode in {"server_debug", "auto"}:
        server_cmd = _server_debug_command(args, jcode_bin, message)
        server_result = _run_subprocess(server_cmd, args, redact_last=True)
        server_result["jcode_bin"] = jcode_bin
        server_result["execution_mode"] = "server_debug"
        server_result["output_mode"] = "text"
        server_result["message_chars"] = len(message)
        if server_result.get("success"):
            server_result["parsed"] = {
                "text": str(server_result.get("stdout") or "").rstrip("\n")
            }
            if debug_socket_attempt is not None:
                server_result["debug_socket_attempt"] = debug_socket_attempt
            if server_start_attempt is not None:
                server_result["server_start_attempt"] = server_start_attempt
            return _json(_attach_safety_audit(server_result, safety_audit))
        if execution_mode == "server_debug":
            if server_start_attempt is not None:
                server_result["server_start_attempt"] = server_start_attempt
            return _json(_attach_safety_audit(server_result, safety_audit))

    cmd = _base_command(args, jcode_bin)
    cmd.append("run")
    if output_mode == "json":
        cmd.append("--json")
    elif output_mode == "ndjson":
        cmd.append("--ndjson")
    cmd.append(message)

    result = _run_subprocess(cmd, args, redact_last=True)
    result["jcode_bin"] = jcode_bin
    result["execution_mode"] = "cli"
    result["output_mode"] = output_mode
    result["message_chars"] = len(message)
    if debug_socket_attempt is not None:
        result["debug_socket_attempt"] = debug_socket_attempt
    if server_start_attempt is not None:
        result["server_start_attempt"] = server_start_attempt
    if execution_mode == "auto":
        result["server_debug_attempt"] = server_result

    if result.get("success") and output_mode in {"json", "ndjson"}:
        try:
            parser = _parse_json_stdout if output_mode == "json" else _parse_ndjson_stdout
            parsed = parser(result.get("stdout", ""))
            contract = (
                validate_run_json_payload(parsed)
                if output_mode == "json"
                else validate_run_ndjson_events(parsed)
            )
            result["contract_version"] = BRIDGE_CONTRACT_VERSION
            if not contract.ok:
                result["success"] = False
                result["error"] = f"jcode {output_mode} output violated bridge contract"
                result["contract_errors"] = contract.errors
            result["parsed"] = parsed
        except json.JSONDecodeError as exc:
            result["success"] = False
            result["error"] = f"failed to parse jcode {output_mode} output: {exc}"

    return _json(_attach_safety_audit(result, safety_audit))


def handle_jcode_contract_check(args: dict[str, Any], **_: Any) -> str:
    """Validate the bridge's versioned compatibility contract."""
    if not isinstance(args, dict):
        args = {}

    fixture_dir = _fixture_dir()
    schema_dir = _schema_dir()
    checks = _contract_fixture_checks(fixture_dir)
    checks.extend(_contract_schema_checks(schema_dir))

    if _truthy(args.get("live")):
        live_args = _contract_live_args(args)
        status = json.loads(handle_jcode_status({
            **live_args,
            "checks": ["version"],
        }))
        checks.append(_contract_check(
            "live:jcode_status_version",
            bool(status.get("success")),
            payload=status,
        ))

        if _truthy(args.get("live_run")):
            message = args.get("live_run_message")
            if not isinstance(message, str) or not message.strip():
                message = "Reply with exactly OK."
            run = json.loads(handle_jcode_run({
                **live_args,
                "message": message,
                "output_mode": "json",
            }))
            checks.append(_contract_check(
                "live:jcode_run_json",
                bool(run.get("success")),
                payload=run,
            ))

    return _json({
        "success": all(item["ok"] for item in checks),
        "contract_version": BRIDGE_CONTRACT_VERSION,
        "fixture_dir": str(fixture_dir),
        "schema_dir": str(schema_dir),
        "schema_files": list(BRIDGE_SCHEMA_FILENAMES),
        "checks": checks,
    })


def handle_jcode_status(args: dict[str, Any], **_: Any) -> str:
    """Run lightweight jcode diagnostics and return a JSON string."""
    if not isinstance(args, dict):
        args = {}

    requested = args.get("checks") or ["version"]
    if not isinstance(requested, list):
        requested = ["version"]

    jcode_bin = _resolve_jcode_bin(args)
    cli_checks = {
        "version",
        "auth_status",
        "provider_current",
        "browser_status",
        "server_list",
    }
    needs_cli = any(check in cli_checks for check in requested)
    if needs_cli and not jcode_bin:
        return _json({
            "success": False,
            "available": False,
            "error": "jcode executable not found; install jcode or set JCODE_BIN",
            "debug_sockets": _debug_socket_status(args)
            if "debug_sockets" in requested
            else None,
        })

    known: dict[str, list[str]] = {
        "version": ["version", "--json"],
        "auth_status": ["auth", "status", "--json"],
        "provider_current": ["provider", "current", "--json"],
        "browser_status": ["browser", "status"],
        "server_list": ["debug", "list"],
    }

    checks: dict[str, Any] = {}
    overall_success = True
    for check in requested:
        if check == "debug_sockets":
            result = _debug_socket_status(args)
            checks[check] = result
            overall_success = overall_success and bool(result.get("success"))
            continue
        if check not in known:
            checks[str(check)] = {"success": False, "error": "unknown check"}
            overall_success = False
            continue
        if not jcode_bin:
            checks[str(check)] = {
                "success": False,
                "error": "jcode executable not found; install jcode or set JCODE_BIN",
            }
            overall_success = False
            continue
        cmd = _base_command(args, jcode_bin) + known[check]
        result = _run_subprocess(cmd, args)
        if result.get("success") and check not in {"browser_status", "server_list"}:
            try:
                result["parsed"] = _parse_json_stdout(result.get("stdout", ""))
            except json.JSONDecodeError as exc:
                result["success"] = False
                result["error"] = f"failed to parse jcode JSON output: {exc}"
        checks[check] = result
        overall_success = overall_success and bool(result.get("success"))

    return _json({
        "success": overall_success,
        "available": bool(jcode_bin) or bool(checks.get("debug_sockets", {}).get("success")),
        "jcode_bin": jcode_bin,
        "checks": checks,
    })
