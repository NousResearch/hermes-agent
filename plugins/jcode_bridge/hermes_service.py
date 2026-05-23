"""Reverse bridge for exposing selected Hermes tools to jcode."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from plugins.jcode_bridge.safety import evaluate_jcode_bridge_safety


HERMES_SERVICE_CONTRACT_VERSION = "hermes-service.v1"
HERMES_SERVICE_SCHEMA_RELATIVE_DIR = Path("contracts") / "hermes_service" / "v1"
HERMES_SERVICE_SCHEMA_FILENAMES = (
    "service_request.schema.json",
    "service_response.schema.json",
)

DEFAULT_ALLOWED_TOOLS = (
    "web_search",
    "web_extract",
    "session_search",
    "memory",
)
CONFIRMATION_REQUIRED_TOOLS = {
    "send_message",
}


@dataclass(frozen=True)
class ServiceValidation:
    ok: bool
    errors: list[str]


def _ok() -> ServiceValidation:
    return ServiceValidation(ok=True, errors=[])


def _fail(*errors: str) -> ServiceValidation:
    return ServiceValidation(ok=False, errors=[error for error in errors if error])


def hermes_service_schema_dir(repo_root: Path | None = None) -> Path:
    """Return the repo-local directory for portable Hermes service schemas."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / HERMES_SERVICE_SCHEMA_RELATIVE_DIR


def validate_service_request(payload: Any) -> ServiceValidation:
    """Validate the jcode -> Hermes service request envelope."""
    if not isinstance(payload, dict):
        return _fail("service request must be an object")

    errors: list[str] = []
    if payload.get("type") != "hermes_service_request":
        errors.append("service request type must be 'hermes_service_request'")

    request_id = payload.get("id")
    if request_id is not None and not isinstance(request_id, (str, int)):
        errors.append("service request id must be a string or integer when present")

    tool = payload.get("tool")
    if not isinstance(tool, str) or not tool:
        errors.append("service request tool must be a non-empty string")

    args = payload.get("args")
    if args is None:
        errors.append("service request args must be present")
    elif not isinstance(args, dict):
        errors.append("service request args must be an object")

    for key in ("session_id", "task_id", "safety_override_reason"):
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"service request {key} must be a string when present")

    for key in ("confirm_outbound_human_contact", "confirm_sensitive_person_data"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            errors.append(f"service request {key} must be a boolean when present")

    return _ok() if not errors else ServiceValidation(ok=False, errors=errors)


def validate_service_response(payload: Any) -> ServiceValidation:
    """Validate the Hermes service response envelope."""
    if not isinstance(payload, dict):
        return _fail("service response must be an object")

    errors: list[str] = []
    if payload.get("type") != "hermes_service_response":
        errors.append("service response type must be 'hermes_service_response'")
    if payload.get("contract_version") != HERMES_SERVICE_CONTRACT_VERSION:
        errors.append(
            f"service response contract_version must be {HERMES_SERVICE_CONTRACT_VERSION}"
        )
    if not isinstance(payload.get("ok"), bool):
        errors.append("service response ok must be a boolean")
    tool = payload.get("tool")
    if tool is not None and not isinstance(tool, str):
        errors.append("service response tool must be a string when present")
    error = payload.get("error")
    if error is not None and not isinstance(error, str):
        errors.append("service response error must be a string when present")
    if not payload.get("ok") and not isinstance(error, str):
        errors.append("failed service response must include string error")
    duration_ms = payload.get("duration_ms")
    if duration_ms is not None and not isinstance(duration_ms, int):
        errors.append("service response duration_ms must be an integer when present")

    return _ok() if not errors else ServiceValidation(ok=False, errors=errors)


def _repo_root() -> Path:
    override = os.getenv("JCODE_BRIDGE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _fixture_dir() -> Path:
    return _repo_root() / "tests" / "fixtures" / "hermes_service"


def _schema_dir() -> Path:
    return hermes_service_schema_dir(_repo_root())


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True)


def _parse_tool_result(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    stripped = raw.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return raw


def _request_text_for_safety(request: dict[str, Any]) -> str:
    args = request.get("args")
    if not isinstance(args, dict):
        return ""
    parts: list[str] = []
    for key in ("content", "message", "text", "body", "query", "prompt"):
        value = args.get(key)
        if isinstance(value, str):
            parts.append(value)
    target = args.get("target") or args.get("chat_id") or args.get("recipient")
    if isinstance(target, str):
        parts.append(target)
    return "\n".join(parts)


def _tool_allowed(tool: str, allowed_tools: Iterable[str]) -> bool:
    return tool in set(allowed_tools)


def _default_dispatch(tool: str, args: dict[str, Any], request: dict[str, Any]) -> str:
    from model_tools import handle_function_call

    return handle_function_call(
        tool,
        args,
        task_id=request.get("task_id") if isinstance(request.get("task_id"), str) else None,
        session_id=request.get("session_id") if isinstance(request.get("session_id"), str) else None,
        skip_pre_tool_call_hook=False,
    )


def dispatch_service_request(
    request: dict[str, Any],
    *,
    allowed_tools: Iterable[str] = DEFAULT_ALLOWED_TOOLS,
    dispatcher: Callable[[str, dict[str, Any], dict[str, Any]], str] | None = None,
) -> dict[str, Any]:
    """Dispatch one jcode -> Hermes service request and return a response envelope."""
    started = time.monotonic()
    validation = validate_service_request(request)
    request_id = request.get("id") if isinstance(request, dict) else None
    tool = request.get("tool") if isinstance(request, dict) else None
    if not validation.ok:
        return {
            "type": "hermes_service_response",
            "contract_version": HERMES_SERVICE_CONTRACT_VERSION,
            "id": request_id,
            "ok": False,
            "tool": tool,
            "error": "Hermes service request violated contract",
            "contract_errors": validation.errors,
        }

    assert isinstance(tool, str)
    args = request.get("args")
    assert isinstance(args, dict)

    effective_allowed = tuple(allowed_tools)
    if not _tool_allowed(tool, effective_allowed):
        return {
            "type": "hermes_service_response",
            "contract_version": HERMES_SERVICE_CONTRACT_VERSION,
            "id": request_id,
            "ok": False,
            "tool": tool,
            "error": "Hermes service tool is not allowed",
            "allowed_tools": list(effective_allowed),
        }

    if tool in CONFIRMATION_REQUIRED_TOOLS:
        safety_text = _request_text_for_safety(request) or tool
        safety = evaluate_jcode_bridge_safety(safety_text, request)
        if not request.get("confirm_outbound_human_contact") or not safety.allowed:
            confirmation_fields = safety.confirmation_fields or [
                "confirm_outbound_human_contact"
            ]
            return {
                "type": "hermes_service_response",
                "contract_version": HERMES_SERVICE_CONTRACT_VERSION,
                "id": request_id,
                "ok": False,
                "tool": tool,
                "error": "Hermes service safety confirmation required",
                "requires_confirmation": True,
                "risk_types": safety.risk_types or ["outbound_human_contact"],
                "confirmation_fields": confirmation_fields,
                "safety_details": safety.details,
            }

    dispatch = dispatcher or _default_dispatch
    try:
        raw = dispatch(tool, args, request)
    except Exception as exc:
        return {
            "type": "hermes_service_response",
            "contract_version": HERMES_SERVICE_CONTRACT_VERSION,
            "id": request_id,
            "ok": False,
            "tool": tool,
            "error": f"Hermes service dispatch failed: {type(exc).__name__}: {exc}",
            "duration_ms": int((time.monotonic() - started) * 1000),
        }

    parsed = _parse_tool_result(raw)
    ok = not (isinstance(parsed, dict) and isinstance(parsed.get("error"), str))
    response: dict[str, Any] = {
        "type": "hermes_service_response",
        "contract_version": HERMES_SERVICE_CONTRACT_VERSION,
        "id": request_id,
        "ok": ok,
        "tool": tool,
        "result": parsed,
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
    if not ok:
        response["error"] = parsed.get("error")
    return response


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "ok": bool(ok)}
    result.update(details)
    return result


def _validation_check(name: str, validation: ServiceValidation) -> dict[str, Any]:
    return _check(name, validation.ok, errors=validation.errors)


def _schema_checks(schema_dir: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for filename in HERMES_SERVICE_SCHEMA_FILENAMES:
        path = schema_dir / filename
        if not path.exists():
            checks.append(_check(f"schema:{filename}", False, errors=["schema file is missing"]))
            continue
        try:
            payload = _load_json(path)
        except Exception as exc:
            checks.append(_check(f"schema:{filename}", False, errors=[str(exc)]))
            continue
        errors: list[str] = []
        if payload.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            errors.append("schema must declare JSON Schema draft 2020-12")
        if payload.get("x-bridge-contract-version") != HERMES_SERVICE_CONTRACT_VERSION:
            errors.append(
                f"schema x-bridge-contract-version must be {HERMES_SERVICE_CONTRACT_VERSION}"
            )
        if not isinstance(payload.get("$id"), str) or not payload.get("$id"):
            errors.append("schema must declare a non-empty $id")
        checks.append(_check(f"schema:{filename}", not errors, errors=errors))
    return checks


def service_contract_report() -> dict[str, Any]:
    """Validate reverse-bridge fixtures and portable schema artifacts."""
    fixture_dir = _fixture_dir()
    schema_dir = _schema_dir()
    checks = [
        _validation_check(
            "fixture:service_request_web_search",
            validate_service_request(_load_json(fixture_dir / "service_request_web_search.json")),
        ),
        _validation_check(
            "fixture:service_response_success",
            validate_service_response(_load_json(fixture_dir / "service_response_success.json")),
        ),
        _validation_check(
            "fixture:service_response_error",
            validate_service_response(_load_json(fixture_dir / "service_response_error.json")),
        ),
    ]
    checks.extend(_schema_checks(schema_dir))
    return {
        "success": all(item["ok"] for item in checks),
        "contract_version": HERMES_SERVICE_CONTRACT_VERSION,
        "fixture_dir": str(fixture_dir),
        "schema_dir": str(schema_dir),
        "schema_files": list(HERMES_SERVICE_SCHEMA_FILENAMES),
        "checks": checks,
    }


def run_stdio_service(*, allowed_tools: Iterable[str] = DEFAULT_ALLOWED_TOOLS) -> int:
    """Run a newline-JSON stdio service for jcode or another local client."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "type": "hermes_service_response",
                "contract_version": HERMES_SERVICE_CONTRACT_VERSION,
                "id": None,
                "ok": False,
                "error": f"failed to parse request JSON: {exc}",
            }
        else:
            response = dispatch_service_request(request, allowed_tools=allowed_tools)
        print(_json(response), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Validate reverse-bridge fixtures and schemas.")

    dispatch_parser = subparsers.add_parser("dispatch", help="Dispatch one request JSON file.")
    dispatch_parser.add_argument("request", help="Path to a service request JSON file.")
    dispatch_parser.add_argument(
        "--allow-tool",
        action="append",
        dest="allow_tools",
        help="Allowed Hermes tool. May be provided multiple times.",
    )

    stdio_parser = subparsers.add_parser("stdio", help="Run the newline-JSON service.")
    stdio_parser.add_argument(
        "--allow-tool",
        action="append",
        dest="allow_tools",
        help="Allowed Hermes tool. May be provided multiple times.",
    )

    ns = parser.parse_args(argv)
    if ns.command == "check":
        report = service_contract_report()
        print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))
        return 0 if report["success"] else 1

    allowed_tools = tuple(ns.allow_tools or DEFAULT_ALLOWED_TOOLS)
    if ns.command == "dispatch":
        request = _load_json(Path(ns.request).expanduser())
        response = dispatch_service_request(request, allowed_tools=allowed_tools)
        print(json.dumps(response, indent=2, ensure_ascii=True, sort_keys=True))
        return 0 if response.get("ok") else 1

    return run_stdio_service(allowed_tools=allowed_tools)


if __name__ == "__main__":
    raise SystemExit(main())
