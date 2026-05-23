"""Compatibility contract helpers for the Hermes <-> jcode bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


BRIDGE_CONTRACT_VERSION = "jcode-bridge.v1"
BRIDGE_SCHEMA_RELATIVE_DIR = Path("contracts") / "jcode_bridge" / "v1"
BRIDGE_SCHEMA_FILENAMES = (
    "debug_command.schema.json",
    "debug_response.schema.json",
    "run_json.schema.json",
    "run_ndjson_event.schema.json",
    "run_ndjson_stream.schema.json",
    "upstream_sync_report.schema.json",
)
FINAL_TEXT_KEYS = ("text", "final_response", "response", "content")


@dataclass(frozen=True)
class ContractValidation:
    ok: bool
    errors: list[str]


def _ok() -> ContractValidation:
    return ContractValidation(ok=True, errors=[])


def _fail(*errors: str) -> ContractValidation:
    return ContractValidation(ok=False, errors=[error for error in errors if error])


def bridge_schema_dir(repo_root: Path | None = None) -> Path:
    """Return the repo-local directory for portable bridge schema artifacts."""
    root = repo_root if repo_root is not None else Path(__file__).resolve().parents[2]
    return root / BRIDGE_SCHEMA_RELATIVE_DIR


def final_text_from_mapping(payload: dict[str, Any]) -> str | None:
    """Return the first supported final-text field from a jcode payload."""
    for key in FINAL_TEXT_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def validate_run_json_payload(payload: Any) -> ContractValidation:
    """Validate the wrapper-oriented `jcode run --json` result."""
    if not isinstance(payload, dict):
        return _fail("run_json payload must be an object")

    errors: list[str] = []
    text = final_text_from_mapping(payload)
    if not isinstance(text, str):
        errors.append(
            "run_json payload must include a string final response field "
            f"({', '.join(FINAL_TEXT_KEYS)})"
        )

    for key in ("session_id", "provider", "model"):
        value = payload.get(key)
        if value is not None and not isinstance(value, str):
            errors.append(f"run_json field '{key}' must be a string when present")

    usage = payload.get("usage")
    if usage is not None and not isinstance(usage, dict):
        errors.append("run_json field 'usage' must be an object when present")

    return _ok() if not errors else ContractValidation(ok=False, errors=errors)


def validate_run_ndjson_events(events: Any) -> ContractValidation:
    """Validate the wrapper-oriented `jcode run --ndjson` event stream."""
    if not isinstance(events, list):
        return _fail("run_ndjson payload must be a list of event objects")
    if not events:
        return _fail("run_ndjson payload must contain at least one event")

    errors: list[str] = []
    done_events: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append(f"run_ndjson event {index} must be an object")
            continue
        event_type = event.get("type")
        if not isinstance(event_type, str) or not event_type:
            errors.append(f"run_ndjson event {index} must include string field 'type'")
        if event_type == "done":
            done_events.append(event)

    if not done_events:
        errors.append("run_ndjson payload must include a final event with type 'done'")
    elif not isinstance(final_text_from_mapping(done_events[-1]), str):
        errors.append("run_ndjson final done event must include string field 'text'")

    return _ok() if not errors else ContractValidation(ok=False, errors=errors)


def make_debug_command_request(
    command: str,
    *,
    request_id: int,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build the newline-JSON debug socket request Hermes sends to jcode."""
    request: dict[str, Any] = {
        "type": "debug_command",
        "id": request_id,
        "command": command,
    }
    if session_id:
        request["session_id"] = session_id
    return request


def validate_debug_command_request(payload: Any) -> ContractValidation:
    """Validate the debug socket request envelope sent by the bridge."""
    if not isinstance(payload, dict):
        return _fail("debug command request must be an object")

    errors: list[str] = []
    if payload.get("type") != "debug_command":
        errors.append("debug command request type must be 'debug_command'")
    if not isinstance(payload.get("id"), int):
        errors.append("debug command request id must be an integer")
    if not isinstance(payload.get("command"), str) or not payload.get("command"):
        errors.append("debug command request command must be a non-empty string")
    session_id = payload.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        errors.append("debug command request session_id must be a string when present")

    return _ok() if not errors else ContractValidation(ok=False, errors=errors)


def validate_debug_response_payload(payload: Any) -> ContractValidation:
    """Validate the debug socket response envelope returned by jcode."""
    if not isinstance(payload, dict):
        return _fail("debug response must be an object")

    if payload.get("type") == "error":
        message = payload.get("message")
        if isinstance(message, str) and message:
            return _ok()
        return _fail("debug error response must include string field 'message'")

    errors: list[str] = []
    if payload.get("type") != "debug_response":
        errors.append("debug response type must be 'debug_response'")
    if not isinstance(payload.get("ok"), bool):
        errors.append("debug response field 'ok' must be a boolean")
    output = payload.get("output")
    if output is not None and not isinstance(output, str):
        errors.append("debug response field 'output' must be a string when present")
    message = payload.get("message")
    if message is not None and not isinstance(message, str):
        errors.append("debug response field 'message' must be a string when present")

    return _ok() if not errors else ContractValidation(ok=False, errors=errors)
