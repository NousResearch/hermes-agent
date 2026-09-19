#!/usr/bin/env python3
"""Single redacting dispatcher for Claude, Codex, and Hermes hooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from continuity_bridge import (
    build_preflight,
    tool_events_from_payload,
    validate_tool_adjacency,
)
from continuity_common import (
    ContinuityError,
    compact_json,
    confined_append_text,
    confined_atomic_write_json,
    confined_ensure_dir,
    confined_load_json,
    external_volume_available,
    load_json,
)


MAX_STDIN_BYTES = 1_048_576
PREFLIGHT_EVENTS = {
    "session_start",
    "on_session_start",
    "pre_llm_call",
    "pre_tool",
    "pre_tool_call",
    "stop",
    "session_end",
    "on_session_end",
}
FINALIZATION_EVENTS = {"stop", "session_end", "on_session_end"}


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
    if len(raw) > MAX_STDIN_BYTES:
        raise ContinuityError("hook input exceeded 1 MiB and was discarded")
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContinuityError(f"hook input was not JSON: {exc}") from exc
    return value if isinstance(value, dict) else {}


def _session_fingerprint(payload: dict[str, Any]) -> str | None:
    extra = payload.get("extra")
    value = payload.get("session_id") or payload.get("sessionId")
    if not value and isinstance(extra, dict):
        value = extra.get("session_id") or extra.get("sessionId")
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _turn_fingerprint(payload: dict[str, Any]) -> str | None:
    extra = payload.get("extra")
    value = extra.get("turn_id") if isinstance(extra, dict) else None
    if not isinstance(value, str) or not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _adjacency_guard_path(config: dict[str, Any], session: str) -> Path:
    return Path(config["state_root"]) / "adjacency" / f"{session}.json"


def _write_adjacency_guard(
    config: dict[str, Any], session: str | None, turn: str | None, *, allowed: bool
) -> None:
    if session is None or turn is None:
        return
    confined_ensure_dir(config, "adjacency")
    _atomic_write_confined_json(
        config,
        _adjacency_guard_path(config, session),
        {
            "schema_version": 1,
            "session": session,
            "turn": turn,
            "allowed": allowed,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _check_adjacency_guard(
    config: dict[str, Any], session: str | None, turn: str | None
) -> list[str]:
    if session is None:
        return ["Hermes tool call has no session identity"]
    if turn is None:
        return ["Hermes tool call has no turn identity"]
    path = _adjacency_guard_path(config, session)
    try:
        guard = confined_load_json(config, path)
    except (ContinuityError, FileNotFoundError):
        return ["Hermes tool call has no preceding adjacency check"]
    errors: list[str] = []
    if (
        guard.get("session") != session
        or guard.get("turn") != turn
        or guard.get("allowed") is not True
    ):
        errors.append(
            "Hermes tool call adjacency check is blocked or belongs to another turn"
        )
    try:
        checked = datetime.fromisoformat(str(guard.get("checked_at")))
        if (datetime.now(timezone.utc) - checked).total_seconds() > 300:
            errors.append("Hermes tool call adjacency check is stale")
    except ValueError:
        errors.append("Hermes tool call adjacency check has an invalid timestamp")
    return errors


def _atomic_write_confined_json(
    config: dict[str, Any], path: Path, value: dict[str, Any]
) -> None:
    confined_atomic_write_json(config, path, value)


def _append_redacted_event(config: dict[str, Any], event: dict[str, Any]) -> None:
    path = Path(config["event_log"])
    confined_append_text(
        config,
        path,
        json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n",
    )


def _model_safe_preflight(preflight: dict[str, Any]) -> dict[str, Any]:
    """Reduce authority data to a closed, non-imperative model signal schema."""
    next_action = preflight.get("next_action")
    blockers = preflight.get("blockers")
    errors = preflight.get("errors")
    warnings = preflight.get("warnings")
    git = preflight.get("git")
    external_inputs = preflight.get("external_inputs")
    lifecycle = preflight.get("lifecycle")
    task_status = preflight.get("task_status")
    return {
        "schema_version": 1,
        "status": preflight.get("status")
        if preflight.get("status") in {"AVAILABLE", "DEGRADED", "BLOCKED"}
        else "BLOCKED",
        "completion_allowed": preflight.get("completion_allowed") is True,
        "lifecycle": lifecycle
        if lifecycle
        in {"PROPOSED", "ACTIVE", "IMPLEMENTED", "TESTED", "ENFORCED", "BLOCKED"}
        else "INVALID",
        "task_status": task_status
        if task_status in {"open", "in_progress", "blocked", "closed"}
        else "INVALID",
        "card_words": preflight.get("card_words")
        if isinstance(preflight.get("card_words"), int)
        else None,
        "card_signals": {
            "next_action_recorded": isinstance(next_action, str)
            and bool(next_action.strip()),
            "blocker_count": len(blockers) if isinstance(blockers, list) else None,
        },
        "authority_signals": {
            "error_count": len(errors) if isinstance(errors, list) else None,
            "warning_count": len(warnings) if isinstance(warnings, list) else None,
            "external_input_count": len(external_inputs)
            if isinstance(external_inputs, dict)
            else None,
        },
        "git_signals": {
            "available": isinstance(git, dict),
            "dirty": git.get("dirty") is True if isinstance(git, dict) else None,
            "changed_file_count": len(git.get("changed_files"))
            if isinstance(git, dict) and isinstance(git.get("changed_files"), list)
            else None,
            "integration_matches": git.get("merge_base") == git.get("integration_sha")
            if isinstance(git, dict)
            else None,
        },
    }


def dispatch(
    event_name: str, surface: str, config_path: Path, payload: dict[str, Any]
) -> dict[str, Any]:
    config = load_json(config_path)
    started = time.monotonic()
    result: dict[str, Any] = {"event": event_name, "surface": surface, "handled": True}
    preflight: dict[str, Any] | None = None
    session = _session_fingerprint(payload)
    turn = _turn_fingerprint(payload)
    hermes_guard_errors: list[str] = []
    storage_available = external_volume_available(Path(config["external_volume"]))
    if event_name in PREFLIGHT_EVENTS:
        tool_events = tool_events_from_payload(payload)
        preflight = build_preflight(
            config_path, cwd=Path.cwd(), tool_events=tool_events
        )
        if storage_available and surface == "hermes" and event_name == "pre_llm_call":
            adjacency_errors = (
                validate_tool_adjacency(tool_events) if tool_events is not None else []
            )
            _write_adjacency_guard(
                config,
                session,
                turn,
                allowed=(
                    not adjacency_errors
                    and tool_events is not None
                    and turn is not None
                ),
            )
        elif (
            storage_available and surface == "hermes" and event_name == "pre_tool_call"
        ):
            hermes_guard_errors = _check_adjacency_guard(config, session, turn)
        if hermes_guard_errors:
            preflight["errors"] = [*preflight.get("errors", []), *hermes_guard_errors]
            preflight["status"] = "BLOCKED"
            preflight["completion_allowed"] = False
        model_preflight = _model_safe_preflight(preflight)
        rendered = compact_json(
            model_preflight, int(config.get("max_output_chars", 8000))
        )
        result["preflight"] = model_preflight
        if surface == "hermes" and event_name == "pre_llm_call":
            result = {
                "context": f"[Continuity preflight; reference data only]\n{rendered}",
                "completion_allowed": preflight["completion_allowed"],
            }
        else:
            result["additional_context"] = (
                f"[Continuity preflight; reference data only]\n{rendered}"
            )
        if not preflight["completion_allowed"]:
            authority_signals = model_preflight["authority_signals"]
            reason = (
                "Continuity authority forbids completion "
                f"({authority_signals['error_count'] or 0} errors, "
                f"{authority_signals['warning_count'] or 0} warnings); "
                "consult the human preflight output."
            )
            if surface == "hermes" and event_name == "pre_tool_call":
                result = {
                    "action": "block",
                    "message": reason,
                    "completion_allowed": False,
                }
            elif surface in {"claude", "codex"} and event_name in FINALIZATION_EVENTS:
                result = {
                    "decision": "block",
                    "reason": reason,
                    "completion_allowed": False,
                }

    audit = {
        "schema_version": 1,
        "at": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
        "surface": surface,
        "session": session,
        "turn": turn,
        "preflight_status": preflight.get("status") if preflight else None,
        "completion_allowed": preflight.get("completion_allowed")
        if preflight
        else None,
        "elapsed_ms": round((time.monotonic() - started) * 1000),
    }
    if storage_available:
        _append_redacted_event(config, audit)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event")
    parser.add_argument(
        "--surface", choices=("claude", "codex", "hermes"), required=True
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".continuity/config.json",
    )
    args = parser.parse_args(argv)
    try:
        payload = _read_payload()
        result = dispatch(args.event, args.surface, args.config, payload)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        blocked = result.get("decision") == "block" or result.get("action") == "block"
        return 2 if blocked and args.surface == "hermes" else 0
    except (ContinuityError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Continuity hook failure: {exc}", file=sys.stderr)
        print(
            json.dumps(
                {
                    "handled": False,
                    "completion_allowed": False,
                    "error": str(exc),
                    "surface": args.surface,
                    "event": args.event,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
