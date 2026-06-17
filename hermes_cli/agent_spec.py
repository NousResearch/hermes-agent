"""CLI handler for read-only ``hermes agent-spec`` commands."""

from __future__ import annotations

import sys

from agent.agent_spec import list_profile_specs, preview_agent_spec, render_json, render_text, validate_spec_path


def _emit(payload: dict, *, as_json: bool) -> None:
    if as_json:
        print(render_json(payload), end="")
    else:
        print(render_text(payload), end="")


def _exit_code(status: str, *, strict: bool = False, warnings: list | None = None) -> int:
    if status == "fail" or (strict and warnings):
        return 1
    return 0


def cmd_agent_spec(args) -> None:
    command = getattr(args, "agent_spec_command", None)
    as_json = bool(getattr(args, "json", False))
    try:
        if command == "validate":
            payload = validate_spec_path(
                getattr(args, "path_or_id"),
                strict=bool(getattr(args, "strict", False)),
                profile_id=getattr(args, "profile", None),
            )
            _emit(payload, as_json=as_json)
            sys.exit(_exit_code(payload.get("status", "fail"), strict=bool(getattr(args, "strict", False)), warnings=payload.get("warnings", [])))
        if command == "preview":
            payload = preview_agent_spec(
                getattr(args, "profile"),
                spec_path=getattr(args, "spec", None),
                strict=bool(getattr(args, "strict", False)),
            )
            _emit(payload, as_json=as_json)
            sys.exit(_exit_code(payload.get("status", "fail"), strict=bool(getattr(args, "strict", False)), warnings=payload.get("warnings", [])))
        if command == "list":
            payload = list_profile_specs()
            _emit(payload, as_json=as_json)
            sys.exit(0)
        print("usage: hermes agent-spec {validate,preview,list} ...", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError as exc:
        payload = {"status": "fail", "errors": [{"code": "file_not_found", "message": str(exc)}], "warnings": [], "read_only_guarantee": True}
        _emit(payload, as_json=as_json)
        sys.exit(1)
    except Exception as exc:
        payload = {"status": "fail", "errors": [{"code": "agent_spec_error", "message": str(exc)}], "warnings": [], "read_only_guarantee": True}
        _emit(payload, as_json=as_json)
        sys.exit(1)
