"""Explicit MCP snapshots and offline compatibility reports for ``hermes mcp``."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli.mcp_schema_diff import compare_manifests, validate_manifest

_MAX_BYTES = 8 * 1024 * 1024
_MAX_DEPTH = 64
_EXIT_CODES = {"unchanged": 0, "compatible": 0, "breaking": 1, "review": 3}


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}.")


def _check_depth(data: object) -> None:
    pending = [(data, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > _MAX_DEPTH:
            raise ValueError(f"Snapshot nesting exceeds {_MAX_DEPTH} levels.")
        children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
        pending.extend((child, depth + 1) for child in children)


def _read_snapshot(path: str) -> dict:
    with Path(path).expanduser().open("rb") as stream:
        raw = stream.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError("Snapshot exceeds the 8 MiB limit.")
    data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    _check_depth(data)
    return validate_manifest(data)


def _snapshot(args) -> int:
    from hermes_cli.config_effective import load_user_config_effective
    from hermes_cli.mcp_config import _probe_single_server
    from utils import atomic_json_write

    servers = load_user_config_effective().get("mcp_servers") or {}
    if not isinstance(servers, dict) or args.name not in servers:
        raise ValueError(f"Server {args.name!r} is not configured in the active profile.")
    config = servers[args.name]
    if not isinstance(config, dict):
        raise ValueError("The configured server entry must be an object.")
    definitions: list[dict] = []
    try:
        _probe_single_server(args.name, config, tool_manifest=definitions)
    except Exception as exc:
        # Transport/SDK exception types vary; normalize only this external I/O boundary.
        raise ValueError(f"Could not capture tool definitions ({type(exc).__name__}): {exc}") from None
    data = {"version": 1, "server": args.name, "captured_at": datetime.now(timezone.utc).isoformat(),
            "tools": sorted(definitions, key=lambda t: t["name"])}
    _check_depth(data)
    validate_manifest(data)
    if len(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False).encode("utf-8")) > _MAX_BYTES:
        raise ValueError("Tool definitions exceed the 8 MiB snapshot limit.")
    # Save only after the complete live probe succeeds. Config, tokens, and the runtime
    # schema cache are not a baseline and must never be copied into the manifest.
    atomic_json_write(Path(args.output).expanduser(), data, mode=0o600, ensure_ascii=True, allow_nan=False)
    print(f"Saved {len(definitions)} tool definitions for {args.name!r} to {args.output!r}.")
    return 0


def _diff(args) -> int:
    report = compare_manifests(_read_snapshot(args.baseline), _read_snapshot(args.current))
    if args.json:
        print(json.dumps(report, ensure_ascii=True, allow_nan=False, indent=2))
    else:
        print(f"MCP compatibility for {report['server']!r}: {report['status']}")
        for change in report["changes"]:
            print(f"  [{change['severity']}] {change['tool']!r} {change['path']!r}: {change['message']}")
        print("Breaking means a potential incompatibility; review means the change needs manual assessment.")
        print("This compares declared contracts, not server behavior or provider acceptance.")
    return _EXIT_CODES[report["status"]]


def run_contract_command(args) -> None:
    """Keep diagnostics out of JSON stdout and distinguish errors from change verdicts."""
    from hermes_cli.mcp_config import redact_mcp_probe_text

    try:
        code = {"snapshot": _snapshot, "diff": _diff}[args.mcp_action](args)
    except (OSError, ValueError, RecursionError) as exc:
        message = redact_mcp_probe_text(exc)
        if getattr(args, "json", False):
            print(json.dumps({"version": 1, "status": "error", "error": message}, ensure_ascii=True))
        else:
            print(f"MCP {args.mcp_action} failed: {message}", file=sys.stderr)
        code = 2
    if code:
        raise SystemExit(code)
