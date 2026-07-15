"""Non-interactive account usage command used by local status consumers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from agent.account_usage import (
    account_usage_snapshot_dict,
    fetch_account_usage,
    render_account_usage_lines,
)
from hermes_cli.auth import AuthError, resolve_provider
from hermes_cli.runtime_provider import resolve_requested_provider


def _effective_provider(explicit: str | None) -> str:
    requested = resolve_requested_provider(explicit)
    try:
        return resolve_provider(requested)
    except AuthError:
        return requested


def _write_json_atomic(path: Path, payload: str) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def account_usage_command(args) -> int:
    provider = _effective_provider(getattr(args, "provider", None))
    snapshot = fetch_account_usage(provider)
    if snapshot is None:
        message = {
            "schemaVersion": 1,
            "provider": provider,
            "unavailableReason": "Account usage is unavailable for this provider or credential type.",
        }
        if getattr(args, "json", False) or getattr(args, "output", None):
            rendered = json.dumps(message, separators=(",", ":"), sort_keys=True)
            if getattr(args, "output", None):
                _write_json_atomic(Path(args.output), rendered)
            if getattr(args, "json", False):
                print(rendered)
        else:
            print(message["unavailableReason"])
        return 2

    payload = account_usage_snapshot_dict(snapshot)
    rendered = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    if getattr(args, "output", None):
        _write_json_atomic(Path(args.output), rendered)
    if getattr(args, "json", False):
        print(rendered)
    elif not getattr(args, "output", None):
        for line in render_account_usage_lines(snapshot):
            print(line)
    return 0
