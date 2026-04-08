#!/usr/bin/env python3
"""Healthcheck/recovery helper for canonical Obsidian sync pipeline."""

from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_API_BASE = "https://127.0.0.1:27124"
DEFAULT_LEDGER = Path.home() / ".hermes" / "state" / "obsidian_bookmark_ledger.json"
DEFAULT_KEY_RELATIVE = Path(".obsidian/plugins/obsidian-local-rest-api/data.json")


def _tcp_probe(api_base: str, timeout: float = 1.5) -> bool:
    parsed = urlparse(api_base)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _key_file_path(vault_path: str | None, explicit_key_file: str | None) -> Path | None:
    if explicit_key_file:
        return Path(explicit_key_file).expanduser()
    if vault_path:
        return Path(vault_path).expanduser() / DEFAULT_KEY_RELATIVE
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Healthcheck for Obsidian canonical pipeline")
    parser.add_argument("--api-base", default=os.getenv("OBSIDIAN_REST_URL", DEFAULT_API_BASE))
    parser.add_argument("--vault-path", default=os.getenv("OBSIDIAN_VAULT"))
    parser.add_argument("--key-file", default=os.getenv("OBSIDIAN_REST_KEY_FILE"))
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    parser.add_argument("--repair", action="store_true", help="Create missing ledger parent/file")
    args = parser.parse_args()

    key_file = _key_file_path(args.vault_path, args.key_file)
    ledger_path = Path(args.ledger).expanduser()

    report = {
        "api_base": args.api_base,
        "tcp_reachable": _tcp_probe(args.api_base),
        "vault_path": args.vault_path,
        "key_file": str(key_file) if key_file else None,
        "key_file_exists": bool(key_file and key_file.exists()),
        "api_key_env_present": bool(os.getenv("OBSIDIAN_API_KEY", "").strip()),
        "ledger_path": str(ledger_path),
        "ledger_exists": ledger_path.exists(),
        "repair_applied": False,
    }

    if args.repair:
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        if not ledger_path.exists():
            ledger_path.write_text("{}\n", encoding="utf-8")
        report["ledger_exists"] = ledger_path.exists()
        report["repair_applied"] = True

    print(json.dumps(report, indent=2))

    healthy = report["tcp_reachable"] and (report["key_file_exists"] or report["api_key_env_present"])
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
