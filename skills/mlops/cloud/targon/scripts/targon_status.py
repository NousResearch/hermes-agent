#!/usr/bin/env python3
"""Targon Compute Status — check CLI health, active rentals, GPU inventory, and billing.

Usage:
    python targon_status.py
    python targon_status.py --json
    python targon_status.py --inventory-only
    python targon_status.py --rentals-only
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Any


# ── helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        env={**os.environ},
    )


def _check_cli() -> dict[str, Any]:
    """Return CLI availability and version info."""
    path = shutil.which("targon")
    if path is None:
        return {
            "installed": False,
            "path": None,
            "version": None,
            "message": "targon CLI not found. Install with: pip install targon-sdk",
        }
    result = _run(["targon", "--version"])
    version = result.stdout.strip() or result.stderr.strip() or "unknown"
    return {"installed": True, "path": path, "version": version, "message": "OK"}


def _check_api_key() -> dict[str, Any]:
    """Check whether TARGON_API_KEY is set (does not validate against server)."""
    key = os.environ.get("TARGON_API_KEY", "")
    config_path = os.path.expanduser("~/.targon/config")
    config_exists = os.path.isfile(config_path)
    if key:
        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        return {"set": True, "source": "env", "masked": masked}
    if config_exists:
        return {"set": True, "source": "config_file", "masked": "(from ~/.targon/config)"}
    return {
        "set": False,
        "source": None,
        "masked": None,
        "message": "Set TARGON_API_KEY env var or run `targon setup`",
    }


def _list_rentals() -> dict[str, Any]:
    """Parse active rentals from `targon list`."""
    result = _run(["targon", "list"])
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip(), "rentals": []}

    lines = result.stdout.strip().splitlines()
    # Best-effort parse: skip header line, split on whitespace
    rentals = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            rentals.append(
                {
                    "id": parts[0],
                    "gpu": parts[1] if len(parts) > 1 else "?",
                    "status": parts[2] if len(parts) > 2 else "?",
                    "started": " ".join(parts[3:]) if len(parts) > 3 else "",
                }
            )
    return {"ok": True, "raw": result.stdout.strip(), "rentals": rentals}


def _list_inventory() -> dict[str, Any]:
    """Parse GPU inventory from `targon inventory`."""
    result = _run(["targon", "inventory"])
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip(), "gpus": []}

    lines = result.stdout.strip().splitlines()
    gpus = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            gpus.append(
                {
                    "type": parts[0],
                    "available": parts[1] if len(parts) > 1 else "?",
                    "price_per_hour": parts[2] if len(parts) > 2 else "?",
                }
            )
    return {"ok": True, "raw": result.stdout.strip(), "gpus": gpus}


def _check_usage() -> dict[str, Any]:
    """Get billing/credit balance from `targon usage`."""
    result = _run(["targon", "usage"])
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip(), "raw": ""}
    return {"ok": True, "raw": result.stdout.strip()}


# ── display ───────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print("─" * 50)


def _display(cli: dict, api_key: dict, rentals: dict, inventory: dict, usage: dict) -> None:
    # CLI health
    _print_section("CLI Status")
    if cli["installed"]:
        print(f"  [OK] targon CLI found at {cli['path']}")
        print(f"       Version: {cli['version']}")
    else:
        print(f"  [MISSING] {cli['message']}")

    # API key
    _print_section("API Key")
    if api_key["set"]:
        print(f"  [OK] Key detected ({api_key['source']}): {api_key['masked']}")
    else:
        print(f"  [NOT SET] {api_key.get('message', '')}")

    if not cli["installed"]:
        print("\n  Cannot query rentals or inventory — install the CLI first.")
        return

    # Active rentals
    _print_section("Active Rentals")
    if not rentals["ok"]:
        print(f"  [ERROR] {rentals['error']}")
    elif not rentals["rentals"]:
        print("  No active rentals.")
    else:
        print(f"  {'ID':<20} {'GPU':<10} {'STATUS':<12} STARTED")
        for r in rentals["rentals"]:
            print(f"  {r['id']:<20} {r['gpu']:<10} {r['status']:<12} {r['started']}")

    # GPU inventory
    _print_section("GPU Inventory")
    if not inventory["ok"]:
        print(f"  [ERROR] {inventory['error']}")
    elif not inventory["gpus"]:
        print("  No inventory data returned.")
        if inventory.get("raw"):
            print(f"\n  Raw output:\n{inventory['raw']}")
    else:
        print(f"  {'GPU TYPE':<16} {'AVAILABLE':<12} PRICE/HR")
        for g in inventory["gpus"]:
            print(f"  {g['type']:<16} {g['available']:<12} {g['price_per_hour']}")

    # Billing
    _print_section("Billing / Usage")
    if not usage["ok"]:
        print(f"  [ERROR] {usage['error']}")
    elif usage["raw"]:
        for line in usage["raw"].splitlines():
            print(f"  {line}")
    else:
        print("  No usage data available.")

    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check Targon CLI health, active rentals, GPU inventory, and billing."
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--inventory-only", action="store_true", help="Show only GPU inventory")
    parser.add_argument("--rentals-only", action="store_true", help="Show only active rentals")
    args = parser.parse_args()

    cli = _check_cli()
    api_key = _check_api_key()

    if cli["installed"]:
        rentals = _list_rentals()
        inventory = _list_inventory()
        usage = _check_usage()
    else:
        rentals = {"ok": False, "error": "CLI not installed", "rentals": []}
        inventory = {"ok": False, "error": "CLI not installed", "gpus": []}
        usage = {"ok": False, "error": "CLI not installed", "raw": ""}

    if args.json:
        output = {
            "cli": cli,
            "api_key": api_key,
            "rentals": rentals,
            "inventory": inventory,
            "usage": usage,
        }
        print(json.dumps(output, indent=2))
        return

    if args.inventory_only:
        _print_section("GPU Inventory")
        if inventory["ok"] and inventory["gpus"]:
            print(f"  {'GPU TYPE':<16} {'AVAILABLE':<12} PRICE/HR")
            for g in inventory["gpus"]:
                print(f"  {g['type']:<16} {g['available']:<12} {g['price_per_hour']}")
        elif inventory["ok"]:
            print(f"\n  Raw output:\n{inventory.get('raw', '')}")
        else:
            print(f"  [ERROR] {inventory['error']}")
        print()
        return

    if args.rentals_only:
        _print_section("Active Rentals")
        if rentals["ok"] and rentals["rentals"]:
            print(f"  {'ID':<20} {'GPU':<10} {'STATUS':<12} STARTED")
            for r in rentals["rentals"]:
                print(f"  {r['id']:<20} {r['gpu']:<10} {r['status']:<12} {r['started']}")
        elif rentals["ok"]:
            print("  No active rentals.")
        else:
            print(f"  [ERROR] {rentals['error']}")
        print()
        return

    _display(cli, api_key, rentals, inventory, usage)

    # Exit with non-zero if CLI is missing or API key is unset
    if not cli["installed"] or not api_key["set"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
