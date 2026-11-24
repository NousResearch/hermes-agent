#!/usr/bin/env python3
"""Verify Discord slash command fingerprint stability.

This script computes what the fingerprint would be for the current
command manifest and compares it with the stored sync state.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from plugins.platforms.discord.discord_command_manifest import build_native_command_manifest


def _canonicalize_payload(payload: dict) -> dict:
    """Standalone canonicalization matching adapter logic."""
    contexts = payload.get("contexts")
    integration_types = payload.get("integration_types")
    return {
        "type": int(payload.get("type", 1) or 1),
        "name": str(payload.get("name", "") or ""),
        "description": str(payload.get("description", "") or ""),
        "default_member_permissions": str(payload.get("default_member_permissions")) if payload.get("default_member_permissions") is not None else None,
        "dm_permission": bool(payload.get("dm_permission", True)),
        "nsfw": bool(payload.get("nsfw", False)),
        "contexts": sorted(int(c) for c in contexts) if contexts else None,
        "integration_types": sorted(int(i) for i in integration_types) if integration_types else None,
        "options": [
            _canonicalize_option(item)
            for item in payload.get("options", []) or []
            if isinstance(item, dict)
        ],
    }


def _canonicalize_option(payload: dict) -> dict:
    return {
        "type": int(payload.get("type", 0) or 0),
        "name": str(payload.get("name", "") or ""),
        "description": str(payload.get("description", "") or ""),
        "required": bool(payload.get("required", False)),
        "autocomplete": bool(payload.get("autocomplete", False)),
        "choices": [
            {"name": str(choice.get("name", "") or ""), "value": choice.get("value")}
            for choice in payload.get("choices", []) or []
            if isinstance(choice, dict)
        ],
        "channel_types": list(payload.get("channel_types", []) or []),
        "min_value": payload.get("min_value"),
        "max_value": payload.get("max_value"),
        "min_length": payload.get("min_length"),
        "max_length": payload.get("max_length"),
        "options": [
            _canonicalize_option(item)
            for item in payload.get("options", []) or []
            if isinstance(item, dict)
        ],
    }


def compute_manifest_fingerprint(language: str = "en") -> str:
    """Compute fingerprint from the stable manifest (native commands only)."""
    manifest = build_native_command_manifest()
    
    # Convert manifest to canonical payload format
    desired = []
    for cmd in manifest:
        # Build a mock payload similar to what Discord would have
        payload = {
            "type": 1,  # CHAT_INPUT
            "name": cmd["name"],
            "description": cmd["description"],
            "options": cmd.get("options", []),
        }
        canonical = _canonicalize_payload(payload)
        desired.append(canonical)
    
    desired.sort(key=lambda item: (item.get("type", 1), item.get("name", "")))
    payload = json.dumps({"language": language, "commands": desired}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_sync_state() -> dict:
    """Load the current Discord sync state from Hermes home."""
    import os
    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    state_path = hermes_home / "gateway" / "discord_command_sync_state.json"
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {}


def main():
    print("=== Discord Command Fingerprint Verification ===\n")
    
    # Compute manifest fingerprint
    manifest_fp = compute_manifest_fingerprint("en")
    print(f"Manifest fingerprint (en): {manifest_fp}")
    
    # Load stored sync state
    sync_state = load_sync_state()
    if sync_state:
        print("\nStored sync state:")
        for app_id, entry in sync_state.items():
            stored_fp = entry.get("fingerprint", "N/A")
            print(f"  App {app_id}:")
            print(f"    Stored fingerprint: {stored_fp}")
            print(f"    Match: {'✓' if stored_fp == manifest_fp else '✗'}")
            if "last_success_at" in entry:
                import datetime
                dt = datetime.datetime.fromtimestamp(entry["last_success_at"])
                print(f"    Last success: {dt}")
            if "summary" in entry:
                print(f"    Last sync: {entry['summary']}")
    else:
        print("\nNo stored sync state found.")
    
    print("\n=== Manifest Commands ===")
    manifest = build_native_command_manifest()
    for cmd in manifest:
        opts = cmd.get("options", [])
        opt_str = f" ({len(opts)} options)" if opts else ""
        print(f"  /{cmd['name']}: {cmd['description'][:60]}{opt_str}")


if __name__ == "__main__":
    main()