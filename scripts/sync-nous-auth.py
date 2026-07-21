#!/usr/bin/env python3
"""
Sync Nous Portal credentials from ~/.hermes/auth.json to every profile that
declares `provider: nous` in its config.yaml.

Part of the hermes-agent core install.  Maintained in the hermes-agent repo at
scripts/sync-nous-auth.py; mirrored into ~/.hermes/scripts/ on install.

Usage:  python sync-nous-auth.py           # dry-run
        python sync-nous-auth.py --write    # update files
"""
import json, sys, os
from pathlib import Path

HERMES = Path.home() / ".hermes"
MAIN_AUTH = HERMES / "auth.json"
DRY = "--write" not in sys.argv

def load(path): return json.loads(path.read_text()) if path.exists() else {}

def sync():
    main = load(MAIN_AUTH)
    nous_cred = main.get("providers", {}).get("nous") or main.get("nous")
    if not nous_cred or not nous_cred.get("access_token"):
        print("WARNING: no Nous credential found in ~/.hermes/auth.json"); return

    patched = []
    for cfg in sorted((HERMES / "profiles").glob("*/config.yaml")):
        if "provider: nous" not in cfg.read_text(): continue
        pdir = cfg.parent
        auth_path = pdir / "auth.json"
        existing = load(auth_path) if auth_path.exists() else {"version": 1, "providers": {}}
        old_token = existing.get("providers", {}).get("nous", {}).get("access_token", "")
        new_token = nous_cred["access_token"]
        if not old_token or old_token != new_token:
            existing.setdefault("providers", {})["nous"] = nous_cred.copy()
            patched.append(str(auth_path))
            if not DRY:
                auth_path.write_text(json.dumps(existing, indent=2) + "\n")
                print(f"  → wrote {auth_path.relative_to(HERMES)}")

    if patched:
        action = "would patch" if DRY else "patched"
        ts = "\n  ".join(patched)
        print(f"{action} {len(patched)} profile(s):\n  {ts}")
    if DRY and patched:
        print("\nDry-run: add --write to apply changes")
    else:
        print("All nous profiles already have the latest token.")

if __name__ == "__main__":
    print("DRY RUN — add --write to actually update files\n" if DRY else "")
    sync()
