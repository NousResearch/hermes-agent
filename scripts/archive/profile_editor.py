#!/usr/bin/env python3
"""
Denji Profile Editor — governed profile mutation with git-backed rollback (WS-7).

Usage:
  profile_editor.py <profile> <file> <key_path> <new_value> <reason>

  profile    — profile name (e.g. "octacon", not the full path)
  file       — one of: config.yaml, SOUL.md, USER.md
  key_path   — dot-separated YAML key path for config.yaml edits
               (e.g. "agent.reasoning_effort"); ignored for .md files
  new_value  — the new value to set
  reason     — human-readable reason for the change (ledger entry)

Examples:
  python3 profile_editor.py octacon config.yaml agent.reasoning_effort high \
    "Octacon consistently fails complex tasks at medium effort"

  python3 profile_editor.py light SOUL.md "" "Updated role description" \
    "Light's SOUL was missing document-indexing scope"

Output:
  JSON on stdout: {"ok": true, "commit": "<sha>", "ledger_entry": "<line>", "profile": "..."}
  Exit 0 on success, non-zero on failure.

Rollback (separate entry point):
  python3 profile_editor.py --rollback <commit_hash>

  Reverts the commit and appends a rollback entry to the ledger.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/home/kensei/.hermes"))
PROFILES_DIR = HERMES_HOME / "profiles"
LEDGER = HERMES_HOME / "governance" / "profile-change-ledger.md"
TZ = datetime.now().astimezone().tzinfo  # system local timezone


def _git(*args, **kwargs):
    """Run a git command in the profiles directory."""
    return subprocess.run(
        ["git"] + list(args),
        cwd=str(PROFILES_DIR),
        capture_output=True,
        text=True,
        **kwargs,
    )


def _commit_hash():
    """Get the latest commit hash."""
    r = _git("rev-parse", "HEAD")
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def _ledger_append(date: str, profile: str, trigger: str, change: str, follow_up: str) -> None:
    """Append a row to the profile-change-ledger.MD table."""
    row = f"| {date} | {profile} | {trigger} | {change} | {follow_up} | Pending |\n"
    with open(LEDGER, "a") as f:
        f.write(row)


def edit(args: list[str]) -> dict:
    """Main edit operation."""
    if len(args) < 5:
        return {"ok": False, "error": "usage: profile_editor.py <profile> <file> <key_path> <new_value> <reason>"}

    # WS-7 safety gates: feature flag + blast-radius cap
    # Feature flag must be explicitly enabled in config.yaml
    config_path = HERMES_HOME / "config.yaml"
    try:
        import yaml as _yaml
        with open(config_path) as _f:
            _cfg = _yaml.safe_load(_f) or {}
        autonomous = (_cfg.get("governance", {}) or {}).get("autonomous_edits_enabled", False)
        if not autonomous:
            return {"ok": False, "error": "autonomous edits disabled — set governance.autonomous_edits_enabled: true in config.yaml to enable"}
    except Exception:
        return {"ok": False, "error": "cannot read config.yaml to check autonomous_edits_enabled flag"}

    # Blast-radius cap: max 3 profile edits per 24h window
    import re as _re
    import time as _time
    now = _time.time()
    try:
        r = subprocess.run(
            ["git", "log", "--oneline", "--since=24 hours ago", "--author=denji"],
            cwd=str(PROFILES_DIR), capture_output=True, text=True, timeout=10,
        )
        recent_count = len(r.stdout.strip().splitlines()) if r.stdout.strip() else 0
    except Exception:
        recent_count = 0
    if recent_count >= 3:
        return {"ok": False, "error": f"blast-radius cap: {recent_count} edits in trailing 24h (max 3). Try again later."}

    profile_name = args[0]
    file_name = args[1]
    key_path = args[2]
    new_value = args[3]
    reason = " ".join(args[4:]) if len(args) > 4 else "No reason provided"

    # Validate profile exists
    profile_dir = PROFILES_DIR / profile_name
    if not profile_dir.is_dir():
        return {"ok": False, "error": f"profile '{profile_name}' not found at {profile_dir}"}

    # Validate file
    if file_name not in ("config.yaml", "SOUL.md", "USER.md"):
        return {"ok": False, "error": f"file must be one of: config.yaml, SOUL.md, USER.md (got {file_name})"}
    target_file = profile_dir / file_name
    if not target_file.exists():
        return {"ok": False, "error": f"{file_name} not found in profile '{profile_name}'"}

    # Read current content
    old_content = target_file.read_text()
    old_hash = _commit_hash()

    # Apply the change
    if file_name.endswith(".yaml") and key_path:
        new_content = _edit_yaml_key(old_content, key_path, new_value)
    elif file_name.endswith(".md"):
        # For .MD files, key_path is ignored; the new_value replaces or appends.
        # If the file has a specific section to update, key_path can name it.
        new_content = old_content  # MD edits are full-replace for v1
    else:
        return {"ok": False, "error": f"cannot determine edit strategy for {file_name}/{key_path}"}

    if new_content == old_content:
        return {"ok": False, "error": "no change detected (value already matches)"}

    # Write the change
    target_file.write_text(new_content)

    # Git commit
    change_desc = f"{file_name} — {key_path}: {new_value}" if key_path else f"{file_name} updated"
    _git("add", str(target_file.relative_to(PROFILES_DIR)))
    commit_msg = f"denji: {profile_name} {change_desc}\n\nReason: {reason}"
    r = _git("commit", "-m", commit_msg)
    if r.returncode != 0:
        # Rollback the file write
        target_file.write_text(old_content)
        return {"ok": False, "error": f"git commit failed: {r.stderr}"}

    new_hash = _commit_hash()

    # Append to ledger
    now = datetime.now(TZ)
    date_str = now.strftime("%d/%m/%y")
    follow_up = (now + timedelta(weeks=2)).strftime("%d/%m/%y")
    _ledger_append(date_str, profile_name, f"Denji WS-7 — {reason[:60]}", change_desc, follow_up)

    return {
        "ok": True,
        "commit": new_hash,
        "previous_commit": old_hash,
        "profile": profile_name,
        "file": file_name,
        "change": change_desc,
        "reason": reason,
        "follow_up": follow_up,
        "ledger_entry": change_desc,
    }


def rollback(args: list[str]) -> dict:
    """Revert a specific commit."""
    if not args:
        return {"ok": False, "error": "usage: profile_editor.py --rollback <commit_hash>"}
    commit = args[0]

    r = _git("revert", "--no-edit", commit)
    if r.returncode != 0:
        return {"ok": False, "error": f"git revert failed: {r.stderr}"}

    new_hash = _commit_hash()
    now = datetime.now(TZ).strftime("%d/%m/%y")
    _ledger_append(now, "(rollback)", f"Denji rollback of {commit[:8]}", f"Reverted commit {commit[:12]}", now)

    return {"ok": True, "commit": new_hash, "reverted": commit[:12]}


def _edit_yaml_key(content: str, key_path: str, new_value: str) -> str:
    """Change a dot-separated key in YAML content. Supports top-level and nested keys."""
    import yaml

    data = yaml.safe_load(content) or {}
    keys = key_path.split(".")
    target = data
    for k in keys[:-1]:
        if k not in target:
            target[k] = {}
        target = target[k]

    last_key = keys[-1]
    old_value = target.get(last_key)
    if old_value == new_value:
        return content

    # Parse the new value — try int, bool, str
    lv = new_value.lower()
    if lv == "true":
        parsed = True
    elif lv == "false":
        parsed = False
    elif lv == "null" or lv == "none":
        parsed = None
    else:
        try:
            parsed = int(new_value)
        except ValueError:
            parsed = new_value

    target[last_key] = parsed

    return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)


def main() -> None:
    if "--rollback" in sys.argv:
        idx = sys.argv.index("--rollback")
        result = rollback(sys.argv[idx + 1:])
    else:
        result = edit(sys.argv[1:])

    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
