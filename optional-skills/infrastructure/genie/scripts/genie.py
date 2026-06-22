#!/usr/bin/env python3
"""
Genie — VPS Disk Guardian
Safe cleanup of Hermes VPS disk space without reducing functionality.

Tiers:
  Tier 1 (zero risk):  State snapshots, log rotation, cron output compression
  Tier 2 (low risk):   Session JSON compression (duplicates of state.db data)
  Tier 3 (read-only):  state.db bloat analysis, commons/ analysis

Usage:
  python3 genie.py --assess                    # Show current disk state and targets
  python3 genie.py --assess --tier 3           # Include Tier 3 analysis
  python3 genie.py --clean                     # Execute Tier 1+2 cleanup
  python3 genie.py --clean --tier 1            # Only Tier 1
  python3 genie.py --clean --dry-run           # Preview without modifying anything
  python3 genie.py --analyze                   # Tier 3 analysis only
  python3 genie.py --discover                  # Map filesystem, create/update FILESYSTEM.md
  python3 genie.py --json                      # Output as JSON

FILESYSTEM.md integration:
  If FILESYSTEM.md exists at ~/.hermes/profiles/<profile>/references/FILESYSTEM.md
  (or the path in GENIE_FILESYSTEM_MD), genie reads the cleanup-manifest section
  and merges it with built-in defaults. Discovered paths extend — never replace —
  the built-in knowledge.

  If FILESYSTEM.md does not exist, genie --discover walks the filesystem,
  creates it with a cleanup-manifest pre-populated from what it found,
  then proceeds with normal assessment/cleanup.

  If FILESYSTEM.md exists but has no cleanup-manifest, genie adds one
  based on built-in defaults + anything new discovered at runtime.
"""

import argparse
import copy
import datetime
import gzip
import json
import os
import re
import shutil
import sqlite3

# ── Configuration ──────────────────────────────────────────────────────────

HERMES_HOME = os.environ.get("HERMES_HOME", "/root/.hermes")
PROFILE = os.environ.get("HERMES_PROFILE", "indigo")

# Handle the case where HERMES_HOME already includes the profile path
# (e.g., HERMES_HOME=/root/.hermes/profiles/indigo when running in profile context)
if HERMES_HOME.endswith(os.path.join("profiles", PROFILE)):
    HERMES_HOME = os.path.dirname(os.path.dirname(HERMES_HOME))

PROFILE_HOME = os.path.join(HERMES_HOME, "profiles", PROFILE)

# FILESYSTEM.md locations to check (in order)
FILESYSTEM_MD_PATHS = [
    os.environ.get("GENIE_FILESYSTEM_MD", ""),
    os.path.join(PROFILE_HOME, "references", "FILESYSTEM.md"),
    os.path.join(HERMES_HOME, "references", "FILESYSTEM.md"),
    os.path.join(HERMES_HOME, "FILESYSTEM.md"),
]

DEFAULTS = {
    "snapshot_max_age_days": int(os.environ.get("GENIE_SNAPSHOT_MAX_AGE_DAYS", 7)),
    "log_compress_age_days": int(os.environ.get("GENIE_LOG_COMPRESS_AGE_DAYS", 7)),
    "log_delete_age_days": int(os.environ.get("GENIE_LOG_DELETE_AGE_DAYS", 30)),
    "cron_output_compress_age_days": int(os.environ.get("GENIE_CRON_OUTPUT_COMPRESS_AGE_DAYS", 7)),
    "session_compress_age_days": int(os.environ.get("GENIE_SESSION_COMPRESS_AGE_DAYS", 14)),
    "dry_run": os.environ.get("GENIE_DRY_RUN", "false").lower() == "true",
    "tier_limit": os.environ.get("GENIE_TIER_LIMIT", "3"),
    "state_db_path": os.environ.get("GENIE_STATE_DB_PATH", os.path.join(PROFILE_HOME, "state.db")),
    "sessions_path": os.environ.get("GENIE_SESSIONS_PATH", os.path.join(PROFILE_HOME, "sessions")),
    "logs_path": os.environ.get("GENIE_LOGS_PATH", os.path.join(PROFILE_HOME, "logs")),
    "cron_output_path": os.environ.get("GENIE_CRON_OUTPUT_PATH", os.path.join(PROFILE_HOME, "cron-output")),
    "snapshots_path": os.environ.get("GENIE_SNAPSHOTS_PATH", os.path.join(PROFILE_HOME, "state-snapshots")),
    "commons_path": os.environ.get("GENIE_COMMONS_PATH", os.path.join(PROFILE_HOME, "commons")),
    "backups_path": os.environ.get("GENIE_BACKUPS_PATH", "/root/backups"),
    "tmp_stale_hours": int(os.environ.get("GENIE_TMP_STALE_HOURS", 24)),
}

# Built-in cleanup targets: path → {tier, action, max_age_days, ...}
# These are the defaults. FILESYSTEM.md manifest EXTENDS this, never replaces.
BUILTIN_TARGETS = {
    "state-snapshots": {
        "tier": 1, "action": "delete_dirs",
        "max_age_days": 7, "path": "state-snapshots",
        "description": "Pre-update rollback snapshots",
        "pattern": "state-snapshots/*/",
    },
    "logs": {
        "tier": 1, "action": "compress_then_delete",
        "compress_age_days": 7, "delete_compressed_age_days": 30,
        "path": "logs", "pattern": "logs/*.log",
        "description": "Log files",
    },
    "cron-output": {
        "tier": 1, "action": "compress",
        "max_age_days": 7, "path": "cron-output",
        "pattern": "cron-output/*",
        "description": "Cron job output files",
    },
    "session-jsons": {
        "tier": 2, "action": "compress",
        "max_age_days": 14, "path": "sessions",
        "pattern": "sessions/*.json",
        "description": "Session JSON files (duplicates of state.db)",
        "requires_confirmation": True,
    },
    "backups": {
        "tier": 1, "action": "delete_dirs",
        "max_age_days": 30, "path": "/root/backups",
        "pattern": "/root/backups/*/",
        "description": "Dated backup directories",
    },
    "tmp": {
        "tier": 1, "action": "delete_files",
        "max_age_hours": 24, "path": "/tmp",
        "pattern": "/tmp/*",
        "description": "Stale temp files",
    },
    "cache-pip": {
        "tier": 1, "action": "delete_dir_contents",
        "path": "~/.cache/pip",
        "description": "pip package cache (rebuildable)",
    },
    "cache-uv": {
        "tier": 1, "action": "delete_dir_contents",
        "path": "~/.cache/uv",
        "description": "uv package cache (rebuildable)",
    },
    "npm-cache": {
        "tier": 1, "action": "delete_dir_contents",
        "path": "~/.npm",
        "description": "npm cache (rebuildable)",
    },
}

# Paths that should never be touched (safety hard-coded)
NEVER_TOUCH = [
    "commons/db/chronicle",
    "auth.json", ".env", "secrets", "credentials",
    "state.db",  # live DB — Tier 3 analysis only
]


# ── Helpers ────────────────────────────────────────────────────────────────

def du(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dp, _, fnames in os.walk(path):
        for f in fnames:
            fp = os.path.join(dp, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def fmt(size_bytes):
    b = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def disk_usage():
    stat = os.statvfs("/")
    total = stat.f_blocks * stat.f_frsize
    free = stat.f_bavail * stat.f_frsize
    used = total - free
    return total, used, free, used / total * 100


def age_days(path):
    return (datetime.datetime.now().timestamp() - os.path.getmtime(path)) / 86400


def age_hours(path):
    return (datetime.datetime.now().timestamp() - os.path.getmtime(path)) / 3600


def gzip_file(src, dry_run):
    dst = src + ".gz"
    if os.path.exists(dst):
        return 0
    if dry_run:
        return int(os.path.getsize(src) * 0.9) if os.path.isfile(src) else 0
    if not os.path.isfile(src) or os.path.getsize(src) == 0:
        return 0
    try:
        size_before = os.path.getsize(src)
        with open(src, "rb") as f_in:
            with gzip.open(dst, "wb", compresslevel=6) as f_out:
                shutil.copyfileobj(f_in, f_out)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            os.remove(src)
            return size_before - os.path.getsize(dst)
        if os.path.exists(dst):
            os.remove(dst)
        return 0
    except Exception:
        if os.path.exists(dst):
            try:
                os.remove(dst)
            except Exception:
                pass
        return 0


def oldest_mtime_in_dir(dir_path):
    """Return the oldest mtime of any file inside dir_path (recursive)."""
    oldest = os.path.getmtime(dir_path)
    for dp, _, fns in os.walk(dir_path):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if os.path.isfile(fp):
                m = os.path.getmtime(fp)
                if m < oldest:
                    oldest = m
    return oldest


# ── FILESYSTEM.md integration ──────────────────────────────────────────────

def find_filesystem_md():
    """Return the path to FILESYSTEM.md if it exists, else None."""
    for p in FILESYSTEM_MD_PATHS:
        if p and os.path.isfile(p):
            return p
    return None


def parse_manifest_from_md(md_path):
    """
    Parse the cleanup-manifest YAML block from FILESYSTEM.md.
    Returns a dict with keys: tier_1_safe, tier_2_low_risk, tier_3_analysis_only, never_touch
    Each value is a dict of entry_id → entry_config.
    Returns None if no manifest found.
    """
    with open(md_path, "r") as f:
        content = f.read()

    # Look for ```yaml ... ``` block with cleanup-manifest
    match = re.search(r'```yaml\s*\n(.*?)```', content, re.DOTALL)
    if not match:
        return None

    yaml_block = match.group(1)

    # Simple YAML parser for our specific format (no pyyaml dependency)
    manifest = {}
    current_section = None
    current_entry = {}
    entry_id = None

    for line in yaml_block.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Top-level key (no leading whitespace, ends with colon)
        if not line.startswith(" ") and not line.startswith("\t") and stripped.endswith(":"):
            # Save previous entry
            if current_entry and current_section and entry_id:
                manifest[current_section][entry_id] = current_entry
            current_section = stripped[:-1].strip()
            if current_section in ("tier_1_safe", "tier_2_low_risk", "tier_3_analysis_only", "never_touch"):
                if current_section not in manifest:
                    manifest[current_section] = {}
                current_entry = {}
                entry_id = None
            else:
                current_entry = {}
                entry_id = None
            continue

        # List item under a section (starts with "- ")
        if stripped.startswith("- ") and current_section and current_section in manifest:
            # Save previous entry
            if current_entry and entry_id:
                manifest[current_section][entry_id] = current_entry
            # Start new entry
            rest = stripped[2:].strip()
            current_entry = {}
            entry_id = None
            if ":" in rest:
                k, v = rest.split(":", 1)
                k = k.strip()
                v = v.strip()
                current_entry[k] = v
                if k == "id":
                    entry_id = v
            else:
                current_entry["path"] = rest
            continue

        # Key: value under a list item (indented, has colon)
        if ":" in stripped and current_entry is not None and current_section in manifest:
            # Only process if indented (part of a list item)
            if line.startswith(" ") or line.startswith("\t"):
                k, v = stripped.split(":", 1)
                k = k.strip()
                v = v.strip()
                # Type coercion
                if v.lower() in ("true", "false"):
                    v = v.lower() == "true"
                else:
                    try:
                        v = int(v)
                    except ValueError:
                        try:
                            v = float(v)
                        except ValueError:
                            pass
                current_entry[k] = v
                if k == "id":
                    entry_id = v
            continue

    # Save last entry
    if current_entry and current_section and entry_id and current_section in manifest:
        manifest[current_section][entry_id] = current_entry

    return manifest if manifest else None


def merge_manifest_into_targets(manifest):
    """
    Merge FILESYSTEM.md manifest entries into BUILTIN_TARGETS.
    Manifest entries get priority (they extend/override builtins).
    Returns the merged targets dict.
    """
    merged = copy.deepcopy(BUILTIN_TARGETS)

    tier_map = {
        "tier_1_safe": 1,
        "tier_2_low_risk": 2,
        "tier_3_analysis_only": 3,
    }

    for section, entries in manifest.items():
        if section == "never_touch":
            for eid, entry in entries.items():
                reason = entry.get("reason", "user-managed")
                path = entry.get("path", eid)
                if path.startswith("~"):
                    path = os.path.expanduser(path)
                NEVER_TOUCH.append(path)
            continue

        tier = tier_map.get(section)
        if tier is None:
            continue

        for eid, entry in entries.items():
            raw_path = entry.get("path", eid)
            # Expand ~ in paths from FILESYSTEM.md
            if raw_path.startswith("~"):
                raw_path = os.path.expanduser(raw_path)
            target = {
                "tier": tier,
                "path": raw_path,
                "description": entry.get("description", ""),
                "source": "filesystem_md",
            }
            # Copy age/threshold fields
            for key in ("max_age_days", "max_age_hours", "compress_age_days",
                        "delete_compressed_age_days", "action", "pattern",
                        "requires_confirmation"):
                if key in entry:
                    target[key] = entry[key]

            # Use provided id or derive from path
            target_id = entry.get("id", eid)
            merged[target_id] = target

    return merged


def generate_manifest_yaml(targets):
    """Generate a cleanup-manifest YAML block from the targets dict."""
    home = os.path.expanduser("~")

    def path_display(p):
        """Use ~ notation for home-relative paths."""
        if p.startswith(home):
            return "~" + p[len(home):]
        return p

    lines = ["```yaml", "# cleanup-manifest — auto-generated by genie --discover", "# Edit this to customize cleanup behavior. Genie reads this at runtime.", ""]

    # Tier 1
    lines.append("tier_1_safe:")
    for tid, t in sorted(targets.items()):
        if t.get("tier") == 1:
            lines.append(f"  - id: {tid}")
            for k in ("path", "description", "max_age_days", "max_age_hours",
                      "compress_age_days", "delete_compressed_age_days", "action",
                      "pattern", "requires_confirmation"):
                if k in t:
                    v = t[k]
                    if k == "path":
                        v = path_display(v)
                    if isinstance(v, bool):
                        v = str(v).lower()
                    lines.append(f"      {k}: {v}")
            lines.append("")
    lines.append("")

    # Tier 2
    lines.append("tier_2_low_risk:")
    has_t2 = False
    for tid, t in sorted(targets.items()):
        if t.get("tier") == 2:
            has_t2 = True
            lines.append(f"  - id: {tid}")
            for k in ("path", "description", "max_age_days", "action", "pattern",
                      "requires_confirmation"):
                if k in t:
                    v = t[k]
                    if k == "path":
                        v = path_display(v)
                    if isinstance(v, bool):
                        v = str(v).lower()
                    lines.append(f"      {k}: {v}")
            lines.append("")
    if not has_t2:
        lines.append("  # (none)")
        lines.append("")
    lines.append("")

    # Tier 3
    lines.append("tier_3_analysis_only:")
    has_t3 = False
    for tid, t in sorted(targets.items()):
        if t.get("tier") == 3:
            has_t3 = True
            lines.append(f"  - id: {tid}")
            for k in ("path", "description"):
                if k in t:
                    v = t[k]
                    if k == "path":
                        v = path_display(v)
                    lines.append(f"      {k}: {v}")
            lines.append("")
    if not has_t3:
        lines.append("  # (none)")
        lines.append("")
    lines.append("")

    # Never touch
    lines.append("never_touch:")
    for item in NEVER_TOUCH:
        lines.append(f"  - path: {path_display(item)}")
        lines.append(f"      reason: user-managed / live data")
    lines.append("")

    lines.append("```")
    return "\n".join(lines)


def create_filesystem_md(md_path, targets):
    """Create a new FILESYSTEM.md with the cleanup manifest."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    content = f"""# FILESYSTEM.md — where things go (Hermes / {PROFILE})

Auto-generated by genie --discover on {now}.
This file is both a human-readable map and a machine-readable cleanup manifest.
Genie reads the cleanup-manifest section at runtime to discover cleanup targets.

## How genie uses this file

1. If this file exists, genie reads the `cleanup-manifest` YAML block below
2. Manifest entries extend genie's built-in cleanup targets
3. If a path doesn't exist on this system, genie skips it silently
4. Run `genie --discover` to regenerate this file with current filesystem state

## Cleanup manifest

{generate_manifest_yaml(targets)}

## Directory map

Generated from filesystem walk at {now}:

"""

    # Add discovered directory tree
    for base_label, base_path in [("~/.hermes", HERMES_HOME), (f"profiles/{PROFILE}", PROFILE_HOME)]:
        if os.path.isdir(base_path):
            content += f"### {base_path}\n\n```\n"
            try:
                entries = sorted(os.listdir(base_path))
                for entry in entries:
                    full = os.path.join(base_path, entry)
                    if os.path.isdir(full):
                        size = du(full)
                        content += f"  {entry}/  ({fmt(size)})\n"
                    elif os.path.isfile(full):
                        size = os.path.getsize(full)
                        if size > 1024 * 1024:  # Only show files > 1MB
                            content += f"  {entry}  ({fmt(size)})\n"
            except PermissionError:
                content += "  (permission denied)\n"
            content += "```\n\n"

    content += f"""
## Never touch (hard-coded safety)

These paths are never cleaned by genie, regardless of manifest:

"""
    for item in NEVER_TOUCH:
        content += f"- `{item}`\n"

    content += f"""
---

_Last updated: {now}_
_Regenerate: genie --discover_
"""

    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as f:
        f.write(content)
    return content


def add_manifest_to_existing_md(md_path, targets):
    """Add a cleanup-manifest block to an existing FILESYSTEM.md that lacks one."""
    with open(md_path, "r") as f:
        content = f.read()

    manifest_yaml = generate_manifest_yaml(targets)

    # Check if there's already a cleanup-manifest block
    if "```yaml" in content and "cleanup-manifest" in content:
        # Replace existing manifest block
        content = re.sub(
            r'```yaml\s*\n# cleanup-manifest.*?\n```',
            manifest_yaml.strip(),
            content,
            flags=re.DOTALL
        )
    else:
        # Append manifest before the last section or at end
        content = content.rstrip() + "\n\n---\n\n## Cleanup manifest\n\n" + manifest_yaml + "\n"

    with open(md_path, "w") as f:
        f.write(content)


# ── Discovery ──────────────────────────────────────────────────────────────

def discover_filesystem():
    """
    Walk the filesystem and discover cleanup targets.
    Returns a dict of target_id → target_config for discovered paths.
    """
    discovered = {}

    # Walk the profile home
    if os.path.isdir(PROFILE_HOME):
        try:
            for entry in os.listdir(PROFILE_HOME):
                full = os.path.join(PROFILE_HOME, entry)
                if not os.path.isdir(full):
                    continue
                size = du(full)

                # Classify known patterns
                if entry == "state-snapshots":
                    discovered["state-snapshots"] = {
                        "tier": 1, "action": "delete_dirs",
                        "max_age_days": 7, "path": full,
                        "description": "Pre-update rollback snapshots",
                        "pattern": "state-snapshots/*/",
                        "source": "discovered",
                    }
                elif entry == "logs":
                    discovered["logs"] = {
                        "tier": 1, "action": "compress_then_delete",
                        "compress_age_days": 7, "delete_compressed_age_days": 30,
                        "path": full, "pattern": "logs/*.log",
                        "description": "Log files", "source": "discovered",
                    }
                elif entry == "sessions":
                    discovered["session-jsons"] = {
                        "tier": 2, "action": "compress",
                        "max_age_days": 14, "path": full,
                        "pattern": "sessions/*.json",
                        "description": "Session JSON files",
                        "requires_confirmation": True, "source": "discovered",
                    }
                elif entry == "cron-output" or entry == "cron" and os.path.isdir(os.path.join(full, "output")):
                    cron_out = os.path.join(full, "output") if entry == "cron" else full
                    discovered["cron-output"] = {
                        "tier": 1, "action": "compress",
                        "max_age_days": 7, "path": cron_out,
                        "pattern": f"{cron_out}/*",
                        "description": "Cron job output", "source": "discovered",
                    }
                elif entry == "commons":
                    # Analyze commons subdirectories
                    for sub in ("data", "db"):
                        sub_path = os.path.join(full, sub)
                        if os.path.isdir(sub_path):
                            sub_size = du(sub_path)
                            if sub_size > 100 * 1024 * 1024:  # > 100MB
                                discovered[f"commons-{sub}"] = {
                                    "tier": 3, "action": "analyze_only",
                                    "path": sub_path,
                                    "description": f"commons/{sub} — {fmt(sub_size)}",
                                    "source": "discovered",
                                }
                elif entry == "home" and size > 1024 * 1024 * 1024:  # > 1GB
                    discovered["profile-home"] = {
                        "tier": 3, "action": "analyze_only",
                        "path": full,
                        "description": f"Profile home — {fmt(size)}",
                        "source": "discovered",
                    }
                elif entry in ("cache", ".cache") and size > 50 * 1024 * 1024:
                    discovered["cache-dir"] = {
                        "tier": 1, "action": "delete_dir_contents",
                        "path": full,
                        "description": f"Package cache — {fmt(size)}",
                        "source": "discovered",
                    }
                elif entry in ("node", "node_modules") and size > 100 * 1024 * 1024:
                    discovered["node-modules"] = {
                        "tier": 3, "action": "analyze_only",
                        "path": full,
                        "description": f"Node modules — {fmt(size)} (rebuildable via npm install)",
                        "source": "discovered",
                    }
                elif entry == "checkpoints" and size > 50 * 1024 * 1024:
                    discovered["checkpoints"] = {
                        "tier": 3, "action": "analyze_only",
                        "path": full,
                        "description": f"Git checkpoints — {fmt(size)}",
                        "source": "discovered",
                    }
                elif entry == "lsp" and size > 50 * 1024 * 1024:
                    discovered["lsp"] = {
                        "tier": 3, "action": "analyze_only",
                        "path": full,
                        "description": f"LSP server data — {fmt(size)}",
                        "source": "discovered",
                    }
        except PermissionError:
            pass

    # Check /root/backups
    if os.path.isdir("/root/backups"):
        size = du("/root/backups")
        if size > 10 * 1024 * 1024:  # > 10MB
            discovered["backups"] = {
                "tier": 1, "action": "delete_dirs",
                "max_age_days": 30, "path": "/root/backups",
                "pattern": "/root/backups/*/",
                "description": f"Backup directories — {fmt(size)}",
                "source": "discovered",
            }

    # Check /tmp
    if os.path.isdir("/tmp"):
        size = du("/tmp")
        if size > 10 * 1024 * 1024:
            discovered["tmp"] = {
                "tier": 1, "action": "delete_files",
                "max_age_hours": 24, "path": "/tmp",
                "pattern": "/tmp/*",
                "description": f"Temp files — {fmt(size)}",
                "source": "discovered",
            }

    # Check package caches
    for cache_label, cache_path in [
        ("cache-pip", os.path.expanduser("~/.cache/pip")),
        ("cache-uv", os.path.expanduser("~/.cache/uv")),
        ("npm-cache", os.path.expanduser("~/.npm")),
    ]:
        if os.path.isdir(cache_path):
            size = du(cache_path)
            if size > 10 * 1024 * 1024:
                discovered[cache_label] = {
                    "tier": 1, "action": "delete_dir_contents",
                    "path": cache_path,
                    "description": f"{os.path.basename(cache_path)} cache — {fmt(size)} (rebuildable)",
                    "source": "discovered",
                }

    # Check for state.db
    for db_path in [os.path.join(PROFILE_HOME, "state.db"), os.path.join(HERMES_HOME, "state.db")]:
        if os.path.isfile(db_path):
            size = os.path.getsize(db_path)
            discovered["state-db"] = {
                "tier": 3, "action": "analyze_only",
                "path": db_path,
                "description": f"Live state database — {fmt(size)}",
                "source": "discovered",
            }
            break

    # Check for large directories at /root/ that aren't in known zones
    if os.path.isdir("/root"):
        known_zones = {"projects", "backups", "trash", "hermes-agent", "hermes-ecosystem"}
        try:
            for entry in os.listdir("/root"):
                full = os.path.join("/root", entry)
                if not os.path.isdir(full):
                    continue
                if entry in known_zones or entry.startswith("."):
                    continue
                size = du(full)
                if size > 100 * 1024 * 1024:  # > 100MB
                    discovered[f"root-{entry}"] = {
                        "tier": 3, "action": "analyze_only",
                        "path": full,
                        "description": f"Unknown /root/ directory — {fmt(size)} (review manually)",
                        "source": "discovered_unexpected",
                    }
        except PermissionError:
            pass

    return discovered


# ── Tier 1: Zero Risk ──────────────────────────────────────────────────────

def clean_snapshots(snapshots_path, max_age_days, dry_run):
    result = {"action": "snapshots", "tier": 1, "files": 0, "bytes_freed": 0, "errors": []}
    if not os.path.isdir(snapshots_path):
        return result

    # Find the most recent snapshot (by oldest mtime inside dir) — always preserved
    entries = []
    for entry in os.listdir(snapshots_path):
        path = os.path.join(snapshots_path, entry)
        if not os.path.isdir(path):
            continue
        oldest = oldest_mtime_in_dir(path)
        entries.append((oldest, entry, path))

    if not entries:
        return result

    # Skip the most recent snapshot (highest mtime = youngest)
    entries.sort(key=lambda x: x[0], reverse=True)
    most_recent_path = entries[0][2]
    result["skipped_most_recent"] = os.path.basename(most_recent_path)

    for oldest, entry, path in entries:
        if path == most_recent_path:
            continue
        snap_age = (datetime.datetime.now().timestamp() - oldest) / 86400
        if snap_age > max_age_days:
            size = du(path)
            result["files"] += 1
            result["bytes_freed"] += size
            if not dry_run:
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    result["errors"].append(f"rmtree {path}: {e}")
                    result["bytes_freed"] -= size
    return result


def clean_logs(logs_path, compress_age_days, delete_age_days, dry_run):
    result = {"action": "logs", "tier": 1, "compressed": 0, "deleted": 0,
              "bytes_freed": 0, "errors": []}
    if not os.path.isdir(logs_path):
        return result

    for entry in os.listdir(logs_path):
        path = os.path.join(logs_path, entry)
        if not os.path.isfile(path):
            continue

        if entry.endswith(".gz"):
            if age_days(path) > delete_age_days:
                size = os.path.getsize(path)
                result["deleted"] += 1
                result["bytes_freed"] += size
                if not dry_run:
                    try:
                        os.remove(path)
                    except Exception as e:
                        result["errors"].append(f"remove {path}: {e}")
            continue

        if entry.endswith(".log") and age_days(path) > compress_age_days:
            size_before = os.path.getsize(path)
            if size_before == 0:
                continue
            saved = gzip_file(path, dry_run)
            result["compressed"] += 1
            result["bytes_freed"] += saved

    # Clean up 0-byte .log files whose .gz counterparts exist and are valid
    if not dry_run:
        for entry in os.listdir(logs_path):
            path = os.path.join(logs_path, entry)
            if entry.endswith(".log") and os.path.isfile(path) and os.path.getsize(path) == 0:
                gz_path = path + ".gz"
                if os.path.exists(gz_path) and os.path.getsize(gz_path) > 100:
                    try:
                        os.remove(path)
                    except Exception:
                        pass

    return result


def clean_cron_output(cron_output_path, compress_age_days, dry_run):
    result = {"action": "cron_output", "tier": 1, "compressed": 0,
              "bytes_freed": 0, "errors": []}
    if not os.path.isdir(cron_output_path):
        return result

    for dp, _, filenames in os.walk(cron_output_path):
        for f in filenames:
            if f.endswith(".gz"):
                continue
            path = os.path.join(dp, f)
            if not os.path.isfile(path):
                continue
            if age_days(path) > compress_age_days:
                size_before = os.path.getsize(path)
                saved = gzip_file(path, dry_run)
                result["compressed"] += 1
                result["bytes_freed"] += saved

    return result


def clean_backups(backups_path, max_age_days, dry_run):
    result = {"action": "backups", "tier": 1, "dirs": 0, "bytes_freed": 0, "errors": []}
    if not os.path.isdir(backups_path):
        return result

    for entry in os.listdir(backups_path):
        path = os.path.join(backups_path, entry)
        if not os.path.isdir(path):
            # Also clean old individual files in backups/
            if os.path.isfile(path) and age_days(path) > max_age_days:
                size = os.path.getsize(path)
                result["dirs"] += 1
                result["bytes_freed"] += size
                if not dry_run:
                    try:
                        os.remove(path)
                    except Exception as e:
                        result["errors"].append(f"remove {path}: {e}")
                        result["bytes_freed"] -= size
            continue

        if age_days(path) > max_age_days:
            size = du(path)
            result["dirs"] += 1
            result["bytes_freed"] += size
            if not dry_run:
                try:
                    shutil.rmtree(path)
                except Exception as e:
                    result["errors"].append(f"rmtree {path}: {e}")
                    result["bytes_freed"] -= size
    return result


def clean_tmp(tmp_path, max_age_hours, dry_run):
    result = {"action": "tmp", "tier": 1, "files": 0, "bytes_freed": 0, "errors": []}
    if not os.path.isdir(tmp_path):
        return result

    for entry in os.listdir(tmp_path):
        path = os.path.join(tmp_path, entry)
        # Skip if still in use (check if any process has it open would be ideal,
        # but for safety we just check age)
        if age_hours(path) > max_age_hours:
            size = du(path)
            result["files"] += 1
            result["bytes_freed"] += size
            if not dry_run:
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                except Exception as e:
                    result["errors"].append(f"remove {path}: {e}")
                    result["bytes_freed"] -= size
    return result


def clean_cache_dir(cache_path, dry_run):
    """Delete contents of a cache directory (rebuildable)."""
    result = {"action": f"cache:{os.path.basename(cache_path)}", "tier": 1,
              "files": 0, "bytes_freed": 0, "errors": []}
    if not os.path.isdir(cache_path):
        return result

    size_before = du(cache_path)
    result["bytes_freed"] = size_before
    result["files"] = 1

    if not dry_run:
        try:
            shutil.rmtree(cache_path)
            os.makedirs(cache_path, exist_ok=True)
        except Exception as e:
            result["errors"].append(f"clean {cache_path}: {e}")
            result["bytes_freed"] = 0

    return result


# ── Tier 2: Low Risk ───────────────────────────────────────────────────────

def clean_session_jsons(sessions_path, compress_age_days, dry_run):
    result = {"action": "session_jsons", "tier": 2, "compressed": 0,
              "bytes_freed": 0, "errors": []}
    if not os.path.isdir(sessions_path):
        return result

    for entry in os.listdir(sessions_path):
        if not entry.endswith(".json"):
            continue
        path = os.path.join(sessions_path, entry)
        if not os.path.isfile(path):
            continue
        if age_days(path) > compress_age_days:
            size_before = os.path.getsize(path)
            saved = gzip_file(path, dry_run)
            result["compressed"] += 1
            result["bytes_freed"] += saved
            if not dry_run:
                gz_path = path + ".gz"
                if not os.path.exists(gz_path):
                    result["errors"].append(f"gzip failed for {path}")
                    result["compressed"] -= 1
                    result["bytes_freed"] -= saved

    return result


# ── Tier 3: Analysis Only ──────────────────────────────────────────────────

def analyze_state_db(db_path):
    result = {"action": "state_db_analysis", "tier": 3, "exists": False,
              "size": 0, "tables": []}
    if not os.path.isfile(db_path):
        return result

    result["exists"] = True
    result["size"] = os.path.getsize(db_path)

    try:
        conn = sqlite3.connect(db_path, timeout=10)
        c = conn.cursor()
        # Get DB size estimate from page_count * page_size (fast, no table scan)
        c.execute("PRAGMA page_count")
        page_count = c.fetchone()[0]
        c.execute("PRAGMA page_size")
        page_size = c.fetchone()[0]
        result["estimated_size"] = page_count * page_size

        c.execute("SELECT name FROM sqlite_master WHERE type='table'")
        for (table_name,) in c.fetchall():
            if table_name.startswith("sqlite_"):
                continue
            # For large tables, skip COUNT(*) — it does a full scan
            # Just report the table exists
            result["tables"].append({"name": table_name, "rows": -1})
        wal_path = db_path + "-wal"
        if os.path.exists(wal_path):
            result["wal_size"] = os.path.getsize(wal_path)
        conn.close()
    except Exception as e:
        result["error"] = str(e)

    return result


def analyze_commons(commons_path):
    result = {"action": "commons_analysis", "tier": 3, "data_size": 0,
              "db_size": 0, "data_files": [], "db_files": []}
    if not os.path.isdir(commons_path):
        return result

    for sub in ("data", "db"):
        sub_path = os.path.join(commons_path, sub)
        if not os.path.isdir(sub_path):
            continue
        for entry in os.listdir(sub_path):
            fp = os.path.join(sub_path, entry)
            if not os.path.isfile(fp):
                continue
            info = {"name": entry, "size": os.path.getsize(fp),
                    "age_days": round(age_days(fp), 1)}
            if sub == "data":
                result["data_size"] += info["size"]
                result["data_files"].append(info)
            else:
                result["db_size"] += info["size"]
                result["db_files"].append(info)

    return result


def analyze_path(path, label):
    """Generic analysis of a filesystem path."""
    result = {"action": f"analyze:{label}", "tier": 3, "path": path,
              "exists": False, "size": 0}
    if os.path.isfile(path):
        result["exists"] = True
        result["size"] = os.path.getsize(path)
        result["type"] = "file"
    elif os.path.isdir(path):
        result["exists"] = True
        result["size"] = du(path)
        result["type"] = "directory"
        try:
            count = sum(1 for _ in os.walk(path))
            result["subdirs"] = count
        except Exception:
            pass
    return result


# ── Assess ─────────────────────────────────────────────────────────────────

def assess(cfg, targets=None):
    lines = []
    total, used, free, pct = disk_usage()
    lines.append(f"Disk: {fmt(used)} / {fmt(total)} ({pct:.0f}% used, {fmt(free)} free)")

    if pct < 50:
        lines.append("Status: OK — disk usage below 50%, no cleanup needed.")
        return lines

    lines.append(f"Status: Disk at {pct:.0f}% — cleanup recommended.")
    lines.append("")

    # Snapshots
    sp = cfg["snapshots_path"]
    if os.path.isdir(sp):
        snap_dirs = [d for d in os.listdir(sp) if os.path.isdir(os.path.join(sp, d))]
        snap_size = du(sp)
        old_snaps = []
        for d in snap_dirs:
            dp = os.path.join(sp, d)
            oldest = oldest_mtime_in_dir(dp)
            age = (datetime.datetime.now().timestamp() - oldest) / 86400
            if age > cfg["snapshot_max_age_days"]:
                old_snaps.append(d)
        old_size = sum(du(os.path.join(sp, d)) for d in old_snaps)
        lines.append(
            f"Snapshots: {len(snap_dirs)} dirs, {fmt(snap_size)} total, "
            f"{len(old_snaps)} older than {cfg['snapshot_max_age_days']}d "
            f"({fmt(old_size)} reclaimable)")

    # Logs
    lp = cfg["logs_path"]
    if os.path.isdir(lp):
        log_size = du(lp)
        old_logs = sum(1 for f in os.listdir(lp)
                       if f.endswith(".log") and os.path.isfile(os.path.join(lp, f))
                       and age_days(os.path.join(lp, f)) > cfg["log_compress_age_days"])
        old_gz = sum(1 for f in os.listdir(lp)
                     if f.endswith(".gz") and os.path.isfile(os.path.join(lp, f))
                     and age_days(os.path.join(lp, f)) > cfg["log_delete_age_days"])
        lines.append(f"Logs: {fmt(log_size)} total, {old_logs} to compress, {old_gz} .gz to delete")

    # Cron output
    cp = cfg["cron_output_path"]
    if os.path.isdir(cp):
        cron_size = du(cp)
        old_cron = sum(1 for dp, _, fs in os.walk(cp) for f in fs
                       if not f.endswith(".gz")
                       and os.path.isfile(os.path.join(dp, f))
                       and age_days(os.path.join(dp, f)) > cfg["cron_output_compress_age_days"])
        lines.append(f"Cron output: {fmt(cron_size)} total, {old_cron} files to compress")

    # Session JSONs
    sesp = cfg["sessions_path"]
    if os.path.isdir(sesp):
        json_files = [f for f in os.listdir(sesp) if f.endswith(".json")]
        gz_files = [f for f in os.listdir(sesp) if f.endswith(".gz")]
        json_size = sum(os.path.getsize(os.path.join(sesp, f)) for f in json_files)
        old_json = sum(1 for f in json_files
                       if age_days(os.path.join(sesp, f)) > cfg["session_compress_age_days"])
        lines.append(
            f"Session JSONs: {len(json_files)} .json + {len(gz_files)} .gz, "
            f"{fmt(json_size)} total, {old_json} older than "
            f"{cfg['session_compress_age_days']}d compressible")

    # Backups
    bp = cfg["backups_path"]
    if os.path.isdir(bp):
        backup_size = du(bp)
        old_backup_dirs = sum(1 for d in os.listdir(bp)
                              if os.path.isdir(os.path.join(bp, d))
                              and age_days(os.path.join(bp, d)) > 30)
        lines.append(f"Backups: {fmt(backup_size)} total, {old_backup_dirs} dirs older than 30d")

    # /tmp
    if os.path.isdir("/tmp"):
        tmp_size = du("/tmp")
        old_tmp = sum(1 for f in os.listdir("/tmp")
                      if age_hours(os.path.join("/tmp", f)) > cfg.get("tmp_stale_hours", 24))
        lines.append(f"/tmp: {fmt(tmp_size)} total, {old_tmp} files/dirs older than {cfg.get('tmp_stale_hours', 24)}h")

    # State DB
    dbp = cfg["state_db_path"]
    if os.path.isfile(dbp):
        db_size = os.path.getsize(dbp)
        wal_size = os.path.getsize(dbp + "-wal") if os.path.exists(dbp + "-wal") else 0
        lines.append(f"State DB: {fmt(db_size)} (+ {fmt(wal_size)} WAL)")

    # Discovered targets (from FILESYSTEM.md or --discover)
    if targets:
        lines.append("")
        lines.append("── Discovered Targets ──")
        for tid, t in sorted(targets.items()):
            path = t.get("path", "")
            if not os.path.exists(path):
                continue
            size = du(path) if os.path.isdir(path) else (os.path.getsize(path) if os.path.isfile(path) else 0)
            if size > 10 * 1024 * 1024:  # Only show if > 10MB
                tier = t.get("tier", "?")
                action = t.get("action", "analyze_only")
                desc = t.get("description", tid)
                source = t.get("source", "builtin")
                tag = f" [{source}]" if source != "builtin" else ""
                lines.append(f"  [T{tier}] {desc}: {fmt(size)} ({action}){tag}")

    return lines


# ── Clean ──────────────────────────────────────────────────────────────────

def clean(cfg):
    tier_limit = int(cfg.get("tier_limit", 3))
    results = []

    results.append(clean_snapshots(cfg["snapshots_path"], cfg["snapshot_max_age_days"], cfg["dry_run"]))
    results.append(clean_logs(cfg["logs_path"], cfg["log_compress_age_days"],
                              cfg["log_delete_age_days"], cfg["dry_run"]))
    results.append(clean_cron_output(cfg["cron_output_path"],
                                     cfg["cron_output_compress_age_days"], cfg["dry_run"]))

    # Backups
    if cfg.get("backups_path") and os.path.isdir(cfg["backups_path"]):
        results.append(clean_backups(cfg["backups_path"], 30, cfg["dry_run"]))

    # /tmp
    if cfg.get("tmp_stale_hours", 0) > 0:
        results.append(clean_tmp("/tmp", cfg["tmp_stale_hours"], cfg["dry_run"]))

    # Package caches
    for cache_key, cache_path in [
        ("cache-pip", os.path.expanduser("~/.cache/pip")),
        ("cache-uv", os.path.expanduser("~/.cache/uv")),
        ("npm-cache", os.path.expanduser("~/.npm")),
    ]:
        if os.path.isdir(cache_path) and du(cache_path) > 10 * 1024 * 1024:
            results.append(clean_cache_dir(cache_path, cfg["dry_run"]))

    if tier_limit >= 2:
        results.append(clean_session_jsons(cfg["sessions_path"],
                                           cfg["session_compress_age_days"], cfg["dry_run"]))

    return results


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genie — VPS Disk Guardian")
    parser.add_argument("--assess", action="store_true", help="Assess disk usage and cleanup targets")
    parser.add_argument("--clean", action="store_true", help="Execute cleanup")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3], default=3,
                        help="Max tier to run (1=safe, 2=+sessions, 3=+analysis)")
    parser.add_argument("--analyze", action="store_true", help="Run Tier 3 analysis (read-only)")
    parser.add_argument("--discover", action="store_true",
                        help="Map filesystem, create/update FILESYSTEM.md, then assess")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    cfg["tier_limit"] = str(args.tier)
    if args.dry_run:
        cfg["dry_run"] = True
    if not args.assess and not args.clean and not args.analyze and not args.discover:
        args.assess = True

    output = []
    targets = None

    # ── FILESYSTEM.md integration ──
    md_path = find_filesystem_md()
    manifest = None

    if md_path:
        output.append(f"FILESYSTEM.md: found at {md_path}")
        manifest = parse_manifest_from_md(md_path)
        if manifest:
            output.append(f"  → Loaded cleanup manifest ({sum(len(v) for v in manifest.values() if isinstance(v, (list, dict)))} entries)")
        else:
            output.append("  → No cleanup manifest found in FILESYSTEM.md")
    else:
        output.append("FILESYSTEM.md: not found (using built-in defaults)")

    # ── Discovery mode ──
    if args.discover or not md_path:
        output.append("")
        output.append("── Discovery ──")
        discovered = discover_filesystem()
        output.append(f"Discovered {len(discovered)} cleanup targets")

        if discovered:
            output.append("")
            for tid, t in sorted(discovered.items()):
                path = t.get("path", "")
                size = du(path) if os.path.exists(path) else 0
                output.append(f"  [T{t.get('tier','?')}] {t.get('description', tid)}: {fmt(size)}")

        # Create or update FILESYSTEM.md
        fs_md_path = os.path.join(PROFILE_HOME, "references", "FILESYSTEM.md")

        if args.discover or not md_path:
            if md_path:
                # File exists but may lack manifest — add it
                output.append(f"\nUpdating FILESYSTEM.md at {fs_md_path}...")
                add_manifest_to_existing_md(fs_md_path, discovered)
                output.append("  → Added/updated cleanup manifest")
            else:
                # No FILESYSTEM.md at all — create one
                output.append(f"\nCreating FILESYSTEM.md at {fs_md_path}...")
                create_filesystem_md(fs_md_path, discovered)
                output.append("  → Created with cleanup manifest")
                md_path = fs_md_path

            # Re-read after creation/update
            manifest = parse_manifest_from_md(fs_md_path)

        targets = discovered

    # Merge manifest into targets
    if manifest:
        targets = merge_manifest_into_targets(manifest)

    # ── Assess ──
    if args.assess or args.discover:
        output.append("")
        output.extend(assess(cfg, targets))

    # ── Clean ──
    if args.clean:
        if not output:
            output.extend(assess(cfg, targets))
        output.append("")
        output.append("── Cleanup Results ──")
        results = clean(cfg)
        total_freed = 0
        for r in results:
            freed = r.get("bytes_freed", 0)
            total_freed += freed
            output.append(f"  [T{r.get('tier','?')}] {r['action']}: freed {fmt(freed)}")
            if r.get("errors"):
                for err in r["errors"]:
                    output.append(f"    ERROR: {err}")
        output.append(f"  Total freed: {fmt(total_freed)}")
        total, used, free, pct = disk_usage()
        output.append(f"  Disk after: {fmt(used)} / {fmt(total)} ({pct:.0f}% used)")

    # ── Analyze ──
    if args.analyze:
        if not output:
            output.extend(assess(cfg, targets))
        output.append("")
        output.append("── Tier 3 Analysis ──")
        db_result = analyze_state_db(cfg["state_db_path"])
        if db_result["exists"]:
            output.append(f"  state.db: {fmt(db_result['size'])}")
            for t in db_result["tables"]:
                output.append(f"    {t['name']}: {t['rows']:,} rows")
            if "wal_size" in db_result:
                output.append(f"    WAL: {fmt(db_result['wal_size'])}")
        else:
            output.append("  state.db: not found")
        commons_result = analyze_commons(cfg["commons_path"])
        if commons_result["data_files"] or commons_result["db_files"]:
            output.append(f"  commons/data: {fmt(commons_result['data_size'])}, "
                          f"{len(commons_result['data_files'])} files (oldest: "
                          f"{max(f['age_days'] for f in commons_result['data_files']):.0f}d)")
            output.append(f"  commons/db: {fmt(commons_result['db_size'])}, "
                          f"{len(commons_result['db_files'])} files (oldest: "
                          f"{max(f['age_days'] for f in commons_result['db_files']):.0f}d)")

        # Analyze discovered Tier 3 targets
        if targets:
            for tid, t in sorted(targets.items()):
                if t.get("tier") == 3 and t.get("action") == "analyze_only":
                    path = t.get("path", "")
                    if os.path.exists(path):
                        analysis = analyze_path(path, tid)
                        output.append(f"  {t.get('description', tid)}: {fmt(analysis['size'])}")

    if args.json:
        print(json.dumps({"lines": output, "config": {k: str(v) for k, v in cfg.items()}}))
    else:
        print("\n".join(output))


if __name__ == "__main__":
    main()
