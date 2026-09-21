#!/usr/bin/env python3
"""
Prune ghost entries from .usage.json — entries whose skills no longer exist on disk.
Runs as a no_agent cron script: stdout = message content, empty = nothing to report.

Scans all known skill directories: main ~/.hermes/skills/, per-profile skills,
hermes-agent skills, and hermes-agent optional skills.

Plugin-provided skills (no file on disk) are explicitly allowed.
"""
import json
import os
from datetime import datetime, timezone

USAGE_PATH = "/home/kensei/.hermes/skills/.usage.json"
BASE = "/home/kensei/.hermes"

# Plugin-provided system skills that don't have a SKILL.md on disk
# but are legitimately part of Hermes' plugin system.
PLUGIN_SKILLS = {
    # Hermes native MCP/TUI plugin skills
    "kanban-db-schema-mismatch",
    "skill-manage-api",
    "voice-ai-agents",

    # Agent-provided skills from Hermes core
    "outlines",
    "axolotl",
    "fine-tuning-with-trl",
    "unsloth",
}


def collect_disk_skills():
    """Walk all known skill directories and return a set of valid keys."""
    disk_keys = set()

    search_roots = [
        os.path.join(BASE, "skills"),
    ]

    # Add profile skill directories
    profiles_dir = os.path.join(BASE, "profiles")
    if os.path.isdir(profiles_dir):
        for p in os.listdir(profiles_dir):
            skill_dir = os.path.join(profiles_dir, p, "skills")
            if os.path.isdir(skill_dir):
                search_roots.append(skill_dir)

    # Add hermes-agent skills
    agent_skills = os.path.join(BASE, "hermes-agent", "skills")
    if os.path.isdir(agent_skills):
        search_roots.append(agent_skills)

    # Add optional skills
    opt_skills = os.path.join(BASE, "hermes-agent", "optional-skills")
    if os.path.isdir(opt_skills):
        search_roots.append(opt_skills)

    for root in search_roots:
        for dirpath, dirs, files in os.walk(root):
            if "SKILL.md" in files:
                rel = os.path.relpath(dirpath, root)
                if rel == ".":
                    continue
                disk_keys.add(rel)
                # Also add flat leaf name for entries keyed without category prefix
                leaf = rel.split("/")[-1]
                disk_keys.add(leaf)

    return disk_keys, search_roots


def main():
    if not os.path.exists(USAGE_PATH):
        print("Usage registry not found at", USAGE_PATH)
        return

    with open(USAGE_PATH) as f:
        usage = json.load(f)

    disk_keys, roots = collect_disk_skills()

    ghosts = []
    for key, entry in sorted(usage.items()):
        if entry.get("state") == "active":
            # Skip explicit plugin skills
            if key in PLUGIN_SKILLS:
                continue
            if key not in disk_keys:
                ghosts.append((key, entry.get("use_count", 0)))

    if not ghosts:
        return

    now = datetime.now(timezone.utc).isoformat()
    for key, uses in ghosts:
        usage[key]["archived_at"] = now
        usage[key]["state"] = "archived"

    with open(USAGE_PATH, "w") as f:
        json.dump(usage, f, indent=2)
        f.write("\n")

    lines = [f"Pruned {len(ghosts)} ghost skill(s) from usage registry:"]
    for key, uses in ghosts:
        lines.append(f"  - {key} ({uses} use(s))")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
