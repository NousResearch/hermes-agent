#!/usr/bin/env python3
"""
verify-sub-agent-configs.py

Read-only verification script that checks all 29 sub-agent profiles
for correct 3-layer equipment (config, SOUL.md, skills, toolsets).

Output: Table with columns: Profile | Config OK | SOUL.md OK | kanban-worker | toolsets OK | terminal OK | Notes
Uses PASS/FAIL for each column.
"""

import os
import sys
from pathlib import Path

# Try PyYAML first, fall back to regex-based parsing
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

PROFILES_HOME = Path.home() / ".hermes" / "profiles"

SUB_AGENTS = [
    "octacon-backend", "octacon-frontend", "octacon-infra", "octacon-testrunner",
    "remii-deep", "remii-digest", "remii-gitradar", "remii-market",
    "quan-arch", "quan-code", "quan-perf", "quan-security", "quan-ux",
    "ceecee-brand", "ceecee-reviewer", "ceecee-social", "ceecee-writer",
    "gojo-admin", "gojo-calendar", "gojo-mailbox",
    "wesker-backup", "wesker-ops", "wesker-scanner",
    "light-archivist", "light-indexer", "light-wiki",
    "dezzy-brand", "dezzy-component-lib", "dezzy-design-system", "dezzy-ux-prototype",
]


def parse_config_regex(text):
    """Minimal YAML-like parser using regex for when PyYAML is unavailable."""
    import re
    result = {}

    # Extract top-level keys and their values (scalar or list)
    # Match top-level keys
    lines = text.split('\n')
    current_key = None
    list_items = None
    in_list = False

    for line in lines:
        stripped = line.rstrip()
        if not stripped or stripped.startswith('#'):
            continue

        # Check if this is a top-level key (no indent)
        if not stripped.startswith(' ') and not stripped.startswith('\t'):
            # Save previous list if any
            if current_key and in_list and list_items is not None:
                result[current_key] = list_items
                list_items = None
                in_list = False

            if ':' in stripped:
                key, _, val = stripped.partition(':')
                key = key.strip()
                val = val.strip()
                if val == '':
                    # Could be a mapping or list parent
                    current_key = key
                    list_items = []
                    in_list = True
                else:
                    result[key] = val
                    current_key = None
                    in_list = False
            else:
                current_key = None
                in_list = False
        elif stripped.startswith('- ') and in_list:
            item = stripped[2:].strip()
            list_items.append(item)
        elif (stripped.startswith('  ') or stripped.startswith('\t')) and in_list:
            # Sub-items under a list item (like nested mappings) - skip for our purposes
            pass
        else:
            # Indented but not a list item - might be sub-keys of a mapping
            # We don't need deep parsing for our checks
            pass

    # Flush last list
    if current_key and in_list and list_items is not None:
        result[current_key] = list_items

    return result


def load_config(profile_path):
    """Load config.yaml from a profile directory."""
    config_path = profile_path / "config.yaml"
    if not config_path.exists():
        return None, "config.yaml not found"

    try:
        text = config_path.read_text(encoding="utf-8")
    except Exception as e:
        return None, f"cannot read config.yaml: {e}"

    if HAS_YAML:
        try:
            data = yaml.safe_load(text)
            if not isinstance(data, dict):
                return None, "config.yaml is not a mapping"
            return data, None
        except yaml.YAMLError as e:
            return None, f"YAML parse error: {e}"
    else:
        data = parse_config_regex(text)
        return data, None


def check_config(data):
    """Check that config has model, provider, toolsets, agent.max_turns."""
    if data is None:
        return False, "no config data"

    issues = []

    if "model" not in data:
        issues.append("missing 'model'")
    if "provider" not in data:
        issues.append("missing 'provider'")
    if "toolsets" not in data or not isinstance(data.get("toolsets"), (list, tuple)):
        issues.append("missing/invalid 'toolsets'")
    if "agent" not in data or not isinstance(data.get("agent"), dict):
        issues.append("missing 'agent' section")
    elif "max_turns" not in data["agent"]:
        issues.append("missing agent.max_turns")

    if issues:
        return False, "; ".join(issues)
    return True, ""


def check_soul_md(profile_path):
    """Check that SOUL.md exists and contains kanban_complete or kanban_block."""
    soul_path = profile_path / "SOUL.md"
    if not soul_path.exists():
        return False, "SOUL.md not found"

    try:
        text = soul_path.read_text(encoding="utf-8")
    except Exception as e:
        return False, f"cannot read SOUL.md: {e}"

    has_complete = "kanban_complete" in text
    has_block = "kanban_block" in text

    if has_complete or has_block:
        return True, ""
    else:
        return False, "no 'kanban_complete' or 'kanban_block' found"


def check_kanban_worker(data):
    """Check that always_skills includes kanban-worker."""
    if data is None:
        return False, "no config data"

    skills = data.get("skills")
    if not isinstance(skills, dict):
        return False, "no 'skills' section"

    always = skills.get("always_skills")
    if not isinstance(always, (list, tuple)):
        return False, "no 'always_skills' list"

    if "kanban-worker" in always:
        return True, ""
    else:
        return False, "'kanban-worker' not in always_skills"


def check_toolsets(data):
    """Check that toolsets includes 'kanban' and 'hermes-cli'."""
    if data is None:
        return False, "no config data"

    toolsets = data.get("toolsets")
    if not isinstance(toolsets, (list, tuple)):
        return False, "no 'toolsets' list"

    missing = []
    if "kanban" not in toolsets:
        missing.append("kanban")
    if "hermes-cli" not in toolsets:
        missing.append("hermes-cli")

    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, ""


def check_terminal(data):
    """Check that terminal toolset is available."""
    if data is None:
        return False, "no config data"

    toolsets = data.get("toolsets")
    if not isinstance(toolsets, (list, tuple)):
        return False, "no 'toolsets' list"

    if "terminal" in toolsets:
        return True, ""
    else:
        return False, "'terminal' not in toolsets"


def main():
    header = f"{'Profile':<25} {'Config OK':<10} {'SOUL.md OK':<10} {'kanban-worker':<14} {'toolsets OK':<12} {'terminal OK':<12} Notes"
    sep = "-" * len(header)
    print(header)
    print(sep)

    total = len(SUB_AGENTS)
    passed = {"config": 0, "soul": 0, "kanban_worker": 0, "toolsets": 0, "terminal": 0}
    all_pass = 0

    for name in SUB_AGENTS:
        profile_path = PROFILES_HOME / name
        notes = []

        # Load config once
        data, err = load_config(profile_path)
        if err:
            notes.append(err)

        # Check 1: config.yaml
        config_ok, config_note = check_config(data)
        if config_ok:
            passed["config"] += 1
        if config_note:
            notes.append(config_note)

        # Check 2: SOUL.md
        soul_ok, soul_note = check_soul_md(profile_path)
        if soul_ok:
            passed["soul"] += 1
        if soul_note:
            notes.append(soul_note)

        # Check 3: kanban-worker in always_skills
        kw_ok, kw_note = check_kanban_worker(data)
        if kw_ok:
            passed["kanban_worker"] += 1
        if kw_note:
            notes.append(kw_note)

        # Check 4: toolsets include kanban + hermes-cli
        ts_ok, ts_note = check_toolsets(data)
        if ts_ok:
            passed["toolsets"] += 1
        if ts_note:
            notes.append(ts_note)

        # Check 5: terminal toolset
        term_ok, term_note = check_terminal(data)
        if term_ok:
            passed["terminal"] += 1
        if term_note:
            notes.append(term_note)

        # Count fully passing profiles
        if config_ok and soul_ok and kw_ok and ts_ok and term_ok:
            all_pass += 1

        config_str = "PASS" if config_ok else "FAIL"
        soul_str = "PASS" if soul_ok else "FAIL"
        kw_str = "PASS" if kw_ok else "FAIL"
        ts_str = "PASS" if ts_ok else "FAIL"
        term_str = "PASS" if term_ok else "FAIL"
        notes_str = "; ".join(notes) if notes else ""

        print(f"{name:<25} {config_str:<10} {soul_str:<10} {kw_str:<14} {ts_str:<12} {term_str:<12} {notes_str}")

    # Summary
    print(sep)
    print(f"\nSummary: {all_pass}/{total} profiles fully pass")
    print(f"  Config OK:     {passed['config']}/{total}")
    print(f"  SOUL.md OK:    {passed['soul']}/{total}")
    print(f"  kanban-worker: {passed['kanban_worker']}/{total}")
    print(f"  toolsets OK:   {passed['toolsets']}/{total}")
    print(f"  terminal OK:   {passed['terminal']}/{total}")

    return 0 if all_pass == total else 1


if __name__ == "__main__":
    sys.exit(main())
