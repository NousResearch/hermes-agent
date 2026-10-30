#!/usr/bin/env python3
"""validate_skill.py - Validate a skill's SKILL.md format and content."""

import argparse
import json
import os
import re
import sys
import yaml


def load_yaml_frontmatter(filepath):
    """Load and parse YAML frontmatter from SKILL.md."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Check first bytes are ---
    if not content.startswith("---"):
        return None, "File must start with '---' (YAML frontmatter delimiter)"

    # Split frontmatter
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None, "Missing closing '---' delimiter for frontmatter"

    yaml_text = parts[1].strip()
    body = parts[2].strip()

    try:
        data = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return None, f"Invalid YAML frontmatter: {e}"

    if not isinstance(data, dict):
        return None, "Frontmatter must be a YAML mapping"

    return data, body


def validate_skill(skill_path):
    """Validate a skill directory."""
    skill_md = os.path.join(skill_path, "SKILL.md")

    if not os.path.isfile(skill_md):
        return False, [f"SKILL.md not found at {skill_md}"]

    errors = []
    warnings = []

    # Parse frontmatter
    data, body_or_err = load_yaml_frontmatter(skill_md)
    if data is None:
        return False, [body_or_err]

    body = body_or_err

    # Required fields
    if not data.get("name"):
        errors.append("Missing required field: 'name'")
    else:
        name = data["name"]
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", name):
            errors.append(f"Invalid name format: '{name}' (must be lowercase with hyphens)")
        if len(name) > 64:
            errors.append(f"Name too long: {len(name)} chars (max 64)")

    if not data.get("description"):
        errors.append("Missing required field: 'description'")
    elif len(data["description"]) > 1024:
        errors.append(f"Description too long: {len(data['description'])} chars (max 1024)")

    # File size check
    file_size = len(body)
    if file_size > 100000:
        errors.append(f"SKILL.md body too large: {file_size} chars (max 100,000)")

    # Check for hardcoded secrets in scripts/
    scripts_dir = os.path.join(skill_path, "scripts")
    if os.path.isdir(scripts_dir):
        secret_patterns = [
            r"ghp_[A-Za-z0-9]{36}",
            r"xox[baprs]-[A-Za-z0-9-]+",
            r"sk-[A-Za-z0-9]{20,}",
            r"password\s*=\s*['\"][^'\"]{8,}['\"]",
            r"api[_-]?key\s*=\s*['\"][^'\"]{8,}['\"]",
        ]
        for root, _, files in os.walk(scripts_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                for pattern in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        warnings.append(
                            f"Possible secret in {os.path.relpath(fpath, skill_path)}: "
                            f"matches pattern '{pattern}'"
                        )

    # Summary
    if errors:
        return False, errors + warnings

    if warnings:
        return True, [f"Valid with {len(warnings)} warning(s):"] + warnings

    return True, ["Valid — no issues found"]


def main():
    parser = argparse.ArgumentParser(description="Validate a Claude Code skill")
    parser.add_argument("skill_path", help="Path to the skill directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    valid, messages = validate_skill(args.skill_path)

    if args.json:
        result = {"valid": valid, "messages": messages}
        print(json.dumps(result, indent=2))
    else:
        status = "PASS" if valid else "FAIL"
        print(f"\nSkill validation: {status}")
        print(f"Path: {args.skill_path}")
        print()
        for msg in messages:
            prefix = "  WARNING" if "warning" in msg.lower() else "  ERROR" if "ERROR" in msg or "Invalid" in msg or "Missing" in msg or "too" in msg.lower() else "  INFO"
            print(f"{prefix}: {msg}")
        print()

    sys.exit(0 if valid else 1)


if __name__ == "__main__":
    main()
