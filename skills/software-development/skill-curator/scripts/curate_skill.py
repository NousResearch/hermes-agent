#!/usr/bin/env python3
"""curate_skill.py — Distill session trajectories into standardized Hermes skills.

Usage:
  curate_skill.py extract <transcript_path> [--output-dir DIR] [--name NAME] [--category CAT] [--json]
  curate_skill.py validate <skill_md_path>
  curate_skill.py scaffold <name> --category CAT [--description DESC] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MARKETING_WORDS = re.compile(
    r"\b(powerful|comprehensive|seamless|revolutionary|cutting-edge|state-of-the-art)\b",
    re.IGNORECASE,
)
MACHINE_LOCAL_PATHS = re.compile(r"/home/(?!runner\b)[a-z0-9_-]+/|[A-Z]:\\+Users\\+(?!<)")


def parse_transcript(transcript_path: Path) -> Dict[str, Any]:
    """Parse JSONL transcript file and extract tool calls and user inputs."""
    user_prompts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    assistant_summaries: List[str] = []

    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue

            event_type = record.get("type", "")
            content = record.get("content", "")

            if event_type == "USER_INPUT" and content:
                user_prompts.append(content)
            elif event_type == "PLANNER_RESPONSE":
                calls = record.get("tool_calls", [])
                for tc in calls:
                    fn_name = tc.get("name") or tc.get("function", {}).get("name")
                    fn_args = tc.get("args") or tc.get("function", {}).get("arguments")
                    if fn_name:
                        tool_calls.append({"name": fn_name, "args": fn_args})
                if content and isinstance(content, str) and len(content) > 50:
                    assistant_summaries.append(content)

    return {
        "user_prompts": user_prompts,
        "tool_calls": tool_calls,
        "assistant_summaries": assistant_summaries,
    }


def validate_skill_file(skill_path: Path) -> Tuple[bool, List[str]]:
    """Validate a SKILL.md against Hermes in-repo authoring standards."""
    errors: List[str] = []
    if not skill_path.exists():
        return False, [f"File does not exist: {skill_path}"]

    content = skill_path.read_text(encoding="utf-8")
    if not content.startswith("---"):
        errors.append("SKILL.md must start with '---' at byte 0.")

    fm_match = re.search(r"\n---\s*\n", content[3:])
    if not fm_match:
        errors.append("Unclosed YAML frontmatter (missing closing '---').")
        return False, errors

    fm_text = content[3 : fm_match.start() + 3]
    try:
        import yaml
        fm = yaml.safe_load(fm_text)
    except Exception as e:
        return False, [f"Invalid YAML frontmatter: {e}"]

    if not isinstance(fm, dict):
        return False, ["Frontmatter must be a YAML mapping."]

    # Check required fields
    for field in ("name", "description", "version", "author", "license", "platforms"):
        if field not in fm:
            errors.append(f"Missing required frontmatter field: '{field}'")

    # Name directory match
    dir_name = skill_path.parent.name
    skill_name = fm.get("name", "")
    if skill_name and skill_name != dir_name:
        errors.append(f"Frontmatter name '{skill_name}' does not match directory '{dir_name}'.")

    # Description rules (<= 60 chars, ends with period, no marketing)
    desc = str(fm.get("description") or "")
    if len(desc) > 60:
        errors.append(f"Description exceeds 60 chars (length: {len(desc)}).")
    if not desc.rstrip().endswith("."):
        errors.append("Description must end with a period.")
    m_market = MARKETING_WORDS.search(desc)
    if m_market:
        errors.append(f"Marketing word '{m_market.group(0)}' forbidden in description.")

    # Check machine-local paths
    m_local = MACHINE_LOCAL_PATHS.search(content)
    if m_local:
        errors.append(f"Machine-local path forbidden: '{m_local.group(0)}'")

    # Size limit
    if len(content) > 100_000:
        errors.append(f"File size exceeds 100,000 characters ({len(content)} chars).")

    return len(errors) == 0, errors


def scaffold_skill(
    name: str,
    category: str,
    *,
    description: str = "",
    output_dir: Optional[Path] = None,
) -> Path:
    """Create a structured in-repo skill scaffold."""
    if not description:
        description = f"Execute {name.replace('-', ' ')} workflows reliably."
        if len(description) > 60:
            description = f"Execute {name} workflows."
        if not description.endswith("."):
            description += "."

    base_dir = output_dir or (Path("skills") / category / name)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "scripts").mkdir(exist_ok=True)
    (base_dir / "references").mkdir(exist_ok=True)

    skill_md = f"""---
name: {name}
description: {description}
version: 0.1.0
author: Thamer (taljeri), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [{name.title()}, Workflow, Automation, {category.title()}]
    category: {category}
    related_skills: [hermes-agent-skill-authoring]
---

# {name.replace('-', ' ').title()} Skill

Provide concise capability summary, core problem solved, and expected behavior.

## When to Use

- "Run {name.replace('-', ' ')}"
- Specific trigger phrases that should load this skill

Don't use for:
- Out of scope tasks

## Prerequisites

- Required CLI utilities, packages, or API tokens.

## How to Run

Execute through the `terminal` tool:

```bash
# Example invocation
python3 skills/{category}/{name}/scripts/{name.replace('-', '_')}.py
```

## Quick Reference

| Task | Command |
|---|---|
| Run default action | `python3 skills/{category}/{name}/scripts/{name.replace('-', '_')}.py run` |

## Procedure

1. **Step One:** Preparation and environment check.
2. **Step Two:** Core execution.
3. **Step Three:** Verification and reporting.

## Pitfalls

- Known edge cases or common configuration mistakes.

## Verification

Run verification command to confirm success:
```bash
python3 skills/{category}/{name}/scripts/{name.replace('-', '_')}.py --verify
```
"""
    skill_file = base_dir / "SKILL.md"
    skill_file.write_text(skill_md, encoding="utf-8")
    return skill_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Hermes Skill Curator CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # extract
    p_ext = subparsers.add_parser("extract", help="Extract tool trajectory from transcript.")
    p_ext.add_argument("transcript", type=Path, help="Path to transcript.jsonl.")
    p_ext.add_argument("--name", type=str, default="curated-skill", help="Skill name.")
    p_ext.add_argument("--category", type=str, default="software-development", help="Skill category.")
    p_ext.add_argument("--json", action="store_true", help="Output parsed trajectory as JSON.")

    # validate
    p_val = subparsers.add_parser("validate", help="Validate a SKILL.md file.")
    p_val.add_argument("skill_md", type=Path, help="Path to SKILL.md.")

    # scaffold
    p_scaf = subparsers.add_parser("scaffold", help="Scaffold a new skill directory.")
    p_scaf.add_argument("name", type=str, help="Skill name.")
    p_scaf.add_argument("--category", type=str, required=True, help="Category name.")
    p_scaf.add_argument("--description", type=str, default="", help="Skill description (<= 60 chars).")
    p_scaf.add_argument("--output-dir", type=Path, default=None, help="Custom output directory.")

    args = parser.parse_args()

    if args.command == "extract":
        data = parse_transcript(args.transcript)
        if args.json:
            print(json.dumps(data, indent=2))
        else:
            print(f"Extracted {len(data['user_prompts'])} user prompts and {len(data['tool_calls'])} tool calls.")
            print(f"User Prompts: {data['user_prompts'][:3]}")
            print(f"Tool Calls summary: {[tc['name'] for tc in data['tool_calls'][:10]]}")

    elif args.command == "validate":
        valid, errors = validate_skill_file(args.skill_md)
        if valid:
            print(f"✓ {args.skill_md} is fully compliant with Hermes authoring standards.")
            sys.exit(0)
        else:
            print(f"✗ Validation errors in {args.skill_md}:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "scaffold":
        out = scaffold_skill(
            args.name,
            args.category,
            description=args.description,
            output_dir=args.output_dir,
        )
        print(f"Created skill scaffold at: {out}")


if __name__ == "__main__":
    main()
