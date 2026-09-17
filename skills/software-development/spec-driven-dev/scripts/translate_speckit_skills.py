#!/usr/bin/env python3
"""
Translate Spec Kit's Claude-Code skills into Hermes SKILL.md frontmatter.

`specify init --integration claude` installs `.claude/skills/speckit-*/SKILL.md`
files using Claude Code's frontmatter shape. Hermes uses a different
frontmatter contract (metadata.hermes.tags/category/related_skills, a
description length cap, etc. -- see skills/AGENTS.md's authoring standards).
This script performs a MECHANICAL field rename only; it does not invent
content, sections, or descriptions beyond truncation for the length cap.

Usage:
  python3 translate_speckit_skills.py <project_dir> [--out <hermes_skills_dir>]

Reads <project_dir>/.claude/skills/speckit-*/SKILL.md and writes translated
copies to <project_dir>/.hermes/skills/speckit-*/SKILL.md by default.
"""

import argparse
import re
import sys
from pathlib import Path

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
DESC_RE = re.compile(r"^description:\s*(.*)$", re.MULTILINE)


def translate_frontmatter(text: str, skill_name: str) -> str:
    match = FRONTMATTER_RE.match(text)
    if not match:
        # No frontmatter to translate; leave body untouched.
        return text

    body = text[match.end():]
    desc_match = DESC_RE.search(match.group(1))
    description = desc_match.group(1).strip().strip('"') if desc_match else (
        f"Spec Kit {skill_name} step."
    )
    if len(description) > 60:
        description = description[:57].rstrip() + "..."
    if not description.endswith("."):
        description += "."

    new_frontmatter = (
        "---\n"
        f"name: {skill_name}\n"
        f'description: "{description}"\n'
        "version: 1.0.0\n"
        "author: GitHub Spec Kit (translated for Hermes)\n"
        "license: MIT\n"
        "platforms: [linux, macos, windows]\n"
        "metadata:\n"
        "  hermes:\n"
        "    tags: [spec-kit, spec-driven-development]\n"
        "    category: software-development\n"
        "    related_skills: [spec-driven-dev]\n"
        "capability_class: LOCAL_MUTATION\n"
        "mixed_capability: false\n"
        "---\n"
    )
    return new_frontmatter + body


def translate_project(project_dir: Path, out_dir: Path) -> list[str]:
    src_root = project_dir / ".claude" / "skills"
    if not src_root.is_dir():
        raise FileNotFoundError(
            f"No .claude/skills found under {project_dir} -- run specify init first."
        )

    written = []
    for skill_md in sorted(src_root.glob("speckit-*/SKILL.md")):
        skill_name = skill_md.parent.name
        text = skill_md.read_text()
        translated = translate_frontmatter(text, skill_name)

        dest_dir = out_dir / skill_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "SKILL.md"
        dest_file.write_text(translated)
        written.append(str(dest_file))

    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    out_dir = (args.out or (project_dir / ".hermes" / "skills")).resolve()

    written = translate_project(project_dir, out_dir)
    for path in written:
        print(path)
    print(f"Translated {len(written)} skill(s) into {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
