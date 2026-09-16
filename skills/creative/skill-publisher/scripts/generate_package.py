#!/usr/bin/env python3
"""generate_package.py - Generate a publish package for any Claude Code skill.

Creates a directory with:
- SKILL.md (copied from source)
- README.md (bilingual EN/ZH, extracted from SKILL.md)
- examples/ (copied from source)
- .clawhub.json (ClawHub manifest)
"""

import argparse
import json
import os
import re
import shutil
import sys
import yaml


def copy_skill(source_dir, output_dir):
    """Copy SKILL.md and scripts/ to the output directory."""
    os.makedirs(output_dir, exist_ok=True)

    src_skill = os.path.join(source_dir, "SKILL.md")
    if not os.path.isfile(src_skill):
        print(f"Error: SKILL.md not found at {src_skill}", file=sys.stderr)
        sys.exit(1)

    shutil.copy2(src_skill, os.path.join(output_dir, "SKILL.md"))

    src_scripts = os.path.join(source_dir, "scripts")
    if os.path.isdir(src_scripts):
        dst_scripts = os.path.join(output_dir, "scripts")
        shutil.copytree(src_scripts, dst_scripts, dirs_exist_ok=True)


def parse_skill(skill_path):
    """Parse SKILL.md into frontmatter data and body sections."""
    with open(skill_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = content.split("---", 2)
    if len(parts) < 3:
        print("Error: Invalid SKILL.md — missing YAML frontmatter", file=sys.stderr)
        sys.exit(1)

    data = yaml.safe_load(parts[1].strip())
    body = parts[2].strip()
    return data, body


def extract_sections(body):
    """Split body into named sections for reuse in README."""
    sections = {}
    current_heading = None
    current_lines = []

    for line in body.split("\n"):
        m = re.match(r"^(#{1,3})\s+(.+)", line)
        if m:
            if current_heading:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = m.group(2).strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def find_section(sections, keywords):
    """Find a section matching any of the keywords (case-insensitive partial match)."""
    for heading, content in sections.items():
        for kw in keywords:
            if kw.lower() in heading.lower():
                return content
    return None


def extract_examples_from_source(source_dir):
    """List example files in the source skill directory."""
    examples_dir = os.path.join(source_dir, "examples")
    if not os.path.isdir(examples_dir):
        return []
    examples = []
    for f in sorted(os.listdir(examples_dir)):
        if f.endswith((".ascii", ".excalidraw", ".md", ".txt", ".png", ".jpg")):
            examples.append(f)
    return examples


def generate_readme(data, body, sections, output_path):
    """Generate a bilingual README.md from any skill's SKILL.md."""
    name = data.get("name", "skill")
    description = data.get("description", "")
    display_name = name.replace("-", " ").title()

    # Try to find overview/when-to-use content
    overview = (
        find_section(sections, ["when to use", "overview", "about"])
        or "See SKILL.md for full documentation."
    )

    # Try to find features section
    features_content = (
        find_section(sections, ["features", "capabilities", "functionality"])
        or find_section(sections, ["pattern", "color"])
        or find_section(sections, ["conversion", "workflow", "step"])
    )

    # Extract tables from sections if present (useful for config/color/palette sections)
    tables = []
    for heading, content in sections.items():
        if any(kw in heading for kw in ["color", "palette", "config", "mapping"]):
            table_lines = [l for l in content.split("\n") if l.strip().startswith("|")]
            if table_lines:
                tables.append((heading, "\n".join(table_lines)))

    # Find example references
    examples = extract_examples_from_source(os.path.dirname(output_path).replace("/tmp/publish-" + name, os.path.join(os.path.expanduser("~"), ".skills", name)))

    readme_parts = []
    readme_parts.append(f"# {display_name} / {display_name}")
    readme_parts.append("")
    readme_parts.append(f"> {description}")
    readme_parts.append("")
    readme_parts.append("---")
    readme_parts.append("")
    readme_parts.append("## Overview / 概述")
    readme_parts.append("")
    readme_parts.append(overview)
    readme_parts.append("")
    readme_parts.append("---")
    readme_parts.append("")
    readme_parts.append("## Features / 功能特性")
    readme_parts.append("")
    if features_content:
        readme_parts.append(features_content)
    else:
        readme_parts.append(f"See SKILL.md for details on {display_name}.")
    readme_parts.append("")

    for heading, table_content in tables:
        readme_parts.append(f"### {heading}")
        readme_parts.append("")
        readme_parts.append(table_content)
        readme_parts.append("")

    readme_parts.append("---")
    readme_parts.append("")
    readme_parts.append("## Quick Start / 快速开始")
    readme_parts.append("")
    readme_parts.append("### English")
    readme_parts.append("")
    readme_parts.append("1. Trigger the skill in Claude Code: `/ {name}`")
    readme_parts.append("2. Follow the interactive prompts")
    readme_parts.append("")
    readme_parts.append("### 中文")
    readme_parts.append("")
    readme_parts.append("1. 在 Claude Code 中触发技能：`/ {name}`")
    readme_parts.append("2. 按照交互提示操作")
    readme_parts.append("")

    if examples:
        readme_parts.append("---")
        readme_parts.append("")
        readme_parts.append("## Examples / 示例")
        readme_parts.append("")
        readme_parts.append("Example files are included in the `examples/` directory.")
        readme_parts.append("")
        readme_parts.append("示例文件位于 `examples/` 目录中。")
        readme_parts.append("")

    readme_parts.append("---")
    readme_parts.append("")
    readme_parts.append("## License / 许可")
    readme_parts.append("")
    readme_parts.append(data.get("license", "MIT"))
    readme_parts.append("")
    readme_parts.append("---")
    readme_parts.append("")
    readme_parts.append("*Generated by skill-publisher*")
    readme_parts.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(readme_parts))

    print(f"Generated README.md at {output_path}")


def generate_clawhub_manifest(data, output_path):
    """Generate .clawhub.json manifest for ClawHub."""
    manifest = {
        "name": data.get("name", ""),
        "description": data.get("description", ""),
        "version": data.get("version", "1.0.0"),
        "author": data.get("author", ""),
        "license": data.get("license", "MIT"),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated .clawhub.json at {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a publish package for any Claude Code skill"
    )
    parser.add_argument("--source", required=True, help="Path to the source skill directory")
    parser.add_argument("--output", required=True, help="Output directory for the publish package")
    parser.add_argument("--platforms", nargs="+", default=["clawhub", "hermes", "anthropics"],
                        help="Target platforms (default: all)")
    parser.add_argument("--examples-dir", help="Directory with example files")
    parser.add_argument("--screenshots", help="Directory with screenshot files")
    args = parser.parse_args()

    print(f"Generating publish package for: {args.source}")
    print(f"Output directory: {args.output}")
    print(f"Platforms: {', '.join(args.platforms)}")
    print()

    # Parse source skill
    src_skill = os.path.join(args.source, "SKILL.md")
    data, body = parse_skill(src_skill)
    sections = extract_sections(body)

    # Step 1: Copy skill files
    copy_skill(args.source, args.output)

    # Step 2: Generate bilingual README
    generate_readme(data, body, sections, os.path.join(args.output, "README.md"))

    # Step 3: Copy examples
    examples_src = args.examples_dir or os.path.join(args.source, "examples")
    if os.path.isdir(examples_src):
        examples_dst = os.path.join(args.output, "examples")
        shutil.copytree(examples_src, examples_dst, dirs_exist_ok=True)
        print(f"Copied examples from {examples_src}")

    # Step 4: Generate ClawHub manifest
    if "clawhub" in args.platforms:
        generate_clawhub_manifest(data, os.path.join(args.output, ".clawhub.json"))

    # Step 5: Copy screenshots
    if args.screenshots and os.path.isdir(args.screenshots):
        screenshots_dst = os.path.join(args.output, "screenshots")
        shutil.copytree(args.screenshots, screenshots_dst, dirs_exist_ok=True)
        print(f"Copied screenshots from {args.screenshots}")

    # Summary
    print(f"\nPublish package ready at: {args.output}")
    print("\nNext steps:")
    if "clawhub" in args.platforms:
        print("  ClawHub:    cd {} && clawhub publish .".format(args.output))
    if "hermes" in args.platforms:
        print("  Hermes:     Copy SKILL.md to skills/creative/<name>/ in NousResearch/hermes-agent fork")
    if "anthropics" in args.platforms:
        print("  Anthropic:  Copy SKILL.md to skills/<name>/ in anthropics/skills fork")


if __name__ == "__main__":
    main()
