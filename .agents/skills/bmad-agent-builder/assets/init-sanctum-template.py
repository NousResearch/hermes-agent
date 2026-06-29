#!/usr/bin/env python3
"""
First Breath — Deterministic sanctum scaffolding.

This script runs BEFORE the conversational awakening. It creates the sanctum
folder structure, copies template files with config values substituted,
copies all capability files and their supporting references into the sanctum,
and auto-generates CAPABILITIES.md from capability prompt frontmatter.

After this script runs, the sanctum is fully self-contained — the agent does
not depend on the skill bundle location for normal operation.

This initializes the agent's runtime sanctum memory, not build-time config. It
reads config.yaml and config.user.yaml strictly to substitute values into the
sanctum templates, and it never writes or authors any config file. Build-time
customization is owned by customize.toml, a separate surface this script never
touches.

Usage:
    uv run init-sanctum.py <project-root> <skill-path>

    project-root: The root of the project (where _bmad/ lives)
    skill-path:   Path to the skill directory (where SKILL.md, references/, assets/ live)
"""

import sys
import re
import shutil
from datetime import date
from pathlib import Path

# --- Agent-specific configuration (set by builder) ---

SKILL_NAME = "{skillName}"
SANCTUM_DIR = SKILL_NAME

# Files that stay in the skill bundle (only used during First Breath)
SKILL_ONLY_FILES = {"{skill-only-files}"}

TEMPLATE_FILES = [
    {template-files-list}
]

# Whether the owner can teach this agent new capabilities
EVOLVABLE = {evolvable}

# --- End agent-specific configuration ---


def parse_yaml_config(config_path: Path) -> dict:
    """Simple YAML key-value parser. Handles top-level scalar values only."""
    config = {}
    if not config_path.exists():
        return config
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, value = line.partition(":")
                value = value.strip().strip("'\"")
                if value:
                    config[key.strip()] = value
    return config


def parse_frontmatter(file_path: Path) -> dict:
    """Extract YAML frontmatter from a markdown file."""
    meta = {}
    with open(file_path) as f:
        content = f.read()

    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return meta

    for line in match.group(1).strip().split("\n"):
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip().strip("'\"")
    return meta


def copy_references(source_dir: Path, dest_dir: Path) -> list[str]:
    """Copy all reference files (except skill-only files) into the sanctum."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for source_file in sorted(source_dir.iterdir()):
        if source_file.name in SKILL_ONLY_FILES:
            continue
        if source_file.is_file():
            shutil.copy2(source_file, dest_dir / source_file.name)
            copied.append(source_file.name)

    return copied


def copy_scripts(source_dir: Path, dest_dir: Path) -> list[str]:
    """Copy any scripts the capabilities might use into the sanctum."""
    if not source_dir.exists():
        return []
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for source_file in sorted(source_dir.iterdir()):
        if source_file.is_file() and source_file.name != "init-sanctum.py":
            shutil.copy2(source_file, dest_dir / source_file.name)
            copied.append(source_file.name)

    return copied


def discover_capabilities(references_dir: Path, sanctum_refs_path: str) -> list[dict]:
    """Scan references/ for capability prompt files with frontmatter."""
    capabilities = []

    for md_file in sorted(references_dir.glob("*.md")):
        if md_file.name in SKILL_ONLY_FILES:
            continue
        meta = parse_frontmatter(md_file)
        if meta.get("name") and meta.get("code"):
            capabilities.append({
                "name": meta["name"],
                "description": meta.get("description", ""),
                "code": meta["code"],
                "source": f"{sanctum_refs_path}/{md_file.name}",
            })
    return capabilities


def generate_capabilities_md(capabilities: list[dict], evolvable: bool) -> str:
    """Generate CAPABILITIES.md content from discovered capabilities."""
    lines = [
        "# Capabilities",
        "",
        "## Built-in",
        "",
        "| Code | Name | Description | Source |",
        "|------|------|-------------|--------|",
    ]
    for cap in capabilities:
        lines.append(
            f"| [{cap['code']}] | {cap['name']} | {cap['description']} | `{cap['source']}` |"
        )

    if evolvable:
        lines.extend([
            "",
            "## Learned",
            "",
            "_Capabilities added by the owner over time. Prompts live in `capabilities/`._",
            "",
            "| Code | Name | Description | Source | Added |",
            "|------|------|-------------|--------|-------|",
            "",
            "## How to Add a Capability",
            "",
            'Tell me "I want you to be able to do X" and we\'ll create it together.',
            "I'll write the prompt, save it to `capabilities/`, and register it here.",
            "Next session, I'll know how.",
            "Load `references/capability-authoring.md` for the full creation framework.",
        ])

    lines.extend([
        "",
        "## Tools",
        "",
        "Prefer crafting your own tools over depending on external ones. A script you wrote "
        "and saved is more reliable than an external API. Use the file system creatively.",
        "",
        "### User-Provided Tools",
        "",
        "_MCP servers, APIs, or services the owner has made available. Document them here._",
    ])

    return "\n".join(lines) + "\n"


def substitute_vars(content: str, variables: dict) -> str:
    """Replace {var_name} placeholders with values from the variables dict."""
    for key, value in variables.items():
        content = content.replace(f"{{{key}}}", value)
    return content


def main():
    if len(sys.argv) < 3:
        print("Usage: uv run init-sanctum.py <project-root> <skill-path>")
        sys.exit(1)

    project_root = Path(sys.argv[1]).resolve()
    skill_path = Path(sys.argv[2]).resolve()

    # Paths
    bmad_dir = project_root / "_bmad"
    memory_dir = bmad_dir / "memory"
    sanctum_path = memory_dir / SANCTUM_DIR
    assets_dir = skill_path / "assets"
    references_dir = skill_path / "references"
    scripts_dir = skill_path / "scripts"

    # Sanctum subdirectories
    sanctum_refs = sanctum_path / "references"
    sanctum_scripts = sanctum_path / "scripts"

    # Relative path for CAPABILITIES.md references (agent loads from within sanctum)
    sanctum_refs_path = "references"

    # Check if sanctum already exists
    if sanctum_path.exists():
        print(f"Sanctum already exists at {sanctum_path}")
        print("This agent has already been born. Skipping First Breath scaffolding.")
        sys.exit(0)

    # Load config
    config = {}
    for config_file in ["config.yaml", "config.user.yaml"]:
        config.update(parse_yaml_config(bmad_dir / config_file))

    # Build variable substitution map
    today = date.today().isoformat()
    variables = {
        "user_name": config.get("user_name", "friend"),
        "communication_language": config.get("communication_language", "English"),
        "birth_date": today,
        "project_root": str(project_root),
        "sanctum_path": str(sanctum_path),
    }

    # Create sanctum structure
    sanctum_path.mkdir(parents=True, exist_ok=True)
    (sanctum_path / "capabilities").mkdir(exist_ok=True)
    (sanctum_path / "sessions").mkdir(exist_ok=True)
    print(f"Created sanctum at {sanctum_path}")

    # Copy reference files (capabilities + techniques + guidance) into sanctum
    copied_refs = copy_references(references_dir, sanctum_refs)
    print(f"  Copied {len(copied_refs)} reference files to sanctum/references/")
    for name in copied_refs:
        print(f"    - {name}")

    # Copy any supporting scripts into sanctum
    copied_scripts = copy_scripts(scripts_dir, sanctum_scripts)
    if copied_scripts:
        print(f"  Copied {len(copied_scripts)} scripts to sanctum/scripts/")
        for name in copied_scripts:
            print(f"    - {name}")

    # Copy and substitute template files
    for template_name in TEMPLATE_FILES:
        template_path = assets_dir / template_name
        if not template_path.exists():
            print(f"  Warning: template {template_name} not found, skipping")
            continue

        # Remove "-template" from the output filename and uppercase it
        output_name = template_name.replace("-template", "").upper()
        # Fix extension casing: .MD -> .md
        output_name = output_name[:-3] + ".md"

        content = template_path.read_text()
        content = substitute_vars(content, variables)

        output_path = sanctum_path / output_name
        output_path.write_text(content)
        print(f"  Created {output_name}")

    # Auto-generate CAPABILITIES.md from references/ frontmatter
    capabilities = discover_capabilities(references_dir, sanctum_refs_path)
    capabilities_content = generate_capabilities_md(capabilities, evolvable=EVOLVABLE)
    (sanctum_path / "CAPABILITIES.md").write_text(capabilities_content)
    print(f"  Created CAPABILITIES.md ({len(capabilities)} built-in capabilities discovered)")

    print()
    print("First Breath scaffolding complete.")
    print("The conversational awakening can now begin.")
    print(f"Sanctum: {sanctum_path}")


if __name__ == "__main__":
    main()
