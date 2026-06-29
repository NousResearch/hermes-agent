#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Enumerate customizable BMad skills installed alongside this one.

Scans a skills directory (by default: the directory this script's own skill
lives in, derived from __file__), finds every sibling directory containing a
`customize.toml`, classifies each as agent and/or workflow based on its
top-level blocks, reads the skill's SKILL.md frontmatter description for a
one-liner, and checks whether override files already exist in
`{project-root}/_bmad/custom/`.

Skills in BMad are loaded either from a project-local location (e.g. the
project's `.claude/skills/` or `.cursor/skills/`) or from a user-global
location (e.g. `~/.claude/skills/`). We do not hardcode those paths — the
running skill's own location is the source of truth for sibling discovery.
`--extra-root` is available for the rare case where skills live in multiple
locations on the same machine.

Output: JSON to stdout. Non-empty `errors[]` in the payload is non-fatal
by contract — the scanner surfaces malformed TOML, missing roots, and
skills with no customization block as data for the caller to display,
and still exits 0. Exit 2 is reserved for invocation errors (e.g.
missing or unreadable `--project-root`) where no useful payload can be
produced.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

# Top-level TOML blocks that indicate a customization surface.
SURFACE_KEYS = ("agent", "workflow")

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def default_skills_root() -> Path:
    """Derive the skills root from this script's location.

    Layout assumption: {skills_root}/bmad-customize/scripts/list_customizable_skills.py.
    So the skills root is three parents up from this file.
    """
    return Path(__file__).resolve().parent.parent.parent


def read_frontmatter_description(skill_md: Path) -> str:
    """Extract the `description:` value from a SKILL.md YAML frontmatter block.

    Returns an empty string if the file is missing, unreadable, or has no
    description field. Intentionally permissive — this is metadata for a
    human-facing list, not a validation target.
    """
    if not skill_md.is_file():
        return ""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return ""
    for line in m.group(1).splitlines():
        stripped = line.strip()
        if stripped.startswith("description:"):
            value = stripped[len("description:") :].strip()
            # Strip surrounding quotes if present.
            if (value.startswith("'") and value.endswith("'")) or (
                value.startswith('"') and value.endswith('"')
            ):
                value = value[1:-1]
            return value
    return ""


def load_customize(toml_path: Path) -> dict | None:
    """Return the parsed TOML, or None if unreadable."""
    try:
        with toml_path.open("rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def scan_skills(
    skills_roots: list[Path],
    project_root: Path,
) -> dict:
    """Scan each skills root for directories that contain a customize.toml."""
    agents: list[dict] = []
    workflows: list[dict] = []
    errors: list[str] = []
    scanned_roots: list[str] = []
    seen_names: set[str] = set()
    custom_dir = project_root / "_bmad" / "custom"

    for root in skills_roots:
        if not root.is_dir():
            errors.append(f"skills root does not exist: {root}")
            continue
        scanned_roots.append(str(root))

        for skill_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            customize_toml = skill_dir / "customize.toml"
            if not customize_toml.is_file():
                continue

            data = load_customize(customize_toml)
            if data is None:
                errors.append(f"failed to parse {customize_toml}")
                continue

            skill_name = skill_dir.name
            # If a skill with this name was already found in an earlier
            # root, skip it — roots are scanned in the order provided, so
            # the first occurrence wins.
            if skill_name in seen_names:
                continue
            seen_names.add(skill_name)

            description = read_frontmatter_description(skill_dir / "SKILL.md")
            team_override = custom_dir / f"{skill_name}.toml"
            user_override = custom_dir / f"{skill_name}.user.toml"

            entry_base = {
                "name": skill_name,
                "install_path": str(skill_dir),
                "skills_root": str(root),
                "description": description,
                "has_team_override": team_override.is_file(),
                "has_user_override": user_override.is_file(),
                "team_override_path": str(team_override),
                "user_override_path": str(user_override),
            }

            # A skill may expose an agent surface, a workflow surface, or
            # both. Emit one entry per surface so the caller can group cleanly.
            surfaces_found = [k for k in SURFACE_KEYS if k in data]
            if not surfaces_found:
                errors.append(
                    f"no [agent] or [workflow] block in {customize_toml}"
                )
                continue
            for surface in surfaces_found:
                entry = dict(entry_base)
                entry["surface"] = surface
                if surface == "agent":
                    agents.append(entry)
                else:
                    workflows.append(entry)

    return {
        "project_root": str(project_root),
        "scanned_roots": scanned_roots,
        "custom_dir": str(custom_dir),
        "agents": agents,
        "workflows": workflows,
        "errors": errors,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List customizable BMad skills installed alongside this one, "
            "grouped by surface (agent vs workflow), with override status "
            "looked up against {project-root}/_bmad/custom/."
        )
    )
    parser.add_argument(
        "--project-root",
        required=True,
        help="Absolute path to the project root (the folder containing _bmad/).",
    )
    parser.add_argument(
        "--skills-root",
        default=None,
        help=(
            "Override the primary skills directory to scan. Defaults to the "
            "directory this script's own skill lives in."
        ),
    )
    parser.add_argument(
        "--extra-root",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Additional skills directory to include (repeatable). Useful "
            "when skills live in multiple locations on the same machine "
            "(e.g. project-local plus a user-global install)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    project_root = Path(args.project_root).expanduser().resolve()
    if not project_root.is_dir():
        print(
            f"error: project-root does not exist or is not a directory: {project_root}",
            file=sys.stderr,
        )
        return 2

    primary = (
        Path(args.skills_root).expanduser().resolve()
        if args.skills_root
        else default_skills_root()
    )
    extras = [Path(p).expanduser().resolve() for p in args.extra_root]
    # Deduplicate in order of appearance.
    roots: list[Path] = []
    for root in [primary, *extras]:
        if root not in roots:
            roots.append(root)

    result = scan_skills(roots, project_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
