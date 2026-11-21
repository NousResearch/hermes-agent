#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# ///
"""Resolve a skill's default, team, and user TOML customization layers."""

import argparse
import json
import sys
from pathlib import Path

# Installed scripts are consumer files, not a location for interpreter caches.
sys.dont_write_bytecode = True

try:
    from config_utils import ConfigError, load_customization
except ModuleNotFoundError as error:
    if error.name != "tomllib":
        raise
    sys.stderr.write("error: Python 3.11+ is required (stdlib `tomllib` not found).\n")
    raise SystemExit(3) from None


_MISSING = object()


def find_project_root(start: Path) -> Path | None:
    """Nearest ancestor holding `_bmad/`, falling back to the nearest holding `.git`.

    `_bmad/` outranks `.git` at every depth: a submodule or nested repo carries
    `.git` without being the BMad project, so treating the two as equal stops the
    walk short of the root that owns `_bmad/custom/`.
    """
    git_root: Path | None = None
    current = start.resolve()
    while True:
        if (current / "_bmad").is_dir():
            return current
        if git_root is None and (current / ".git").exists():
            git_root = current
        if current.parent == current:
            return git_root
        current = current.parent


def script_project_root() -> Path | None:
    """Project root implied by this script's own install path.

    Skills invoke `{project-root}/_bmad/scripts/resolve_customization.py`, so when
    this file sits at that path its grandparent is a project root the caller already
    resolved.
    """
    parents = Path(__file__).resolve().parents
    if len(parents) >= 3 and parents[0].name == "scripts" and parents[1].name == "_bmad":
        return parents[2]
    return None


def candidate_project_roots(skill_dir: Path) -> list[Path]:
    """Plausible project roots, most trustworthy first.

    The working directory leads because the project is where the user is working,
    not where the skill happens to be installed — a home-installed skill walks up to
    `~`, and any `~/_bmad` there would otherwise mask the real project's overrides.
    """
    ordered: list[Path] = []
    for root in (
        find_project_root(Path.cwd()),
        script_project_root(),
        find_project_root(skill_dir),
    ):
        if root is not None and root not in ordered:
            ordered.append(root)
    return ordered


def has_override(root: Path, skill_name: str) -> bool:
    custom_dir = root / "_bmad" / "custom"
    return any(
        (custom_dir / name).is_file()
        for name in (f"{skill_name}.toml", f"{skill_name}.user.toml")
    )


def warn_on_masked_override(chosen: Path, rejected: list[Path], skill_name: str) -> None:
    """Break the silence when a real override exists under a root we did not pick."""
    if has_override(chosen, skill_name):
        return
    for root in rejected:
        if has_override(root, skill_name):
            sys.stderr.write(
                f"note: resolved project root {chosen} has no customization for "
                f"`{skill_name}`, but {root} does. Using {chosen}; pass "
                f"--project-root to select the other explicitly.\n"
            )
            return


def extract_key(data, dotted_key: str):
    current = data
    for part in dotted_key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


def write_json_stdout(output) -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")
    sys.stdout.write(json.dumps(output, indent=2, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve skill customization using three-layer TOML merge."
    )
    parser.add_argument(
        "--skill", "-s", required=True, help="Absolute path to the skill directory"
    )
    parser.add_argument(
        "--project-root",
        "-p",
        help="Explicit project root containing _bmad/ (recommended)",
    )
    parser.add_argument(
        "--key",
        "-k",
        action="append",
        default=[],
        help="Dotted field path to resolve (repeatable). Omit for full dump.",
    )
    args = parser.parse_args()

    skill_dir = Path(args.skill).resolve()
    if args.project_root:
        project_root = Path(args.project_root).resolve()
    else:
        candidates = candidate_project_roots(skill_dir)
        project_root = candidates[0] if candidates else None
        if project_root is not None:
            warn_on_masked_override(project_root, candidates[1:], skill_dir.name)

    try:
        merged = load_customization(project_root, skill_dir)
    except ConfigError as error:
        sys.stderr.write(f"error: {error}\n")
        return 1

    output = merged
    if args.key:
        output = {}
        for key in args.key:
            value = extract_key(merged, key)
            if value is not _MISSING:
                output[key] = value
    write_json_stdout(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
