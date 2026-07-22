#!/usr/bin/env python3
"""Generate the bundled Hermes skill contract consumed by GBrain."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKILLS_DIR = REPO_ROOT / "skills"
RESOLVER_FILENAME = "RESOLVER.md"
MANIFEST_FILENAME = "manifest.json"
SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class ContractError(ValueError):
    """Raised when a bundled skill cannot produce an unambiguous contract."""


@dataclass(frozen=True)
class SkillEntry:
    manifest_name: str
    relative_path: str
    frontmatter_name: str
    triggers: tuple[str, ...]


@dataclass(frozen=True)
class Contract:
    manifest: dict[str, object]
    resolver: str


def _error(path: Path, message: str) -> ContractError:
    return ContractError(f"{path}: {message}")


def _normalize_trigger(value: str) -> str:
    return " ".join(value.split())


def _relative_skill_path(skills_dir: Path, path: Path) -> Path:
    try:
        relative = path.relative_to(skills_dir)
    except ValueError as exc:
        raise _error(path, "skill is outside the bundled skills directory") from exc

    if skills_dir.is_symlink():
        raise _error(skills_dir, "symbolic links are not allowed in skill paths")

    current = skills_dir
    for segment in relative.parts:
        current /= segment
        if current.is_symlink():
            raise _error(current, "symbolic links are not allowed in skill paths")

    resolved_skills_dir = skills_dir.resolve()
    try:
        resolved_path = path.resolve(strict=True)
    except OSError as exc:
        raise _error(path, f"could not resolve skill path: {exc}") from exc
    try:
        resolved_path.relative_to(resolved_skills_dir)
    except ValueError as exc:
        raise _error(
            path, "resolved skill path is outside the bundled skills directory"
        ) from exc

    return relative


def _discover_skill_files(skills_dir: Path) -> list[Path]:
    if skills_dir.is_symlink():
        raise _error(skills_dir, "symbolic links are not allowed in skill paths")

    skill_files: list[Path] = []
    for candidate in sorted(skills_dir.rglob("*")):
        if candidate.is_symlink():
            if candidate.name == "SKILL.md" or candidate.is_dir():
                raise _error(candidate, "symbolic links are not allowed in skill paths")
            continue
        if candidate.name == "SKILL.md":
            if not candidate.is_file():
                raise _error(candidate, "SKILL.md must be a regular file")
            skill_files.append(candidate)
    return skill_files


def _load_frontmatter(path: Path) -> dict[str, object]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise _error(path, f"could not read skill: {exc}") from exc

    if not lines or lines[0].strip() != "---":
        raise _error(path, "YAML frontmatter is required")

    try:
        closing_index = next(
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.strip() == "---"
        )
    except StopIteration as exc:
        raise _error(path, "YAML frontmatter is not closed") from exc

    raw_frontmatter = "\n".join(lines[1:closing_index])
    try:
        frontmatter = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError as exc:
        raise _error(path, "invalid YAML frontmatter") from exc

    if not isinstance(frontmatter, dict):
        raise _error(path, "frontmatter must be a mapping")
    return frontmatter


def _validated_triggers(
    path: Path,
    frontmatter: dict[str, object],
    description: str,
) -> tuple[str, ...]:
    if "triggers" not in frontmatter:
        raw_triggers: list[object] = [description]
    else:
        declared = frontmatter["triggers"]
        if not isinstance(declared, list) or not declared:
            raise _error(path, "triggers must be a non-empty list")
        raw_triggers = declared

    normalized_triggers: list[str] = []
    seen: set[str] = set()
    for raw_trigger in raw_triggers:
        if not isinstance(raw_trigger, str):
            raise _error(path, "trigger must be a string")
        trigger = _normalize_trigger(raw_trigger)
        if not trigger:
            raise _error(path, "trigger must not be empty")
        if "|" in trigger:
            raise _error(path, "trigger must not contain |")
        if trigger in seen:
            raise _error(path, f"duplicate trigger: {trigger!r}")
        seen.add(trigger)
        normalized_triggers.append(trigger)
    return tuple(normalized_triggers)


def parse_skill(skills_dir: Path, path: Path) -> SkillEntry:
    """Parse one SKILL.md into its path-derived runtime contract entry."""

    relative = _relative_skill_path(skills_dir, path)
    if relative.name != "SKILL.md" or len(relative.parts) < 2:
        raise _error(path, "skill must be nested below the bundled skills directory")
    for segment in relative.parts[:-1]:
        if not SKILL_NAME_PATTERN.fullmatch(segment):
            raise _error(path, f"invalid skill path segment: {segment!r}")

    frontmatter = _load_frontmatter(path)
    name = frontmatter.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _error(path, "skill name is required")
    name = name.strip()
    if not SKILL_NAME_PATTERN.fullmatch(name):
        raise _error(path, f"invalid skill name: {name!r}")

    raw_description = frontmatter.get("description")
    if not isinstance(raw_description, str):
        raise _error(path, "skill description is required")
    description = _normalize_trigger(raw_description)
    if not description:
        raise _error(path, "skill description is required")

    relative_path = relative.as_posix()
    manifest_name = relative.parent.as_posix()
    triggers = _validated_triggers(path, frontmatter, description)
    return SkillEntry(
        manifest_name=manifest_name,
        relative_path=relative_path,
        frontmatter_name=name,
        triggers=triggers,
    )


def render_resolver(entries: Sequence[SkillEntry]) -> str:
    lines = [
        "# Hermes Runtime Skill Resolver",
        "",
        "Generated by `scripts/build_gbrain_skill_contract.py`; do not hand-edit.",
        "",
        "## Bundled skills",
        "",
        "| Trigger | Skill |",
        "|---|---|",
    ]
    for entry in sorted(entries, key=lambda item: item.manifest_name):
        skill_path = f"skills/{entry.relative_path}"
        for trigger in entry.triggers:
            lines.append(f"| {trigger} | `{skill_path}` |")
    return "\n".join(lines) + "\n"


def build_contract(skills_dir: Path) -> Contract:
    """Build a deterministic contract from bundled SKILL.md files only."""

    skills_dir = Path(skills_dir)
    skill_files = _discover_skill_files(skills_dir)
    entries = [parse_skill(skills_dir, path) for path in skill_files]

    identities: dict[str, Path] = {}
    triggers: dict[str, str] = {}
    for entry in entries:
        prior_path = identities.get(entry.frontmatter_name)
        if prior_path is not None:
            raise ContractError(
                f"duplicate skill name {entry.frontmatter_name!r}: "
                f"{prior_path} and {entry.relative_path}"
            )
        identities[entry.frontmatter_name] = Path(entry.relative_path)
        for trigger in entry.triggers:
            prior_skill = triggers.get(trigger)
            if prior_skill is not None:
                raise ContractError(
                    f"duplicate trigger {trigger!r}: {prior_skill} and {entry.relative_path}"
                )
            triggers[trigger] = entry.relative_path

    sorted_entries = sorted(entries, key=lambda item: item.manifest_name)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "skills": [
            {
                "frontmatter_name": entry.frontmatter_name,
                "name": entry.manifest_name,
                "path": entry.relative_path,
            }
            for entry in sorted_entries
        ],
    }
    return Contract(manifest=manifest, resolver=render_resolver(sorted_entries))


def render_manifest_json(manifest: dict[str, object]) -> str:
    return (
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _desired_artifacts(skills_dir: Path) -> dict[Path, str]:
    contract = build_contract(skills_dir)
    return {
        skills_dir / RESOLVER_FILENAME: contract.resolver,
        skills_dir / MANIFEST_FILENAME: render_manifest_json(contract.manifest),
    }


def _is_current(path: Path, desired: str) -> bool:
    try:
        return path.read_text(encoding="utf-8") == desired
    except (OSError, UnicodeError):
        return False


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate deterministic GBrain artifacts for bundled Hermes skills."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify generated artifacts without writing files",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    skills_dir = args.skills_dir
    try:
        artifacts = _desired_artifacts(skills_dir)
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.check:
        drifted = [
            path.name
            for path, content in artifacts.items()
            if not _is_current(path, content)
        ]
        if drifted:
            print(
                "generated GBrain skill contract is out of date: " + ", ".join(drifted),
                file=sys.stderr,
            )
            return 1
        return 0

    skills_dir.mkdir(parents=True, exist_ok=True)
    for path, content in artifacts.items():
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
