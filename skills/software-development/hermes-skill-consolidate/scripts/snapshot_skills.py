#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path

SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
MANIFEST_NAME = "manifest.json"


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink is not allowed in snapshot input: {path}")
        if path.is_file():
            yield path


def validate_skill_name(name: str) -> None:
    if not SKILL_NAME.fullmatch(name):
        raise ValueError(f"invalid skill name: {name}")


def snapshot(skills_root: Path, output: Path, skill_names: list[str]) -> dict:
    skills_root = skills_root.resolve()
    output = output.resolve()

    if not skills_root.is_dir():
        raise ValueError(f"skills root does not exist: {skills_root}")
    if is_relative_to(output, skills_root):
        raise ValueError("snapshot output must be outside the live skills tree")
    if output.exists():
        raise ValueError(f"snapshot output already exists: {output}")
    if not skill_names:
        raise ValueError("at least one --skill is required")
    if len(skill_names) != len(set(skill_names)):
        raise ValueError("duplicate --skill values are not allowed")

    sources: list[tuple[str, Path]] = []
    for name in skill_names:
        validate_skill_name(name)
        source = (skills_root / name).resolve()
        if source.parent != skills_root:
            raise ValueError(f"skill path escaped skills root: {name}")
        if not source.is_dir():
            raise ValueError(f"skill directory does not exist: {name}")
        if source.is_symlink():
            raise ValueError(f"skill directory cannot be a symlink: {name}")
        list(iter_files(source))
        sources.append((name, source))

    output.mkdir(parents=True, mode=0o700)
    copied_root = output / "skills"
    copied_root.mkdir(mode=0o700)

    entries: list[dict] = []
    for name, source in sources:
        destination = copied_root / name
        shutil.copytree(source, destination, copy_function=shutil.copy2)
        for copied in iter_files(destination):
            relative_to_skill = copied.relative_to(destination).as_posix()
            entries.append(
                {
                    "skill": name,
                    "path": relative_to_skill,
                    "size": copied.stat().st_size,
                    "sha256": sha256(copied),
                }
            )

    manifest = {
        "schema_version": "1.0",
        "skills_root": str(skills_root),
        "skills": skill_names,
        "files": entries,
    }
    manifest_path = output / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        os.chmod(output, stat.S_IRWXU)
        os.chmod(copied_root, stat.S_IRWXU)
        os.chmod(manifest_path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    return manifest


def verify(snapshot_root: Path) -> dict:
    snapshot_root = snapshot_root.resolve()
    manifest_path = snapshot_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"manifest is missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0":
        raise ValueError("unsupported manifest schema")
    skills = manifest.get("skills")
    files = manifest.get("files")
    if not isinstance(skills, list) or not skills:
        raise ValueError("manifest skills must be a nonempty array")
    if not isinstance(files, list):
        raise ValueError("manifest files must be an array")

    expected: set[tuple[str, str]] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("invalid manifest file entry")
        skill = entry.get("skill")
        rel = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size")
        if skill not in skills or not isinstance(rel, str) or not rel:
            raise ValueError("invalid manifest file identity")
        candidate = (snapshot_root / "skills" / skill / Path(rel)).resolve()
        skill_root = (snapshot_root / "skills" / skill).resolve()
        if not is_relative_to(candidate, skill_root):
            raise ValueError(f"manifest path escaped snapshot: {skill}/{rel}")
        if not candidate.is_file() or candidate.is_symlink():
            raise ValueError(f"snapshot file missing or unsafe: {skill}/{rel}")
        if candidate.stat().st_size != size:
            raise ValueError(f"snapshot size mismatch: {skill}/{rel}")
        if sha256(candidate) != digest:
            raise ValueError(f"snapshot hash mismatch: {skill}/{rel}")
        expected.add((skill, Path(rel).as_posix()))

    actual: set[tuple[str, str]] = set()
    for skill in skills:
        validate_skill_name(skill)
        skill_root = snapshot_root / "skills" / skill
        if not skill_root.is_dir() or skill_root.is_symlink():
            raise ValueError(f"snapshot skill missing or unsafe: {skill}")
        for path in iter_files(skill_root):
            actual.add((skill, path.relative_to(skill_root).as_posix()))

    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        raise ValueError(f"snapshot manifest mismatch: extra={extra} missing={missing}")

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify rollback snapshots for Hermes skill consolidation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a new rollback snapshot.")
    create.add_argument("--skills-root", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--skill", action="append", required=True, dest="skills")

    check = sub.add_parser("verify", help="Verify a rollback snapshot manifest.")
    check.add_argument("--snapshot", required=True, type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "create":
            manifest = snapshot(args.skills_root, args.output, args.skills)
            print(
                f"PASS: snapshot created for {len(manifest['skills'])} skill(s); "
                f"{len(manifest['files'])} file(s)"
            )
        else:
            manifest = verify(args.snapshot)
            print(
                f"PASS: snapshot verified for {len(manifest['skills'])} skill(s); "
                f"{len(manifest['files'])} file(s)"
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
