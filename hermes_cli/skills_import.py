"""Validate and add scanned portable skill contents to the active profile."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

from hermes_cli.archive_safe import normalize_archive_parts, safe_extract_targz
from hermes_constants import get_hermes_home

# Bound work before reading file bodies. These are import format limits, not
# evidence that an archive's publisher or skill instructions can be trusted.
MAX_MEMBERS = 4096
MAX_CONTENT_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


def _canonical_path(name: str) -> str:
    if not isinstance(name, str) or "/".join(normalize_archive_parts(name)) != name:
        raise ValueError("Archive paths must be canonical relative POSIX paths")
    return name


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key!r}")
        result[key] = value
    return result


def validate_archive(path: Path) -> list[str]:
    """Return selected skill paths after checking schema and exact hash coverage.

    Reads regular file streams only: no extraction, code execution, profile
    access, or scan-cache writes. Hashes check integrity, not authenticity.
    """
    files = {}
    members = set()
    directories = set()
    total = 0
    manifest = None
    with tarfile.open(path.expanduser(), "r:gz") as archive:
        for member in archive:
            name = _canonical_path(member.name)
            if name in members:
                raise ValueError(f"Duplicate archive member: {name!r}")
            members.add(name)
            if len(members) > MAX_MEMBERS:
                raise ValueError("Archive exceeds member limit")
            if member.isdir():
                directories.add(name)
                continue
            if not member.isfile() or member.issparse():
                raise ValueError(f"Unsupported archive member type: {name!r}")
            total += member.size
            if member.size < 0 or total > MAX_CONTENT_BYTES:
                raise ValueError("Archive exceeds content size limit")
            if name == "manifest.json" and member.size > MAX_MANIFEST_BYTES:
                raise ValueError("Archive manifest exceeds size limit")
            content = archive.extractfile(member)
            if content is None:
                raise ValueError(f"Cannot read archive member: {name!r}")
            with content:
                if name == "manifest.json":
                    manifest = json.loads(content.read(), object_pairs_hook=_unique_object)
                else:
                    files[name] = hashlib.file_digest(content, "sha256").hexdigest()

    if not isinstance(manifest, dict) or set(manifest) != {"format_version", "skills", "files"}:
        raise ValueError("Invalid skill archive manifest")
    if type(manifest["format_version"]) is not int or manifest["format_version"] != 1:
        raise ValueError("Unsupported skill archive format version")
    names = manifest["skills"]
    if not isinstance(names, list) or not names:
        raise ValueError("Manifest must select at least one skill")
    selected = [_canonical_path(name) for name in names]
    if len(set(selected)) != len(selected):
        raise ValueError("Manifest contains duplicate skill selections")
    if not isinstance(manifest["files"], dict) or manifest["files"] != files:
        raise ValueError("Manifest file hashes do not match archive contents")
    roots = ["skills/" + name for name in selected]
    if any(root + "/SKILL.md" not in files for root in roots):
        raise ValueError("Every selected skill must contain SKILL.md")
    if any(not any(name.startswith(root + "/") for root in roots) for name in files):
        raise ValueError("Archive contains files outside selected skills")
    # Category ancestors and empty subdirectories are valid exporter output.
    if any(not any(name == root or root.startswith(name + "/") or
                   name.startswith(root + "/") for root in roots) for name in directories):
        raise ValueError("Archive contains directories outside selected skills")
    return selected


def import_skills(path: Path) -> list[str]:
    """Add scanned local copies, without assigning Hub update ownership.

    Reserve each destination exclusively and publish SKILL.md last, after its
    supporting files. This is additive, not an atomic multi-skill transaction.
    """
    from tools.skills_guard import scan_skill, should_allow_install
    from tools.skills_hub_install import _is_path_redirect

    with tempfile.TemporaryDirectory(prefix="hermes-skills-import-") as temp:
        staging = Path(temp)
        snapshot = staging / "archive.tar.gz"
        # All subsequent reads use the same private copy, not a mutable source.
        with path.expanduser().open("rb") as source, snapshot.open("wb") as target:
            remaining = MAX_CONTENT_BYTES + MAX_MEMBERS * 2048
            while chunk := source.read(min(1024 * 1024, remaining + 1)):
                remaining -= len(chunk)
                if remaining < 0:
                    raise ValueError("Archive exceeds compressed size limit")
                target.write(chunk)
        names = validate_archive(snapshot)
        if any(a != b and b.startswith(a + "/") for a in names for b in names):
            raise ValueError("Cannot import overlapping skill selections")
        payload = staging / "payload"
        safe_extract_targz(snapshot, payload)
        for name in names:
            result = scan_skill(payload / "skills" / name, source="community")
            allowed, reason = should_allow_install(result)
            if allowed is not True:
                raise ValueError(f"Skills Guard rejected {name!r}: {reason}")

        root = get_hermes_home() / "skills"

        def check_target(name):
            target = root
            if _is_path_redirect(root):
                raise ValueError("Cannot import through a redirected skills directory")
            for part in name.split("/"):
                if (target / "SKILL.md").exists():
                    raise ValueError(f"Cannot nest an import inside an existing skill: {name!r}")
                target = target / part
                if _is_path_redirect(target):
                    raise ValueError(f"Cannot import through a redirected path: {name!r}")
            if target.exists():
                raise ValueError(f"Import destination already exists: {name!r}")
            return target

        # Refuse known conflicts before installing any selection.
        for name in names:
            check_target(name)
        for name in names:
            target = check_target(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir()  # Exclusive reservation: never merge into an existing tree.
            source = payload / "skills" / name
            entries = sorted(source.rglob("*"), key=lambda p: p.relative_to(source).as_posix())
            for entry in entries:
                relative = entry.relative_to(source)
                if relative.as_posix() == "SKILL.md":
                    continue
                destination = target / relative
                if entry.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with entry.open("rb") as src, destination.open("xb") as dst:
                        shutil.copyfileobj(src, dst)
                    destination.chmod(entry.stat().st_mode & 0o777)
            # Never expose a partially written instructional file to discovery.
            # Close before link/unlink for Windows, as in skills_export.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=target, prefix=".import-", delete=False) as dst:
                    temporary = Path(dst.name)
                    with (source / "SKILL.md").open("rb") as src:
                        shutil.copyfileobj(src, dst)
                os.link(temporary, target / "SKILL.md")
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return names


def import_command(args) -> None:
    try:
        names = (validate_archive(Path(args.archive)) if args.dry_run
                 else import_skills(Path(args.archive)))
    except (OSError, ValueError, tarfile.TarError, UnicodeError) as exc:
        print(f"Skill import failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not args.dry_run:
        print("Imported scanned local skills (no Hub update ownership):")
        for name in names:
            print(f"  {name!r}")
        return
    print("Archive integrity verified; no skill contents installed:")
    for name in names:
        print(f"  {name!r}")
    print("Integrity is not publisher authentication or a security scan.")
    print("Normal CLI startup may initialize profile directories and logs.")
