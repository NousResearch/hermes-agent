"""Explicit, content-preserving exports of profile-local skill directories."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile
import tempfile

from hermes_cli.archive_safe import normalize_archive_parts
from hermes_constants import get_hermes_home


def export_skills(names: list[str], output: Path) -> Path:
    """Stage selected contents before publishing a new archive; never overwrite."""
    root = get_hermes_home() / "skills"
    output = output.expanduser().absolute()
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Export destination must be outside the skills directory")
    if not names:
        raise ValueError("Select at least one skill directory")
    if not output.parent.is_dir():
        raise ValueError(f"Export parent directory does not exist: {output.parent}")
    if root.is_symlink():
        raise ValueError("Cannot export through a symlinked skills directory")
    if output.exists() or output.is_symlink():
        raise ValueError(f"Export already exists: {output}")
    selected = []
    for name in names:
        parts = normalize_archive_parts(name)
        source = root
        for part in parts:
            source = source / part
            if source.is_symlink():
                raise ValueError(f"Cannot export symlink: {source}")
        if not (source / "SKILL.md").is_file():
            raise ValueError(f"Not a skill directory: {name}")
        relative = Path(*parts).as_posix()
        if relative not in selected:
            selected.append(relative)

    manifest = {"format_version": 1, "skills": selected, "files": {}}
    # A private staging tree avoids exposing partial archives and keeps the hashes
    # tied to the actual archived bytes, not a second read of a changing source.
    with tempfile.TemporaryDirectory(prefix="hermes-skills-export-") as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        for name in selected:
            source = root / name
            for path in sorted(source.rglob("*")):
                mode = path.lstat().st_mode
                relative = "skills/" + path.relative_to(root).as_posix()
                if "/".join(normalize_archive_parts(relative)) != relative:
                    raise ValueError(f"Nonportable archive path: {path}")
                target = staged / relative
                if stat.S_ISDIR(mode):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not stat.S_ISREG(mode):
                    raise ValueError(f"Cannot export link or special file: {path}")
                if path.name.lower() in {".env", "auth.json"}:
                    raise ValueError(f"Remove credential file from selected skill: {path}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                target.chmod(stat.S_IMODE(mode) & 0o777)
                with target.open("rb") as content:
                    manifest["files"][relative] = hashlib.file_digest(content, "sha256").hexdigest()
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        # Publish with an exclusive link, unlike replace(), so a destination
        # created by another process during staging is never clobbered.
        fd, archive_path = tempfile.mkstemp(dir=output.parent, prefix=".skills-export-", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as temp_archive:
                with tarfile.open(fileobj=temp_archive, mode="w:gz", format=tarfile.GNU_FORMAT) as archive:
                    for path in sorted(staged.rglob("*")):
                        info = archive.gettarinfo(path, arcname=path.relative_to(staged).as_posix())
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        if info.isfile():
                            with path.open("rb") as content:
                                archive.addfile(info, content)
                        else:
                            archive.addfile(info)
            # Close before linking/unlinking: Windows denies deletion of open files.
            try:
                os.link(archive_path, output)
            except OSError as exc:
                detail = exc.strerror or "hard links are unavailable"
                raise ValueError(f"Cannot publish export at {output}: {detail}") from exc
        finally:
            Path(archive_path).unlink(missing_ok=True)
    return output


def export_command(args) -> None:
    try:
        output = export_skills(args.names, Path(args.output))
    except (OSError, ValueError) as exc:
        print(f"Skill export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Skills exported: {output}")
    print("Review before sharing: skill contents are not secret- or PII-scrubbed.")
