#!/usr/bin/env python3
"""Extract a checksummed Electron ZIP while preserving POSIX modes."""

from __future__ import annotations

import os
import shutil
import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath


def contained_path(root: Path, name: str) -> Path:
    member = PurePosixPath(name)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError(f"unsafe ZIP member: {name!r}")
    target = root.joinpath(*member.parts)
    if os.path.commonpath((root, target.resolve(strict=False))) != str(root):
        raise ValueError(f"ZIP member escapes output directory: {name!r}")
    return target


def extract(archive: Path, output: Path) -> None:
    root = output.resolve()
    root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = contained_path(root, member.filename)
            mode = (member.external_attr >> 16) & 0xFFFF

            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif stat.S_ISLNK(mode):
                target.parent.mkdir(parents=True, exist_ok=True)
                link_target = source.read(member).decode("utf-8")
                resolved = (target.parent / link_target).resolve(strict=False)
                if os.path.commonpath((root, resolved)) != str(root):
                    raise ValueError(f"unsafe ZIP symlink target: {member.filename!r}")
                target.unlink(missing_ok=True)
                target.symlink_to(link_target)
            elif mode == 0 or stat.S_IFMT(mode) == 0 or stat.S_ISREG(mode):
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            else:
                raise ValueError(f"unsupported ZIP member type: {member.filename!r}")

            permissions = stat.S_IMODE(mode)
            if permissions and not target.is_symlink():
                target.chmod(permissions)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract.py ARCHIVE OUTPUT_DIRECTORY")
    extract(Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2]))
