from __future__ import annotations

import tarfile
from pathlib import Path


def create_tar_gz(source_dir: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_dir, arcname=source_dir.name)


def extract_tar_gz(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(destination)
    members = [path for path in destination.iterdir()]
    if len(members) != 1:
        raise ValueError(f"Unexpected archive layout in {archive_path}")
    return members[0]
