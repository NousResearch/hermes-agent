"""Helpers for the temporary psutil-on-Android compatibility installer."""

from __future__ import annotations

import shutil

import logging
logger = logging.getLogger(__name__)

import tarfile
from pathlib import Path, PurePosixPath

# Pin a version we know patches cleanly. Update when a newer psutil
# changes the marker line shape and we need to follow upstream.
PSUTIL_URL = (
    "https://files.pythonhosted.org/packages/aa/c6/"
    "d1ddf4abb55e93cebc4f2ed8b5d6dbad109ecb8d63748dd2b20ab5e57ebe/"
    "psutil-7.2.2.tar.gz"
)

MARKER = 'LINUX = sys.platform.startswith("linux")'
REPLACEMENT = 'LINUX = sys.platform.startswith(("linux", "android"))'


class PsutilAndroidInstallError(RuntimeError):
    """Raised when the pinned psutil sdist is missing or unsafe."""


def _normalize_member_parts(member_name: str) -> tuple[str, ...]:
    path = PurePosixPath(member_name)
    parts = tuple(part for part in path.parts if part not in ("", "."))
    if path.is_absolute() or ".." in parts or not parts:
        raise PsutilAndroidInstallError(
            f"Unsafe archive member path: {member_name!r}"
        )
    return parts


def _safe_extract_tar_gz(archive: Path, destination: Path) -> None:
    """Extract a tar.gz without allowing traversal or link members."""
    with tarfile.open(archive, "r:gz") as tf:
        for member in tf.getmembers():
            parts = _normalize_member_parts(member.name)
            target = destination.joinpath(*parts)

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                raise PsutilAndroidInstallError(
                    f"Unsupported archive member type: {member.name}"
                )

            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tf.extractfile(member)
            if extracted is None:
                raise PsutilAndroidInstallError(
                    f"Cannot read archive member: {member.name}"
                )

            with extracted, open(target, "wb") as dst:
                shutil.copyfileobj(extracted, dst)

            try:
                target.chmod(member.mode & 0o777)
            except OSError:
                logger.debug("Suppressed exception", exc_info=True)


def prepare_patched_psutil_sdist(archive: Path, destination: Path) -> Path:
    """Safely extract the pinned psutil sdist and patch it for Android."""
    _safe_extract_tar_gz(archive, destination)

    src_roots = sorted(
        (
            path for path in destination.iterdir()
            if path.is_dir() and path.name.startswith("psutil-")
        ),
        key=lambda path: path.name,
    )
    if not src_roots:
        raise PsutilAndroidInstallError(
            "psutil sdist did not contain a psutil-* directory"
        )

    src_root = src_roots[0]
    common_py = src_root / "psutil" / "_common.py"
    if not common_py.is_file():
        raise PsutilAndroidInstallError(
            f"psutil sdist did not contain {common_py.relative_to(src_root)!s}"
        )
    try:
        content = common_py.read_text(encoding="utf-8")
    except OSError as exc:
        raise PsutilAndroidInstallError(
            f"Failed to read {common_py.relative_to(src_root)!s}"
        ) from exc
    if MARKER not in content:
        raise PsutilAndroidInstallError(
            "psutil Android compatibility patch marker not found"
        )
    try:
        common_py.write_text(
            content.replace(MARKER, REPLACEMENT),
            encoding="utf-8",
        )
    except OSError as exc:
        raise PsutilAndroidInstallError(
            f"Failed to write {common_py.relative_to(src_root)!s}"
        ) from exc
    return src_root
