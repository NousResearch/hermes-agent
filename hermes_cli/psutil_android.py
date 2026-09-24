"""Helpers for the temporary psutil-on-Android compatibility installer."""

from __future__ import annotations

from pathlib import Path

from hermes_cli.archive_safe import safe_extract_targz

# Pinned to a version whose marker line patches cleanly; bump when upstream changes its shape.
PSUTIL_URL = (
    "https://files.pythonhosted.org/packages/aa/c6/"
    "d1ddf4abb55e93cebc4f2ed8b5d6dbad109ecb8d63748dd2b20ab5e57ebe/psutil-7.2.2.tar.gz"
)

MARKER = 'LINUX = sys.platform.startswith("linux")'
REPLACEMENT = 'LINUX = sys.platform.startswith(("linux", "android"))'


class PsutilAndroidInstallError(RuntimeError):
    """Raised when the pinned psutil sdist is missing or unsafe."""


def prepare_patched_psutil_sdist(archive: Path, destination: Path) -> Path:
    """Safely extract the pinned psutil sdist and patch it for Android."""
    try:
        safe_extract_targz(archive, destination)  # rejects traversal, links and device nodes
    except ValueError as exc:
        raise PsutilAndroidInstallError(str(exc)) from exc
    src_roots = [path for path in destination.iterdir() if path.is_dir() and path.name.startswith("psutil-")]
    if not src_roots:
        raise PsutilAndroidInstallError("psutil sdist did not contain a psutil-* directory")
    src_root = min(src_roots, key=lambda path: path.name)
    common_py = src_root / "psutil" / "_common.py"
    rel = common_py.relative_to(src_root)
    if not common_py.is_file():
        raise PsutilAndroidInstallError(f"psutil sdist did not contain {rel!s}")
    try:
        content = common_py.read_text(encoding="utf-8")
    except OSError as exc:
        raise PsutilAndroidInstallError(f"Failed to read {rel!s}") from exc
    if MARKER not in content:
        raise PsutilAndroidInstallError("psutil Android compatibility patch marker not found")
    try:
        common_py.write_text(content.replace(MARKER, REPLACEMENT), encoding="utf-8")
    except OSError as exc:
        raise PsutilAndroidInstallError(f"Failed to write {rel!s}") from exc
    return src_root
