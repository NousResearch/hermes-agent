#!/usr/bin/env python3
"""Install psutil on Termux/Android by patching upstream platform detection.

psutil's setup currently gates Linux sources behind
``sys.platform.startswith('linux')``. On Termux, Python reports
``sys.platform == 'android'``, so ``pip install psutil`` aborts with
"platform android is not supported" — even though psutil compiles fine
when the Linux source path is reused.

This script downloads the official psutil sdist, applies a one-line
patch (``LINUX = sys.platform.startswith(("linux", "android"))``), and
installs the patched tree with ``pip install --no-build-isolation``.

Usage:
    python scripts/install_psutil_android.py [--pip "/path/to/pip"] [--uv]

When neither flag is given, the script auto-detects ``uv`` on PATH and
falls back to ``<sys.executable> -m pip``.

This is a stopgap. Remove once psutil upstream merges
https://github.com/giampaolo/psutil/pull/2762 and ships a release.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# Pin a version we know patches cleanly. Update when a newer psutil
# changes the marker line shape and we need to follow upstream.
PSUTIL_URL = (
    "https://files.pythonhosted.org/packages/aa/c6/"
    "d1ddf4abb55e93cebc4f2ed8b5d6dbad109ecb8d63748dd2b20ab5e57ebe/"
    "psutil-7.2.2.tar.gz"
)
PSUTIL_SHA256 = "0746f5f8d406af344fd547f1c8daa5f5c33dbc293bb8d6a16d80b4bb88f59372"

MARKER = 'LINUX = sys.platform.startswith("linux")'
REPLACEMENT = 'LINUX = sys.platform.startswith(("linux", "android"))'


def _resolve_install_cmd(pip_arg: str | None, prefer_uv: bool) -> list[str]:
    if pip_arg:
        return pip_arg.split()
    if prefer_uv:
        uv = shutil.which("uv")
        if not uv:
            sys.exit("--uv requested but no uv on PATH")
        return [uv, "pip"]
    auto_uv = shutil.which("uv")
    if auto_uv:
        return [auto_uv, "pip"]
    return [sys.executable, "-m", "pip"]


def _verify_archive_checksum(archive: Path, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise RuntimeError(
            f"psutil sdist checksum mismatch: expected {expected_sha256}, got {actual}"
        )


def _safe_extract_tar(tar: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    members = tar.getmembers()
    for member in members:
        name = member.name
        target = (destination / name).resolve()
        try:
            target.relative_to(destination)
        except ValueError:
            raise RuntimeError(f"Unsafe path in psutil sdist: {name}") from None
        if Path(name).is_absolute():
            raise RuntimeError(f"Unsafe path in psutil sdist: {name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Unsafe link in psutil sdist: {name}")
        if not (member.isfile() or member.isdir()):
            raise RuntimeError(f"Unsupported tar entry in psutil sdist: {name}")
    tar.extractall(destination, members=members)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pip",
        help="Explicit installer command (e.g. '/usr/bin/uv pip' or 'python -m pip')",
    )
    parser.add_argument(
        "--uv",
        action="store_true",
        help="Force using uv (errors out if uv is not on PATH)",
    )
    args = parser.parse_args()

    install_cmd_prefix = _resolve_install_cmd(args.pip, args.uv)

    print(
        "→ Termux/Android: prebuilding psutil with Linux source path "
        "compatibility shim (see psutil#2762)..."
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "psutil.tar.gz"
        urllib.request.urlretrieve(PSUTIL_URL, archive)
        _verify_archive_checksum(archive, PSUTIL_SHA256)
        with tarfile.open(archive) as tar:
            _safe_extract_tar(tar, tmp_path)

        try:
            src_root = next(
                p for p in tmp_path.iterdir()
                if p.is_dir() and p.name.startswith("psutil-")
            )
        except StopIteration:
            sys.exit("psutil sdist did not contain a psutil-* directory")

        common_py = src_root / "psutil" / "_common.py"
        content = common_py.read_text(encoding="utf-8")
        if MARKER not in content:
            sys.exit(
                "psutil Android compatibility patch marker not found — "
                "upstream may have changed the LINUX detection line. "
                "Update MARKER/REPLACEMENT in this script."
            )
        common_py.write_text(content.replace(MARKER, REPLACEMENT), encoding="utf-8")

        cmd = install_cmd_prefix + ["install", "--no-build-isolation", str(src_root)]
        print(f"  $ {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode

    print("✓ psutil installed via Android compatibility shim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
