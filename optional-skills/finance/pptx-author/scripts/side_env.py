#!/usr/bin/env python3
"""Build (or reuse) a pinned side environment under ~/.cache and print its interpreter path.

The fallback for a terminal without Hermes (a Docker sandbox); where `hermes` is on PATH,
`hermes pm environment` does this job instead. The same file ships in the pdf,
microsoft-office, excel-author and pptx-author skills; keep the copies identical.

Safe to run from several agents at once: a builder holds `<name>.lock`, re-checks the
published environment after waiting for it, builds into a private stage and publishes it by
atomically swapping the `<name>` symlink. Stages left by a crashed build and replaced
generations are deleted once they are an hour old.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

try:
    import fcntl
except ImportError:  # not POSIX: no lock; the atomic publish still keeps the env whole
    fcntl = None

# "documents" is the pyproject `documents` + `documents-pdf` extras (a test keeps them equal).
PRESETS = {
    "documents": ("hermes-docs-venv", (
        "python-docx==1.2.0", "openpyxl==3.1.5", "python-pptx==1.0.2", "reportlab==5.1.0",
        "pypdf==6.21.2", "pdfplumber==0.11.10", "pypdfium2==5.14.0",
    ), ("docx", "openpyxl", "pptx", "reportlab", "pypdf", "pdfplumber", "pypdfium2")),
    "pdf-pymupdf": ("hermes-pdf-pymupdf-venv", ("pymupdf>=1.26,<3", "pymupdf4llm>=0.2,<3"),
                    ("pymupdf", "pymupdf4llm")),
    "pdf-marker": ("hermes-pdf-marker-venv", ("marker-pdf>=1.8,<3",), ("marker",)),
}
STALE_AFTER_S = 3600


def _imports_ok(python: Path, modules) -> bool:
    try:
        return subprocess.run([str(python), "-c", "import " + ", ".join(modules)], check=False, timeout=300,
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _make_venv(stage: Path, requirements) -> None:
    # stdout goes to stderr: the caller captures stdout as the interpreter path
    subprocess.run([sys.executable, "-m", "venv", str(stage)], check=True, stdin=subprocess.DEVNULL,
                   stdout=sys.stderr, timeout=600)
    subprocess.run([str(stage / "bin" / "python"), "-m", "pip", "install", "-q", "--disable-pip-version-check",
                    *requirements], check=True, stdin=subprocess.DEVNULL, stdout=sys.stderr,
                   timeout=3 * 3600)  # marker pulls PyTorch


@contextmanager
def _lock(path: Path, wait: bool = True):
    """Hold an exclusive lock on ``path``; yields False when ``wait`` is off and it is taken."""
    with open(path, "a", encoding="utf-8") as handle:  # closing the file releases the lock
        held = True
        if fcntl is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held = False
        yield held


def _sweep(cache: Path, name: str) -> None:
    """Delete stages and generations of ``name`` that are not published and are over an hour old."""
    try:
        published = os.path.basename(os.readlink(cache / name))
    except OSError:
        published = None
    cutoff = time.time() - STALE_AFTER_S
    for entry in cache.glob(name + ".*"):
        if entry.name in (published, name + ".lock") or entry.lstat().st_mtime > cutoff:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)


def ensure(preset: str, cache: Path) -> Path:
    """The interpreter of ``preset``'s published environment under ``cache``, built if needed."""
    name, requirements, modules = PRESETS[preset]
    cache.mkdir(parents=True, exist_ok=True)
    python = cache / name / "bin" / "python"
    if _imports_ok(python, modules):
        with _lock(cache / f"{name}.lock", wait=False) as held:
            if held:
                _sweep(cache, name)
        return python
    with _lock(cache / f"{name}.lock"):
        if _imports_ok(python, modules):  # another caller published it while this one waited
            return python
        _sweep(cache, name)
        stage = Path(tempfile.mkdtemp(prefix=name + ".", dir=cache))
        try:
            _make_venv(stage, requirements)
            if not _imports_ok(stage / "bin" / "python", modules):
                raise RuntimeError(f"{stage.name}: the installed packages do not import")
            try:
                old = os.path.basename(os.readlink(cache / name))
            except OSError:
                old = None
            link = cache / (stage.name + ".lnk")
            os.symlink(stage.name, link)
            os.replace(link, cache / name)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
        if old and old.startswith(name + "."):
            shutil.rmtree(cache / old, ignore_errors=True)
    return python


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build or reuse a pinned side environment; "
                                     "prints its interpreter path.")
    parser.add_argument("preset", choices=sorted(PRESETS))
    args = parser.parse_args(argv)
    try:
        python = ensure(args.preset, Path.home() / ".cache")
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"could not build the {args.preset} environment: {exc}", file=sys.stderr)
        return 1
    print(python)
    return 0


if __name__ == "__main__":
    sys.exit(main())
