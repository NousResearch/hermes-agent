"""Stable macOS TCC anchor for the uv-managed Python interpreter (issue #85345).

macOS keys TCC grants (Files & Folders, Photos, Media Library, Automation,
...) to the *resolved absolute path* of the client binary.  Hermes' interpreter
is managed by uv and lives at ``~/.local/share/uv/python/cpython-<ver>-macos-*/
bin/python*``; every patch bump materializes a NEW versioned directory, so the
TCC client string changes and every prior grant is orphaned — macOS re-prompts
for all permissions after each update.

Symlinks do not help: TCC resolves through them to the versioned store path
before matching (the venv's ``bin/python`` -> store symlink is exactly why the
client is reported as ``.../cpython-3.11.15-macos-.../bin/python3.11``).

The anchor: replace the venv's ``bin/python`` symlink with a *real-file copy*
of the interpreter binary.  The venv path (``<checkout>/venv/bin/python``) is
stable across ``hermes update``, and because it is a regular file there is no
symlink for TCC to resolve — so the TCC client path stays constant across
interpreter patch bumps.  ``pyvenv.cfg`` keeps pointing at the uv store (``home``),
which still provides the stdlib exactly as it does today.

The anchor self-heals: when ``hermes update`` / ``hermes doctor`` runs and the
venv python is a symlink again (uv re-created it) or the recorded source no
longer matches the current interpreter (patch bump), the copy is refreshed.
Versioned alias symlinks (``python3``, ``python3.11``, ...) inside the venv bin
dir are re-pointed at the anchor so no alias resolves back into the versioned
store.

All functions are no-ops on non-macOS and for interpreters that are not
uv-managed (Homebrew/system Python has a stable path already).  This module is
pure/best-effort: it never raises to callers (update/doctor must never break
because of it).
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import tempfile
from pathlib import Path

from hermes_constants import venv_python_path

logger = logging.getLogger(__name__)

# Marker file (inside the venv bin dir) recording the uv-store interpreter
# file the anchor copy was taken from.  Used to detect patch-bump staleness.
_MARKER_NAME = ".tcc-anchor-source"

# Alias symlinks uv creates inside the venv bin dir.  They resolve into the
# versioned store today; re-pointed at the stable anchor when it is installed.
_SIBLING_NAMES = ("python3", "python3.11", "python3.12", "python3.13")

# Path fragments that identify a uv-managed macOS CPython store layout:
# ``.../uv/python/cpython-<version>-macos-<arch>/bin/python*``.
_UV_STORE_MARKERS = ("/uv/python/", "cpython-", "-macos-")

# Preferred interpreter file names inside a store ``bin`` dir (versioned
# names first so the real binary is picked over the ``python3`` alias).
_STORE_BIN_NAMES = ("python3.11", "python3.12", "python3.13", "python3", "python")


def is_macos() -> bool:
    """True on macOS (the only platform with TCC)."""
    return platform.system() == "Darwin"


def _is_uv_macos_store(path: str | Path) -> bool:
    """True when *path* lives inside a uv-managed macOS CPython store."""
    text = str(path).replace("\\", "/")
    return all(marker in text for marker in _UV_STORE_MARKERS)


def _venv_dir(project_root: Path | None = None) -> Path | None:
    """Return the checkout's venv dir, mirroring ``managed_uv``'s probing.

    ``venv`` wins when it holds an interpreter (managed layout takes
    precedence); otherwise fall back to ``.venv`` (uv-default/dev checkouts).
    Returns None when neither holds an interpreter.
    """
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    for name in ("venv", ".venv"):
        candidate = root / name
        venv_py = venv_python_path(candidate)
        if venv_py.is_file() or venv_py.is_symlink():
            return candidate
    return None


def _interpreter_file(src: str | Path) -> Path | None:
    """Return the interpreter binary file at/inside *src*.

    *src* is either a resolved store binary path (symlinked venv layout) or a
    store ``bin`` dir read from ``pyvenv.cfg`` ``home`` (anchored layout).
    """
    p = Path(src)
    if p.is_file():
        return p
    if not p.is_dir():
        return None
    for name in _STORE_BIN_NAMES:
        candidate = p / name
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _interpreter_source(venv_dir: Path) -> str | None:
    """Return the interpreter file the venv currently resolves to.

    A symlinked ``bin/python`` (uv's layout) resolves to the versioned store
    binary.  A regular-file anchor instead reads ``pyvenv.cfg`` ``home`` (the
    base interpreter's bin dir) — that is what the anchor copy was taken from
    and where the stdlib still comes from.
    """
    venv_py = venv_python_path(venv_dir)
    if venv_py.is_symlink():
        try:
            resolved = venv_py.resolve(strict=False)
        except OSError:
            return None
        if resolved.is_file():
            return str(resolved)
        return None
    cfg = venv_dir / "pyvenv.cfg"
    try:
        text = cfg.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        if line.strip().lower().startswith("home"):
            _, _, value = line.partition("=")
            home = value.strip()
            if home:
                return str(_interpreter_file(Path(home)))
    return None


def _anchor_marker(venv_bin: Path) -> Path:
    return venv_bin / _MARKER_NAME


def _repoint_aliases(venv_bin: Path, anchor: Path) -> None:
    """Re-point uv alias symlinks at the stable anchor.

    ``python3`` / ``python3.11`` inside the venv bin dir currently resolve into
    the versioned store; anything spawned through them would still churn TCC.
    Only symlinks that resolve into the uv store are touched.
    """
    for name in _SIBLING_NAMES:
        alias = venv_bin / name
        try:
            if not alias.is_symlink():
                continue
            if not _is_uv_macos_store(str(alias.resolve(strict=False))):
                continue
            tmp = venv_bin / f".{name}.tcc-tmp"
            try:
                os.symlink(anchor.name, tmp)
                os.replace(tmp, alias)
            except OSError:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            continue


def _install_anchor(venv_dir: Path, source_file: Path) -> None:
    """Replace ``bin/python`` with a real-file copy of *source_file*.

    Atomic (temp file + rename) so a crash mid-copy cannot leave the venv
    interpreter half-written.  Writes the source marker and re-points alias
    symlinks so the whole venv bin dir resolves to stable paths.
    """
    venv_py = venv_python_path(venv_dir)
    venv_bin = venv_py.parent
    venv_bin.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".python-tcc-", dir=str(venv_bin))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(source_file, tmp_path)
        os.chmod(tmp_path, source_file.stat().st_mode | 0o111)
        os.replace(tmp_path, venv_py)
        _anchor_marker(venv_bin).write_text(str(source_file), encoding="utf-8")
        _repoint_aliases(venv_bin, venv_py)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def ensure_tcc_anchor(project_root: Path | None = None) -> Path | None:
    """Pin a stable interpreter anchor for macOS TCC (issue #85345).

    No-op (returns None) on non-macOS, when no venv interpreter exists, or when
    the interpreter is not uv-managed.  Otherwise makes the venv's ``bin/python``
    a real-file copy of the current uv-store interpreter and returns its path.
    Idempotent: a fresh anchor is returned unchanged.  Best-effort — returns
    None (and logs) if the copy fails; callers must never depend on success.
    """
    if not is_macos():
        return None
    venv_dir = _venv_dir(project_root)
    if venv_dir is None:
        return None
    venv_py = venv_python_path(venv_dir)
    if not (venv_py.is_file() or venv_py.is_symlink()):
        return None
    source = _interpreter_source(venv_dir)
    if source is None or not _is_uv_macos_store(source):
        return None
    source_file = _interpreter_file(source)
    if source_file is None:
        return None
    if not venv_py.is_symlink():
        # Already anchored — refresh only when the interpreter changed.
        marker = _anchor_marker(venv_py.parent)
        try:
            if marker.is_file() and marker.read_text(encoding="utf-8").strip() == str(
                source_file
            ):
                return venv_py
        except OSError:
            pass
    try:
        _install_anchor(venv_dir, source_file)
    except Exception as exc:  # best-effort: never break update/doctor
        logger.warning("macOS TCC anchor install failed: %s", exc)
        return None
    return venv_py


def tcc_anchor_state(project_root: Path | None = None) -> tuple[str, str]:
    """Report the anchor state for ``hermes doctor``.

    Returns ``(status, detail)`` with status one of:

    - ``"skip"``    — not applicable (non-macOS, no venv, or not uv-managed)
    - ``"active"``  — venv interpreter is pinned at a stable real-file anchor
    - ``"stale"``   — pinned but the interpreter changed since the last copy
    - ``"missing"`` — uv-managed interpreter with no stable anchor installed
    """
    if not is_macos():
        return "skip", "not macOS"
    venv_dir = _venv_dir(project_root)
    if venv_dir is None:
        return "skip", "no venv interpreter"
    venv_py = venv_python_path(venv_dir)
    if not (venv_py.is_file() or venv_py.is_symlink()):
        return "skip", "no venv interpreter"
    source = _interpreter_source(venv_dir)
    if source is None or not _is_uv_macos_store(source):
        return "skip", "interpreter not uv-managed (stable path)"
    if not venv_py.is_symlink():
        marker = _anchor_marker(venv_py.parent)
        try:
            if marker.is_file() and marker.read_text(encoding="utf-8").strip() == str(
                source
            ):
                return "active", str(venv_py)
        except OSError:
            pass
        return "stale", str(venv_py)
    return "missing", str(venv_py)
