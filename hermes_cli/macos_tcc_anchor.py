"""Stable macOS TCC anchor for the uv-managed Python interpreter.

macOS TCC grants are keyed to the interpreter binary's path; a venv ``bin/python`` that symlinks
into uv's store changes identity on every interpreter upgrade. The anchor replaces it with a
signed real-file copy, gated on a real boot, with uv's alias names (``python3``,
``python3.N``) materialized as real-file copies too (never symlinks — the #95541 crash shape)
and the store's ``libpython*`` hardlinked into ``venv/lib/`` (existing ``LC_RPATH`` already points
at ``@executable_path/../lib``). All functions are no-ops off macOS and for non-uv interpreters,
and never raise to callers.
"""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from hermes_constants import venv_python_path
from hermes_cli.managed_uv import _RUNTIME_DIR_NAME
from utils import atomic_write_text

logger = logging.getLogger(__name__)

_MARKER_NAME = ".tcc-anchor-source"

_STORE_COMMON_MARKERS = ("cpython-", "-macos-")
# Derived from managed_uv so a rename of the repair-generation directory cannot silently stop
# the anchor from matching.
_STORE_ROOT_MARKERS = ("/uv/python/", f"/{_RUNTIME_DIR_NAME}/python/")

_ALIAS_NAMES = ("python3", f"python3.{sys.version_info.minor}")
_STORE_BIN_NAMES = (f"python3.{sys.version_info.minor}", "python3", "python")


class _BootGateFailed(Exception):
    """Staged copy refused to boot; the live venv must stay untouched."""


def _marker_value(source_file: Path) -> str:
    """Fully resolved so symlinked spellings of the same store binary compare equal."""
    return os.path.realpath(str(source_file))


def is_macos() -> bool:
    return platform.system() == "Darwin"


def _is_uv_macos_store(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if not all(marker in normalized for marker in _STORE_COMMON_MARKERS):
        return False
    return any(marker in normalized for marker in _STORE_ROOT_MARKERS)


def _venv_dir(project_root: Path | None = None) -> Path | None:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    return next((root / n for n in ("venv", ".venv") if _present(venv_python_path(root / n))), None)


def _present(path: Path) -> bool:
    return path.is_file() or path.is_symlink()


def _interpreter_file(src: str | Path) -> Path | None:
    """Return the interpreter binary file at/inside *src*."""
    p = Path(src)
    if p.is_file():
        return p
    if not p.is_dir():
        return None
    try:
        candidates = [p / n for n in _STORE_BIN_NAMES] + sorted(
            c for c in p.glob("python3.*") if not c.name.endswith((".dSYM", ".txt"))
        )
        return next((c for c in candidates if c.is_file()), None)
    except OSError:
        return None


def _interpreter_source(venv_dir: Path) -> str | None:
    """Return the interpreter file the venv currently resolves to (symlink target or pyvenv.cfg home)."""
    venv_py = venv_python_path(venv_dir)
    if venv_py.is_symlink():
        try:
            return str(venv_py.resolve(strict=False))
        except OSError:
            return None
    cfg = venv_dir / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    try:
        lines = cfg.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    home = next((l.partition("=")[2].strip() for l in lines if l.lower().startswith("home")), "")
    interp = _interpreter_file(home) if home else None
    return str(interp) if interp is not None else None


def _managed_venv(project_root: Path | None) -> tuple[Path, Path, str] | str:
    """``(venv_dir, venv_py, source)`` for a uv-managed macOS venv, else the skip reason."""
    if not is_macos():
        return "not macOS"
    venv_dir = _venv_dir(project_root)
    if venv_dir is None:
        return "no venv interpreter"
    source = _interpreter_source(venv_dir)
    if source is None or not _is_uv_macos_store(source):
        return "interpreter not uv-managed (stable path)"
    return venv_dir, venv_python_path(venv_dir), source


def _anchor_marker(venv_bin: Path) -> Path:
    return venv_bin / _MARKER_NAME


def _marker_matches(venv_bin: Path, expected: str) -> bool:
    marker = _anchor_marker(venv_bin)
    try:
        return marker.is_file() and marker.read_text(encoding="utf-8").strip() == expected
    except OSError:
        return False


def _write_marker(venv_bin: Path, source_file: Path) -> None:
    """Atomic: a concurrent ensure (update + doctor --fix) must never read a torn marker, which
    would compare unequal and trigger a spurious reinstall."""
    atomic_write_text(
        _anchor_marker(venv_bin),
        _marker_value(source_file),
        tmp_prefix=f"{_MARKER_NAME}.",
    )


def _store_root(source_file: Path) -> Path:
    # .../cpython-<ver>-macos-*/bin/python3.N → store root
    return source_file.resolve(strict=False).parent.parent


def _provision_libpython(venv_dir: Path, source_file: Path, *, refresh: bool = False) -> None:
    """Hardlink (else copy) store ``libpython*`` into ``venv/lib/``.

    Provision-if-present: a surplus hardlink on a statically-linked build is free; a missed
    detection is the only way the dylib-not-found crash returns.

    See #95425.
    """
    src_lib = _store_root(source_file) / "lib"
    if not src_lib.is_dir():
        return
    dst_lib = venv_dir / "lib"
    try:
        dst_lib.mkdir(parents=True, exist_ok=True)
        for src in src_lib.glob("libpython*"):
            if not src.is_file():
                continue
            dst = dst_lib / src.name
            if dst.exists() or dst.is_symlink():
                if not refresh:
                    continue
                try:
                    dst.unlink()
                except OSError:
                    continue
            try:
                os.link(src, dst)
            except OSError:
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    logger.debug("libpython provision failed for %s", src, exc_info=True)
    except OSError:
        logger.debug("libpython provision skipped", exc_info=True)


def _stage_copy(venv_bin: Path, prefix: str, source: Path) -> Path:
    """Copy *source* to a unique (mkstemp) executable staging file in *venv_bin*.

    Unique names mean a concurrent ensure cannot promote another run's truncated interim copy.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=prefix, dir=str(venv_bin))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(source, tmp_path)
        os.chmod(tmp_path, source.stat().st_mode | 0o111)
    except BaseException:
        _discard(tmp_path)
        raise
    return tmp_path


def _discard(tmp_path: Path | None) -> None:
    if tmp_path is not None:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)


def _copy_alias(venv_bin: Path, name: str, anchor: Path) -> bool:
    """Materialize *name* as a real-file copy of *anchor* (atomic rename).

    Returns False (and warns) on failure: a leftover alias *symlink* to the anchor is the exact
    crash shape, so callers must know when the alias set is incomplete.
    """
    tmp_path: Path | None = None
    try:
        tmp_path = _stage_copy(venv_bin, f".{name}.tcc-", anchor)
        os.replace(tmp_path, venv_bin / name)
        return True
    except OSError as exc:
        logger.warning("TCC anchor alias %s not materialized: %s", name, exc)
        _discard(tmp_path)
        return False


def _materialize_aliases(venv_bin: Path, anchor: Path, *, refresh: bool = False) -> bool:
    """Materialize uv alias names as real-file copies; True only if every needed one succeeded."""
    names = set(_ALIAS_NAMES)
    try:
        names.update(
            p.name for p in venv_bin.glob("python3*") if re.fullmatch(r"python3(\.\d+)?", p.name)
        )
    except OSError:
        pass
    ok = True
    for name in sorted(names):
        alias = venv_bin / name
        try:
            if refresh or alias.is_symlink() or not alias.exists():
                ok = _copy_alias(venv_bin, name, anchor) and ok
        except OSError:
            ok = False
    return ok


def _passes_boot_gate(staged: Path, venv_dir: Path) -> bool:
    """Launch *staged* and demand encodings + the venv prefix.

    Runs with ``PYTHONHOME``/``PYTHONPATH`` scrubbed: an inherited PYTHONHOME papers over the very
    prefix failure this gate exists to catch. ``ENOENT``/``ENOEXEC`` (fixture binaries, foreign
    arch) mean the binary can't run here at all, so the symlinked venv was equally dead: skip.
    Anything else (notably ``EACCES`` after our own chmod) is a broken install going live: refuse.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "__PYVENV_LAUNCHER__")
    }
    try:
        proc = subprocess.run(
            [str(staged), "-c", "import encodings, sys; print(sys.prefix)"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ENOEXEC):
            logger.debug("boot gate skipped: cannot execute %s (%s)", staged, exc)
            return True
        logger.warning("boot gate: staged copy not executable: %s", exc)
        return False
    except subprocess.TimeoutExpired:
        return False
    if proc.returncode != 0:
        return False
    printed = (proc.stdout or "").strip().splitlines()
    if not printed:
        return False
    try:
        return Path(printed[-1]).resolve() == venv_dir.resolve()
    except OSError:
        return str(venv_dir) in printed[-1]


def _install_anchor(venv_dir: Path, source_file: Path) -> None:
    """Replace ``bin/python`` with a signed copy, gated on a real boot."""
    venv_py = venv_python_path(venv_dir)
    venv_bin = venv_py.parent
    venv_bin.mkdir(parents=True, exist_ok=True)

    _provision_libpython(venv_dir, source_file, refresh=True)

    tmp_path = _stage_copy(venv_bin, ".python-tcc-", source_file)
    try:
        try:
            from hermes_cli.managed_uv import _macos_sign_managed_python

            _macos_sign_managed_python(tmp_path)
        except Exception:  # pragma: no cover - never block the anchor
            logger.debug("anchor copy signing skipped", exc_info=True)
        if not _passes_boot_gate(tmp_path, venv_dir):
            raise _BootGateFailed(f"staged copy at {tmp_path} failed encodings/prefix probe")
        os.replace(tmp_path, venv_py)
        if _materialize_aliases(venv_bin, venv_py, refresh=True):
            # Marker last, atomically: it asserts the WHOLE layout (anchor + aliases) is complete.
            # A partial alias set must not read "active" in doctor; an absent marker makes the
            # next ensure retry the install.
            _write_marker(venv_bin, source_file)
        else:
            logger.warning(
                "TCC anchor installed but alias materialization was "
                "incomplete; leaving anchor unmarked so the next run retries"
            )
    except Exception:
        _discard(tmp_path)
        raise


def ensure_tcc_anchor(project_root: Path | None = None) -> Path | None:
    """Pin a dylib-complete interpreter anchor for macOS TCC.

    No-op (None) on non-macOS, without a venv interpreter, or when the interpreter is not
    uv-managed. Idempotent. Best-effort — None (and logs) if the copy or boot-gate fails; callers
    must never depend on success.

    See #95596.
    """
    found = _managed_venv(project_root)
    if isinstance(found, str):
        return None
    venv_dir, venv_py, source = found
    source_file = _interpreter_file(source)
    if source_file is None:
        return None
    if not venv_py.is_symlink() and _marker_matches(venv_py.parent, _marker_value(source_file)):
        try:
            _provision_libpython(venv_dir, source_file, refresh=False)
            if _passes_boot_gate(venv_py, venv_dir):
                _materialize_aliases(venv_py.parent, venv_py)
                return venv_py
        except OSError:
            pass
    try:
        _install_anchor(venv_dir, source_file)
    except _BootGateFailed as exc:
        logger.warning("macOS TCC anchor boot-gate refused install: %s", exc)
        return None
    except Exception as exc:  # best-effort: never break update/doctor
        logger.warning("macOS TCC anchor install failed: %s", exc)
        return None
    return venv_py


def tcc_anchor_state(project_root: Path | None = None) -> tuple[str, str]:
    """Report the anchor state for ``hermes doctor`` as ``(status, detail)``.

    ``skip`` = not applicable; ``active`` = pinned at a stable real-file anchor; ``stale`` = pinned
    but the interpreter changed since the last copy; ``missing`` = uv-managed with no anchor.
    """
    found = _managed_venv(project_root)
    if isinstance(found, str):
        return "skip", found
    _venv, venv_py, source = found
    if venv_py.is_symlink():
        return "missing", str(venv_py)
    source_file = _interpreter_file(source)
    expected = _marker_value(source_file) if source_file is not None else source
    status = "active" if _marker_matches(venv_py.parent, expected) else "stale"
    return status, str(venv_py)
