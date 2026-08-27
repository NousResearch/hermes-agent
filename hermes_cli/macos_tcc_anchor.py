"""Stable macOS TCC anchor for the uv-managed Python interpreter (#95596).

Re-land of the interpreter anchor reverted in #95563.  macOS keys TCC grants
to the resolved absolute path of the client binary.  Hermes' interpreter is
managed by uv and lives at a versioned store path; every patch bump orphans
every prior grant (#85345).

The first landing copied the interpreter into ``venv/bin/python`` but left
two holes that bricked real Macs:

* Dynamically-linked builds look up ``libpython`` via
  ``@executable_path/../lib``.  That resolved into ``venv/lib/``, which had
  no dylib — every hermes command, including update/doctor, died in dyld
  (#95425).
* Alias names (``python3``, ``python3.N``) were re-pointed at the copy as
  *symlinks*.  Invoking the copied interpreter through a symlink makes
  CPython getpath lose the venv prefix on affected python-build-standalone
  builds — startup dies with ``ModuleNotFoundError: encodings`` and the
  stdlib resolves to the build-time ``/install`` prefix (#95541).  Console
  scripts exec ``python3``, so the entire CLI surface died.

This re-land keeps the copy + identifier-pinned signature (TCC attribution
stays on the stable venv path) and closes both holes:

1. Aliases are materialized as real-file copies of the anchor, never
   symlinks.
2. If the store ships ``libpython*``, it is hardlinked into ``venv/lib/``
   (copy if the store is on another device).  Existing ``LC_RPATH`` already
   points at ``@executable_path/../lib`` — no rewrite.
3. A pre-install boot gate actually launches the staged copy and demands
   ``import encodings`` plus ``sys.prefix == <venv>``.  Failure rolls the
   staging file back and leaves the live interpreter untouched (a surplus
   provisioned dylib in ``venv/lib/`` may remain — harmless), so a bad
   anchor can never brick update/doctor again.

All functions are no-ops on non-macOS and for interpreters that are not
uv-managed.  Best-effort: never raises to callers.
"""

from __future__ import annotations

import errno
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from hermes_constants import venv_python_path
from hermes_cli.managed_uv import _RUNTIME_DIR_NAME

logger = logging.getLogger(__name__)

_MARKER_NAME = ".tcc-anchor-source"

_STORE_COMMON_MARKERS = ("cpython-", "-macos-")
# The runtime-store marker is derived from managed_uv so a rename of the
# repair-generation directory cannot silently stop the anchor from matching.
_STORE_ROOT_MARKERS = ("/uv/python/", f"/{_RUNTIME_DIR_NAME}/python/")


class _BootGateFailed(Exception):
    """Staged copy refused to boot; the live venv must stay untouched."""


def _marker_value(source_file: Path) -> str:
    """Canonical marker value: fully resolved so symlinked spellings of the
    same store binary (``cpython-3.11-macos-*`` → ``cpython-3.11.15-macos-*``)
    compare equal."""
    return os.path.realpath(str(source_file))


def is_macos() -> bool:
    return platform.system() == "Darwin"


def _sibling_names() -> tuple[str, ...]:
    """Alias names uv creates inside the venv bin dir."""
    import sys as _sys

    return ("python3", f"python3.{_sys.version_info.minor}")


def _store_bin_names() -> tuple[str, ...]:
    """Preferred interpreter file names inside a store ``bin`` dir."""
    import sys as _sys

    return (f"python3.{_sys.version_info.minor}", "python3", "python")


def _is_uv_macos_store(path: str) -> bool:
    normalized = path.replace("\\", "/")
    if not all(marker in normalized for marker in _STORE_COMMON_MARKERS):
        return False
    return any(marker in normalized for marker in _STORE_ROOT_MARKERS)


def _venv_dir(project_root: Path | None = None) -> Path | None:
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
    """Return the interpreter binary file at/inside *src*."""
    p = Path(src)
    if p.is_file():
        return p
    if not p.is_dir():
        return None
    for name in _store_bin_names():
        candidate = p / name
        if candidate.is_file():
            return candidate
    try:
        for candidate in sorted(p.glob("python3.*")):
            if candidate.is_file() and not candidate.name.endswith((".dSYM", ".txt")):
                return candidate
    except OSError:
        return None
    return None


def _interpreter_source(venv_dir: Path) -> str | None:
    """Return the interpreter file the venv currently resolves to."""
    venv_py = venv_python_path(venv_dir)
    if venv_py.is_symlink():
        try:
            resolved = venv_py.resolve(strict=False)
        except OSError:
            return None
        return str(resolved)
    cfg = venv_dir / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    home = ""
    try:
        for line in cfg.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("home"):
                _, _, home = line.partition("=")
                home = home.strip()
                break
    except OSError:
        return None
    if not home:
        return None
    interp = _interpreter_file(home)
    return str(interp) if interp is not None else None


def _anchor_marker(venv_bin: Path) -> Path:
    return venv_bin / _MARKER_NAME


def _write_marker(venv_bin: Path, source_file: Path) -> None:
    """Write the anchor marker atomically (write-then-rename).

    A concurrent ensure (update + doctor --fix) must never observe a
    partially-written marker: a torn read would compare unequal and trigger
    a spurious reinstall, and ``write_text`` alone is not atomic.
    """
    fd, tmp_name = tempfile.mkstemp(prefix=f"{_MARKER_NAME}.", dir=str(venv_bin))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(_marker_value(source_file))
        os.replace(tmp_name, _anchor_marker(venv_bin))
    except OSError:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _store_root(source_file: Path) -> Path:
    # .../cpython-<ver>-macos-*/bin/python3.N → store root
    return source_file.resolve(strict=False).parent.parent


def _provision_libpython(
    venv_dir: Path, source_file: Path, *, refresh: bool = False
) -> None:
    """Hardlink (else copy) store ``libpython*`` into ``venv/lib/``.

    Provision-if-present: a surplus hardlink on a statically-linked build is
    free; a missed detection is the only way #95425 returns.
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


def _copy_alias(venv_bin: Path, name: str, anchor: Path) -> bool:
    """Materialize *name* as a real-file copy of *anchor* (atomic rename).

    Returns False (and warns) on failure: a leftover alias *symlink* to the
    anchor is the exact #95541 crash shape, so callers must know when the
    alias set is incomplete.  The staging name is unique (mkstemp) so a
    concurrent ensure (update + doctor --fix) cannot promote a truncated
    interim copy.
    """
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{name}.tcc-", dir=str(venv_bin))
        os.close(fd)
        tmp_path = Path(tmp_name)
        shutil.copy2(anchor, tmp_path)
        os.chmod(tmp_path, anchor.stat().st_mode | 0o111)
        os.replace(tmp_path, venv_bin / name)
        return True
    except OSError as exc:
        logger.warning("TCC anchor alias %s not materialized: %s", name, exc)
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return False


def _materialize_aliases(
    venv_bin: Path, anchor: Path, *, refresh: bool = False
) -> bool:
    """Materialize uv alias names as real-file copies of the anchor.

    Returns True only when every alias that needed materializing succeeded.
    """
    names = set(_sibling_names())
    try:
        names.update(
            p.name
            for p in venv_bin.glob("python3*")
            if re.fullmatch(r"python3(\.\d+)?", p.name)
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
            continue
    return ok


def _passes_boot_gate(staged: Path, venv_dir: Path) -> bool:
    """Launch *staged* and demand encodings + the venv prefix.

    The probe runs with ``PYTHONHOME``/``PYTHONPATH`` scrubbed: an inherited
    ``PYTHONHOME`` papers over exactly the prefix-resolution failure the gate
    exists to catch.  ``OSError`` is split by errno — ``ENOENT``/``ENOEXEC``
    (fixture binaries, foreign-arch images) means the binary cannot run here
    at all, so the symlinked venv was equally dead and installing cannot make
    things worse: skip.  Anything else (notably ``EACCES`` after our own
    chmod) is a broken install about to go live: refuse.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP",
                     "__PYVENV_LAUNCHER__")
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

    fd, tmp_name = tempfile.mkstemp(prefix=".python-tcc-", dir=str(venv_bin))
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(source_file, tmp_path)
        os.chmod(tmp_path, source_file.stat().st_mode | 0o111)
        try:
            from hermes_cli.managed_uv import _macos_sign_managed_python

            _macos_sign_managed_python(tmp_path)
        except Exception:  # pragma: no cover - never block the anchor
            logger.debug("anchor copy signing skipped", exc_info=True)
        if not _passes_boot_gate(tmp_path, venv_dir):
            raise _BootGateFailed(
                f"staged copy at {tmp_path} failed encodings/prefix probe"
            )
        os.replace(tmp_path, venv_py)
        aliases_ok = _materialize_aliases(venv_bin, venv_py, refresh=True)
        if aliases_ok:
            # Marker last, atomically: it asserts the WHOLE layout (anchor +
            # aliases) is complete.  A partially-materialized alias set (the
            # #95541 crash shape when an alias stays a symlink) must not read
            # "active" in doctor — leaving the marker absent makes the next
            # ensure retry the install.
            _write_marker(venv_bin, source_file)
        else:
            logger.warning(
                "TCC anchor installed but alias materialization was "
                "incomplete; leaving anchor unmarked so the next run retries"
            )
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def ensure_tcc_anchor(project_root: Path | None = None) -> Path | None:
    """Pin a dylib-complete interpreter anchor for macOS TCC (#95596).

    No-op (returns None) on non-macOS, when no venv interpreter exists, or
    when the interpreter is not uv-managed.  Idempotent.  Best-effort —
    returns None (and logs) if the copy or boot-gate fails; callers must
    never depend on success.
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
        marker = _anchor_marker(venv_py.parent)
        try:
            if marker.is_file() and marker.read_text(encoding="utf-8").strip() == (
                _marker_value(source_file)
            ):
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
        source_file = _interpreter_file(source)
        expected = _marker_value(source_file) if source_file is not None else source
        try:
            if marker.is_file() and marker.read_text(encoding="utf-8").strip() == expected:
                return "active", str(venv_py)
        except OSError:
            pass
        return "stale", str(venv_py)
    return "missing", str(venv_py)
