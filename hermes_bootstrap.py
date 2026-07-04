"""Windows UTF-8 bootstrap for Hermes entry points.

Python on Windows has two long-standing text-encoding footguns:

1. ``sys.stdout`` / ``sys.stderr`` are bound to the console code page
   (``cp1252`` on US-locale installs), so ``print("caf\u00e9")`` crashes with
   ``UnicodeEncodeError: 'charmap' codec can't encode character``.

2. Child processes spawned via ``subprocess`` don't know to use UTF-8
   unless ``PYTHONUTF8`` and/or ``PYTHONIOENCODING`` are set in their
   environment \u2014 so any Python subprocess (the execute_code sandbox,
   delegation children, linter subprocesses, etc.) inherits the same
   cp1252 defaults and hits the same UnicodeEncodeError.

This module fixes both on Windows *only* \u2014 POSIX is untouched.  It
should be imported at the very top of every Hermes entry point
(``hermes``, ``hermes-agent``, ``hermes-acp``, ``python -m gateway.run``,
``batch_runner.py``, ``cron/scheduler.py``) before any other imports
that might do file I/O or print to stdout.

What this module does on Windows:

  - Sets ``os.environ["PYTHONUTF8"] = "1"`` (PEP 540 UTF-8 mode) so
    every child process we spawn uses UTF-8 for ``open()`` and stdio.
  - Sets ``os.environ["PYTHONIOENCODING"] = "utf-8"`` for belt-and-
    suspenders \u2014 some tools read this instead of / in addition to
    ``PYTHONUTF8``.
  - Reconfigures ``sys.stdout`` / ``sys.stderr`` to UTF-8 in the current
    process, using the ``reconfigure()`` API (Python 3.7+).  This fixes
    ``print("caf\u00e9")`` in the parent without a re-exec.
  - On Windows, clears stale ``__pycache__`` directories when the source
    revision changes, preventing ``TypeError`` from cached bytecode
    mismatches after upgrades.

What this module does NOT do:

  - It does not re-exec Python with ``-X utf8``, so ``open()`` calls in
    the *current* process still default to locale encoding.  Those need
    an explicit ``encoding="utf-8"`` at the call site (lint rule
    ``PLW1514`` / ``PYI058``).  Ruff is the right tool for that sweep.

What this module does on POSIX:

  - Nothing.  POSIX systems are already UTF-8 by default in 99% of cases,
    and we don't want to touch ``LANG``/``LC_*`` behavior that users may
    have configured intentionally.  If someone hits a C/POSIX locale on
    Linux, they can export ``PYTHONUTF8=1`` themselves \u2014 we won't override.

Idempotent: safe to call multiple times.  ``_bootstrap_once`` guards
against double-reconfigure.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

_IS_WINDOWS = sys.platform == "win32"
_bootstrap_applied = False
_BOOTSTRAP_ROOT = Path(__file__).resolve().parent
_PYCACHE_REVISION_MARKER = ".hermes-pycache-revision"
_PYCACHE_SKIP_DIRS = {"venv", ".venv", "node_modules", ".git", ".worktrees"}


def apply_windows_utf8_bootstrap() -> bool:
    """Apply the Windows UTF-8 bootstrap if we're on Windows.

    Returns True if bootstrap was applied (i.e. we're on Windows and
    haven't already done this), False otherwise.  The return value is
    advisory \u2014 callers normally don't need it, but tests may want to
    assert the path was taken.

    Idempotent: subsequent calls after the first are a no-op.
    """
    global _bootstrap_applied

    if not _IS_WINDOWS:
        return False
    if _bootstrap_applied:
        return False

    # 1. Child processes inherit these and run in UTF-8 mode.
    #    We use setdefault() rather than overwriting so the user can
    #    explicitly opt out by setting PYTHONUTF8=0 in their environment
    #    (or PYTHONIOENCODING=something-else) if they really want to.
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # 2. Reconfigure the current process's stdio to UTF-8.  Needed
    #    because os.environ changes don't retroactively rebind sys.stdout
    #    \u2014 those were bound at interpreter startup based on the console
    #    code page.  ``reconfigure`` is a TextIOWrapper method since 3.7.
    #
    #    errors="replace" means that if we ever *read* something from
    #    stdin that isn't UTF-8 (unlikely but possible with piped input
    #    from legacy tools), we'll get U+FFFD replacement chars rather
    #    than a crash.  Output is pure UTF-8.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            # Not a TextIOWrapper (could be redirected to a BytesIO in
            # tests, or a non-standard stream in some embedded cases).
            # Skip silently \u2014 the env-var fix is still in effect for
            # child processes, which is the bigger win.
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Already closed, or someone replaced it with something
            # non-reconfigurable.  Non-fatal.
            pass

    # Clear stale Windows bytecode before any other imports that may
    # load compiled modules from a previous revision.
    _clear_windows_pycache_after_source_update(_BOOTSTRAP_ROOT)

    # 3. stdin is reconfigured separately with errors="replace" too \u2014
    #    input from a legacy pipe shouldn't crash the process.
    stdin = getattr(sys, "stdin", None)
    if stdin is not None:
        reconfigure = getattr(stdin, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

    _bootstrap_applied = True
    return True


def harden_import_path(src_root: str | None = None) -> None:
    """Stop a package in the current directory from shadowing Hermes modules.

    Hermes ships top-level modules with common names (``utils``, ``proxy``,
    ``ui``).  Python always seeds ``sys.path`` with the current directory, so
    launching an entry point from a project that has its own ``utils/`` package
    makes ``from utils import ...`` resolve to the *user's* package and crash
    with an ImportError before the gateway can even start.

    The current directory reaches ``sys.path`` two ways, and a complete guard
    has to handle both:

      - As the empty string ``""`` (or ``"."``) that Python inserts at
        ``sys.path[0]`` for ``-m`` / script launches.
      - As its own *absolute* path, when a venv activation or a project that
        adds itself to ``PYTHONPATH`` puts the directory there explicitly.

    We drop the relative forms outright, then force the real Hermes source root
    to the front \u2014 relocating it ahead of any absolute cwd entry rather than
    only inserting when absent, so an absolute cwd path can't keep winning.

    ``src_root`` defaults to the directory this module lives in, which is the
    repository root for every shipped entry point, so the guard is
    self-sufficient and does not depend on the spawner exporting an env var.
    """
    root = src_root or os.environ.get("HERMES_PYTHON_SRC_ROOT") or os.path.dirname(
        os.path.abspath(__file__)
    )

    sys.path[:] = [p for p in sys.path if p not in ("", ".")]

    root_abs = os.path.abspath(root)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != root_abs]
    sys.path.insert(0, root)


def activate_durable_lazy_target() -> None:
    """Put the durable lazy-install dir on ``sys.path`` if one is configured.

    On immutable Docker images the agent venv is sealed and lazy installs
    are redirected to a writable dir on the data volume
    (``HERMES_LAZY_INSTALL_TARGET``, e.g. ``/opt/data/lazy-packages``).
    Packages installed there on a previous run must be importable on this
    run, so we activate the dir here \u2014 at the very first import, before any
    backend module imports its SDK.

    The activation appends to the END of ``sys.path`` so the core venv
    always wins name collisions (see ``tools.lazy_deps`` for the full
    security rationale). Never raises; a missing/empty target is a no-op.
    """
    if not os.environ.get("HERMES_LAZY_INSTALL_TARGET", "").strip():
        return
    try:
        from tools import lazy_deps
        lazy_deps.activate_durable_lazy_target()
    except Exception:
        # Bootstrap must never crash an entry point. If activation fails the
        # backend simply reports itself unavailable, exactly as before.
        pass


def _git_dir_for(root: Path) -> Path | None:
    git_path = root / ".git"
    if git_path.is_dir():
        return git_path
    if not git_path.is_file():
        return None
    try:
        text = git_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text.startswith("gitdir:"):
        return None
    git_dir = Path(text.split(":", 1)[1].strip())
    if not git_dir.is_absolute():
        git_dir = (root / git_dir).resolve()
    return git_dir


def _source_revision(root: Path) -> str | None:
    git_dir = _git_dir_for(root)
    if git_dir is None:
        return None
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not head:
        return None
    if head.startswith("ref:"):
        ref = head.split(":", 1)[1].strip()
        try:
            return (git_dir / ref).read_text(encoding="utf-8").strip() or None
        except OSError:
            return _packed_ref_revision(git_dir, ref)
    return head


def _packed_ref_revision(git_dir: Path, ref: str) -> str | None:
    try:
        lines = (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    suffix = f" {ref}"
    for line in lines:
        if line.startswith("#") or line.startswith("^"):
            continue
        if line.endswith(suffix):
            return line.split(" ", 1)[0].strip() or None
    return None


def _clear_pycache_dirs(root: Path) -> int:
    removed = 0
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _PYCACHE_SKIP_DIRS]
        if os.path.basename(dirpath) != "__pycache__":
            continue
        try:
            shutil.rmtree(dirpath)
            removed += 1
        except OSError:
            pass
        dirnames.clear()
    return removed


def _clear_windows_pycache_after_source_update(root: Path) -> bool:
    """Clear stale bytecode once per source revision on Windows startup."""
    if not _IS_WINDOWS:
        return False

    revision = _source_revision(root)
    if not revision:
        return False

    git_dir = _git_dir_for(root)
    if git_dir is None:
        return False

    marker = git_dir / _PYCACHE_REVISION_MARKER
    try:
        if marker.read_text(encoding="utf-8").strip() == revision:
            return False
    except FileNotFoundError:
        pass
    except OSError:
        return False

    _clear_pycache_dirs(root)
    try:
        marker.write_text(f"{revision}\n", encoding="utf-8")
    except OSError:
        return False
    return True


# Apply on import \u2014 entry points just need ``import hermes_bootstrap``
# (or ``from hermes_bootstrap import apply_windows_utf8_bootstrap``) at
# the very top of their module, before importing anything else.  The
# import side effect does the right thing.
apply_windows_utf8_bootstrap()

# Activate the durable lazy-install target (immutable Docker images) so
# packages installed into the data volume on a previous run are importable
# this run, before any backend module imports its SDK. No-op when unset.
activate_durable_lazy_target()
