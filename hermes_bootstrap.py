"""Windows UTF-8 bootstrap for Hermes entry points.

Python on Windows has two long-standing text-encoding footguns:

1. ``sys.stdout`` / ``sys.stderr`` are bound to the console code page
   (``cp1252`` on US-locale installs), so ``print("café")`` crashes with
   ``UnicodeEncodeError: 'charmap' codec can't encode character``.

2. Child processes spawned via ``subprocess`` don't know to use UTF-8
   unless ``PYTHONUTF8`` and/or ``PYTHONIOENCODING`` are set in their
   environment — so any Python subprocess (the execute_code sandbox,
   delegation children, linter subprocesses, etc.) inherits the same
   cp1252 defaults and hits the same UnicodeEncodeError.

This module fixes both on Windows *only* — POSIX is untouched.  It
should be imported at the very top of every Hermes entry point
(``hermes``, ``hermes-agent``, ``hermes-acp``, ``python -m gateway.run``,
``batch_runner.py``, ``cron/scheduler.py``) before any other imports
that might do file I/O or print to stdout.

What this module does on Windows:

  - Sets ``os.environ["PYTHONUTF8"] = "1"`` (PEP 540 UTF-8 mode) so
    every child process we spawn uses UTF-8 for ``open()`` and stdio.
  - Sets ``os.environ["PYTHONIOENCODING"] = "utf-8"`` for belt-and-
    suspenders — some tools read this instead of / in addition to
    ``PYTHONUTF8``.
  - Reconfigures ``sys.stdout`` / ``sys.stderr`` to UTF-8 in the current
    process, using the ``reconfigure()`` API (Python 3.7+).  This fixes
    ``print("café")`` in the parent without a re-exec.

What this module does NOT do:

  - It does not re-exec Python with ``-X utf8``, so ``open()`` calls in
    the *current* process still default to locale encoding.  Those need
    an explicit ``encoding="utf-8"`` at the call site (lint rule
    ``PLW1514`` / ``PYI058``).  Ruff is the right tool for that sweep.

What this module does on POSIX:

  - Nothing.  POSIX systems are already UTF-8 by default in 99% of cases,
    and we don't want to touch ``LANG``/``LC_*`` behavior that users may
    have configured intentionally.  If someone hits a C/POSIX locale on
    Linux, they can export ``PYTHONUTF8=1`` themselves — we won't override.

Idempotent: safe to call multiple times.  ``_bootstrap_once`` guards
against double-reconfigure.
"""

from __future__ import annotations

import logging
import os
import sys

_IS_WINDOWS = sys.platform == "win32"
_bootstrap_applied = False
_dual_venv_warned = False

logger = logging.getLogger(__name__)


def apply_windows_utf8_bootstrap() -> bool:
    """Apply the Windows UTF-8 bootstrap if we're on Windows.

    Returns True if bootstrap was applied (i.e. we're on Windows and
    haven't already done this), False otherwise.  The return value is
    advisory — callers normally don't need it, but tests may want to
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
    #    — those were bound at interpreter startup based on the console
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
            # Skip silently — the env-var fix is still in effect for
            # child processes, which is the bigger win.
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Already closed, or someone replaced it with something
            # non-reconfigurable.  Non-fatal.
            pass

    # stdin is reconfigured separately with errors="replace" too — input
    # from a legacy pipe shouldn't crash the process.
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
    to the front — relocating it ahead of any absolute cwd entry rather than
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


def _is_reparse_point(path: str) -> bool:
    """Return True if ``path`` is a Windows directory junction or symlink.

    Uses ``GetFileAttributesW`` to inspect ``FILE_ATTRIBUTE_REPARSE_POINT``
    (0x400). On POSIX every symlink counts as a reparse point. On Windows
    this distinguishes a *materialized* ``.venv\\`` directory (no reparse
    point) from a junction ``venv\\ -> .venv\\`` (reparse point set).

    Returns False on any error (missing path, permission denied, non-Windows
    ctypes import failure) so a transient filesystem hiccup never turns into
    a bootstrap-time crash.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            FILE_ATTRIBUTE_REPARSE_POINT = 0x400
            attrs = ctypes.windll.kernel32.GetFileAttributesW(path)
            if attrs == 0xFFFFFFFF:  # INVALID_FILE_ATTRIBUTES
                return False
            return bool(attrs & FILE_ATTRIBUTE_REPARSE_POINT)
        except (OSError, ImportError, AttributeError):
            return False
    # POSIX: a symlink is the only candidate; bare directories are not
    # reparse points.
    try:
        return os.path.islink(path)
    except OSError:
        return False


def _has_pyvenv_cfg(directory: str) -> bool:
    """Return True if ``directory/pyvenv.cfg`` exists in a materialized venv.

    A *materialized* venv has its own real ``pyvenv.cfg`` file. A junction
    or symlink that aliases another venv resolves ``pyvenv.cfg`` through the
    reparse point and would otherwise count as drift — this helper filters
    those out by first checking that ``directory`` itself is not a reparse
    point.  This is the core of the dual-venv-collapse fix: post-fix,
    ``venv/`` is a junction and is correctly *not* counted as a second venv.
    """
    if _is_reparse_point(directory):
        # It's a junction / symlink.  The pyvenv.cfg it resolves to belongs
        # to whatever it points at, not to this directory.
        return False
    cfg_path = os.path.join(directory, "pyvenv.cfg")
    try:
        return os.path.isfile(cfg_path)
    except OSError:
        # Broken symlink, permission error, etc.  Treat as "not a real venv"
        # — never raise from a bootstrap-time detector.
        return False


def detect_dual_venv_drift() -> bool:
    """Detect whether two materialized venvs (``venv/`` and ``.venv/``) coexist.

    Recon 2026-06-30 confirmed the repo shipped *two* independent copies of
    ``pyvenv.cfg`` at the repository root — one in ``.venv/`` (uv/pytest) and
    one in ``venv/`` (the ``hermes.EXE`` runtime).  Either copy can be
    regenerated independently by ``uv sync`` / ``python -m venv`` / a manual
    ``hermes update``, leaving the other at a stale ``version_info`` and
    producing daemon-startup failures, broken editable-install ``.pth`` files,
    and ``hermes doctor`` reporting the wrong interpreter.

    This helper walks one level up from ``sys.prefix`` (the canonical venv
    directory) and checks whether both ``venv/pyvenv.cfg`` and
    ``.venv/pyvenv.cfg`` are materialized as real files.  If only one exists,
    or one of them is a junction/symlink to the other (the post-fix state),
    drift is absent.

    Returns True on drift, False otherwise.  Never raises — every I/O path
    swallows ``OSError`` so the bootstrap can't be blocked by a transient
    filesystem hiccup.
    """
    if os.environ.get("HERMES_ALLOW_DUAL_VENV", "").strip() == "1":
        # Opt-out for users who legitimately keep two venvs in sync by hand.
        return False

    parent = os.path.dirname(sys.prefix)
    if not parent:
        return False

    candidates = ("venv", ".venv")
    found = []
    for name in candidates:
        path = os.path.join(parent, name)
        if _has_pyvenv_cfg(path):
            found.append(path)

    return len(found) >= 2


def warn_dual_venv_drift() -> bool:
    """Emit a one-shot WARNING if dual-venv drift is detected.

    Idempotent: the warning fires at most once per process, regardless of how
    many times the caller invokes this function.  A module-level
    ``_dual_venv_warned`` flag guards against log spam on entry points that
    import ``hermes_bootstrap`` transitively from several submodules.

    Returns True when drift was detected and a warning was emitted, False
    otherwise (no drift, or already warned).
    """
    global _dual_venv_warned

    if _dual_venv_warned:
        return False
    if not detect_dual_venv_drift():
        return False

    parent = os.path.dirname(sys.prefix)
    locations = []
    for name in ("venv", ".venv"):
        path = os.path.join(parent, name)
        if _has_pyvenv_cfg(path):
            locations.append(path)

    logger.warning(
        "dual venv detected at %s — both venv/ and .venv/ contain an "
        "independent pyvenv.cfg, risking stale version_info after a partial "
        "rebuild. Set HERMES_ALLOW_DUAL_VENV=1 to silence, or collapse to "
        "a single .venv/ with `venv/` as a junction. See plans/001.",
        parent,
    )
    for loc in locations:
        logger.warning("  venv source of truth candidate: %s", loc)

    _dual_venv_warned = True
    return True


def activate_durable_lazy_target() -> None:
    """Put the durable lazy-install dir on ``sys.path`` if one is configured.

    On immutable Docker images the agent venv is sealed and lazy installs
    are redirected to a writable dir on the data volume
    (``HERMES_LAZY_INSTALL_TARGET``, e.g. ``/opt/data/lazy-packages``).
    Packages installed there on a previous run must be importable on this
    run, so we activate the dir here — at the very first import, before any
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


# Apply on import — entry points just need ``import hermes_bootstrap``
# (or ``from hermes_bootstrap import apply_windows_utf8_bootstrap``) at
# the very top of their module, before importing anything else.  The
# import side effect does the right thing.
apply_windows_utf8_bootstrap()

# Activate the durable lazy-install target (immutable Docker images) so
# packages installed into the data volume on a previous run are importable
# this run, before any backend module imports its SDK. No-op when unset.
activate_durable_lazy_target()

# Detect dual-venv drift (independent pyvenv.cfg in both venv/ and .venv/)
# and emit a one-shot warning. Idempotent — safe to call repeatedly, but
# the underlying detector runs at most once per process via its own guard.
# Never raises: a broken filesystem path will surface as no warning, not a
# crash on every Hermes entry point.
warn_dual_venv_drift()
