"""Runtime bootstrap for Hermes entry points.

Imported at the very top of every Hermes entry point (``hermes``,
``hermes-agent``, ``hermes-acp``, ``python -m gateway.run``,
``batch_runner.py``, ``cli.py``) before any other imports that might
do file I/O or print to stdout, or load YAML.

This module currently applies two unrelated runtime fixes:

1. Windows UTF-8 stdio bootstrap — see ``apply_windows_utf8_bootstrap``.
2. PyYAML CSafeLoader shim — see ``apply_yaml_csafe_shim``.

Both are idempotent and silent (no logging, no raise) so they're safe
to call from any entry point on any platform.

==============================================================================
1. Windows UTF-8 bootstrap
==============================================================================

Python on Windows has two long-standing text-encoding footguns:

a. ``sys.stdout`` / ``sys.stderr`` are bound to the console code page
   (``cp1252`` on US-locale installs), so ``print("café")`` crashes with
   ``UnicodeEncodeError: 'charmap' codec can't encode character``.

b. Child processes spawned via ``subprocess`` don't know to use UTF-8
   unless ``PYTHONUTF8`` and/or ``PYTHONIOENCODING`` are set in their
   environment — so any Python subprocess (the execute_code sandbox,
   delegation children, linter subprocesses, etc.) inherits the same
   cp1252 defaults and hits the same UnicodeEncodeError.

What ``apply_windows_utf8_bootstrap`` does on Windows:

  - Sets ``os.environ["PYTHONUTF8"] = "1"`` (PEP 540 UTF-8 mode) so
    every child process we spawn uses UTF-8 for ``open()`` and stdio.
  - Sets ``os.environ["PYTHONIOENCODING"] = "utf-8"`` for belt-and-
    suspenders — some tools read this instead of / in addition to
    ``PYTHONUTF8``.
  - Reconfigures ``sys.stdout`` / ``sys.stderr`` to UTF-8 in the current
    process, using the ``reconfigure()`` API (Python 3.7+).  This fixes
    ``print("café")`` in the parent without a re-exec.

It does NOT re-exec Python with ``-X utf8``, so ``open()`` calls in the
*current* process still default to locale encoding.  Those need an
explicit ``encoding="utf-8"`` at the call site (lint rule ``PLW1514`` /
``PYI058``).  Ruff is the right tool for that sweep.

On POSIX it is a complete no-op — POSIX systems are already UTF-8 by
default in 99% of cases, and we don't want to touch ``LANG``/``LC_*``
behavior that users may have configured intentionally.  If someone
hits a C/POSIX locale on Linux, they can export ``PYTHONUTF8=1``
themselves — we won't override.

==============================================================================
2. PyYAML CSafeLoader shim
==============================================================================

PyYAML 6.0.x's pure-Python ``safe_load`` parser corrupts CPython's
internal object/refcount state under sustained load on at least some
WSL2 / glibc / kernel combinations — observed as ``TypeError: 'X'
object is not subscriptable`` mid-parse, ``AttributeError: 'dict'
object has no attribute 'match'`` from compiled-regex caches getting
overwritten, and segfaults inside ``Py_INCREF``/``Py_DECREF``.

The same crashes reproduce across:
  - uv-managed cpython-3.11.14 / 3.11.15 (clang, python-build-standalone)
  - Ubuntu 24.04 system python3.12.3 (gcc, apt)

The libyaml-backed C parser (``yaml.CSafeLoader`` / ``yaml._yaml``)
does NOT exhibit this — thousands of repeated parses are clean.
PyYAML keeps ``safe_load`` and ``load(..., Loader=CSafeLoader)`` as
separate code paths for legacy compatibility; users of plain
``safe_load`` get the broken pure-Python implementation by default
even when libyaml is installed.

``apply_yaml_csafe_shim`` rebinds ``yaml.safe_load`` and
``yaml.safe_load_all`` to use ``CSafeLoader`` whenever the libyaml C
extension is available.  This makes every ``yaml.safe_load(...)`` in
Hermes (and in any third-party dependency that ``import yaml`` after
us) take the stable C path automatically — no call-site changes
needed.

If libyaml isn't available (rare — wheels ship it), the shim is a
silent no-op and the original behavior is preserved.

==============================================================================
"""

from __future__ import annotations

import os
import sys

_IS_WINDOWS = sys.platform == "win32"
_bootstrap_applied = False
_yaml_shim_applied = False


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


def apply_yaml_csafe_shim() -> bool:
    """Rebind ``yaml.safe_load`` and ``yaml.safe_load_all`` to use the
    libyaml-backed ``CSafeLoader`` when available.

    PyYAML's pure-Python ``safe_load`` non-deterministically corrupts
    CPython object state under sustained use on some WSL2 / glibc
    combinations (segfaults inside ``Py_INCREF``, ``TypeError`` on
    mid-parse parser-event objects, ``AttributeError`` on cached
    compiled regexes that get clobbered).  The C-backed loader is
    immune.  See the module docstring for the full rationale.

    Returns True if the rebind was applied, False otherwise.  Reasons
    for returning False:

      - Already applied on a previous call (idempotent).
      - PyYAML isn't importable in this venv.
      - libyaml C extension isn't available (no ``CSafeLoader``).
      - Something else already replaced ``yaml.safe_load`` with a
        non-PyYAML implementation; we leave that alone.

    This function is platform-agnostic — the underlying bug is in
    PyYAML's parser code, not in any OS-specific layer, so the shim
    runs on Linux, macOS, and Windows alike.  The performance side
    effect is also positive: CSafeLoader is roughly 5-7× faster than
    the pure-Python loader on typical config files.
    """
    global _yaml_shim_applied

    if _yaml_shim_applied:
        return False

    try:
        import yaml  # type: ignore[import-not-found]
    except Exception:
        return False

    csafe = getattr(yaml, "CSafeLoader", None)
    if csafe is None:
        return False

    current_safe_load = getattr(yaml, "safe_load", None)
    if current_safe_load is None:
        return False
    if getattr(current_safe_load, "__module__", "") != "yaml":
        return False

    def safe_load(stream):  # type: ignore[no-redef]
        return yaml.load(stream, Loader=csafe)

    def safe_load_all(stream):  # type: ignore[no-redef]
        return yaml.load_all(stream, Loader=csafe)

    safe_load.__name__ = "safe_load"
    safe_load.__qualname__ = "safe_load"
    safe_load.__doc__ = (
        "Parse a YAML document via the libyaml CSafeLoader.\n\n"
        "Rebound by hermes_bootstrap.apply_yaml_csafe_shim to dodge "
        "PyYAML pure-Python parser memory-corruption bugs."
    )
    safe_load_all.__name__ = "safe_load_all"
    safe_load_all.__qualname__ = "safe_load_all"
    safe_load_all.__doc__ = safe_load.__doc__

    yaml.safe_load = safe_load
    yaml.safe_load_all = safe_load_all

    _yaml_shim_applied = True
    return True


apply_windows_utf8_bootstrap()
apply_yaml_csafe_shim()
