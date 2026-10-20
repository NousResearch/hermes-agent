"""Runner-independent safety net for tests that need a Hermes home.

Why this exists
---------------
``tests/conftest.py`` sandboxes ``HERMES_HOME`` and installs a real-home tripwire
over every file operation — but ``conftest.py`` is a *pytest* mechanism. A run
that bypasses it (``python -m unittest ...``, a direct ``python tests/...`` file
run, a bespoke runner) gets neither the sandbox nor the tripwire, so a test that
writes ``get_hermes_home()/config.yaml`` writes the operator's LIVE config.

That is not theoretical. On 2026-10-10 a unittest-style run of
``tests/tools/test_delegate.py`` replaced a 29,352-byte live ``config.yaml`` with
a 107-byte provider fixture. The loss surfaced two days later as "no model
configured": the runtime fell back to an ``openrouter`` credential, got HTTP 401,
and could not compact the conversation. Root cause was the missing test/runtime
boundary, not the credential.

Contract
--------
``production_roots()``  the homes a test must never touch.
``guarded_home()``      the home a test MAY write into; raises if it resolves to
                        a production root, so a bypassing run fails loudly
                        instead of destroying live state.
``arm()``               refuse file writes under those roots for the whole
                        process, under ANY runner (audit hook, not a fixture).

Exemptions mirror ``tests/home_io_guard.py``: the running interpreter's own
installation and the checkout itself are NOT Hermes state. That matters here
because the default install checks the repo out *inside* the home
(``$HERMES_HOME/hermes-agent``) — a guard without those exemptions would block
every test file operation.

Use ``guarded_home()`` instead of ``get_hermes_home()`` in every test that
writes. ``tests/__init__.py`` arms the hook for non-pytest runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["production_roots", "guarded_home", "arm"]

_MUTATING_EVENTS = frozenset(
    {
        "os.remove",
        "os.rename",
        "os.mkdir",
        "os.rmdir",
        "os.truncate",
        "os.link",
        "os.symlink",
        "os.chmod",
        "os.chown",
        "shutil.rmtree",
        "shutil.copyfile",
        "shutil.move",
        "shutil.copytree",
    }
)


def production_roots() -> tuple[Path, ...]:
    """The real Hermes homes a test must never touch.

    ``~/.hermes`` is the platform default; ``HERMES_TEST_REAL_HOME`` lets an
    operator name an extra root (custom installs) without changing defaults.
    """
    roots: list[Path] = []
    candidates = [Path.home() / ".hermes", os.environ.get("HERMES_TEST_REAL_HOME") or ""]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:
            continue
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def guarded_home() -> Path:
    """The Hermes home a test may write into.

    Returns the effective home, so behaviour under pytest (whose conftest
    sandboxes ``HERMES_HOME``) is unchanged — but refuses a production root, so a
    runner that skipped the sandbox fails here instead of writing live config.
    """
    from hermes_constants import get_hermes_home

    home = Path(get_hermes_home()).resolve()
    for root in production_roots():
        if home == root or root in home.parents:
            raise RuntimeError(
                f"refusing to use the production Hermes home from a test: {home}\n"
                "Run the suite via scripts/run_tests.sh (or pytest, whose conftest "
                "sandboxes HERMES_HOME). See tests/safe_home.py."
            )
    # Asking for a home means this process writes: protect it end to end, so a
    # later write from any module in the same process hits the same refusal.
    arm()
    return home


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _exempt_prefixes() -> tuple[str, ...]:
    """Paths that are the interpreter's installation or the checkout, not state."""
    prefixes: set[str] = set()
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        if prefix:
            prefixes.add(_norm(prefix))
    for entry in sys.path:
        if entry and Path(entry).name in ("site-packages", "dist-packages"):
            prefixes.add(_norm(entry))
    # The checkout (install.sh checks it out inside HERMES_HOME on a default install).
    prefixes.add(_norm(os.fspath(Path(__file__).resolve().parent.parent)))
    return tuple(prefixes)


def _within(path: str, prefix: str) -> bool:
    if path == prefix:
        return True
    return path.startswith(prefix) if prefix == os.sep else path.startswith(prefix + os.sep)


def _is_guarded(path_text: str, roots: tuple[str, ...], exempt: tuple[str, ...]) -> bool:
    try:
        absolute = _norm(path_text)
    except (OSError, ValueError):
        return False
    for prefix in exempt:
        if _within(absolute, prefix):
            return False
    for root in roots:
        if _within(absolute, root):
            return True
    try:
        resolved = os.path.normcase(os.path.realpath(absolute))
    except OSError:
        return False
    if resolved != absolute:
        for prefix in exempt:
            if _within(resolved, prefix):
                return False
        for root in roots:
            if _within(resolved, root):
                return True
    return False


def _opens_for_write(mode, flags) -> bool:
    if isinstance(mode, str):
        return any(char in mode for char in "wax+")
    if isinstance(flags, int):
        write_bits = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        return bool(flags & write_bits)
    if isinstance(mode, int):
        return _opens_for_write(None, mode)
    return False


_ARMED = False


def arm() -> None:
    """Refuse writes under a production root, for the whole process.

    Armed by importing this module rather than by the pytest conftest, so it
    protects any runner. Idempotent. Respects the interpreter/checkout
    exemptions, so the suite itself keeps working from a default install.
    """
    global _ARMED
    if _ARMED:
        return
    roots = tuple(_norm(os.fspath(root)) for root in production_roots())
    exempt = _exempt_prefixes()

    def _hook(event: str, args: tuple) -> None:
        if event == "open":
            if not _opens_for_write(args[1] if len(args) > 1 else None,
                                    args[2] if len(args) > 2 else None):
                return
        elif event not in _MUTATING_EVENTS:
            return
        raw = args[0] if args else None
        if isinstance(raw, bytes):
            try:
                raw = raw.decode()
            except UnicodeDecodeError:
                return
        if isinstance(raw, int):
            return  # file descriptor, not a path
        if not isinstance(raw, (str, os.PathLike)):
            return
        text = os.fspath(raw)
        if _is_guarded(text, roots, exempt):
            raise RuntimeError(
                f"a test tried to write the production Hermes home: {text}\n"
                "Use tests.safe_home.guarded_home() for a writable home "
                "(or run the suite via scripts/run_tests.sh)."
            )

    sys.addaudithook(_hook)
    _ARMED = True
