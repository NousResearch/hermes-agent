"""Enforcement for the "no visible terminal on Windows" invariant.

Windows console-subsystem programs (``taskkill``, ``schtasks``, ``agent-browser``,
``git-bash`` …) pop a console window unless spawned with ``CREATE_NO_WINDOW``.
Relying on each call site to remember the flag is how cron-driven and future
spawns leaked terminal windows. The durable fix is a single chokepoint —
``hermes_cli._subprocess_compat.run`` / ``.popen`` — that always injects the
flag on Windows, plus the ``FreeConsole`` catch-all in ``hermes_bootstrap`` for
Python children.

These tests pin both halves of that contract:

1. The primitive actually injects ``CREATE_NO_WINDOW`` (and merges, so detach
   callers still work).
2. No source file spawns a known console exe with a *raw* ``subprocess`` call,
   which would bypass the primitive and reintroduce the window.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from hermes_cli import _subprocess_compat

REPO_ROOT = Path(__file__).resolve().parent.parent
_CREATE_NO_WINDOW = 0x08000000


class TestPrimitiveInjectsNoWindow:
    def test_run_injects_create_no_window_on_windows(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
        monkeypatch.setattr(
            _subprocess_compat.subprocess, "run", lambda cmd, **kw: captured.update(kw) or "ok"
        )

        _subprocess_compat.run(["taskkill"], timeout=5)

        assert captured["creationflags"] & _CREATE_NO_WINDOW

    def test_popen_injects_create_no_window_on_windows(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
        monkeypatch.setattr(
            _subprocess_compat.subprocess, "Popen", lambda cmd, **kw: captured.update(kw) or "ok"
        )

        _subprocess_compat.popen(["agent-browser"])

        assert captured["creationflags"] & _CREATE_NO_WINDOW

    def test_merges_with_existing_detach_flags(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", True)
        monkeypatch.setattr(
            _subprocess_compat.subprocess, "run", lambda cmd, **kw: captured.update(kw) or "ok"
        )

        detach = _subprocess_compat.windows_detach_flags()
        _subprocess_compat.run(["x"], creationflags=detach)

        assert captured["creationflags"] & _CREATE_NO_WINDOW
        assert captured["creationflags"] & detach == detach

    def test_no_op_on_posix(self, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(_subprocess_compat, "IS_WINDOWS", False)
        monkeypatch.setattr(
            _subprocess_compat.subprocess, "run", lambda cmd, **kw: captured.update(kw) or "ok"
        )

        _subprocess_compat.run(["x"])

        assert "creationflags" not in captured


# Windows-only console tools — they have no POSIX use, so a raw ``subprocess``
# spawn is unambiguously a Windows path that flashes a terminal. Banning them
# repo-wide is a pure win (cross-platform tools like git/ffmpeg/node are NOT
# listed: they have legitimate foreground/POSIX uses a blanket ban would break;
# their Windows-background call sites are routed through the primitive instead).
# ``_subprocess_compat.run/.popen`` calls never match these (different prefix).
_WINDOWS_ONLY_CONSOLE_EXES = ("taskkill", "schtasks", "wmic", "netstat", "tasklist")
_RAW_CONSOLE_SPAWNS = [
    re.compile(rf"""subprocess\.(?:run|Popen|call)\(\s*\[\s*["']{exe}["']""")
    for exe in _WINDOWS_ONLY_CONSOLE_EXES
]

# The primitive itself is allowed to call raw subprocess — it IS the chokepoint.
_ALLOWED = {REPO_ROOT / "hermes_cli" / "_subprocess_compat.py"}


# Dev/CI tooling that never ships to a user's Windows desktop, where a flashing
# console is irrelevant and importing hermes_cli would be inappropriate.
_SKIP_DIRS = {"tests", "node_modules", ".venv", "venv", "scripts"}


def _python_sources():
    for path in REPO_ROOT.rglob("*.py"):
        if _SKIP_DIRS & set(path.parts):
            continue
        if path in _ALLOWED:
            continue
        yield path


@pytest.mark.parametrize("pattern", _RAW_CONSOLE_SPAWNS, ids=_WINDOWS_ONLY_CONSOLE_EXES)
def test_no_raw_console_exe_spawns(pattern):
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _python_sources()
        if pattern.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]

    assert not offenders, (
        "Console-subsystem exe spawned via raw subprocess (flashes a terminal on "
        f"Windows). Route through hermes_cli._subprocess_compat.run/.popen instead: {offenders}"
    )
