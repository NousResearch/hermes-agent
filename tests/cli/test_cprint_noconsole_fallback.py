"""_cprint must degrade to a plain write whenever prompt_toolkit can't render.

prompt_toolkit raises ``NoConsoleScreenBufferError`` (Windows) or ``OSError``
when stdout is not a real console — piped output, a subprocess worker logging
to a file, a service with no attached console. ``isatty()`` can report True
while the console handle is still unusable, so detection up front is not
enough: *every* emission arm has to degrade, including the ones taken while a
prompt_toolkit Application is running (#65558).
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import cli


@pytest.fixture(autouse=True)
def reset_output_history():
    cli._configure_output_history(False, 200)
    yield
    cli._configure_output_history(True, 200)


class _NoConsole(Exception):
    """Stand-in for prompt_toolkit's NoConsoleScreenBufferError."""


@pytest.fixture()
def broken_renderer(monkeypatch):
    """Every prompt_toolkit emission raises; capture what reaches stdout."""
    plain = []
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: t)

    def _boom(_value):
        raise _NoConsole("no console screen buffer")

    monkeypatch.setattr(cli, "_pt_print", _boom)
    monkeypatch.setattr(cli, "_plain_print", lambda t: plain.append(t))
    return plain


class TestEveryArmDegrades:
    def test_no_app(self, broken_renderer, monkeypatch):
        fake_pt_app = types.ModuleType("prompt_toolkit.application")
        fake_pt_app.get_app_or_none = lambda: None
        fake_pt_app.run_in_terminal = lambda *a, **kw: None
        monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

        cli._cprint("line")

        assert broken_renderer == ["line"]

    def test_prompt_toolkit_import_failure(self, broken_renderer, monkeypatch):
        def _raise(*_a, **_kw):
            raise ImportError("no prompt_toolkit")

        monkeypatch.setitem(sys.modules, "prompt_toolkit.application", None)
        monkeypatch.setattr(cli, "_record_output_history", lambda *_a, **_kw: None)
        cli._cprint("line")

        assert broken_renderer == ["line"]

    def test_active_app_same_loop(self, broken_renderer, monkeypatch):
        """The arm the previous fix never covered: app running, our thread."""
        class FakeLoop:
            def is_running(self):
                return True

            def call_soon_threadsafe(self, cb, *args):
                raise AssertionError("same-thread must not schedule")

        fake_loop = FakeLoop()
        fake_asyncio = types.ModuleType("asyncio")
        fake_asyncio.get_running_loop = lambda: fake_loop
        fake_asyncio.ensure_future = lambda c: None
        monkeypatch.setitem(sys.modules, "asyncio", fake_asyncio)

        fake_app = SimpleNamespace(_is_running=True, loop=fake_loop)
        fake_pt_app = types.ModuleType("prompt_toolkit.application")
        fake_pt_app.get_app_or_none = lambda: fake_app
        fake_pt_app.run_in_terminal = lambda *a, **kw: None
        monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

        cli._cprint("line")

        assert broken_renderer == ["line"]

    def test_active_app_cross_thread_inner_emit(self, broken_renderer, monkeypatch):
        """The run_in_terminal inner emit must degrade too, not raise."""
        scheduled = []

        class FakeLoop:
            def is_running(self):
                return True

            def call_soon_threadsafe(self, cb, *args):
                scheduled.append(cb)
                cb(*args)  # run it here so the inner emit is exercised

        fake_loop = FakeLoop()
        other_loop = SimpleNamespace(is_running=lambda: True)
        fake_asyncio = types.ModuleType("asyncio")
        fake_asyncio.get_running_loop = lambda: other_loop  # different thread
        fake_asyncio.ensure_future = lambda c: None
        monkeypatch.setitem(sys.modules, "asyncio", fake_asyncio)

        fake_app = SimpleNamespace(_is_running=True, loop=fake_loop)
        fake_pt_app = types.ModuleType("prompt_toolkit.application")
        fake_pt_app.get_app_or_none = lambda: fake_app
        # run_in_terminal invokes the callable synchronously (mock behaviour).
        fake_pt_app.run_in_terminal = lambda fn, *a, **kw: fn()
        monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

        cli._cprint("line")

        assert scheduled, "cross-thread emission should have been scheduled"
        assert broken_renderer == ["line"]

    def test_scheduling_failure_still_emits(self, broken_renderer, monkeypatch):
        class FakeLoop:
            def is_running(self):
                return True

            def call_soon_threadsafe(self, cb, *args):
                raise RuntimeError("loop closed")

        fake_loop = FakeLoop()
        other_loop = SimpleNamespace(is_running=lambda: True)
        fake_asyncio = types.ModuleType("asyncio")
        fake_asyncio.get_running_loop = lambda: other_loop
        fake_asyncio.ensure_future = lambda c: None
        monkeypatch.setitem(sys.modules, "asyncio", fake_asyncio)

        fake_app = SimpleNamespace(_is_running=True, loop=fake_loop)
        fake_pt_app = types.ModuleType("prompt_toolkit.application")
        fake_pt_app.get_app_or_none = lambda: fake_app
        fake_pt_app.run_in_terminal = lambda *a, **kw: None
        monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

        cli._cprint("line")

        assert broken_renderer == ["line"]

    def test_missing_loop_attribute(self, broken_renderer, monkeypatch):
        fake_app = SimpleNamespace(_is_running=True, loop=None)
        fake_pt_app = types.ModuleType("prompt_toolkit.application")
        fake_pt_app.get_app_or_none = lambda: fake_app
        fake_pt_app.run_in_terminal = lambda *a, **kw: None
        monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

        cli._cprint("line")

        assert broken_renderer == ["line"]


class TestPlainPrintIsEncodeSafe:
    def test_falls_back_to_replacement_on_legacy_codepage(self, monkeypatch, capsys):
        """cp1252 can't encode Hermes' box-drawing/emoji output."""
        real_print = print
        calls = {"n": 0}

        def _print(text=""):
            calls["n"] += 1
            if calls["n"] == 1:
                raise UnicodeEncodeError("charmap", "x", 0, 1, "unmapped")
            real_print(text)

        monkeypatch.setattr("builtins.print", _print)
        monkeypatch.setattr(sys, "stdout", SimpleNamespace(encoding="cp1252"))

        cli._plain_print("café ✓")

        assert calls["n"] == 2, "the encode-safe retry never ran"

    def test_plain_print_never_raises(self, monkeypatch):
        def _always_boom(*_a, **_kw):
            raise OSError("stdout gone")

        monkeypatch.setattr("builtins.print", _always_boom)
        cli._plain_print("anything")  # must not raise
