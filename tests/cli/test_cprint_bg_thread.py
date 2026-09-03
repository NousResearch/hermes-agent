"""Tests for cli._cprint's bg-thread cooperation with prompt_toolkit.

Background: when a prompt_toolkit Application is running, a bg thread that
calls ``_pt_print`` directly can race with the input-area redraw and the
printed line can end up visually buried behind the prompt.  ``_cprint`` now
routes cross-thread prints through ``run_in_terminal`` via
``loop.call_soon_threadsafe`` so the self-improvement background review's
``💾 Self-improvement review: …`` summary actually surfaces to the user.

These tests verify the routing logic without spinning up a real PT app.
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


def test_cprint_no_app_direct_print(monkeypatch):
    """No active app → direct _pt_print, no run_in_terminal involvement."""
    calls = []
    monkeypatch.setattr(cli, "_pt_print", lambda x: calls.append(("pt_print", x)))
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: ("ANSI", t))

    # Patch the prompt_toolkit import the function performs internally.
    fake_pt_app = types.ModuleType("prompt_toolkit.application")
    fake_pt_app.get_app_or_none = lambda: None
    fake_pt_app.run_in_terminal = lambda *a, **kw: calls.append(("run_in_terminal",))
    monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

    cli._cprint("hello")

    assert calls == [("pt_print", ("ANSI", "hello"))]


def test_cprint_app_not_running_direct_print(monkeypatch):
    """App exists but not running (e.g. teardown) → direct print."""
    calls = []
    monkeypatch.setattr(cli, "_pt_print", lambda x: calls.append(("pt_print", x)))
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: t)

    fake_app = SimpleNamespace(_is_running=False, loop=None)
    fake_pt_app = types.ModuleType("prompt_toolkit.application")
    fake_pt_app.get_app_or_none = lambda: fake_app
    fake_pt_app.run_in_terminal = lambda *a, **kw: calls.append(("run_in_terminal",))
    monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

    cli._cprint("x")

    assert calls == [("pt_print", "x")]




def test_cprint_same_thread_as_app_loop_direct_print(monkeypatch):
    """App running on same thread → direct print (no scheduling)."""
    direct_prints = []
    monkeypatch.setattr(cli, "_pt_print", lambda x: direct_prints.append(x))
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: t)

    class FakeLoop:
        def is_running(self):
            return True

        def call_soon_threadsafe(self, cb, *args):
            raise AssertionError(
                "call_soon_threadsafe must not be used on the app's own thread"
            )

    fake_loop = FakeLoop()
    fake_asyncio = types.ModuleType("asyncio")

    class _Policy:
        def get_event_loop(self):
            return fake_loop  # same as app loop

    fake_asyncio.get_event_loop_policy = lambda: _Policy()
    monkeypatch.setitem(sys.modules, "asyncio", fake_asyncio)

    fake_app = SimpleNamespace(_is_running=True, loop=fake_loop)
    fake_pt_app = types.ModuleType("prompt_toolkit.application")
    fake_pt_app.get_app_or_none = lambda: fake_app
    fake_pt_app.run_in_terminal = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

    cli._cprint("x")

    assert direct_prints == ["x"]




def test_cprint_swallows_prompt_toolkit_import_error(monkeypatch):
    """If prompt_toolkit.application itself fails to import, fall back."""
    direct_prints = []
    monkeypatch.setattr(cli, "_pt_print", lambda x: direct_prints.append(x))
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: t)

    # Drop cached prompt_toolkit.application AND install a meta-path finder
    # that raises ImportError on re-import.
    monkeypatch.delitem(sys.modules, "prompt_toolkit.application", raising=False)

    class _BlockFinder:
        def find_module(self, name, path=None):
            if name == "prompt_toolkit.application":
                return self
            return None

        def load_module(self, name):
            raise ImportError("blocked for test")

        def find_spec(self, name, path=None, target=None):
            if name == "prompt_toolkit.application":
                # Returning a bogus spec that will fail on load works too,
                # but raising here keeps the test simple.
                raise ImportError("blocked for test")
            return None

    blocker = _BlockFinder()
    sys.meta_path.insert(0, blocker)
    try:
        cli._cprint("fallback2")
    finally:
        sys.meta_path.remove(blocker)

    assert direct_prints == ["fallback2"]






def test_replay_output_history_rerenders_callable_entries(monkeypatch):
    cli._configure_output_history(True, 10)
    widths_seen = []
    printed = []

    def _render_current_width():
        widths_seen.append("called")
        return ["top border", "body"]

    cli._record_output_history_entry(_render_current_width)
    monkeypatch.setattr(cli, "_pt_print", lambda value: printed.append(value))
    monkeypatch.setattr(cli, "_PT_ANSI", lambda text: text)

    cli._replay_output_history()

    assert widths_seen == ["called"]
    assert printed == ["top border\nbody"]
    assert list(cli._OUTPUT_HISTORY) == [_render_current_width]




def test_chat_console_records_rich_ansi_for_resize_replay(monkeypatch):
    cli._configure_output_history(True, 10)
    monkeypatch.setattr(cli, "_pt_print", lambda *_args, **_kwargs: None)

    cli.ChatConsole().print("[bold red]Hello[/]")

    assert cli._OUTPUT_HISTORY
    assert any("\x1b[" in line for line in cli._OUTPUT_HISTORY)


def test_suspend_output_history_blocks_recording():
    cli._configure_output_history(True, 10)

    with cli._suspend_output_history():
        cli._record_output_history("hidden")
        cli._record_output_history_entry("also hidden")

    assert list(cli._OUTPUT_HISTORY) == []


def test_clear_output_history_removes_replayable_lines():
    cli._configure_output_history(True, 10)
    cli._record_output_history("before clear")

    cli._clear_output_history()

    assert list(cli._OUTPUT_HISTORY) == []


# ── Flush regression tests (#43356 review follow-up) ──────────────────────
# Every _cprint output path must flush stdout after writing, otherwise
# streamed tokens sit in the stdio buffer on Windows PowerShell and on
# headless/orchestrated stdout pipes.

class _FlushTracker:
    """Replacement for sys.stdout that records flush() calls."""

    def __init__(self):
        self.flushes = 0

    def flush(self):
        self.flushes += 1

    # Other stdout attributes are left unneeded by _cprint itself
    # (the real writes go through _pt_print, which we monkeypatch away).


def test_cprint_no_app_flushes_stdout(monkeypatch):
    """Direct path (no app running) must flush after _pt_print."""
    monkeypatch.setattr(cli, "_pt_print", lambda x: None)
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: t)

    fake_pt_app = types.ModuleType("prompt_toolkit.application")
    fake_pt_app.get_app_or_none = lambda: None
    fake_pt_app.run_in_terminal = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

    tracker = _FlushTracker()
    monkeypatch.setattr(sys, "stdout", tracker)

    cli._cprint("direct")
    assert tracker.flushes >= 1, "no-app path must call sys.stdout.flush()"


def test_cprint_fallback_print_uses_flush(monkeypatch, capsys):
    """When _pt_print raises (no console), the print() fallback uses flush=True."""
    def _boom(_text):
        raise OSError("no console")

    monkeypatch.setattr(cli, "_pt_print", _boom)
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: t)

    fake_pt_app = types.ModuleType("prompt_toolkit.application")
    fake_pt_app.get_app_or_none = lambda: None
    fake_pt_app.run_in_terminal = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

    # spy on builtins print to assert flush=True
    print_calls = []
    import builtins
    real_print = builtins.print

    def _spy_print(*args, **kwargs):
        print_calls.append(kwargs)
        # swallow the actual write so capsys stays clean
    monkeypatch.setattr(builtins, "print", _spy_print)

    cli._cprint("fallback-text")

    assert len(print_calls) == 1
    assert print_calls[0].get("flush") is True, (
        "no-console fallback must call print(..., flush=True)"
    )


def test_cprint_cross_thread_callback_flushes(monkeypatch):
    """The run_in_terminal callback on the cross-thread path must flush."""
    monkeypatch.setattr(cli, "_pt_print", lambda x: None)
    monkeypatch.setattr(cli, "_PT_ANSI", lambda t: t)

    scheduled = []

    class FakeLoop:
        def is_running(self):
            return True

        def call_soon_threadsafe(self, cb, *args):
            scheduled.append(cb)

    fake_loop = FakeLoop()

    # Different-thread current loop so the cross-thread branch is taken.
    fake_current_loop = SimpleNamespace(is_running=lambda: True)
    fake_asyncio = types.ModuleType("asyncio")

    class _Policy:
        def get_event_loop(self):
            return fake_current_loop

    fake_asyncio.get_event_loop_policy = lambda: _Policy()
    fake_asyncio.ensure_future = lambda coro: None
    monkeypatch.setitem(sys.modules, "asyncio", fake_asyncio)

    fake_app = SimpleNamespace(_is_running=True, loop=fake_loop)
    fake_pt_app = types.ModuleType("prompt_toolkit.application")
    fake_pt_app.get_app_or_none = lambda: fake_app
    fake_pt_app.run_in_terminal = lambda func, **kw: (func(), None)[1]
    monkeypatch.setitem(sys.modules, "prompt_toolkit.application", fake_pt_app)

    tracker = _FlushTracker()
    monkeypatch.setattr(sys, "stdout", tracker)

    cli._cprint("cross-thread")
    # Kick the scheduled callback (simulates the app loop picking it up).
    assert scheduled, "cross-thread path must schedule a callback"
    scheduled[0]()

    assert tracker.flushes >= 1, (
        "run_in_terminal callback must flush after _pt_print"
    )
