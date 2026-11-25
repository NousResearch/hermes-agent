"""Plugin terminal chrome in the classic attached chat: ``plugin.<id>.terminal`` events become an
animated tab title and a skin-coloured status-bar segment, untrusted text never reaches the terminal
raw, and the terminal's own title comes back when the chat ends."""
import asyncio

import pytest

from hermes_cli import gateway_chat_chrome as chrome_mod
from hermes_cli.gateway_chat_chrome import TerminalChrome, plugin_of, sanitize
from tests.waiting import async_wait_until


def test_only_plugin_terminal_events_are_chrome():
    assert plugin_of("plugin.pantheon.terminal") == "pantheon"
    assert plugin_of("plugin.my_plugin-2.terminal") == "my_plugin-2"
    for kind in ("plugin.pantheon.scene", "message.delta", "plugin..terminal", "plugin.Bad.terminal",
                 "plugin.a.b.terminal", None, 3):
        assert plugin_of(kind) is None, kind


def test_payload_text_is_sanitized_and_capped():
    entry = sanitize({"frames": ["\x1b]0;evil\x07⚒", "ok", "", 7] + ["x"] * 30, "interval_ms": 5,
                      "label": "a\x1b[31mred\x1b[0m \n label" + "z" * 100, "status": "\x07hi", "tone": "neon"})
    assert all("\x1b" not in f and "\x07" not in f for f in entry["frames"])
    assert len(entry["frames"]) <= chrome_mod.MAX_FRAMES
    assert all(len(f) <= chrome_mod.MAX_FRAME_CHARS for f in entry["frames"])
    assert entry["interval_ms"] == chrome_mod.MIN_INTERVAL_MS
    assert entry["label"].startswith("ared label") and len(entry["label"]) == chrome_mod.MAX_TEXT_CHARS
    assert entry["status"] == "hi" and entry["tone"] == ""
    assert sanitize({"clear": True}) is None and sanitize("nope") is None
    assert sanitize({"interval_ms": True})["interval_ms"] == 0


def test_newest_plugin_owns_the_title_and_clear_hands_it_back():
    chrome = TerminalChrome()
    chrome.update("a", {"frames": ["A1", "A2"], "label": "first"})
    chrome.update("b", {"label": "second"})
    assert chrome.title() == "second"
    chrome.update("b", {"clear": True})
    assert chrome.title() == "A1 first"
    chrome.update("a", {"status": "only status"})  # replaces a's entry: no title left
    assert chrome.title() is None


def test_status_segments_carry_tone_and_current_frame():
    chrome = TerminalChrome()
    chrome.update("p", {"frames": ["⚒✦"], "status": "forging", "tone": "strong"})
    chrome.update("q", {"status": "plain"})
    assert chrome.segments() == [("strong", "⚒✦ forging"), ("", "plain")]


class _Output:
    def __init__(self):
        self.titles = []

    def set_title(self, title):
        self.titles.append(title)

    def clear_title(self):
        self.titles.append(None)

    def flush(self):
        pass


class _App:
    is_running = False


class _Session:
    def __init__(self):
        self.output, self.app = _Output(), _App()


@pytest.mark.asyncio
async def test_frames_animate_and_the_original_title_returns(monkeypatch):
    monkeypatch.setattr(chrome_mod, "_console_title", lambda: "Windows PowerShell")
    session, chrome = _Session(), TerminalChrome()
    chrome.update("pantheon", {"frames": ["⚒ϟ", "⚒✦"], "interval_ms": 100, "label": "quant check"})
    chrome.start(session.output, session.app)
    await async_wait_until(lambda: {"⚒ϟ quant check", "⚒✦ quant check"} <= set(session.output.titles),
                           timeout=5, what="both frames shown")
    chrome.update("pantheon", {"frames": ["⚒?"], "label": "quant check"})
    await async_wait_until(lambda: session.output.titles[-1] == "⚒? quant check", timeout=5,
                           what="the waiting frame")
    await chrome.stop()
    assert session.output.titles[-1] == "Windows PowerShell"


@pytest.mark.asyncio
async def test_untouched_title_is_left_alone(monkeypatch):
    monkeypatch.setattr(chrome_mod, "_console_title", lambda: None)
    session, chrome = _Session(), TerminalChrome()
    chrome.start(session.output, session.app)
    await asyncio.sleep(0)  # let the render task take its first (empty) pass
    await chrome.stop()
    assert session.output.titles == []


def test_chrome_updates_from_another_thread_wake_the_loop(monkeypatch):
    import threading
    monkeypatch.setattr(chrome_mod, "_console_title", lambda: None)  # no title to read: clear on hand-back

    async def scenario():
        session, chrome = _Session(), TerminalChrome()
        chrome.start(session.output, session.app)
        thread = threading.Thread(target=chrome.update, args=("p", {"frames": ["☾ᶻ"], "label": "t"}))
        thread.start()
        thread.join()
        await async_wait_until(lambda: session.output.titles[-1:] == ["☾ᶻ t"], timeout=5, what="cross-thread update")
        chrome.reset()
        await async_wait_until(lambda: session.output.titles[-1] is None, timeout=5, what="title handed back")
        await chrome.stop()

    asyncio.run(scenario())


def test_session_state_feeds_chrome_listeners_on_every_frame_even_between_turns():
    """The connection's reader applies each frame to its session state; a turn's queue pump only runs
    during a turn, so chrome rides the state listeners (like usage) to change while idle too."""
    from hermes_cli.gateway_agent import GatewayAgent
    from hermes_cli.gateway_connection import SessionState
    seen = []
    state = SessionState("s1")
    agent = GatewayAgent.__new__(GatewayAgent)  # the listener path needs no connection
    agent.terminal_chrome_callback = lambda plugin_id, payload: seen.append((plugin_id, payload))
    state.chrome_listeners.append(agent._terminal_chrome)
    state.apply_event("plugin.pantheon.terminal", {"frames": ["⚒✦"]})
    state.apply_event("plugin.pantheon.scene", {"frames": ["x"]})  # not chrome
    state.apply_event("message.delta", {"text": "hi"})
    assert seen == [("pantheon", {"frames": ["⚒✦"]})]
    agent.terminal_chrome_callback = lambda *a: 1 / 0
    state.apply_event("plugin.pantheon.terminal", {})  # a failing callback never reaches the reader


def test_classic_status_bar_shows_plugin_segments_in_skin_classes():
    from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin

    class Bar(CLIStatusBarMixin):
        pass

    bar = Bar()
    bar._on_terminal_chrome("pantheon", {"frames": ["⚒?"], "status": "awaits your word", "tone": "warn"})
    assert bar._plugin_status_segments(None, styled=True) == [[("class:status-bar-warn", "⚒? awaits your word")]]
    assert bar._plugin_status_segments(None, styled=False) == [[("", "⚒? awaits your word")]]
    assert bar._plugin_status_segments(frozenset({"model"}), styled=True) == []  # a fields list without it
    bar._on_terminal_chrome(None, None)
    assert bar._plugin_status_segments(None, styled=True) == []


def test_console_title_is_read_on_windows_only(monkeypatch):
    import ctypes

    class Kernel32:
        def GetConsoleTitleW(self, buf, size):
            buf.value = "PowerShell"
            return len(buf.value)

    monkeypatch.setattr(chrome_mod.os, "name", "nt")
    monkeypatch.setattr(ctypes, "windll", type("WinDLL", (), {"kernel32": Kernel32()})(), raising=False)
    assert chrome_mod._console_title() == "PowerShell"
    monkeypatch.setattr(ctypes, "windll", None, raising=False)
    assert chrome_mod._console_title() is None
    monkeypatch.setattr(chrome_mod.os, "name", "posix")
    assert chrome_mod._console_title() is None


class _RunningApp:
    is_running = True

    def __init__(self, fail=False):
        self.repaints, self.fail = 0, fail

    def invalidate(self):
        if self.fail:
            raise RuntimeError("app is closing")
        self.repaints += 1


@pytest.mark.asyncio
async def test_frames_repaint_the_running_app_and_survive_a_refusing_terminal(monkeypatch):
    monkeypatch.setattr(chrome_mod, "_console_title", lambda: None)

    class RefusingOutput(_Output):
        def set_title(self, title):
            raise OSError("terminal gone")

    app, chrome = _RunningApp(), TerminalChrome()
    chrome.update("p", {"frames": ["a", "b"], "interval_ms": 100, "status": "busy"})
    chrome.start(RefusingOutput(), app)
    chrome.start(_Output(), app)  # a second start keeps the first renderer
    await async_wait_until(lambda: app.repaints >= 2, timeout=5, what="repaints per frame")
    await chrome.stop()
    failing = TerminalChrome()
    failing.start(_Output(), _RunningApp(fail=True))
    await asyncio.sleep(0)
    await failing.stop()
