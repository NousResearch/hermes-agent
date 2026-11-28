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


class _Connection:
    def __init__(self):
        from hermes_cli.gateway_connection import SessionState
        self.states = {}
        self._make = SessionState

    def state(self, session_id):
        return self.states.setdefault(session_id, self._make(session_id))

    def call(self, *a, **k):
        return {}


def _agent(connection, session_id="s1", seen=None, callback=None):
    """A ``GatewayAgent`` with just the chrome wiring ``__init__`` and the CLI give it."""
    from hermes_cli.gateway_agent import GatewayAgent
    agent = GatewayAgent.__new__(GatewayAgent)
    agent.connection, agent._session_id = connection, session_id
    connection.state(session_id).chrome_listeners.append(agent._terminal_chrome)
    if seen is not None:
        def callback(plugin_id, payload):
            seen.append((plugin_id, payload))
    if callback is not None:
        agent.terminal_chrome_callback = callback
    return agent


def _status_bar():
    from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin

    class Bar(CLIStatusBarMixin):
        pass

    return Bar()


def test_session_state_feeds_chrome_listeners_on_every_frame_even_between_turns():
    """The connection's reader applies each frame to its session state; a turn's queue pump only runs
    during a turn, so chrome rides the state listeners (like usage) to change while idle too."""
    connection, seen = _Connection(), []
    agent = _agent(connection, seen=seen)
    state = connection.state("s1")
    state.apply_event("plugin.pantheon.terminal", {"frames": ["⚒✦"]})
    state.apply_event("plugin.pantheon.scene", {"frames": ["x"]})  # not chrome
    state.apply_event("message.delta", {"text": "hi"})
    assert seen == [("pantheon", {"frames": ["⚒✦"]})]
    agent.terminal_chrome_callback = lambda *a: 1 / 0
    state.apply_event("plugin.pantheon.terminal", {})  # a failing callback never reaches the reader


def test_a_viewer_attaching_later_starts_from_each_plugins_latest_chrome():
    connection, seen = _Connection(), []
    state = connection.state("s1")
    state.apply_event("plugin.a.terminal", {"label": "a1"})
    state.apply_event("plugin.b.terminal", {"label": "b1"})
    state.apply_event("plugin.gone.terminal", {"label": "x"})
    state.apply_event("plugin.gone.terminal", {"clear": True})
    state.apply_event("plugin.a.terminal", {"label": "a2"})  # now the newest
    _agent(connection, seen=seen)
    assert seen == [("b", {"label": "b1"}), ("a", {"label": "a2"})]


def test_a_rebuilt_agent_takes_over_without_a_frozen_tab_or_a_second_listener():
    """The CLI retires its agent (model or reasoning switch) and builds the next one on the next turn;
    the tab must keep following the session in between, and the successor replaces the old listener."""
    connection, bar = _Connection(), _status_bar()
    state = connection.state("s1")
    first = _agent(connection, callback=bar._on_terminal_chrome)
    first.release_clients()
    state.apply_event("plugin.p.terminal", {"label": "summoned while retired"})
    state.apply_event("plugin.q.terminal", {"status": "brief"})
    state.apply_event("plugin.q.terminal", {"clear": True})
    assert bar._terminal_chrome().title() == "summoned while retired"
    assert bar._terminal_chrome().segments() == []
    second = _agent(connection, callback=bar._on_terminal_chrome)
    assert state.chrome_listeners == [second._terminal_chrome]
    assert bar._terminal_chrome().title() == "summoned while retired"
    second.close()
    assert state.chrome_listeners == []


def test_switching_session_resets_the_tab_and_replays_the_new_sessions_chrome():
    connection, seen = _Connection(), []
    agent = _agent(connection, seen=seen)
    agent._session_db = None
    agent._apply_info = lambda info: None
    agent._sync_show_reasoning = lambda: None
    agent._lost_turn_notice = lambda: ""
    connection.state("s1").apply_event("plugin.p.terminal", {"label": "old"})
    connection.state("s2").apply_event("plugin.p.terminal", {"label": "new"})
    agent.session_id = "s2"
    assert seen[-2:] == [(None, None), ("p", {"label": "new"})]
    assert connection.state("s1").chrome_listeners == []
    connection.state("s1").apply_event("plugin.p.terminal", {"label": "stale"})
    assert seen[-1] == ("p", {"label": "new"})


def test_frames_from_a_session_the_agent_left_are_dropped():
    connection, seen = _Connection(), []
    agent = _agent(connection, seen=seen)
    connection.state("s2").apply_event("plugin.p.terminal", {"label": "new"})
    in_flight = agent._terminal_chrome  # the reader copied s1's listener list before the switch
    agent._session_id = "s2"
    in_flight("s1", "p", {"label": "stale-old-session"})
    agent._show_chrome(reset=True)
    assert seen == [(None, None), ("p", {"label": "new"})]


def test_untrusted_text_is_bounded_before_escapes_are_stripped(monkeypatch):
    lengths = []
    real = chrome_mod.strip_ansi
    monkeypatch.setattr(chrome_mod, "strip_ansi", lambda text: lengths.append(len(text)) or real(text))
    entry = sanitize({"label": "\x1b]" * 100_000, "frames": ["\x1b]" * 100_000]})
    assert lengths and max(lengths) <= chrome_mod.MAX_RAW_CHARS
    assert "\x1b" not in entry["label"]


def test_interval_is_clamped_to_a_ceiling():
    chrome = TerminalChrome(clock=lambda: 0.0)
    chrome.update("p", {"frames": ["a", "b"], "interval_ms": 10**400})
    assert sanitize({"interval_ms": 10**400})["interval_ms"] == chrome_mod.MAX_INTERVAL_MS
    assert chrome.next_change() == chrome_mod.MAX_INTERVAL_MS / 1000


def test_each_plugin_animates_on_its_own_clock():
    now = [0.0]
    chrome = TerminalChrome(clock=lambda: now[0])
    chrome.update("fast", {"frames": ["f0", "f1", "f2", "f3"], "interval_ms": 100, "status": "s"})
    chrome.update("slow", {"frames": ["s0", "s1"], "interval_ms": 500, "label": "t"})
    now[0] = 0.34
    assert chrome.title() == "s0 t"
    assert chrome.segments() == [("", "f3 s")]
    assert chrome.next_change() == pytest.approx(0.06)
    chrome.update("fast", {"status": "still"})
    assert chrome.next_change() == pytest.approx(0.16)


def test_classic_status_bar_shows_plugin_segments_in_skin_classes():
    bar = _status_bar()
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
