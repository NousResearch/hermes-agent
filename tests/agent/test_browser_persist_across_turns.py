"""A conversation's browser outlives its turns with ``browser.persist_across_turns`` and closes when the
conversation ends — every task id its turns ran under, on every session-end path (agent close, a /new on
the same agent, a gateway boundary evict) — never on a plain cache evict (the session may resume)."""

import threading
from types import SimpleNamespace

import pytest

from agent.chat_completion_helpers import cleanup_task_resources
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from run_agent import AIAgent
from tools import browser_session


@pytest.fixture
def sessions(monkeypatch):
    """Real session registry with record-only sessions (nothing launched, so release has nothing to stop)."""
    active: dict = {}
    monkeypatch.setattr(browser_session, "_active_sessions", active)
    monkeypatch.setattr(browser_session, "_session_last_activity", {})
    monkeypatch.setattr(browser_session, "_session_owner_homes", {})

    def add(*keys):
        for key in keys:
            active[key] = {"session_name": f"h_{key}", "cdp_url": "ws://127.0.0.1:1/devtools/browser/x",
                           "features": {"local": True}}
        return active

    return add


@pytest.fixture
def home(tmp_path):
    def write(config_text: str):
        (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
        return set_hermes_home_override(str(tmp_path))
    return write


def _turn_end(home, config_text: str, task_id: str) -> None:
    token = home(config_text)
    try:
        cleanup_task_resources(SimpleNamespace(verbose_logging=False), task_id)
    finally:
        reset_hermes_home_override(token)


def test_persist_across_turns_keeps_the_page_for_the_next_turn(sessions, home):
    active = sessions("sid-1", "sid-1@p-home")
    _turn_end(home, "browser:\n  persist_across_turns: true\n", "sid-1")
    assert set(active) == {"sid-1", "sid-1@p-home"}


def test_turns_without_a_caller_task_id_do_not_pile_up_kept_browsers(sessions, home):
    """chat() and Kanban turns omit task_id, so each turn gets a fresh one no later turn can reach: its
    browser closes at turn end; a caller-supplied id (CLI, TUI, gateway, ACP) keeps the page."""
    from agent.turn_context import _bind_turn_identity
    agent = SimpleNamespace(verbose_logging=False, session_id="sid-1")
    active = sessions("sid-1")
    token = home("browser:\n  persist_across_turns: true\n")
    try:
        for _ in range(2):
            generated, _turn = _bind_turn_identity(agent, None, None, None, None, None)
            sessions(generated)
            cleanup_task_resources(agent, generated)
        assert set(active) == {"sid-1"}
        supplied, _turn = _bind_turn_identity(agent, "sid-1", None, None, None, None)
        cleanup_task_resources(agent, supplied)
    finally:
        reset_hermes_home_override(token)
    assert set(active) == {"sid-1"}


def test_default_still_closes_the_browser_at_turn_end(sessions, home):
    active = sessions("sid-1")
    _turn_end(home, "browser:\n  inactivity_timeout: 120\n", "sid-1")
    assert active == {}


def test_agent_close_releases_every_task_id_its_turns_ran_under(sessions):
    """A compression rotation (or a delegated/cron run) keys the kept browser by a task id other than the
    final session_id; close() must not leave it to the idle reaper."""
    active = sessions("sid-before-rotation", "sid-after-rotation", "other-conversation")
    agent = SimpleNamespace(_process_owner_task_ids={"sid-before-rotation", "sid-after-rotation"})
    AIAgent._close_task_resources(agent, "sid-after-rotation")
    assert set(active) == {"other-conversation"}


def test_new_on_the_same_agent_closes_the_ended_conversations_browser(sessions):
    """/new, /resume and /branch keep the agent and switch session_id, then reset_session_state()."""
    active = sessions("old-sid", "other-conversation")
    agent = SimpleNamespace(session_id="new-sid", _process_owner_task_ids={"old-sid"}, context_compressor=None,
                            _transition_context_engine_session=lambda **kw: None)
    AIAgent.reset_session_state(agent)
    assert set(active) == {"other-conversation"}


def test_session_end_closes_the_named_sessions_the_conversation_drove_last(sessions, home):
    """``browser_exec(session="x")`` keys its browser ``bu-named-x``, not by task: the turn end leaves it (as
    before), and the end of the conversation that drove it last closes it; another conversation's stays."""
    active = sessions("bu-named-default", "bu-named-research", "bu-named-theirs")
    browser_session.record_session_driver("bu-named-default", "sid-1")
    browser_session.record_session_driver("bu-named-research", "sid-1")
    browser_session.record_session_driver("bu-named-theirs", "sid-1")
    browser_session.record_session_driver("bu-named-theirs", "other-sid")  # driven since by another conversation
    _turn_end(home, "browser:\n  inactivity_timeout: 120\n", "sid-1")
    assert len(active) == 3
    AIAgent._close_task_resources(SimpleNamespace(_process_owner_task_ids={"sid-1"}), "sid-1")
    assert set(active) == {"bu-named-theirs"}


def _gateway_runner(monkeypatch):
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()
    runner._peek_session_state = lambda key: None
    runner._running_agent_ids = lambda: set()
    monkeypatch.setattr(runner, "_spawn_release_thread",
                        lambda target, args, name, *, inline_fallback, session_key=None: target(*args))
    return runner


def _cached_agent(task_ids):
    return SimpleNamespace(_memory_manager=None, _session_messages=[], release_clients=lambda: None,
                           _process_owner_task_ids=set(task_ids))


def test_gateway_session_boundary_closes_the_kept_browser(sessions, monkeypatch):
    """A viewer's /new or exit (CLI, TUI) ends the session by a boundary evict, which only soft-releases the
    agent: without closing the browser here it would outlive the conversation until the idle reaper."""
    active = sessions("local-abc", "other-conversation")
    runner = _gateway_runner(monkeypatch)
    runner._agent_cache["route"] = (_cached_agent({"local-abc"}), "sig")
    runner._evict_cached_agent_at_boundary("route")
    assert set(active) == {"other-conversation"}


def test_gateway_cache_evict_keeps_the_browser_for_a_resume(sessions, monkeypatch):
    active = sessions("local-abc")
    runner = _gateway_runner(monkeypatch)
    runner._agent_cache["route"] = (_cached_agent({"local-abc"}), "sig")
    runner._evict_cached_agent("route")
    assert set(active) == {"local-abc"}


def test_boundary_after_a_plain_evict_still_closes_the_kept_browser(sessions, monkeypatch):
    """An idle/LRU evict drops the agent but keeps its browser (the chat may resume); a later /new or close
    finds no cached agent and must still close what that conversation's turns kept."""
    active = sessions("local-abc", "other-conversation")
    runner = _gateway_runner(monkeypatch)
    runner._agent_cache["route"] = (_cached_agent({"local-abc"}), "sig")
    runner._evict_cached_agent("route")
    assert set(active) == {"local-abc", "other-conversation"}
    runner._evict_cached_agent_at_boundary("route")
    assert set(active) == {"other-conversation"}


def test_in_place_compression_keeps_the_browser_until_the_conversation_ends(sessions, monkeypatch):
    """An in-place /compress keeps the session id the next turn runs under: its browser stays, and the
    conversation's real end still closes it."""
    active = sessions("local-abc", "other-conversation")
    runner = _gateway_runner(monkeypatch)
    runner._agent_cache["route"] = (_cached_agent({"local-abc"}), "sig")
    runner._evict_cached_agent_at_boundary("route", keep_browsers=True)
    assert set(active) == {"local-abc", "other-conversation"}
    runner._evict_cached_agent_at_boundary("route")
    assert set(active) == {"other-conversation"}


def test_ended_conversations_slow_close_spares_the_next_conversations_browser(sessions, monkeypatch):
    """A reset frees the route while the old conversation's memory commit still runs; the next conversation
    browses and is plain-evicted (a /model) under the same key. The old close must take only its own ids."""
    active = sessions("old-sid", "new-sid")
    runner = _gateway_runner(monkeypatch)
    deferred = []
    monkeypatch.setattr(runner, "_spawn_release_thread",
                        lambda target, args, name, *, inline_fallback, session_key=None: deferred.append((target, args)))
    runner._agent_cache["route"] = (_cached_agent({"old-sid"}), "sig")
    runner._evict_cached_agent_at_boundary("route")
    runner._agent_cache["route"] = (_cached_agent({"new-sid"}), "sig")
    runner._evict_cached_agent("route")
    for target, args in deferred:
        target(*args)
    assert set(active) == {"new-sid"}


def test_conversation_close_spares_a_named_session_another_conversation_took_over(sessions, monkeypatch):
    """The close scans, then another conversation drives the named browser before it is claimed: the
    claim re-checks the driver under the registry lock and leaves that browser running."""
    from tools.browser_session_lifecycle import cleanup_conversation_browsers

    active = sessions("bu-named-shared")
    browser_session.record_session_driver("bu-named-shared", "sid-a")

    class _TakeoverAfterScan:
        def __init__(self):
            self._lock, self._hook = threading.RLock(), self._take_over

        def _take_over(self):
            browser_session.record_session_driver("bu-named-shared", "sid-b")

        def __enter__(self):
            self._lock.acquire()

        def __exit__(self, *exc):
            self._lock.release()
            hook, self._hook = self._hook, None
            if hook:
                hook()

    monkeypatch.setattr(browser_session, "_cleanup_lock", _TakeoverAfterScan())
    cleanup_conversation_browsers({"sid-a"})
    assert set(active) == {"bu-named-shared"}
