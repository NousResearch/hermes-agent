import inspect
from datetime import datetime
from types import SimpleNamespace

from gateway.config import GatewayConfig, Platform
from gateway.run import GatewayRunner
from gateway.session import SessionEntry, SessionSource, SessionStore


SESSION_KEY = "agent:main:telegram:dm:123"
NOTICE = "Codex gpt-5.5 caps context at 272K"


def _store(tmp_path) -> SessionStore:
    store = SessionStore(tmp_path, GatewayConfig())
    now = datetime.now()
    store._entries[SESSION_KEY] = SessionEntry(
        session_key=SESSION_KEY,
        session_id="20260608_test",
        created_at=now,
        updated_at=now,
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="123"),
        platform=Platform.TELEGRAM,
    )
    store._loaded = True
    store._save()
    return store


def _agent(warning: str = NOTICE):
    return SimpleNamespace(
        _compression_threshold_autoraised={"from": 0.5, "to": 0.85},
        _compression_warning=warning,
    )


def _runner(store: SessionStore):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = store
    return runner


def test_codex_autoraise_notice_is_kept_for_first_fresh_agent(tmp_path):
    store = _store(tmp_path)
    agent = _agent()

    GatewayRunner._apply_gateway_notice_latch(_runner(store), agent, SESSION_KEY)

    assert agent._compression_warning == NOTICE
    assert store._entries[SESSION_KEY].notices_shown == {"codex_gpt55_autoraise": True}


def test_codex_autoraise_notice_is_cleared_for_rebuilt_agent_same_session(tmp_path):
    store = _store(tmp_path)
    first_agent = _agent()
    second_agent = _agent()

    GatewayRunner._apply_gateway_notice_latch(_runner(store), first_agent, SESSION_KEY)
    GatewayRunner._apply_gateway_notice_latch(_runner(store), second_agent, SESSION_KEY)

    assert first_agent._compression_warning == NOTICE
    assert second_agent._compression_warning is None


def test_codex_autoraise_notice_is_once_per_durable_session(tmp_path):
    store = _store(tmp_path)
    other_key = "agent:main:telegram:dm:456"
    now = datetime.now()
    store._entries[other_key] = SessionEntry(
        session_key=other_key,
        session_id="20260608_other",
        created_at=now,
        updated_at=now,
        origin=SessionSource(platform=Platform.TELEGRAM, chat_id="456"),
        platform=Platform.TELEGRAM,
    )
    store._save()

    first_session_agent = _agent()
    other_session_agent = _agent()

    GatewayRunner._apply_gateway_notice_latch(_runner(store), first_session_agent, SESSION_KEY)
    GatewayRunner._apply_gateway_notice_latch(_runner(store), other_session_agent, other_key)

    assert first_session_agent._compression_warning == NOTICE
    assert other_session_agent._compression_warning == NOTICE
    assert store._entries[SESSION_KEY].notices_shown == {"codex_gpt55_autoraise": True}
    assert store._entries[other_key].notices_shown == {"codex_gpt55_autoraise": True}


def test_agents_without_codex_autoraise_warning_are_not_latched(tmp_path):
    store = _store(tmp_path)
    agent = SimpleNamespace(_compression_threshold_autoraised=None, _compression_warning=NOTICE)

    GatewayRunner._apply_gateway_notice_latch(_runner(store), agent, SESSION_KEY)

    assert agent._compression_warning == NOTICE
    assert store._entries[SESSION_KEY].notices_shown == {}


def test_gateway_latch_tolerates_minimal_test_runner_without_session_store():
    runner = GatewayRunner.__new__(GatewayRunner)
    agent = _agent()

    GatewayRunner._apply_gateway_notice_latch(runner, agent, SESSION_KEY)

    assert agent._compression_warning == NOTICE


def test_latch_allows_exactly_one_lifecycle_replay_for_rebuilt_agents(tmp_path):
    store = _store(tmp_path)
    delivered = []

    for agent in (_agent(), _agent()):
        GatewayRunner._apply_gateway_notice_latch(_runner(store), agent, SESSION_KEY)
        if agent._compression_warning:
            delivered.append(("lifecycle", agent._compression_warning))
            agent._compression_warning = None

    assert delivered == [("lifecycle", NOTICE)]


def test_gateway_fresh_agent_path_applies_latch_before_turn_callbacks():
    source = inspect.getsource(GatewayRunner._run_agent)
    latch_call = "self._apply_gateway_notice_latch(agent, session_key)"

    assert latch_call in source
    assert source.index("agent = AIAgent(") < source.index(latch_call)
    assert source.index(latch_call) < source.index("agent.status_callback = _status_callback_sync")
    assert source.index(latch_call) < source.index("agent.run_conversation")
