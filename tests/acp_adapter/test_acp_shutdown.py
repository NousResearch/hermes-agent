import acp

from acp_adapter import entry, server
from acp_adapter.server import HermesACPAgent
from acp_adapter.session import SessionManager


class NoopDb:
    def get_session(self, *_args, **_kwargs):
        return None

    def create_session(self, *_args, **_kwargs):
        return None

    def update_session(self, *_args, **_kwargs):
        return None


class ShutdownAgent:
    def __init__(self):
        self.model = "fake-model"
        self.shutdown_calls = []

    def shutdown_memory_provider(self, messages=None):
        self.shutdown_calls.append(list(messages or []))


class RaisingShutdownAgent(ShutdownAgent):
    def shutdown_memory_provider(self, messages=None):
        super().shutdown_memory_provider(messages)
        raise RuntimeError("boom")


class EntrypointAgent:
    def __init__(self):
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1


def prepare_entrypoint(monkeypatch, agent, run_agent):
    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(server, "HermesACPAgent", lambda: agent)
    monkeypatch.setattr(acp, "run_agent", run_agent)


def test_session_manager_shutdown_sessions_flushes_agent_memory_with_history():
    fake = ShutdownAgent()
    manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=NoopDb())
    state = manager.create_session(cwd=".")
    state.history = [
        {"role": "user", "content": "remember this"},
        {"role": "assistant", "content": "stored"},
    ]

    manager.shutdown_sessions()

    assert fake.shutdown_calls == [state.history]


def test_acp_agent_shutdown_delegates_to_session_manager():
    fake = ShutdownAgent()
    manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=NoopDb())
    state = manager.create_session(cwd=".")
    state.history = [{"role": "assistant", "content": "done"}]
    acp_agent = HermesACPAgent(session_manager=manager)

    acp_agent.shutdown()

    assert fake.shutdown_calls == [state.history]


def test_session_manager_shutdown_sessions_is_best_effort():
    fake = RaisingShutdownAgent()
    manager = SessionManager(agent_factory=lambda **_kwargs: fake, db=NoopDb())
    manager.create_session(cwd=".")

    manager.shutdown_sessions()

    assert len(fake.shutdown_calls) == 1


def test_entrypoint_shutdown_runs_after_normal_return(monkeypatch):
    agent = EntrypointAgent()

    async def run_agent(actual_agent, *, use_unstable_protocol):
        assert actual_agent is agent
        assert use_unstable_protocol is True

    prepare_entrypoint(monkeypatch, agent, run_agent)

    entry.main([])

    assert agent.shutdown_calls == 1


def test_entrypoint_shutdown_runs_after_keyboard_interrupt(monkeypatch):
    agent = EntrypointAgent()

    async def run_agent(_agent, *, use_unstable_protocol):
        assert use_unstable_protocol is True
        raise KeyboardInterrupt

    prepare_entrypoint(monkeypatch, agent, run_agent)

    entry.main([])

    assert agent.shutdown_calls == 1
