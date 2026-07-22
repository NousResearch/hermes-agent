import threading

from run_agent import AIAgent


class _FakeCodexSession:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def test_agent_close_releases_codex_app_server_session(monkeypatch):
    agent = AIAgent.__new__(AIAgent)
    agent.session_id = "test-codex-lifecycle"
    agent.client = None
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()
    agent._end_session_on_close = False
    agent._session_messages = ["retained"]
    codex_session = _FakeCodexSession()
    agent._codex_session = codex_session

    monkeypatch.setattr("run_agent.cleanup_vm", lambda _task_id: None)
    monkeypatch.setattr("run_agent.cleanup_browser", lambda _task_id: None)

    agent.close()
    agent.close()

    assert codex_session.close_calls == 1
    assert agent._codex_session is None
    assert agent._session_messages == []
