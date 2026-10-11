"""Resuming a long-running thread must not hydrate an unused transcript in the RPC response."""
from agent.transports.codex_app_server_session import CodexAppServerSession
from tests.agent.transports.test_codex_app_server_session import FakeClient


def test_resume_preserves_thread_without_returning_full_history():
    class RecordingClient(FakeClient):
        def request(self, method, params=None, timeout=30):
            if method == 'thread/resume':
                assert params['excludeTurns'] is True
                assert timeout == 90
                self.requests.append((method,params))
                return {'thread': {'id': params['threadId']}}
            return super().request(method, params, timeout)
    client = RecordingClient()
    session = CodexAppServerSession(resume_thread_id='thread-fake-001',client_factory=lambda **kw:client)
    assert session.ensure_started() == 'thread-fake-001'
    assert not any(method == 'thread/start' for method,_ in client.requests)
