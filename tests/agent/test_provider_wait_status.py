import time
from types import SimpleNamespace

from agent.chat_completion_helpers import interruptible_api_call


class _FakeCompletions:
    def __init__(self, delay: float):
        self.delay = delay

    def create(self, **_kwargs):
        time.sleep(self.delay)
        return SimpleNamespace(ok=True)


class _FakeClient:
    def __init__(self, delay: float):
        self.chat = SimpleNamespace(completions=_FakeCompletions(delay))


class _FakeAgent:
    def __init__(self, *, platform: str = "tui", delay: float = 0.35):
        self.api_mode = "chat_completions"
        self.platform = platform
        self._interrupt_requested = False
        self._client = _FakeClient(delay)
        self.status_events = []
        self.activities = []

    def _create_request_openai_client(self, *, reason: str, api_kwargs=None):
        return self._client

    def _close_request_openai_client(self, client, *, reason: str):
        return None

    def _abort_request_openai_client(self, client, *, reason: str):
        return None

    def _compute_non_stream_stale_timeout(self, api_kwargs):
        return 2.0

    def _touch_activity(self, desc: str):
        self.activities.append(desc)

    def status_callback(self, kind: str, text: str):
        self.status_events.append((kind, text))



def test_tui_provider_wait_status_heartbeats_during_silent_model_call(monkeypatch):
    monkeypatch.setenv("HERMES_PROVIDER_WAIT_STATUS_INTERVAL_SECONDS", "0.05")
    agent = _FakeAgent(platform="tui", delay=0.35)

    response = interruptible_api_call(agent, {"model": "test-model"})

    assert response.ok is True
    assert any(kind == "provider_wait" for kind, _ in agent.status_events)
    assert any("test-model" in text and "等待" in text for _, text in agent.status_events)



def test_provider_wait_status_is_not_emitted_to_messaging_platforms(monkeypatch):
    monkeypatch.setenv("HERMES_PROVIDER_WAIT_STATUS_INTERVAL_SECONDS", "0.05")
    agent = _FakeAgent(platform="telegram", delay=0.35)

    response = interruptible_api_call(agent, {"model": "test-model"})

    assert response.ok is True
    assert agent.status_events == []
