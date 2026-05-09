import queue
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from run_agent import AIAgent


class _NeverReturningCompletions:
    def __init__(self, started: threading.Event, stop: threading.Event):
        self.started = started
        self.stop = stop

    def create(self, **kwargs):
        self.started.set()
        # Simulate the OpenAI/httpx worker being stuck in an SSL read even after
        # another thread closes the client. The caller must still regain control.
        while not self.stop.wait(0.05):
            pass
        return iter(())


class _NeverReturningClient:
    def __init__(self, started: threading.Event, stop: threading.Event):
        self.chat = SimpleNamespace(
            completions=_NeverReturningCompletions(started, stop)
        )
        self.closed = False

    def close(self):
        self.closed = True


def _minimal_agent(client: _NeverReturningClient) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    agent.base_url = "https://api.example.test/v1"
    agent._base_url = agent.base_url
    agent.provider = "openrouter"
    agent.model = "test/model"
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent.reasoning_callback = None
    agent.tool_gen_callback = None
    agent._current_streamed_assistant_text = ""
    agent._create_request_openai_client = lambda **kwargs: client
    agent._close_request_openai_client = lambda c, reason=None: c.close()
    agent._replace_primary_openai_client = lambda reason=None: None
    agent._emit_status = lambda message: None
    agent._touch_activity = lambda message: None
    return agent


def test_stream_stale_timeout_returns_control_when_worker_thread_ignores_close(monkeypatch):
    """A stale streaming call must not keep the WebUI/API turn busy forever.

    Closing an OpenAI/httpx client from the watchdog thread is best-effort. In
    the failure seen in production the worker remained blocked in SSL_read after
    tool completion, leaving active_stream_id and pending_user_message set while
    the WebUI still looked busy. The watchdog must synthesize a timeout and
    return control even if the provider worker does not exit promptly.
    """
    monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.01")
    monkeypatch.setenv("HERMES_STREAM_READ_TIMEOUT", "60")

    worker_started = threading.Event()
    worker_stop = threading.Event()
    client = _NeverReturningClient(worker_started, worker_stop)
    agent = _minimal_agent(client)
    result_q: queue.Queue[BaseException | str] = queue.Queue()

    def _run_call():
        try:
            agent._interruptible_streaming_api_call({"model": "test/model", "messages": []})
        except BaseException as exc:  # noqa: BLE001 - test captures exact outcome
            result_q.put(exc)
        else:
            result_q.put("returned")

    caller = threading.Thread(target=_run_call, daemon=True)
    caller.start()

    assert worker_started.wait(1.0), "fake provider call did not start"
    caller.join(timeout=4.0)
    worker_stop.set()

    assert not caller.is_alive(), "stale stream watchdog did not return control"
    outcome = result_q.get_nowait()
    assert isinstance(outcome, TimeoutError)
    assert "Streaming API call timed out" in str(outcome)
    assert client.closed is True
