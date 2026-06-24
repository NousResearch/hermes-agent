"""Regression test: stale-stream cleanup rebuilds the correct client per api_mode.

When a streaming connection goes stale (no chunks within the stale timeout),
``interruptible_streaming_api_call`` kills the connection and rebuilds the
primary client so the retry loop gets a fresh connection pool.

Before the fix, the rebuild unconditionally called
``_replace_primary_openai_client``. In ``anthropic_messages`` mode the primary
client is ``_anthropic_client`` (the Anthropic SDK client), NOT ``self.client``
(the OpenAI SDK client). The OpenAI rebuild failed with
"OPENAI_API_KEY must be set" for non-OpenAI providers (e.g. ZAI/GLM served via
Z.AI's Anthropic-compatible endpoint), and — worse — the actually-stale
``_anthropic_client`` was never rebuilt, so every retry reused the dead
connection and looped indefinitely.

The fix branches on ``api_mode``: ``anthropic_messages``/``bedrock_converse``
rebuild ``_anthropic_client`` via ``_rebuild_anthropic_client``; all other
modes rebuild the OpenAI client as before.

The streaming stale-detection poll loop is deeply coupled to SDK streaming
context managers, transport objects, and per-attempt retry state, so driving
it end-to-end in a hermetic unit test is brittle. Instead these tests exercise
the real ``interruptible_streaming_api_call`` stale path by patching the
inner worker so it holds the connection open without emitting chunks —
reproducing the exact condition the stale detector is built for — and assert
which client-rebuild method the cleanup selected.
"""
import threading
from unittest.mock import MagicMock

import pytest


def _make_agent(api_mode: str):
    """MagicMock agent carrying the attributes the streaming poll loop reads.

    Recording mocks for the two client-rebuild methods let each test assert
    which one the stale path selected for the active ``api_mode``.
    """
    agent = MagicMock(name=f"agent[{api_mode}]")
    agent.api_mode = api_mode
    agent.base_url = "https://example.test/v1"
    agent.provider = "zai"
    agent.model = "glm-5.2"
    agent.quiet_mode = True
    agent._interrupt_requested = False
    agent._rebuild_anthropic_client = MagicMock(name="_rebuild_anthropic_client")
    agent._replace_primary_openai_client = MagicMock(
        name="_replace_primary_openai_client", return_value=True
    )
    for attr in (
        "_buffer_status", "_touch_activity", "_emit_status", "_safe_print",
        "_reset_stream_delivery_tracking", "_try_refresh_anthropic_client_credentials",
    ):
        setattr(agent, attr, MagicMock())
    agent._has_stream_consumers = MagicMock(return_value=False)
    agent._should_start_quiet_spinner = MagicMock(return_value=False)
    agent._create_request_openai_client = MagicMock(return_value=MagicMock())
    agent._close_request_openai_client = MagicMock()
    agent._abort_request_openai_client = MagicMock()
    agent._is_openai_client_closed = MagicMock(return_value=False)
    agent._get_transport = MagicMock()
    agent.stream_delta_callback = None
    agent._stream_callback = None
    agent._disable_streaming = False
    agent._extract_reasoning = MagicMock(return_value=None)
    agent._strip_think_blocks = MagicMock(side_effect=lambda x: x)
    agent._needs_thinking_reasoning_pad = MagicMock(return_value=False)
    agent._split_responses_tool_id = MagicMock(return_value=(None, None))
    agent._deterministic_call_id = MagicMock(return_value="call_test")
    agent._derive_responses_function_call_id = MagicMock(return_value="resp_test")
    agent._copy_reasoning_content_for_api = MagicMock()
    agent._supports_reasoning_extra_body = MagicMock(return_value=False)
    agent._max_tokens_param = MagicMock(return_value=None)
    agent._ollama_num_ctx = None
    agent._resolved_api_call_timeout = MagicMock(return_value=30)
    return agent


def _hung_stream_worker():
    """A worker callable that holds the connection open without emitting a
    chunk, until its release Event is set. This is the exact condition the
    stale detector targets: a live connection that produces no data."""

    release = threading.Event()

    def _worker(*args, **kwargs):
        # Block until released. The stale detector closes the (mocked)
        # request client after the timeout; we then unblock so the worker
        # thread exits and the call returns instead of hanging the test.
        release.wait(timeout=20)

    _worker.release = release
    return _worker


def _drive_to_stale(agent, api_kwargs, monkeypatch):
    """Run interruptible_streaming_api_call, let the stale detector fire, and
    return once the worker has been released. Asserts neither rebuild method;
    callers check ``agent._rebuild_anthropic_client`` / ``_replace_primary_openai_client``."""
    from agent import chat_completion_helpers as cch

    # 1s stale timeout so the detector fires fast; small context so the
    # large-context floor (300s) is never applied (see _stream_stale_timeout).
    monkeypatch.setattr(
        cch, "env_float", lambda name, default: 1.0 if name == "HERMES_STREAM_STALE_TIMEOUT" else default
    )

    # Retry-once then give up so a flaky mock doesn't loop for 500 attempts.
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    result = {}

    def _run():
        try:
            cch.interruptible_streaming_api_call(agent, api_kwargs)
            result["ok"] = True
        except BaseException as exc:  # noqa: BLE001
            result["exc"] = exc

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # 1s stale timeout + 0.3s poll cadence: ~1.5s to fire, plus rebuild.
    t.join(timeout=10)
    return result, t


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_stale_stream_rebuilds_anthropic_client_in_anthropic_mode(monkeypatch):
    """anthropic_messages: stale stream → _rebuild_anthropic_client (not OpenAI).

    Reproduces the production failure: ZAI/GLM via Z.AI's Anthropic endpoint
    streams through ``_anthropic_client``; a stale kill that rebuilds the
    OpenAI client instead leaves the dead Anthropic client in place and
    loops forever.
    """
    from agent import chat_completion_helpers as cch

    agent = _make_agent("anthropic_messages")

    # anthropic_messages mode streams via agent._anthropic_client.messages.stream(**api_kwargs)
    # (a context manager whose body iterates events). Stub it to open but yield
    # NO events, then block until released — the exact stale-stream condition.
    worker = _hung_stream_worker()

    class _StaleStreamCM:
        """Context manager that opens, yields itself (iterating produces no
        events until release), and blocks on close."""

        response = MagicMock()

        def __enter__(self):
            worker()  # hold the connection open with no chunks emitted
            return self

        def __iter__(self):
            return iter(())  # no events → outer poll loop sees no chunks

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return MagicMock()

    agent._anthropic_client = MagicMock()
    agent._anthropic_client.messages.stream = MagicMock(return_value=_StaleStreamCM())
    # _call_anthropic() also touches these diagnostic stubs.
    agent._stream_diag_init = MagicMock(return_value={})
    agent._stream_diag_capture_response = MagicMock()
    agent._stream_diag_record_event = MagicMock()
    agent.log_prefix = ""
    agent._anthropic_preserve_dots = MagicMock(return_value=False)
    agent._copy_reasoning_content_for_api = MagicMock()

    api_kwargs = {"model": "glm-5.2", "messages": [{"role": "user", "content": "hi"}]}

    result, t = _drive_to_stale(agent, api_kwargs, monkeypatch)
    worker.release.set()
    t.join(timeout=5)

    assert agent._rebuild_anthropic_client.called, (
        "anthropic_messages mode must rebuild _anthropic_client on a stale stream; "
        "the pre-fix code rebuilt the OpenAI client instead and left the stale "
        "Anthropic client in place, causing an unrecoverable retry loop on "
        "non-OpenAI providers (ZAI/GLM via the Anthropic endpoint)."
    )
    agent._replace_primary_openai_client.assert_not_called()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_stale_stream_rebuilds_openai_client_in_chat_completions_mode(monkeypatch):
    """chat_completions: stale stream → _replace_primary_openai_client (legacy path preserved)."""
    agent = _make_agent("chat_completions")
    worker = _hung_stream_worker()
    # chat_completions streams via client.chat.completions.create(stream=True).
    fake_client = MagicMock()
    fake_client.chat.completions.create = MagicMock(side_effect=worker)
    agent._create_request_openai_client.return_value = fake_client

    api_kwargs = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}]}

    result, t = _drive_to_stale(agent, api_kwargs, monkeypatch)
    worker.release.set()
    t.join(timeout=5)

    assert agent._replace_primary_openai_client.called, (
        "chat_completions mode must rebuild the OpenAI client on a stale stream "
        "(legacy behaviour must be unchanged by the api_mode branch)."
    )
    agent._rebuild_anthropic_client.assert_not_called()
