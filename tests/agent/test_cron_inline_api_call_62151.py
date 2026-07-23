"""Regression guard for #62151 — gateway cron must not wedge on the 2nd+ call.

Gateway-fired cron jobs hung forever on the 2nd+ API call because both the
non-streaming (``interruptible_api_call``) and the default streaming
(``interruptible_streaming_api_call``) paths run the request on a spawned
daemon worker thread. Inside the gateway's nested cron thread pools that extra
worker wedged before the socket opened; the same job succeeded via ``hermes
cron tick`` (foreground, no nested pools). Cron has no interactive interrupt
surface, so both paths now run inline on the conversation thread for the
``cron`` platform.

These tests pin: (1) the inline gate is cron-only, (2) the inline call runs on
the *calling* thread — no worker is spawned — for both entry points, and (3)
the shared dispatch closes the per-request client.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from run_agent import AIAgent

from agent.chat_completion_helpers import (
    direct_api_call,
    interruptible_api_call,
    interruptible_streaming_api_call,
    should_use_direct_api_call,
)


def _make_agent(*, platform="cron"):
    agent = MagicMock()
    agent.platform = platform
    agent.api_mode = "chat_completions"
    agent.provider = "openrouter"
    agent._interrupt_requested = False
    agent._consecutive_stale_streams = 0
    agent._touch_activity = MagicMock()
    agent._close_request_openai_client = MagicMock()
    return agent


def test_should_use_direct_api_call_only_for_cron_openai_wire():
    assert should_use_direct_api_call(_make_agent(platform="cron")) is True
    assert should_use_direct_api_call(_make_agent(platform="cli")) is False
    assert should_use_direct_api_call(_make_agent(platform="telegram")) is False
    assert should_use_direct_api_call(_make_agent(platform=None)) is False

    for api_mode in ("anthropic_messages", "bedrock_converse"):
        agent = _make_agent(platform="cron")
        agent.api_mode = api_mode
        assert should_use_direct_api_call(agent) is False

    moa = _make_agent(platform="cron")
    moa.provider = "moa"
    assert should_use_direct_api_call(moa) is False


def test_direct_api_call_runs_inline_and_closes_client():
    agent = _make_agent()
    caller_tid = threading.get_ident()
    ran_on = {}
    fake_client = MagicMock()

    def _create(**_kwargs):
        ran_on["tid"] = threading.get_ident()
        return fake_client

    fake_client.chat.completions.create.return_value = SimpleNamespace(id="resp")
    agent._create_request_openai_client.side_effect = _create

    resp = direct_api_call(agent, {"model": "m", "messages": []})

    assert resp.id == "resp"
    # Inline: the request ran on the calling thread, no worker was spawned.
    assert ran_on["tid"] == caller_tid
    assert agent._close_request_openai_client.call_count == 1


def test_interruptible_api_call_routes_cron_inline_no_worker_thread():
    agent = _make_agent()
    caller_tid = threading.get_ident()
    fake_client = MagicMock()
    ran_on = {}

    def _create(**_kwargs):
        ran_on["tid"] = threading.get_ident()
        return fake_client

    fake_client.chat.completions.create.return_value = SimpleNamespace(id="first")
    agent._create_request_openai_client.side_effect = _create

    resp = interruptible_api_call(agent, {"model": "m", "messages": []})

    assert resp.id == "first"
    assert ran_on["tid"] == caller_tid  # no daemon worker thread


def test_interruptible_api_call_routes_cron_codex_inline_no_worker_thread():
    """Codex cron calls must avoid the same nested worker as chat completions."""
    agent = _make_agent()
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    agent._compute_non_stream_stale_timeout.return_value = 180.0
    caller_tid = threading.get_ident()
    fake_client = MagicMock()
    created_on = {}
    streamed_on = {}

    def _create(**_kwargs):
        created_on["tid"] = threading.get_ident()
        return fake_client

    def _run_codex_stream(_api_kwargs, **_kwargs):
        streamed_on["tid"] = threading.get_ident()
        return SimpleNamespace(id="codex")

    agent._create_request_openai_client.side_effect = _create
    agent._run_codex_stream.side_effect = _run_codex_stream

    resp = interruptible_api_call(agent, {"model": "gpt-5.6-sol", "input": []})

    assert resp is not None
    assert resp.id == "codex"
    assert created_on["tid"] == caller_tid
    assert streamed_on["tid"] == caller_tid  # no daemon worker thread
    agent._run_codex_stream.assert_called_once()


def test_interruptible_api_call_routes_sequential_cron_codex_calls_inline():
    """The reported 2nd+ Codex call must stay on the cron conversation thread."""
    agent = _make_agent()
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    caller_tid = threading.get_ident()
    clients = [MagicMock(name="first_client"), MagicMock(name="second_client")]
    created_on = []
    streamed_on = []

    def _create(**_kwargs):
        created_on.append(threading.get_ident())
        return clients[len(created_on) - 1]

    def _run_codex_stream(api_kwargs, **_kwargs):
        streamed_on.append(threading.get_ident())
        return SimpleNamespace(id=api_kwargs["input"][0])

    agent._create_request_openai_client.side_effect = _create
    agent._run_codex_stream.side_effect = _run_codex_stream

    first = interruptible_api_call(
        agent, {"model": "gpt-5.6-sol", "input": ["first"]}
    )
    second = interruptible_api_call(
        agent, {"model": "gpt-5.6-sol", "input": ["second"]}
    )

    assert first is not None
    assert second is not None
    assert (first.id, second.id) == ("first", "second")
    assert created_on == [caller_tid, caller_tid]
    assert streamed_on == [caller_tid, caller_tid]
    assert agent._close_request_openai_client.call_args_list == [
        call(clients[0], reason="request_complete"),
        call(clients[1], reason="request_complete"),
    ]


def test_direct_api_call_interrupt_aborts_active_client_and_raises():
    """Cron's outer watchdog interrupts from another thread while inline."""
    agent = _make_agent()
    client_ready = threading.Event()
    release_request = threading.Event()
    fake_client = MagicMock()

    def _create(**_kwargs):
        client_ready.set()
        return fake_client

    def _request(**_kwargs):
        assert release_request.wait(timeout=2)
        raise RuntimeError("socket closed")

    fake_client.chat.completions.create.side_effect = _request
    agent._create_request_openai_client.side_effect = _create
    result = {}

    def _run():
        try:
            direct_api_call(agent, {"model": "m", "messages": []})
        except Exception as exc:
            result["exception"] = exc

    worker = threading.Thread(target=_run)
    worker.start()
    assert client_ready.wait(timeout=1)
    assert callable(agent._active_request_abort)
    AIAgent.interrupt(agent, "cron timeout")
    agent._abort_request_openai_client.assert_called_once_with(
        fake_client, reason="interrupt_abort"
    )
    release_request.set()
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert isinstance(result.get("exception"), InterruptedError)
    assert agent._active_request_abort is None


def test_direct_codex_call_interrupt_aborts_and_closes_on_owner_thread():
    """Cron timeout interrupts inline Codex and leaves cleanup to its owner."""
    agent = _make_agent()
    agent.api_mode = "codex_responses"
    agent.provider = "openai-codex"
    stream_ready = threading.Event()
    release_stream = threading.Event()
    fake_client = MagicMock()
    thread_ids = {}
    result = {}

    def _create(**_kwargs):
        thread_ids["create"] = threading.get_ident()
        return fake_client

    def _run_codex_stream(_api_kwargs, **_kwargs):
        thread_ids["stream"] = threading.get_ident()
        stream_ready.set()
        assert release_stream.wait(timeout=2)
        raise RuntimeError("socket closed")

    def _abort(_client, *, reason):
        thread_ids["abort"] = threading.get_ident()
        assert reason == "interrupt_abort"

    def _close(_client, *, reason):
        thread_ids["close"] = threading.get_ident()
        assert reason == "request_complete"

    agent._create_request_openai_client.side_effect = _create
    agent._run_codex_stream.side_effect = _run_codex_stream
    agent._abort_request_openai_client.side_effect = _abort
    agent._close_request_openai_client.side_effect = _close

    def _run():
        thread_ids["owner"] = threading.get_ident()
        try:
            direct_api_call(agent, {"model": "gpt-5.6-sol", "input": []})
        except Exception as exc:
            result["exception"] = exc

    owner = threading.Thread(target=_run)
    owner.start()
    assert stream_ready.wait(timeout=1)
    assert callable(agent._active_request_abort)

    interrupter_tid = threading.get_ident()
    AIAgent.interrupt(agent, "cron inactivity timeout")
    release_stream.set()
    owner.join(timeout=2)

    assert not owner.is_alive()
    assert isinstance(result.get("exception"), InterruptedError)
    assert agent._active_request_abort is None
    agent._abort_request_openai_client.assert_called_once_with(
        fake_client, reason="interrupt_abort"
    )
    agent._close_request_openai_client.assert_called_once_with(
        fake_client, reason="request_complete"
    )
    assert thread_ids["abort"] == interrupter_tid
    assert thread_ids["create"] == thread_ids["owner"]
    assert thread_ids["stream"] == thread_ids["owner"]
    assert thread_ids["close"] == thread_ids["owner"]


def test_interruptible_streaming_api_call_routes_cron_via_nonstream_method():
    """Streaming is the default even for cron — the gate must catch it too.

    It delegates to the ``_interruptible_api_call`` method (which itself runs
    inline for cron) rather than calling ``direct_api_call`` directly, so the
    outer loop's per-request retry/refresh seam — which patches that method —
    stays intact (regression from the codex 401-refresh path).
    """
    agent = _make_agent()
    sentinel = SimpleNamespace(id="via-nonstream")
    agent._interruptible_api_call = MagicMock(return_value=sentinel)

    resp = interruptible_streaming_api_call(
        agent, {"model": "m", "messages": []}, on_first_delta=lambda: None
    )

    assert resp is sentinel
    agent._interruptible_api_call.assert_called_once()
