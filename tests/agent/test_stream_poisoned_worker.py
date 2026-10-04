"""A poisoned managed llama-server worker is recycled once, never retried forever (#132778).

Incident shape (Mac Studio, Metal, 2026-10): after an OOM the child's backend logs "backend is in
error state from a previous command buffer failure - recreate the backend to recover" and answers
every request HTTP 500 "Compute error" before a single token. The server declared the worker dead
until recreated, yet the stream retries, the 5xx unmask probe and the turn loop's retry/fallback
machinery kept re-issuing requests against it (1,062 500s in one log window).

Contract: for a MANAGED worker (this process's supervisor owns ``agent.base_url``) a pre-delivery
Compute error with that verdict in the router log replaces the worker (unload + load: a fresh
child process and backend generation), gets ONE retry on the replacement, and a second strike
stops the turn with a non-retryable error instead of burning the retry budget on the corpse.
Everything else (healthy-server 5xx, a Compute error with no backend verdict, an unmanaged
route, a pending /stop) keeps the ordinary retry policy.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent import chat_completion_helpers as h
from agent.error_classifier import FailoverReason, ManagedWorkerPoisonedError, classify_api_error
from hermes_cli.local_runtime import bootstrap

MODEL = "qwen3-coder-30b"
ROUTER = "http://127.0.0.1:18434/v1"
METAL_VERDICT = ("E ggml_metal_graph_compute: backend is in error state from a previous command "
                 "buffer failure - recreate the backend to recover")


class _ComputeError(Exception):
    """llama-server's 500 as the OpenAI SDK raises it."""

    def __init__(self):
        super().__init__("Error code: 500 - {'error': {'code': 500, 'message': 'Compute error.', 'type': 'server_error'}}")
        self.status_code = 500


class _OpaqueServerError(Exception):
    def __init__(self):
        super().__init__("Error code: 500 - something went wrong")
        self.status_code = 500


class _FakeSupervisor:
    """The process-local supervisor: evidence scan + child replacement, recorded."""

    def __init__(self, *, verdict=True, fail_recycle=None, base_url=ROUTER):
        self.base_url = base_url
        self.recycled = []
        self.generation = 0
        self.evidence_calls = 0
        self._verdict = verdict
        self._fail_recycle = fail_recycle

    def backend_error_evidence(self):
        self.evidence_calls += 1
        return METAL_VERDICT if self._verdict else None

    def recycle_model(self, model):
        if self._fail_recycle is not None:
            raise self._fail_recycle
        self.recycled.append(model)
        self.generation += 1
        return self.generation

    def worker_generation(self, model):
        return self.generation


@pytest.fixture
def supervisor(monkeypatch):
    def install(**kwargs):
        sup = _FakeSupervisor(**kwargs)
        monkeypatch.setattr(bootstrap, "_SUPERVISOR", sup)  # what get_supervisor() reads
        return sup
    return install


def _make_call(*, base_url=ROUTER, interrupted=False):
    call = h._StreamingCall.__new__(h._StreamingCall)
    call.warnings, call.closed, call.buffered = [], [], []
    call.agent = SimpleNamespace(
        provider="custom", model=MODEL, base_url=base_url, api_mode="chat_completions",
        _interrupt_requested=interrupted, _stream_options_unsupported=False,
        _is_provider_stream_parse_error=lambda e: False,
        _emit_warning=lambda text: call.warnings.append(text),
        _buffer_status=lambda text: call.buffered.append(text),
        _disable_streaming=False, _stream_5xx_probe_ts=None,
    )
    call.api_kwargs = {"model": MODEL, "messages": [], "stream": True}
    call.result = {"response": None, "error": None, "partial_tool_names": []}
    call.deltas_were_sent = {"yes": False}
    call.first_delta_fired = {"done": False}
    call.provider_tool_in_flight = {"yes": False}
    call._request_cancelled = {"value": False}
    call.stream_attempt_lock = threading.Lock()
    call.stream_attempt_state = {"current": 1, "cancelled": set(), "discarded_chunks": 0, "discarded_bytes": 0}
    call.clients = SimpleNamespace(close_once=lambda reason: call.closed.append(reason), diag=None)
    call.last_chunk_time = {"t": 0.0}
    call._stream_stale_timeout = 180.0
    call._compat_retries = 0
    call._worker_recycled = None
    return call


def _strike(call, e=None):
    return call._handle_stream_error(e or _ComputeError(), attempt=0, max_retries=2)


# ── _handle_stream_error contract ──────────────────────────────────────────

def test_first_strike_replaces_the_worker_and_grants_one_retry(supervisor):
    sup = supervisor()
    call = _make_call()

    assert _strike(call) is True  # retry — on the replacement child
    assert sup.recycled == [MODEL]
    assert call._worker_recycled == 1
    # Like the stream_options compat retry: not a network retry, must not eat the transient budget.
    assert call._compat_retries == 1
    assert call.result["error"] is None
    assert call.closed == ["poisoned_worker_recycle"]  # the dead attempt's client is torn down
    assert call._stream_stale_timeout == 180.0  # suspended for the model load, then restored
    assert call.last_chunk_time["t"] > 0.0  # a model load is not the dead attempt's silence
    assert call.warnings and MODEL in call.warnings[0]


def test_second_strike_on_the_replacement_fails_visibly_and_non_retryably(supervisor):
    sup = supervisor()
    call = _make_call()
    assert _strike(call) is True

    assert _strike(call) is False  # stop: the budget is one replacement, one retry
    err = call.result["error"]
    assert isinstance(err, ManagedWorkerPoisonedError)
    assert err.model == MODEL and err.generation == 1
    assert sup.recycled == [MODEL]  # no second replacement
    assert call._compat_retries == 1
    classified = classify_api_error(err, provider="custom", model=MODEL, base_url=ROUTER)
    assert classified.reason is FailoverReason.local_backend_poisoned
    assert classified.retryable is False


def test_failed_replacement_is_terminal_not_retried(supervisor):
    sup = supervisor(fail_recycle=TimeoutError("llama-server did not load the model within 600s"))
    call = _make_call()

    assert _strike(call) is False
    err = call.result["error"]
    assert isinstance(err, ManagedWorkerPoisonedError)
    assert "could not be started" in str(err) and "600s" in str(err)
    assert sup.recycled == []


@pytest.mark.parametrize("case", ["no_backend_verdict", "unmanaged_route", "pending_interrupt"])
def test_other_compute_errors_keep_the_ordinary_policy(supervisor, monkeypatch, case):
    """A Compute error is only a poisoned worker when the router log says so, the route is
    the managed one, and no /stop is pending; otherwise nothing is recycled and the error
    takes the pre-existing path (here: the 5xx unmask probe, then propagation)."""
    sup = supervisor(verdict=case != "no_backend_verdict")
    call = _make_call(base_url="https://api.example.com/v1" if case == "unmanaged_route" else ROUTER,
                      interrupted=case == "pending_interrupt")
    probe_error = _OpaqueServerError()
    probes = []

    def fake_probe(agent, kwargs):
        probes.append(kwargs)
        raise probe_error

    monkeypatch.setattr(h, "interruptible_api_call", fake_probe)
    e = _ComputeError()

    assert _strike(call, e) is False
    assert sup.recycled == []
    assert call._worker_recycled is None and call._compat_retries == 0
    assert call.result["error"] is e  # propagated to the turn loop's retry/fallback policy
    if case == "unmanaged_route":
        assert sup.evidence_calls == 0  # another server's 500 never touches the supervisor
    if case == "pending_interrupt":
        assert probes == []  # the loop's pre-retry interrupt check owns a pending /stop


def test_healthy_server_5xx_never_consults_the_supervisor(supervisor, monkeypatch):
    sup = supervisor()
    call = _make_call()
    monkeypatch.setattr(h, "interruptible_api_call", lambda agent, kwargs: (_ for _ in ()).throw(_OpaqueServerError()))

    assert _strike(call, _OpaqueServerError()) is False
    assert sup.evidence_calls == 0 and sup.recycled == []


# ── end to end through the stream retry loop ──────────────────────────────

def _make_agent():
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="managed-key", base_url=ROUTER, model=MODEL, provider="custom", quiet_mode=True,
        skip_context_files=True, skip_memory=True, enabled_toolsets=[], max_iterations=1,
    )
    agent.api_mode = "chat_completions"
    return agent


class _OkStream:
    """A two-chunk SSE stream ("ok", then stop). A plain iterator, not a MagicMock: the relay
    treats anything with a ``choices`` attribute as an already-completed response."""

    def __init__(self):
        self._chunks = iter([
            SimpleNamespace(choices=[SimpleNamespace(index=0, delta=SimpleNamespace(
                content="ok", tool_calls=None, reasoning_content=None, reasoning=None), finish_reason=None)],
                model=MODEL, usage=None),
            SimpleNamespace(choices=[SimpleNamespace(index=0, delta=SimpleNamespace(
                content=None, tool_calls=None, reasoning_content=None, reasoning=None), finish_reason="stop")],
                model=MODEL, usage=None),
        ])
        self.response = SimpleNamespace(headers={})

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)

    def close(self):
        pass


def _ok_stream():
    return _OkStream()


def _run(agent, on_create):
    client = MagicMock()
    client.chat.completions.create.side_effect = on_create
    with patch("run_agent.AIAgent._create_request_openai_client", return_value=client), \
            patch("run_agent.AIAgent._close_request_openai_client"):
        return agent._interruptible_streaming_api_call(
            {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]})


@pytest.mark.parametrize("stream_retries", ["2", "0"])
def test_retry_after_compute_error_lands_on_the_replacement_worker(supervisor, monkeypatch, stream_retries):
    """The acceptance test from the issue: the request after the Compute error reaches a NEW
    backend generation, not the child that logged the error state. Also with the transient
    stream budget at zero: the replacement retry is granted on top of it."""
    monkeypatch.setenv("HERMES_STREAM_RETRIES", stream_retries)
    sup = supervisor()
    agent = _make_agent()
    served_by = []

    def poisoned_until_replaced(**kwargs):
        served_by.append(sup.generation)
        if sup.generation == 0:
            raise _ComputeError()
        return _ok_stream()

    response = _run(agent, poisoned_until_replaced)

    assert response is not None and response.choices[0].message.content == "ok"
    assert served_by == [0, 1]
    assert sup.recycled == [MODEL]


def test_worker_poisoned_again_after_replacement_stops_after_exactly_one_retry(supervisor):
    sup = supervisor()
    agent = _make_agent()
    served_by = []

    def always_poisoned(**kwargs):
        served_by.append(sup.generation)
        raise _ComputeError()

    with pytest.raises(ManagedWorkerPoisonedError) as info:
        _run(agent, always_poisoned)

    assert served_by == [0, 1]  # the original, one retry on the replacement — then stop
    assert sup.recycled == [MODEL]
    assert info.value.generation == 1
    assert classify_api_error(info.value, provider="custom", model=MODEL, base_url=ROUTER).retryable is False
