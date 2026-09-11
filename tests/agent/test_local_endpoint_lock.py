from __future__ import annotations

import threading
from types import SimpleNamespace

from agent.local_endpoint_lock import local_endpoint_lock
from agent.turn_api_call import perform_api_call


def test_local_endpoint_lock_is_reentrant_for_nested_same_thread_calls():
    notices: list[str] = []

    with local_endpoint_lock("http://localhost:11434/v1", on_wait=notices.append):
        with local_endpoint_lock("http://127.0.0.1:11434", on_wait=notices.append):
            pass

    assert notices == []


def test_turn_calls_to_same_local_backend_queue_across_threads(monkeypatch):
    release_first = threading.Event()
    first_entered = threading.Event()
    second_entered = threading.Event()
    second_waiting = threading.Event()
    interrupted_waiting = threading.Event()
    interrupt_observed = threading.Event()
    results: list[str] = []
    errors: list[BaseException] = []

    def run_middleware(request, call, **_kwargs):
        return call(request)

    monkeypatch.setattr(
        "hermes_cli.middleware.run_llm_execution_middleware", run_middleware
    )

    def make_agent(base_url: str, entered: threading.Event, wait=None):
        def api_call(_kwargs):
            entered.set()
            release_first.wait(timeout=2)
            return base_url

        return SimpleNamespace(
            api_mode="chat_completions",
            base_url=base_url,
            provider="custom",
            model="local-model",
            session_id=base_url,
            platform="cli",
            _disable_streaming=True,
            _interrupt_requested=False,
            _model_request_active=None,
            _pending_redirect_lock=None,
            _pending_redirect=None,
            _has_pending_redirect=lambda: False,
            _has_stream_consumers=lambda: False,
            _interruptible_api_call=api_call,
            _emit_status=(wait or (lambda _message: None)),
        )

    first = make_agent("http://localhost:11434/v1", first_entered)
    second = make_agent(
        "http://127.0.0.1:11434/api", second_entered,
        lambda message: second_waiting.set(),
    )
    interrupted = make_agent(
        "http://LOCALHOST:11434/v1", threading.Event(),
        lambda message: interrupted_waiting.set(),
    )

    def invoke(agent):
        try:
            verdict = perform_api_call(
                agent,
                api_kwargs={"model": "local-model"},
                _original_api_kwargs={},
                _llm_middleware_trace=[],
                _moa_prepared_request=None,
                _retry=None,
                thinking_spinner=None,
                retry_count=0,
                api_call_count=0,
                api_request_id="request",
                effective_task_id=None,
                turn_id="turn",
                interrupted=False,
            )
            results.append(verdict.response)
        except BaseException as exc:
            errors.append(exc)
            interrupt_observed.set()

    owner = threading.Thread(target=invoke, args=(first,))
    waiter = threading.Thread(target=invoke, args=(second,))
    cancelled_waiter = threading.Thread(target=invoke, args=(interrupted,))
    owner.start()
    assert first_entered.wait(timeout=1)
    waiter.start()
    assert second_waiting.wait(timeout=1)
    assert not second_entered.is_set()
    cancelled_waiter.start()
    assert interrupted_waiting.wait(timeout=1)
    interrupted._interrupt_requested = True
    assert interrupt_observed.wait(timeout=1)

    release_first.set()
    owner.join(timeout=2)
    waiter.join(timeout=2)
    cancelled_waiter.join(timeout=2)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert not cancelled_waiter.is_alive()
    assert second_entered.is_set()
    assert sorted(results) == sorted([first.base_url, second.base_url])
    assert len(errors) == 1
    assert isinstance(errors[0], InterruptedError)
