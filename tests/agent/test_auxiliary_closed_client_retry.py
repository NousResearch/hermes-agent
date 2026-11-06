"""A Codex timeout that retires its client must not retry that same client."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent import auxiliary_client as aux


@pytest.mark.parametrize("async_mode,initial_blip", [(False, False), (True, False), (False, True)])
def test_retired_codex_timeout_reaches_recovery_without_closed_client_retry(monkeypatch, async_mode, initial_blip):
    calls = []
    leaf = SimpleNamespace(api_key="test", base_url="https://example.invalid/v1", closed=False)

    def close():
        leaf.closed = True

    def create(**kwargs):
        calls.append(kwargs)
        if initial_blip and len(calls) == 1:
            raise ConnectionError("transient connection reset")
        if leaf.closed:
            raise RuntimeError("Cannot send a request, as the client has been closed.")
        return iter([SimpleNamespace(type="response.in_progress")])

    leaf.close = close
    leaf.responses = SimpleNamespace(create=create)
    client = aux.CodexAuxiliaryClient(leaf, "test-model")
    if async_mode:
        client = aux.AsyncCodexAuxiliaryClient(client)
    kwargs = {"messages": [{"role": "user", "content": "test"}], "timeout": 120.0}
    req = SimpleNamespace(client=client, kwargs=kwargs, request_provider="custom",
                          resolved_api_mode="codex_responses", base_info=leaf.base_url,
                          resolved_base_url=leaf.base_url)
    monkeypatch.setattr(aux, "_plan_aux_call", lambda *a, **kw: (req, {}, {}))
    monkeypatch.setattr(aux, "_TRANSIENT_RETRY_BACKOFF_BASE", 0)
    monkeypatch.setattr(aux, "_transient_retry_count", lambda: 2)
    monkeypatch.setattr(aux._CodexCompletionsAdapter, "_build_responses_kwargs",
                        lambda self, kw: ({"model": "test-model"}, "test-model", 120.0))
    # Deterministic owner-thread timeout on the first stream event, without a timer or sleep.
    monkeypatch.setattr(aux._CodexStreamGuard, "start", lambda self: None)
    monkeypatch.setattr(aux._CodexStreamGuard, "effective_deadline", lambda self: 0.0)
    monkeypatch.setattr(aux, "_evict_cached_client_instance", Mock())

    def consume(stream, *, model, on_event):
        for event in stream:
            on_event(event)

    monkeypatch.setattr("agent.codex_runtime._consume_codex_event_stream", consume)
    monkeypatch.setattr(aux, "_relay_sync_completion", lambda client, kw, **opts: client.chat.completions.create(**kw))

    async def relay_async(client, kw, **opts):
        return await client.chat.completions.create(**kw)

    monkeypatch.setattr(aux, "_relay_async_completion", relay_async)
    recovery = []

    def recover(exc, *args, **kw):
        recovery.append(exc)
        raise exc

    monkeypatch.setattr(aux, "_start_recovery_ladder", recover)
    with pytest.raises(TimeoutError, match="no-progress timeout"):
        if async_mode:
            asyncio.run(aux._async_call_llm_impl(task="vision", messages=kwargs["messages"]))
        else:
            aux._call_llm_impl(task="vision", messages=kwargs["messages"])
    assert leaf.closed
    assert len(calls) == 1 + int(initial_blip)
    assert len(recovery) == 1


@pytest.mark.parametrize("exc,expected", [
    (ConnectionError("connection reset"), True),
    (TimeoutError("no-progress timeout"), True),
    (TimeoutError("hard ceiling"), False),
    (RuntimeError("Cannot send a request, as the client has been closed."), False),
    (aux.AuxiliaryExplicitCancellation(), False),
])
def test_unrelated_failures_keep_existing_retry_policy(exc, expected):
    assert aux._should_retry_same_provider("vision", exc, "") is expected


def test_cancelled_attempt_does_not_retire_client_or_leak_to_next_guard(monkeypatch):
    leaf = SimpleNamespace(close=Mock())
    guard = aux._CodexStreamGuard(leaf, 120.0)
    guard._protected_cancel_check = lambda: True
    monkeypatch.setattr(aux, "_evict_cached_client_instance", Mock())
    guard._close_client_on_timeout()
    guard.finish()
    assert guard.timed_out.is_set()
    assert not guard.timeout_release_pending.is_set()
    leaf.close.assert_not_called()
    aux._evict_cached_client_instance.assert_not_called()
    next_guard = aux._CodexStreamGuard(leaf, 120.0)
    assert not next_guard.timed_out.is_set()
    assert not next_guard.timeout_release_pending.is_set()


def test_late_retirement_cleanup_only_targets_failed_leaf(monkeypatch):
    failed = SimpleNamespace(close=Mock())
    replacement = SimpleNamespace(close=Mock())
    old_wrapper = aux.CodexAuxiliaryClient(
        SimpleNamespace(api_key="test", base_url="https://example.invalid", close=failed.close), "test")
    failed = old_wrapper._real_client
    cache_key = ("isolated-profile", "custom")
    monkeypatch.setattr(aux, "_client_cache", {cache_key: (replacement, "test", None)})
    guard = aux._CodexStreamGuard(failed, 120.0)
    # A previously queued timeout callback completes after another caller installed a replacement.
    guard._close_client_on_timeout()
    guard.finish()
    assert aux._client_cache[cache_key][0] is replacement
    replacement.close.assert_not_called()
    failed.close.assert_called()


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("has_fallback", [False, True])
def test_real_recovery_ladder_preserves_timeout_or_configured_fallback(monkeypatch, async_mode, has_fallback):
    error = aux._CodexClientRetiredTimeout("no-progress timeout")
    cause = ConnectionError("stream interrupted")
    primary = SimpleNamespace(base_url="https://primary.invalid/v1", api_key="test")
    fallback = SimpleNamespace(base_url="https://fallback.invalid/v1", api_key="test")
    messages = [{"role": "user", "content": "test"}]
    kwargs = dict(model="primary-model", messages=messages, temperature=0.2,
                  max_tokens=37, timeout=120.0)
    req = SimpleNamespace(client=primary, kwargs=kwargs, request_provider="custom",
                          resolved_provider="custom", resolved_model="primary-model",
                          final_model="primary-model", resolved_api_mode="chat_completions",
                          resolved_api_key="test", base_info=primary.base_url,
                          resolved_base_url=primary.base_url)
    candidate = dict(task="vision", messages=messages, temperature=0.2, max_tokens=37,
                     tools=None, effective_timeout=120.0, effective_extra_body={}, reasoning_config=None)
    monkeypatch.setattr(aux, "_plan_aux_call", lambda *a, **kw: (req, dict(max_tokens=37, main_runtime={}), candidate))
    monkeypatch.setattr(aux, "_get_auxiliary_task_config", lambda task: {
        "fallback_chain": [{"provider": "custom", "model": "fallback-model",
                            "base_url": fallback.base_url, "timeout": 17.0}]
    })
    configured = Mock(return_value=(fallback, "fallback-model", "fallback_chain[0](custom)")
                      if has_fallback else (None, None, ""))
    monkeypatch.setattr(aux, "_try_configured_fallback_chain", configured)
    main_fallback = Mock(return_value=(None, None, ""))
    monkeypatch.setattr(aux, "_try_main_agent_model_fallback", main_fallback)
    monkeypatch.setattr(aux, "_recoverable_pool_provider", lambda *a, **kw: None)
    monkeypatch.setattr(aux, "_evict_cached_client_instance", Mock())
    sent = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="fallback"),
                                                        finish_reason="stop")], model="fallback-model", usage=None)

    def send(client, request, **opts):
        sent.append((client, request, opts))
        if client is primary:
            raise error from cause
        assert client is fallback
        return response

    async def async_send(client, request, **opts):
        return send(client, request, **opts)

    monkeypatch.setattr(aux, "_relay_sync_completion", send)
    monkeypatch.setattr(aux, "_relay_async_completion", async_send)
    route = {"provider": "custom", "model": "primary-model"}

    def run():
        if async_mode:
            return asyncio.run(aux._async_call_llm_impl(task="vision", messages=messages, route_info=route))
        return aux._call_llm_impl(task="vision", messages=messages, route_info=route)

    if has_fallback:
        assert run() is response
        assert sent[1][1]["messages"] == messages
        assert sent[1][1]["model"] == "fallback-model"
        assert sent[1][1]["temperature"] == 0.2
        assert sent[1][1]["max_completion_tokens"] == 37
        assert sent[1][1]["timeout"] == 17.0
        assert route["model"] == "fallback-model"
        main_fallback.assert_not_called()
    else:
        with pytest.raises(aux._CodexClientRetiredTimeout) as caught:
            run()
        assert caught.value is error
        assert caught.value.__cause__ is cause
        assert route["model"] == "primary-model"
    assert sum(client is primary for client, _, _ in sent) == 1
    assert len(sent) == 1 + int(has_fallback)
    assert configured.call_args.args == ("vision", "custom")
    assert configured.call_args.kwargs["failed_base_url"] == primary.base_url
    assert configured.call_args.kwargs["failed_model"] == "primary-model"
    assert configured.call_args.kwargs["reason"] == "request timed out"


def test_timeout_publication_gap_remains_plain_timeout(monkeypatch):
    """Characterize the remaining race, not a claim that retirement is always observed."""
    import threading

    published = threading.Event()
    release = threading.Event()
    guards = []
    workers = []
    leaf = SimpleNamespace(base_url="https://example.invalid", close=Mock())

    def start(guard):
        guards.append(guard)
        original_set = guard.timed_out.set

        def publish_then_pause():
            original_set()
            published.set()
            assert release.wait(5)

        guard.timed_out.set = publish_then_pause
        worker = threading.Thread(target=guard._close_client_on_timeout)
        workers.append(worker)
        worker.start()
        assert published.wait(5)

    def create(**kwargs):
        raise ConnectionError("stream interrupted before retirement publication")

    leaf.responses = SimpleNamespace(create=create)
    monkeypatch.setattr(aux._CodexStreamGuard, "start", start)
    monkeypatch.setattr(aux._CodexCompletionsAdapter, "_build_responses_kwargs",
                        lambda self, kw: ({"model": "test"}, "test", 120.0))
    monkeypatch.setattr(aux, "_evict_cached_client_instance", Mock())
    monkeypatch.setattr("agent.agent_runtime_helpers.force_close_tcp_sockets", lambda client: 0)
    try:
        with pytest.raises(TimeoutError) as caught:
            aux._CodexCompletionsAdapter(leaf, "test").create(messages=[])
        assert type(caught.value) is TimeoutError
        assert not guards[0].timeout_release_pending.is_set()
        leaf.close.assert_not_called()
    finally:
        release.set()
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
    assert guards[0].timeout_release_pending.is_set()
    # A later cleanup still targets only the old attempt's captured leaf.
    guards[0].finish()
    leaf.close.assert_called_once()
