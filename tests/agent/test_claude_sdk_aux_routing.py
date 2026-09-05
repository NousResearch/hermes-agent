"""Subscription-safe auxiliary routing for the Claude Agent SDK."""

from __future__ import annotations

import asyncio
import builtins
import concurrent.futures
import sys
import threading
from types import ModuleType

import pytest

from agent import auxiliary_client as M
from agent import claude_sdk_aux_client as AUX
from agent.claude_sdk_aux_client import ClaudeSdkAuxClient, ClaudeSdkAuxError
from agent.transports import claude_agent_sdk_session as SESSION


def _plant_sdk(monkeypatch, messages):
    """Install the optional SDK's minimal typed surface for one test."""
    monkeypatch.setattr("tools.lazy_deps.ensure", lambda *_args, **_kwargs: None)
    module = ModuleType("claude_agent_sdk")

    class TextBlock:
        def __init__(self, text):
            self.text = text

    class AssistantMessage:
        def __init__(self, content):
            self.content = content

    class ResultMessage:
        def __init__(
            self,
            *,
            subtype="success",
            is_error=False,
            result="",
            errors=None,
            usage=None,
            stop_reason=None,
        ):
            self.subtype = subtype
            self.is_error = is_error
            self.result = result
            self.errors = errors or []
            self.usage = usage
            self.stop_reason = stop_reason

    captured = {}

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    async def query(*, prompt, options):
        captured["prompt"] = prompt
        captured["options"] = options
        for message in messages:
            yield message

    module.TextBlock = TextBlock
    module.AssistantMessage = AssistantMessage
    module.ResultMessage = ResultMessage
    module.ClaudeAgentOptions = ClaudeAgentOptions
    module.query = query
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", module)
    return module, captured


class _SequenceAsyncIterator:
    """Minimal declared-shape AsyncIterator for SDK query contract tests."""

    def __init__(self, messages):
        self._messages = iter(messages)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._messages)
        except StopIteration:
            raise StopAsyncIteration from None


class _FailingCloseAsyncIterator(_SequenceAsyncIterator):
    def __init__(self, messages, close_error):
        super().__init__(messages)
        self._close_error = close_error

    async def aclose(self):
        raise self._close_error


def test_auto_sdk_runtime_uses_one_shot_subscription_aux(monkeypatch):
    monkeypatch.setattr(
        M,
        "_normalize_main_runtime",
        lambda runtime: {
            "api_mode": "claude_agent_sdk",
            "model": "claude-sonnet-5",
            "provider": "claude-agent-sdk",
        },
    )

    client, model, provider = M._resolve_auto_route(main_runtime={})

    assert isinstance(client, ClaudeSdkAuxClient)
    assert model == "claude-sonnet-5"
    assert provider == "claude-agent-sdk"


def test_explicit_sdk_aux_provider_returns_sdk_client():
    client, model = M.resolve_provider_client(
        "claude-agent-sdk", model="claude-sonnet-5"
    )

    assert isinstance(client, ClaudeSdkAuxClient)
    assert model == "claude-sonnet-5"


def test_explicit_sdk_async_aux_provider_uses_subscription_facade(monkeypatch):
    """Async SDK aux calls must not fall through to AsyncOpenAI."""
    client, model = M.resolve_provider_client(
        "claude-agent-sdk", model="claude-sonnet-5", async_mode=True
    )

    assert type(client).__name__ == "AsyncClaudeSdkAuxClient"
    assert model == "claude-sonnet-5"
    assert client.base_url == ""

    async def _fake_collect(
        prompt, *, model, cancel_check=None, progress_hook=None
    ):
        assert prompt
        assert model == "claude-sonnet-5"
        assert cancel_check is None
        assert progress_hook is None
        return "summary", {"input_tokens": 2}, "stop"

    monkeypatch.setattr(AUX, "_collect_text", _fake_collect)
    result = asyncio.run(
        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "summarize"}],
        )
    )

    assert result.choices[0].message.content == "summary"


def test_auto_sdk_async_aux_provider_retains_subscription_identity():
    client, model = M.resolve_provider_client(
        "auto",
        async_mode=True,
        main_runtime={
            "api_mode": "claude_agent_sdk",
            "model": "claude-sonnet-5",
            "provider": "claude-agent-sdk",
        },
    )

    assert type(client).__name__ == "AsyncClaudeSdkAuxClient"
    assert model == "claude-sonnet-5"
    assert getattr(client, "_hermes_aux_effective_provider", None) == "claude-agent-sdk"


def _payment_error():
    error = RuntimeError("Payment Required: subscription request failed")
    error.status_code = 402
    return error


def _fail_if_called(*_args, **_kwargs):
    raise AssertionError("metered fallback helper must not run for claude-agent-sdk")


def test_auto_sdk_sync_failure_is_fail_closed_before_every_fallback(monkeypatch):
    """An auto-selected subscription client must never open a metered route."""
    client = type("SdkClient", (), {})()
    client.base_url = ""
    client._hermes_aux_effective_provider = "claude-agent-sdk"
    client.chat = type("Chat", (), {})()
    client.chat.completions = type("Completions", (), {})()
    client.chat.completions.create = lambda **_kwargs: (_ for _ in ()).throw(_payment_error())

    monkeypatch.setattr(
        M,
        "_resolve_task_provider_model",
        lambda *_args, **_kwargs: ("auto", None, None, None, None),
    )
    monkeypatch.setattr(M, "_get_cached_client", lambda *_args, **_kwargs: (client, "claude-sonnet-5"))
    for helper in (
        "_try_configured_fallback_chain",
        "_try_main_fallback_chain",
        "_try_payment_fallback",
        "_try_main_agent_model_fallback",
    ):
        monkeypatch.setattr(M, helper, _fail_if_called)

    with pytest.raises(RuntimeError, match="Payment Required"):
        M.call_llm(task="compression", messages=[{"role": "user", "content": "summarize"}])


def test_auto_sdk_async_failure_is_fail_closed_before_every_fallback(monkeypatch):
    """Async auto SDK failures retain the subscription identity too."""
    class Completions:
        async def create(self, **_kwargs):
            raise _payment_error()

    client = type("AsyncSdkClient", (), {})()
    client.base_url = ""
    client._hermes_aux_effective_provider = "claude-agent-sdk"
    client.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(
        M,
        "_resolve_task_provider_model",
        lambda *_args, **_kwargs: ("auto", None, None, None, None),
    )
    monkeypatch.setattr(M, "_get_cached_client", lambda *_args, **_kwargs: (client, "claude-sonnet-5"))
    for helper in (
        "_try_configured_fallback_chain",
        "_try_main_fallback_chain",
        "_try_payment_fallback",
        "_try_main_agent_model_fallback",
    ):
        monkeypatch.setattr(M, helper, _fail_if_called)

    with pytest.raises(RuntimeError, match="Payment Required"):
        asyncio.run(M.async_call_llm(
            task="compression", messages=[{"role": "user", "content": "summarize"}]
        ))


def test_async_sdk_explicit_cancellation_bypasses_fallbacks(monkeypatch):
    class Completions:
        async def create(self, **_kwargs):
            raise M.AuxiliaryExplicitCancellation()

    client = type("AsyncSdkClient", (), {})()
    client.base_url = ""
    client._hermes_aux_effective_provider = "claude-agent-sdk"
    client.chat = type("Chat", (), {"completions": Completions()})()

    monkeypatch.setattr(
        M,
        "_resolve_task_provider_model",
        lambda *_args, **_kwargs: ("auto", None, None, None, None),
    )
    monkeypatch.setattr(
        M,
        "_get_cached_client",
        lambda *_args, **_kwargs: (client, "claude-sonnet-5"),
    )
    for helper in (
        "_try_configured_fallback_chain",
        "_try_main_fallback_chain",
        "_try_payment_fallback",
        "_try_main_agent_model_fallback",
    ):
        monkeypatch.setattr(M, helper, _fail_if_called)

    with pytest.raises(M.AuxiliaryExplicitCancellation):
        asyncio.run(
            M.async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )
        )


def test_prompt_formatter_preserves_roles_and_only_text_content():
    prompt = AUX._messages_to_prompt(
        [
            {"role": "system", "content": "Summarize precisely."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "secret-url"}},
                ],
            },
            {"role": "tool", "content": {"content": "tool output"}},
        ]
    )

    assert "System:\nSummarize precisely." in prompt
    assert "User:\nhello" in prompt
    assert "Tool result:\ntool output" in prompt
    assert "secret-url" not in prompt


def test_sdk_aux_query_reports_progress_for_each_consumed_message(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("part one")]),
        module.AssistantMessage([module.TextBlock("part two")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    pulses = []
    monkeypatch.setattr(M, "_notify_aux_progress", lambda: pulses.append("progress"))

    text, usage, _ = asyncio.run(
        AUX._collect_text("prompt", model="claude-sonnet-5")
    )

    assert text == "part onepart two"
    assert usage == {"input_tokens": 2}
    assert len(pulses) == len(messages)
    assert captured["include_partial_messages"] is True


def test_sdk_aux_query_honors_cancellation_before_stream_start(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [module.ResultMessage(usage={"input_tokens": 2})]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    checks = []
    pulses = []
    monkeypatch.setattr(
        M,
        "_aux_interrupt_cancel_requested",
        lambda: checks.append("checked") or True,
    )
    monkeypatch.setattr(M, "_notify_aux_progress", lambda: pulses.append("progress"))

    with pytest.raises(M.AuxiliaryExplicitCancellation) as raised:
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))

    assert raised.value.cause == "explicit_host_cancel"
    assert checks == ["checked"]
    assert "prompt" not in captured
    assert pulses == []


def test_sdk_aux_query_honors_cancellation_after_message_arrives(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("message content")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    stream_state = []

    async def _messages_after_start(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        captured["options"] = kwargs["options"]
        for message in messages:
            stream_state.append("message arrived")
            yield message

    monkeypatch.setattr(module, "query", _messages_after_start)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    pulses = []
    monkeypatch.setattr(
        M,
        "_aux_interrupt_cancel_requested",
        lambda: bool(stream_state),
    )
    monkeypatch.setattr(M, "_notify_aux_progress", lambda: pulses.append("progress"))

    with pytest.raises(M.AuxiliaryExplicitCancellation) as raised:
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))

    assert raised.value.cause == "explicit_host_cancel"
    assert captured["prompt"] == "prompt"
    assert stream_state == ["message arrived"]
    # Progress is reported after the cancellation checkpoint, so an empty
    # pulse list proves the arrived message was not processed.
    assert pulses == []


def test_sync_sdk_aux_facade_propagates_cancel_inside_running_loop(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("must not complete")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    cancel_event = threading.Event()
    stream_started = threading.Event()

    async def _messages_after_cancel(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        captured["options"] = kwargs["options"]
        stream_started.set()
        while not cancel_event.is_set():
            await asyncio.sleep(0.01)
        for message in messages:
            yield message

    monkeypatch.setattr(module, "query", _messages_after_cancel)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    client = ClaudeSdkAuxClient()
    stream_started_in_time = []

    def _cancel_after_stream_start():
        stream_started_in_time.append(stream_started.wait(timeout=5))
        cancel_event.set()

    cancel_worker = threading.Thread(target=_cancel_after_stream_start)
    cancel_worker.start()

    async def _call_sync_facade():
        with M.aux_interrupt_protection(cancel_event=cancel_event):
            return client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "summarize"}],
            )

    try:
        with pytest.raises(M.AuxiliaryExplicitCancellation):
            asyncio.run(_call_sync_facade())
    finally:
        cancel_worker.join(timeout=5)

    assert not cancel_worker.is_alive()
    assert stream_started_in_time == [True]
    assert captured["prompt"]


def test_async_sdk_aux_facade_propagates_cancel_across_worker(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("must not complete")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    cancel_event = threading.Event()
    cancel_event.set()
    client = AUX.AsyncClaudeSdkAuxClient(ClaudeSdkAuxClient())

    async def _call_async_facade():
        with M.aux_interrupt_protection(cancel_check=cancel_event.is_set):
            return await client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "summarize"}],
            )

    with pytest.raises(M.AuxiliaryExplicitCancellation):
        asyncio.run(_call_async_facade())

    assert "prompt" not in captured


def test_async_sdk_aux_facade_propagates_cancel_event_across_worker(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [module.ResultMessage(usage={"input_tokens": 2})]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    cancel_event = threading.Event()
    cancel_event.set()
    client = AUX.AsyncClaudeSdkAuxClient(ClaudeSdkAuxClient())

    async def _call_async_facade():
        with M.aux_interrupt_protection(cancel_event=cancel_event):
            return await client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "summarize"}],
            )

    with pytest.raises(M.AuxiliaryExplicitCancellation):
        asyncio.run(_call_async_facade())

    assert "prompt" not in captured


def test_async_sdk_aux_cancel_event_precedes_cancel_check(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("summary")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    cancel_event = threading.Event()
    check_calls = 0

    def _cancel_check():
        nonlocal check_calls
        check_calls += 1
        return True

    client = AUX.AsyncClaudeSdkAuxClient(ClaudeSdkAuxClient())

    async def _call_async_facade():
        with M.aux_interrupt_protection(
            cancel_check=_cancel_check,
            cancel_event=cancel_event,
        ):
            return await client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "summarize"}],
            )

    result = asyncio.run(_call_async_facade())

    assert result.choices[0].message.content == "summary"
    assert check_calls == 0


def test_async_sdk_aux_facade_inactive_with_source_still_cancels(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [module.ResultMessage(usage={"input_tokens": 2})]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    client = AUX.AsyncClaudeSdkAuxClient(ClaudeSdkAuxClient())

    async def _call_async_facade():
        with M.aux_interrupt_protection(
            active=False,
            cancel_check=lambda: True,
        ):
            return await client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "summarize"}],
            )

    with pytest.raises(M.AuxiliaryExplicitCancellation):
        asyncio.run(_call_async_facade())

    assert "prompt" not in captured


def test_async_sdk_aux_facade_active_without_source_does_not_cancel(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("summary")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    observed = []

    async def _query_with_state(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        captured["options"] = kwargs["options"]
        observed.append(
            (
                M._aux_interrupt_protected(),
                M._capture_aux_cancel_check(),
            )
        )
        for message in messages:
            yield message

    monkeypatch.setattr(module, "query", _query_with_state)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    client = AUX.AsyncClaudeSdkAuxClient(ClaudeSdkAuxClient())

    async def _call_async_facade():
        with M.aux_interrupt_protection(active=True):
            return await client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "summarize"}],
            )

    result = asyncio.run(_call_async_facade())

    assert result.choices[0].message.content == "summary"
    assert observed == [(True, None)]


def test_async_sdk_aux_facade_preserves_cancel_identity_and_fails_open(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("summary")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    class RaisingCancelCheck:
        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            raise RuntimeError("predicate failure must not escape")

    cancel_check = RaisingCancelCheck()
    evaluated_sources = []
    original_evaluator = M._captured_aux_cancel_requested

    def _record_source(source):
        evaluated_sources.append(source)
        return original_evaluator(source)

    monkeypatch.setattr(M, "_captured_aux_cancel_requested", _record_source)
    client = AUX.AsyncClaudeSdkAuxClient(ClaudeSdkAuxClient())

    async def _call_async_facade():
        with M.aux_interrupt_protection(cancel_check=cancel_check):
            return await client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "summarize"}],
            )

    result = asyncio.run(_call_async_facade())

    assert result.choices[0].message.content == "summary"
    assert cancel_check.calls >= 1
    assert evaluated_sources
    assert all(source is cancel_check for source in evaluated_sources)


@pytest.mark.parametrize("async_facade", [False, True])
def test_sdk_aux_facade_propagates_progress_across_worker(
    monkeypatch, async_facade
):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("summary")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    sync_client = ClaudeSdkAuxClient()
    pulses = []

    async def _call_facade():
        with M.aux_progress_hook(lambda: pulses.append("pulse")):
            if async_facade:
                async_client = AUX.AsyncClaudeSdkAuxClient(sync_client)
                return await async_client.chat.completions.create(
                    model="claude-sonnet-5",
                    messages=[{"role": "user", "content": "summarize"}],
                )
            return sync_client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "summarize"}],
            )

    result = asyncio.run(_call_facade())

    assert result.choices[0].message.content == "summary"
    assert pulses == ["pulse", "pulse"]


def test_sdk_aux_none_progress_hook_preserves_consuming_thread_hook(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("summary")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    pulses = []

    with M.aux_progress_hook(lambda: pulses.append("worker")):
        text, _, _ = asyncio.run(
            AUX._collect_text(
                "prompt",
                model="claude-sonnet-5",
                progress_hook=None,
            )
        )

    assert text == "summary"
    assert pulses == ["worker", "worker"]


def test_sdk_aux_progress_hook_failure_does_not_abort_stream(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("summary")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    calls = []

    def _raising_progress():
        calls.append("pulse")
        raise RuntimeError("progress hooks are advisory")

    client = ClaudeSdkAuxClient()
    with M.aux_progress_hook(_raising_progress):
        result = client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "summarize"}],
        )

    assert result.choices[0].message.content == "summary"
    assert calls == ["pulse", "pulse"]


@pytest.mark.parametrize("async_facade", [False, True])
def test_sdk_aux_composes_with_protected_provider_worker(monkeypatch, async_facade):
    captured = {}

    async def _fake_collect(
        prompt, *, model, cancel_check=None, progress_hook=None
    ):
        captured["prompt"] = prompt
        captured["model"] = model
        captured["cancel_check"] = cancel_check
        captured["progress_hook"] = progress_hook
        return "summary", {"input_tokens": 2}, "stop"

    monkeypatch.setattr(AUX, "_collect_text", _fake_collect)
    client = ClaudeSdkAuxClient()
    async_client = AUX.AsyncClaudeSdkAuxClient(client)

    def _never_cancel():
        return False

    def _progress():
        return None

    def _provider_call(kwargs):
        if async_facade:
            return asyncio.run(async_client.chat.completions.create(**kwargs))
        return client.chat.completions.create(**kwargs)

    kwargs = {
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": "summarize"}],
    }
    with (
        M.aux_progress_hook(_progress),
        M.aux_interrupt_protection(cancel_check=_never_cancel),
    ):
        result = M._run_protected_sync_provider_call(_provider_call, kwargs)

    assert result.choices[0].message.content == "summary"
    assert isinstance(captured["cancel_check"], M._AuxiliaryCancellationDecision)
    assert captured["cancel_check"]._source_cancel_check is _never_cancel
    assert captured["progress_hook"] is _progress


def test_sdk_aux_composed_hard_cancel_latches_without_timeout_cleanup(monkeypatch):
    source_event = threading.Event()
    worker_started = threading.Event()
    worker_unwound = threading.Event()
    captured = {}
    begin_timeout_calls = []
    original_begin_timeout = M._AuxiliaryCancellationDecision.begin_timeout_cleanup

    def _record_begin_timeout(self):
        begin_timeout_calls.append(self)
        return original_begin_timeout(self)

    async def _fake_collect(
        prompt, *, model, cancel_check=None, progress_hook=None
    ):
        captured["prompt"] = prompt
        captured["cancel_check"] = cancel_check
        assert callable(cancel_check)
        worker_started.set()
        try:
            while not cancel_check():
                await asyncio.sleep(0.005)
            raise M.AuxiliaryExplicitCancellation()
        finally:
            worker_unwound.set()

    monkeypatch.setattr(
        M._AuxiliaryCancellationDecision,
        "begin_timeout_cleanup",
        _record_begin_timeout,
    )
    monkeypatch.setattr(AUX, "_collect_text", _fake_collect)
    client = ClaudeSdkAuxClient()

    def _provider_call(kwargs):
        return client.chat.completions.create(**kwargs)

    def _cancel_after_worker_starts():
        assert worker_started.wait(timeout=5)
        source_event.set()

    cancel_worker = threading.Thread(target=_cancel_after_worker_starts)
    cancel_worker.start()
    try:
        with M.aux_interrupt_protection(cancel_event=source_event):
            with pytest.raises(M.AuxiliaryExplicitCancellation):
                M._run_protected_sync_provider_call(
                    _provider_call,
                    {
                        "model": "claude-sonnet-5",
                        "messages": [{"role": "user", "content": "summarize"}],
                    },
                )
    finally:
        cancel_worker.join(timeout=5)

    assert not cancel_worker.is_alive()
    assert worker_unwound.wait(timeout=5)
    decision = captured["cancel_check"]
    assert isinstance(decision, M._AuxiliaryCancellationDecision)
    assert decision._outcome == "cancelled"
    assert begin_timeout_calls == []


def test_async_sdk_aux_facade_honors_public_sync_create_wrapper(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("summary")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    sync_client = ClaudeSdkAuxClient()
    original_create = sync_client.chat.completions.create
    calls = []

    def _wrapped_create(**kwargs):
        calls.append(kwargs)
        return original_create(**kwargs)

    sync_client.chat.completions.create = _wrapped_create
    client = AUX.AsyncClaudeSdkAuxClient(sync_client)
    result = asyncio.run(
        client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "summarize"}],
        )
    )

    assert result.choices[0].message.content == "summary"
    assert len(calls) == 1


def test_async_sdk_aux_worker_context_is_restored_between_calls(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("summary")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    observed = []

    async def _query_with_worker_state(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        captured["options"] = kwargs["options"]
        observed.append(
            (
                threading.get_ident(),
                getattr(M._aux_progress, "hook", None),
                getattr(M._aux_interrupt_protection, "cancel_check", None),
                M._aux_interrupt_protected(),
            )
        )
        for message in messages:
            yield message

    monkeypatch.setattr(module, "query", _query_with_worker_state)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    client = AUX.AsyncClaudeSdkAuxClient(ClaudeSdkAuxClient())
    pulses = []

    def _progress():
        pulses.append("pulse")

    def _never_cancel():
        return False

    async def _call_twice():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        with (
            M.aux_progress_hook(_progress),
            M.aux_interrupt_protection(active=False, cancel_check=_never_cancel),
        ):
            first = await client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "first"}],
            )
        pulses_after_first = list(pulses)
        second = await client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "second"}],
        )
        return first, second, pulses_after_first

    first, second, pulses_after_first = asyncio.run(_call_twice())

    assert first.choices[0].message.content == "summary"
    assert second.choices[0].message.content == "summary"
    assert pulses_after_first == ["pulse", "pulse"]
    assert pulses == pulses_after_first
    assert len(observed) == 2
    assert observed[0][0] == observed[1][0]
    assert observed[0][1] is _progress
    assert observed[0][2] is _never_cancel
    assert observed[0][3] is False
    assert observed[1][1:] == (None, None, False)


@pytest.mark.parametrize(
    "raised_error",
    [
        ClaudeSdkAuxError("expected first-call failure"),
        M.AuxiliaryExplicitCancellation(),
    ],
)
def test_async_sdk_aux_worker_context_is_restored_after_create_raises(
    monkeypatch, raised_error
):
    sync_client = ClaudeSdkAuxClient()
    async_client = AUX.AsyncClaudeSdkAuxClient(sync_client)
    original_create = sync_client.chat.completions.create
    observed = []
    calls = 0

    def _recording_create(**kwargs):
        nonlocal calls
        calls += 1
        observed.append(
            (
                threading.get_ident(),
                getattr(M._aux_progress, "hook", None),
                getattr(M._aux_interrupt_protection, "cancel_check", None),
                M._aux_interrupt_protected(),
            )
        )
        if calls == 1:
            raise raised_error
        return original_create(**kwargs)

    sync_client.chat.completions.create = _recording_create
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("summary")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    def _progress():
        return None

    def _never_cancel():
        return False

    async def _call_after_failure():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1))
        with (
            M.aux_progress_hook(_progress),
            M.aux_interrupt_protection(active=False, cancel_check=_never_cancel),
        ):
            with pytest.raises(type(raised_error)):
                await async_client.chat.completions.create(
                    model="claude-sonnet-5",
                    messages=[{"role": "user", "content": "first"}],
                )
        return await async_client.chat.completions.create(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "second"}],
        )

    result = asyncio.run(_call_after_failure())

    assert result.choices[0].message.content == "summary"
    assert len(observed) == 2
    assert observed[0][0] == observed[1][0]
    assert observed[0][1:] == (_progress, _never_cancel, False)
    assert observed[1][1:] == (None, None, False)


def test_collect_text_closes_query_generator_before_cancellation_escapes(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    generator_closed = threading.Event()
    checks = 0

    async def _query_with_cleanup(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        captured["options"] = kwargs["options"]
        try:
            yield module.AssistantMessage([module.TextBlock("partial")])
            yield module.ResultMessage(usage={"input_tokens": 2})
        finally:
            generator_closed.set()

    def _cancel_after_stream_starts():
        nonlocal checks
        checks += 1
        return checks > 1

    monkeypatch.setattr(module, "query", _query_with_cleanup)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    async def _exercise():
        with pytest.raises(M.AuxiliaryExplicitCancellation):
            await AUX._collect_text(
                "prompt",
                model="claude-sonnet-5",
                cancel_check=_cancel_after_stream_starts,
            )
        assert generator_closed.is_set()

    asyncio.run(_exercise())


def test_query_close_failure_does_not_mask_explicit_cancellation(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])
    checks = 0

    async def _query_with_failing_cleanup(**_kwargs):
        try:
            yield module.AssistantMessage([module.TextBlock("partial")])
            yield module.ResultMessage(usage={"input_tokens": 2})
        finally:
            raise RuntimeError("transport teardown failed")

    def _cancel_after_stream_starts():
        nonlocal checks
        checks += 1
        return checks > 1

    monkeypatch.setattr(module, "query", _query_with_failing_cleanup)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(M.AuxiliaryExplicitCancellation):
        asyncio.run(
            AUX._collect_text(
                "prompt",
                model="claude-sonnet-5",
                cancel_check=_cancel_after_stream_starts,
            )
        )


def test_query_close_timeout_does_not_block_explicit_cancellation(
    monkeypatch, caplog
):
    module, _ = _plant_sdk(monkeypatch, [])
    checks = 0
    never_finishes = asyncio.Event()

    async def _query_with_blocked_cleanup(**_kwargs):
        try:
            yield module.AssistantMessage([module.TextBlock("partial")])
            yield module.ResultMessage(usage={"input_tokens": 2})
        finally:
            await never_finishes.wait()

    def _cancel_after_stream_starts():
        nonlocal checks
        checks += 1
        return checks > 1

    monkeypatch.setattr(module, "query", _query_with_blocked_cleanup)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    monkeypatch.setattr(AUX, "_QUERY_CLOSE_TIMEOUT", 0.01)

    async def _exercise():
        return await asyncio.wait_for(
            AUX._collect_text(
                "prompt",
                model="claude-sonnet-5",
                cancel_check=_cancel_after_stream_starts,
            ),
            timeout=0.2,
        )

    with pytest.raises(M.AuxiliaryExplicitCancellation):
        asyncio.run(_exercise())

    assert "query close timed out" in caplog.text


def test_query_close_timeout_after_success_returns_result(monkeypatch, caplog):
    module, _ = _plant_sdk(monkeypatch, [])
    never_finishes = asyncio.Event()

    class BlockingCloseIterator(_SequenceAsyncIterator):
        async def aclose(self):
            await never_finishes.wait()

    stream = BlockingCloseIterator(
        [
            module.AssistantMessage([module.TextBlock("summary")]),
            module.ResultMessage(usage={"input_tokens": 2}),
        ]
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    monkeypatch.setattr(AUX, "_QUERY_CLOSE_TIMEOUT", 0.01)

    text, usage, stop_reason = asyncio.run(
        AUX._collect_text("prompt", model="claude-sonnet-5")
    )

    assert (text, usage, stop_reason) == (
        "summary",
        {"input_tokens": 2},
        "stop",
    )
    assert "model=claude-sonnet-5, timeout=0.01s" in caplog.text
    assert "prompt" not in caplog.text
    assert "summary" not in caplog.text


def test_outer_deadline_during_close_does_not_mask_explicit_cancellation(
    monkeypatch,
):
    module, _ = _plant_sdk(monkeypatch, [])
    checks = 0
    never_finishes = asyncio.Event()

    async def _query_with_blocked_cleanup(**_kwargs):
        try:
            yield module.AssistantMessage([module.TextBlock("partial")])
            yield module.ResultMessage(usage={"input_tokens": 2})
        finally:
            await never_finishes.wait()

    def _cancel_after_stream_starts():
        nonlocal checks
        checks += 1
        return checks > 1

    monkeypatch.setattr(module, "query", _query_with_blocked_cleanup)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    monkeypatch.setattr(AUX, "_QUERY_CLOSE_TIMEOUT", 1.0)

    async def _exercise():
        return await asyncio.wait_for(
            AUX._collect_text(
                "prompt",
                model="claude-sonnet-5",
                cancel_check=_cancel_after_stream_starts,
            ),
            timeout=0.02,
        )

    with pytest.raises(M.AuxiliaryExplicitCancellation):
        asyncio.run(_exercise())


def test_query_close_failure_does_not_mask_terminal_error(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])

    stream = _FailingCloseAsyncIterator(
        [module.ResultMessage(is_error=True, result="sdk failed safely")],
        RuntimeError("transport teardown failed"),
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(ClaudeSdkAuxError, match="sdk failed safely"):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_query_plain_async_iterator_without_aclose_is_supported(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])

    stream = _SequenceAsyncIterator(
        [
            module.AssistantMessage([module.TextBlock("summary")]),
            module.ResultMessage(usage={"input_tokens": 2}),
        ]
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    text, usage, stop_reason = asyncio.run(
        AUX._collect_text("prompt", model="claude-sonnet-5")
    )

    assert text == "summary"
    assert usage == {"input_tokens": 2}
    assert stop_reason == "stop"


def test_query_plain_async_iterator_cancellation_needs_no_close(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])
    checks = 0

    stream = _SequenceAsyncIterator(
        [module.AssistantMessage([module.TextBlock("partial")])]
    )

    def _cancel_after_stream_starts():
        nonlocal checks
        checks += 1
        return checks > 1

    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(M.AuxiliaryExplicitCancellation):
        asyncio.run(
            AUX._collect_text(
                "prompt",
                model="claude-sonnet-5",
                cancel_check=_cancel_after_stream_starts,
            )
        )


def test_query_slow_close_finishes_under_default_bound(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])
    checks = 0
    generator_closed = asyncio.Event()

    async def _query_with_slow_cleanup(**_kwargs):
        try:
            yield module.AssistantMessage([module.TextBlock("partial")])
            yield module.ResultMessage(usage={"input_tokens": 2})
        finally:
            await asyncio.sleep(0.01)
            generator_closed.set()

    def _cancel_after_stream_starts():
        nonlocal checks
        checks += 1
        return checks > 1

    assert AUX._QUERY_CLOSE_TIMEOUT == 5.0
    monkeypatch.setattr(module, "query", _query_with_slow_cleanup)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    async def _exercise():
        with pytest.raises(M.AuxiliaryExplicitCancellation):
            await AUX._collect_text(
                "prompt",
                model="claude-sonnet-5",
                cancel_check=_cancel_after_stream_starts,
            )
        assert generator_closed.is_set()

    asyncio.run(_exercise())


def test_query_close_failure_after_success_returns_result(monkeypatch, caplog):
    module, _ = _plant_sdk(monkeypatch, [])
    stream = _FailingCloseAsyncIterator(
        [
            module.AssistantMessage([module.TextBlock("summary")]),
            module.ResultMessage(usage={"input_tokens": 2}),
        ],
        RuntimeError("transport teardown failed"),
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    text, usage, stop_reason = asyncio.run(
        AUX._collect_text("prompt", model="claude-sonnet-5")
    )

    assert (text, usage, stop_reason) == (
        "summary",
        {"input_tokens": 2},
        "stop",
    )
    assert "query close failed" in caplog.text
    assert "model=claude-sonnet-5" in caplog.text
    assert "error=RuntimeError" in caplog.text
    assert "prompt" not in caplog.text
    assert "summary" not in caplog.text


def test_query_close_cancelled_error_outranks_terminal_error(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])
    stream = _FailingCloseAsyncIterator(
        [module.ResultMessage(is_error=True, result="sdk failed safely")],
        asyncio.CancelledError(),
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_query_sdk_close_timeout_error_is_not_our_bound(monkeypatch, caplog):
    module, _ = _plant_sdk(monkeypatch, [])
    stream = _FailingCloseAsyncIterator(
        [
            module.AssistantMessage([module.TextBlock("summary")]),
            module.ResultMessage(usage={"input_tokens": 2}),
        ],
        TimeoutError("transport timeout"),
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    text, usage, stop_reason = asyncio.run(
        AUX._collect_text("prompt", model="claude-sonnet-5")
    )

    assert (text, usage, stop_reason) == (
        "summary",
        {"input_tokens": 2},
        "stop",
    )
    assert "query close timed out" not in caplog.text


@pytest.mark.parametrize("host_stop", [KeyboardInterrupt(), SystemExit()])
def test_query_close_host_interrupt_outranks_terminal_error(
    monkeypatch, host_stop
):
    module, _ = _plant_sdk(monkeypatch, [])
    stream = _FailingCloseAsyncIterator(
        [module.ResultMessage(is_error=True, result="sdk failed safely")],
        host_stop,
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(type(host_stop)):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_query_close_explicit_cancellation_outranks_terminal_error(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])
    stream = _FailingCloseAsyncIterator(
        [module.ResultMessage(is_error=True, result="sdk failed safely")],
        M.AuxiliaryExplicitCancellation(),
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(M.AuxiliaryExplicitCancellation):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_query_close_cancel_precedence_is_independent_of_base_class(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])

    class FutureAuxiliaryExplicitCancellation(Exception):
        pass

    monkeypatch.setattr(
        M,
        "AuxiliaryExplicitCancellation",
        FutureAuxiliaryExplicitCancellation,
    )
    stream = _FailingCloseAsyncIterator(
        [module.ResultMessage(is_error=True, result="sdk failed safely")],
        FutureAuxiliaryExplicitCancellation(),
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(FutureAuxiliaryExplicitCancellation):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_query_close_base_exception_does_not_mask_terminal_error(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])

    class CleanupFailure(BaseException):
        pass

    stream = _FailingCloseAsyncIterator(
        [module.ResultMessage(is_error=True, result="sdk failed safely")],
        CleanupFailure("cleanup base exception"),
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(ClaudeSdkAuxError, match="sdk failed safely"):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_query_close_base_exception_surfaces_after_success(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])

    class CleanupFailure(BaseException):
        pass

    stream = _FailingCloseAsyncIterator(
        [
            module.AssistantMessage([module.TextBlock("summary")]),
            module.ResultMessage(usage={"input_tokens": 2}),
        ],
        CleanupFailure("cleanup base exception"),
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(CleanupFailure, match="cleanup base exception"):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_query_close_cancelled_error_propagates_inside_ancestor_except(monkeypatch):
    module, _ = _plant_sdk(monkeypatch, [])
    stream = _FailingCloseAsyncIterator(
        [
            module.AssistantMessage([module.TextBlock("summary")]),
            module.ResultMessage(usage={"input_tokens": 2}),
        ],
        asyncio.CancelledError(),
    )
    monkeypatch.setattr(module, "query", lambda **_kwargs: stream)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    try:
        raise RuntimeError("ancestor exception")
    except RuntimeError:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_sync_sdk_aux_cancellation_closes_query_generator(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    cancel_event = threading.Event()
    stream_started = threading.Event()
    generator_closed = threading.Event()

    async def _query_until_cancelled(**kwargs):
        captured["prompt"] = kwargs["prompt"]
        captured["options"] = kwargs["options"]
        try:
            stream_started.set()
            yield module.AssistantMessage([module.TextBlock("partial")])
            while not cancel_event.is_set():
                await asyncio.sleep(0.01)
            yield module.ResultMessage(usage={"input_tokens": 2})
        finally:
            generator_closed.set()

    monkeypatch.setattr(module, "query", _query_until_cancelled)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})
    client = ClaudeSdkAuxClient()

    def _cancel_after_start():
        assert stream_started.wait(timeout=5)
        cancel_event.set()

    cancel_worker = threading.Thread(target=_cancel_after_start)
    cancel_worker.start()

    async def _call_sync_facade():
        with M.aux_interrupt_protection(cancel_event=cancel_event):
            return client.chat.completions.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": "summarize"}],
            )

    try:
        with pytest.raises(M.AuxiliaryExplicitCancellation):
            asyncio.run(_call_sync_facade())
    finally:
        cancel_worker.join(timeout=5)

    assert not cancel_worker.is_alive()
    assert generator_closed.wait(timeout=5)
def test_sdk_aux_cold_start_ensures_dependency_before_import(monkeypatch):
    """The auxiliary path must install the optional SDK before importing it."""
    monkeypatch.delitem(sys.modules, "claude_agent_sdk", raising=False)
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    real_import = builtins.__import__
    sdk_available = False
    ensure_calls = []
    captured = {}

    def guarded_import(name, *args, **kwargs):
        if name == "claude_agent_sdk" and not sdk_available:
            raise ModuleNotFoundError("No module named 'claude_agent_sdk'")
        return real_import(name, *args, **kwargs)

    def ensure(feature, *, prompt):
        nonlocal sdk_available, captured
        ensure_calls.append((feature, prompt))
        sdk_available = True
        module, captured = _plant_sdk(monkeypatch, [])
        messages = [
            module.AssistantMessage([module.TextBlock("answer")]),
            module.ResultMessage(usage={"input_tokens": 2}),
        ]
        monkeypatch.setattr(
            module,
            "query",
            lambda **call_kwargs: _async_messages(messages, captured, call_kwargs),
        )

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr("tools.lazy_deps.ensure", ensure)

    text, usage, _ = asyncio.run(
        AUX._collect_text("prompt", model="claude-sonnet-5")
    )

    assert ensure_calls == [("provider.claude_agent_sdk", False)]
    assert text == "answer"
    assert usage == {"input_tokens": 2}
    assert captured["prompt"] == "prompt"


def test_one_shot_query_has_no_tools_and_scrubs_child_env(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("answer")]),
        module.ResultMessage(usage={"input_tokens": 2}),
    ]
    # The async generator closes over this list, so populate it after the
    # stand-in classes are available.
    monkeypatch.setattr(
        sys.modules["claude_agent_sdk"],
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(
        SESSION,
        "_sdk_env_overrides",
        lambda: {"ANTHROPIC_API_KEY": ""},
    )

    text, usage, _ = asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))

    assert text == "answer"
    assert usage == {"input_tokens": 2}
    assert captured["tools"] == []
    assert captured["allowed_tools"] == []
    assert captured["mcp_servers"] == {}
    assert captured["env"] == {"ANTHROPIC_API_KEY": ""}


async def _async_messages(messages, captured, kwargs):
    captured["prompt"] = kwargs["prompt"]
    captured["options"] = kwargs["options"]
    for message in messages:
        yield message


@pytest.mark.parametrize(
    "result_kwargs",
    [
        {"is_error": True, "subtype": "error_during_execution", "result": "boom"},
        {"subtype": "error_max_turns", "result": "limit"},
    ],
)
def test_terminal_error_never_returns_partial_text(monkeypatch, result_kwargs):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [
        module.AssistantMessage([module.TextBlock("partial")]),
        module.ResultMessage(**result_kwargs),
    ]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(ClaudeSdkAuxError):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_terminal_error_is_redacted(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"
    messages = [module.ResultMessage(is_error=True, result=f"bad key {secret}")]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(ClaudeSdkAuxError) as raised:
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))

    assert secret not in str(raised.value)


def test_stream_ending_without_result_is_not_partial_success(monkeypatch):
    module, captured = _plant_sdk(monkeypatch, [])
    messages = [module.AssistantMessage([module.TextBlock("partial")])]
    monkeypatch.setattr(
        module,
        "query",
        lambda **kwargs: _async_messages(messages, captured, kwargs),
    )
    monkeypatch.setattr(SESSION, "_sdk_env_overrides", lambda: {})

    with pytest.raises(ClaudeSdkAuxError, match="without a terminal result"):
        asyncio.run(AUX._collect_text("prompt", model="claude-sonnet-5"))


def test_sdk_exception_is_redacted_at_openai_facade(monkeypatch):
    secret = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789"

    async def _boom(prompt, *, model):
        raise RuntimeError(f"transport exposed {secret}")

    monkeypatch.setattr(AUX, "_collect_text", _boom)
    client = ClaudeSdkAuxClient()

    with pytest.raises(ClaudeSdkAuxError) as raised:
        client.chat.completions.create(
            messages=[{"role": "user", "content": "summarize this"}]
        )

    assert secret not in str(raised.value)
