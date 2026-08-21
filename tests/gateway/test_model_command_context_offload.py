"""``/model`` context-length resolution must not block the gateway event loop.

Behavioral regression tests for the offload of
``resolve_display_context_length`` (blocking provider probe ladder) out of the
async ``/model`` handlers, and for the offload of
``enrich_model_switch_warnings_for_gateway`` (which reaches the same sync
resolver via ``merge_preflight_compression_warning``).

These drive the real ``_handle_model_command`` with a mocked switch pipeline —
no source-reading assertions; reverting either offload makes the corresponding
test fail because the blocking work lands back on the loop thread.
"""

import asyncio
import threading

import pytest
from unittest.mock import AsyncMock, MagicMock

import gateway.slash_commands as slash_commands
from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


class _PickerAdapter:
    def __init__(self):
        self.callback = None

    async def send_model_picker(self, *, on_model_selected, **_kwargs):
        self.callback = on_model_selected
        return type("_PickerResult", (), {"success": True})()


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="u1",
        chat_id="c1",
        user_name="tester",
        chat_type="dm",
    )


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_make_source(),
    )


def _runner_with_store(
    tmp_path,
    monkeypatch,
    *,
    config=None,
    switch_result=None,
):
    """Minimal GatewayRunner harness driving the real /model handler."""
    import yaml as _yaml

    import gateway.run as gateway_run
    from gateway.run import GatewayRunner
    from hermes_cli.model_switch import ModelSwitchResult

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    if config is None:
        config = {"model": {"default": "old-model", "provider": "openrouter"}}
    (hermes_home / "config.yaml").write_text(_yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", hermes_home)
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    if switch_result is None:
        switch_result = ModelSwitchResult(
            success=True,
            new_model="gpt-5.5",
            target_provider="openrouter",
            provider_changed=False,
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
            api_mode="chat_completions",
            provider_label="OpenRouter",
        )
    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **kw: switch_result,
    )
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: hermes_home)
    # No expensive-model confirmation detour.
    monkeypatch.setattr(
        "hermes_cli.model_cost_guard.expensive_model_warning",
        lambda *a, **k: None,
    )

    runner = object.__new__(GatewayRunner)
    runner.adapters = {}
    runner._voice_mode = {}
    runner._session_model_overrides = {}
    runner._pending_one_turn_model_restores = {}
    runner._running_agents = {}
    _store = MagicMock()
    _store.set_model_override = AsyncMock()
    _store._store = None
    runner.session_store = None
    runner._async_session_store = _store
    return runner


@pytest.mark.asyncio
async def test_context_resolution_runs_off_the_loop_thread(tmp_path, monkeypatch):
    """The sync resolver must execute on a worker thread when the /model
    handler resolves the display context length for the switch reply."""
    from hermes_cli import model_switch

    seen = {}
    loop_thread = threading.current_thread()

    def _recording_resolver(model, provider, **kwargs):
        seen.setdefault("threads", []).append(threading.current_thread())
        return 128000

    monkeypatch.setattr(
        model_switch, "resolve_display_context_length", _recording_resolver
    )

    runner = _runner_with_store(tmp_path, monkeypatch)
    result = await runner._handle_model_command(_event("/model gpt-5.5"))

    assert result is not None and "gpt-5.5" in result
    assert seen.get("threads"), "handler never resolved the context length"
    assert all(th is not loop_thread for th in seen["threads"]), (
        "resolve_display_context_length ran on the event loop thread — "
        "the /model handler must offload it via "
        "resolve_display_context_length_async"
    )


@pytest.mark.asyncio
async def test_warning_enrichment_is_offloaded(tmp_path, monkeypatch):
    """enrich_model_switch_warnings_for_gateway reaches the same sync resolver
    via merge_preflight_compression_warning, so the handler must dispatch it
    through asyncio.to_thread rather than calling it inline on the loop."""
    from hermes_cli import context_switch_guard

    offloaded = []
    real_to_thread = asyncio.to_thread

    async def _spy_to_thread(func, /, *args, **kwargs):
        offloaded.append(func)
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(slash_commands.asyncio, "to_thread", _spy_to_thread)

    runner = _runner_with_store(tmp_path, monkeypatch)
    result = await runner._handle_model_command(_event("/model gpt-5.5"))

    assert result is not None and "gpt-5.5" in result
    assert context_switch_guard.enrich_model_switch_warnings_for_gateway in offloaded, (
        "enrich_model_switch_warnings_for_gateway must be dispatched via "
        "asyncio.to_thread (it was called inline on the event loop instead)"
    )


@pytest.mark.asyncio
async def test_typed_model_uses_named_provider_model_context(tmp_path, monkeypatch):
    from hermes_cli import model_switch
    from hermes_cli.model_switch import ModelSwitchResult

    model = "vllm/DeepSeek-V4-Flash-0731"
    provider = "bifrost"
    base_url = "http://da-aihost01:4000/v1"
    providers = {
        provider: {
            "base_url": base_url,
            "models": {model: {"context_length": 1_048_576}},
        }
    }
    switch_result = ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider=provider,
        provider_changed=True,
        api_key="sk-test",
        base_url=base_url,
        api_mode="chat_completions",
        provider_label="Bifrost",
    )

    def _resolver(selected_model, selected_provider, **kwargs):
        metadata = (
            kwargs.get("user_providers", {})
            .get(selected_provider, {})
            .get("models", {})
            .get(selected_model, {})
        )
        return metadata.get("context_length", 1_000_000)

    monkeypatch.setattr(model_switch, "resolve_display_context_length", _resolver)
    runner = _runner_with_store(
        tmp_path,
        monkeypatch,
        config={
            "model": {"default": "old-model", "provider": "openrouter"},
            "providers": providers,
        },
        switch_result=switch_result,
    )

    result = await runner._handle_model_command(
        _event(f"/model {model} --provider {provider} --session")
    )

    assert result is not None
    assert "Context: 1,048,576" in result
    assert "Context: 1,000,000" not in result


@pytest.mark.asyncio
async def test_picker_model_uses_named_provider_model_context(tmp_path, monkeypatch):
    from hermes_cli import model_switch
    from hermes_cli.model_switch import ModelSwitchResult

    model = "vllm/DeepSeek-V4-Flash-0731"
    provider = "bifrost"
    base_url = "http://da-aihost01:4000/v1"
    providers = {
        provider: {
            "base_url": base_url,
            "models": {model: {"context_length": 1_048_576}},
        }
    }
    switch_result = ModelSwitchResult(
        success=True,
        new_model=model,
        target_provider=provider,
        provider_changed=True,
        api_key="sk-test",
        base_url=base_url,
        api_mode="chat_completions",
        provider_label="Bifrost",
    )

    def _resolver(selected_model, selected_provider, **kwargs):
        metadata = (
            kwargs.get("user_providers", {})
            .get(selected_provider, {})
            .get("models", {})
            .get(selected_model, {})
        )
        return metadata.get("context_length", 1_000_000)

    monkeypatch.setattr(model_switch, "resolve_display_context_length", _resolver)
    monkeypatch.setattr(
        model_switch,
        "list_picker_providers",
        lambda **_kwargs: [
            {
                "slug": provider,
                "name": "Bifrost",
                "models": [model],
                "total_models": 1,
            }
        ],
    )
    runner = _runner_with_store(
        tmp_path,
        monkeypatch,
        config={
            "model": {"default": "old-model", "provider": "openrouter"},
            "providers": providers,
        },
        switch_result=switch_result,
    )
    adapter = _PickerAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}

    sent = await runner._handle_model_command(_event("/model --session"))
    assert sent is None
    assert adapter.callback is not None

    result = await adapter.callback("c1", model, provider)
    assert "Context: 1,048,576" in result
    assert "Context: 1,000,000" not in result


def test_gateway_context_authority_uses_named_provider_identity(monkeypatch):
    import gateway.run as gateway_run

    model = "vllm/DeepSeek-V4-Flash-0731"
    base_url = "http://da-aihost01:4000/v1"
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "model": {"default": model, "provider": "bifrost"},
            "providers": {
                "other": {
                    "base_url": base_url,
                    "models": {model: {"context_length": 1_000_000}},
                },
                "bifrost": {
                    "base_url": base_url,
                    "models": {model: {"context_length": 1_048_576}},
                },
            },
        },
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: model)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom",
            "requested_provider": "bifrost",
            "base_url": base_url,
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, config_context_length=None, **_kwargs: (
            config_context_length or 1_000_000
        ),
    )

    context = gateway_run._resolve_gateway_model_context()

    assert context.context_length == 1_048_576
    assert context.context_source == "config"


def test_gateway_context_endpoint_fallback_no_sibling_leak(monkeypatch):
    """After an exact named-provider miss, the endpoint fallback must NOT reach
    through a converted sibling ``providers:`` entry that shares the base_url.

    ``beta`` is the selected provider and declares the model but no
    ``context_length``; ``alpha`` shares the endpoint and declares 1,048,576.
    ``beta`` must not borrow ``alpha``'s window via the compatible endpoint
    match (Medium review finding on #89714)."""
    import gateway.run as gateway_run

    model = "vllm/DeepSeek-V4-Flash-0731"
    base_url = "http://da-aihost01:4000/v1"
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {
            "model": {"default": model, "provider": "beta"},
            "providers": {
                "alpha": {
                    "base_url": base_url,
                    "models": {model: {"context_length": 1_048_576}},
                },
                "beta": {
                    "base_url": base_url,
                    "models": {model: {}},
                },
            },
        },
    )
    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda: model)
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom",
            "requested_provider": "beta",
            "base_url": base_url,
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, config_context_length=None, **_kwargs: (
            config_context_length or 512_000
        ),
    )

    context = gateway_run._resolve_gateway_model_context()

    assert context.context_length != 1_048_576
    assert context.context_length == 512_000
