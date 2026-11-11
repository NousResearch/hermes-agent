"""Diagnostic: /model must inherit a provider-only channel route (#149132)."""

import pytest

from tests.gateway.test_messaging_session_picks import _compressed_chat


@pytest.mark.asyncio
@pytest.mark.parametrize("named_custom", [False, True])
@pytest.mark.parametrize("entry", ["context", "picker"])
async def test_provider_only_channel_route_reaches_model_switch_context(tmp_path, monkeypatch, named_custom, entry):
    import gateway.run as gateway_run
    from hermes_cli import runtime_provider
    from gateway.config import ChannelOverride, PlatformConfig
    from gateway.slash_commands_model import _ModelSwitchContext

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    endpoint = "http://127.0.0.1:9/v1"
    identity = "custom:channel" if named_custom else "anthropic"
    wire_provider = "custom" if named_custom else identity
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model:\n  default: configured-model\n  provider: openrouter\n"
        "providers:\n  channel:\n    base_url: http://127.0.0.1:9/v1\n"
        "    api_key: diagnostic-placeholder\n"
    )

    # Transport resolution alone is synthetic. Keep the gateway's provider wrapper,
    # channel lookup, precedence, custom identity and switch context real.
    def resolve_provider(**kwargs):
        return {"provider": wire_provider, "requested_provider": kwargs["requested"],
                "base_url": endpoint, "api_key": "diagnostic-placeholder"}

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", resolve_provider)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {
        "provider": "openrouter", "base_url": "http://127.0.0.1:8/v1"})
    runner, source, route, viewer, _root, _fences = await _compressed_chat(tmp_path)
    runner.config.platforms[source.platform] = PlatformConfig(channel_overrides={
        source.chat_id: ChannelOverride(provider=identity)})
    try:
        model, runtime = runner._resolve_session_agent_runtime(
            source=source, session_key=route,
            user_config={"model": {"default": "configured-model", "provider": "openrouter"}})
        assert (model, runtime["provider"], runtime["base_url"]) == (
            "configured-model", wire_provider, endpoint)

        if entry == "context":
            ctx = _ModelSwitchContext(route, source, config_path, persist_global=False)
            ctx.read_config()
            await runner._apply_effective_model_route(ctx, source)
            # These fields also feed switch_model's route on a typed switch.
            assert (ctx.current_model, ctx.current_provider, ctx.current_base_url) == (
                model, identity, runtime["base_url"])
        else:
            from types import SimpleNamespace
            from gateway.platforms.event import MessageEvent
            from hermes_cli import model_switch_providers

            delivered = []

            class PickerAdapter:
                async def send_model_picker(self, **kwargs):
                    delivered.append((kwargs["current_model"], kwargs["current_provider"]))
                    return SimpleNamespace(success=True)

            monkeypatch.setattr(runner, "_delivery_adapter_for", lambda source: PickerAdapter())
            monkeypatch.setattr(model_switch_providers, "list_picker_providers", lambda **kwargs: [
                {"provider": identity, "models": [{"id": model}]}])
            assert await runner._handle_model_command(MessageEvent(text="/model", source=source)) is None
            assert delivered == [(model, identity)]
    finally:
        await viewer.close()
        runner.session_store.close_all_db_handles()
