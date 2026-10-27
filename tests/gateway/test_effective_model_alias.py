"""The final gateway-selected alias owns the entire provider route."""
from types import SimpleNamespace

import pytest

from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
from gateway.run import _profile_runtime_scope
from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from gateway.session import SessionSource


def test_signal_profile_alias_precedence_and_credentials(tmp_path, monkeypatch):
    from gateway import run
    home = tmp_path / "sharik"
    home.mkdir()
    (home / "config.yaml").write_text(
        'model:\n  default: default-alias\nmodel_aliases:\n'
        '  default-alias: {model: default/wire, provider: custom, base_url: "https://default.example/v1", api_key: default-key}\n'
        '  channel-alias: {model: channel/wire, provider: custom, base_url: "https://channel.example/v1", api_key: channel-key}\n'
        '  session-alias: {model: session/wire, provider: custom, base_url: "https://session.example/v1", api_key: session-key}\n'
    )
    source = SessionSource(platform=Platform.SIGNAL, chat_id="family", user_id="user")
    runner = SimpleNamespace(
        config=GatewayConfig(platforms={Platform.SIGNAL: PlatformConfig(
            enabled=True, channel_overrides={"family": ChannelOverride(model="channel-alias")})}),
        _pre_agent_fallback_notice=None,
        _resolve_session_key_or_none=lambda source, session_key: "signal-family",
        _rehydrate_session_model_override=lambda key: None,
        _session_state=lambda key: SimpleNamespace(conversation=SimpleNamespace(last_resolved_model=None)),
    )
    selected = {"override": None}
    runner._peek_session_state = lambda key: SimpleNamespace(conversation=SimpleNamespace(
        model_override=selected["override"], last_resolved_model=None)) if key == "signal-family" else None
    monkeypatch.setattr(run, "_resolve_runtime_agent_kwargs", lambda: pytest.fail("alias must not use default route"))
    with _profile_runtime_scope(home):
        from hermes_cli.config import load_config
        cfg = load_config()
        for channel_model, override, expected_model, expected_key, expected_host in (
            (None, None, "default/wire", "default-key", "default.example"),
            ("channel-alias", None, "channel/wire", "channel-key", "channel.example"),
            ("channel-alias", {"model": "session-alias", "provider": "custom", "api_key": "old-key",
                               "base_url": "https://old.example/v1"}, "session/wire", "session-key", "session.example"),
        ):
            runner.config.platforms[Platform.SIGNAL].channel_overrides = (
                {"family": ChannelOverride(model=channel_model)} if channel_model else {})
            selected["override"] = override
            model, runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime(
                runner, source=source, user_config=cfg)
            assert model == expected_model
            assert runtime["api_key"] == expected_key
            assert expected_host in runtime["base_url"]
            assert runtime["provider"] == "custom"
            assert runtime["api_mode"] == "chat_completions"
            assert "old-key" not in str(runtime)


def test_malformed_declared_alias_fails_before_provider_resolution(tmp_path, monkeypatch):
    from gateway import run
    home = tmp_path / "sharik"
    home.mkdir()
    (home / "config.yaml").write_text(
        'model:\n  default: broken\nmodel_aliases:\n  broken: {provider: openrouter, base_url: "https://openrouter.ai/api/v1"}\n')
    from hermes_cli import runtime_provider
    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", lambda **kw: pytest.fail("provider request"))
    with _profile_runtime_scope(home), pytest.raises(RuntimeError, match="alias 'broken'"):
        run._resolve_direct_alias_agent_runtime("broken")
