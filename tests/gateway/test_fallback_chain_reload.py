"""Regression tests for #60955: gateway must not freeze fallback_providers.

Cron reloads ``fallback_providers`` from disk on every job. The gateway used to
freeze ``self._fallback_model`` at process start, so a chain configured (or
edited) after ``hermes gateway`` was already running never reached messaging
sessions — even though cron in the same process fell back correctly.

These tests pin the reload + cached-agent apply helpers without driving the
full Feishu session path.
"""

from __future__ import annotations

import time
from types import SimpleNamespace


def test_refresh_fallback_model_rereads_config(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(
        _fallback_model=None,
    )
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    chain = bound()

    assert chain == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    assert runner._fallback_model == chain

    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: anthropic/claude-sonnet-4.6\n"
    )
    updated = bound()
    assert updated == [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}
    ]
    assert runner._fallback_model == updated


def test_apply_fallback_chain_skips_while_cooldown_holds_fallback():
    """Do not clobber a live fallback activation during its cooldown window."""
    from gateway.run import GatewayRunner

    live = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    agent = SimpleNamespace(
        _fallback_chain=live,
        _fallback_model=live[0],
        _fallback_index=1,
        _fallback_activated=True,
        _rate_limited_until=time.monotonic() + 30,
    )
    GatewayRunner._apply_fallback_chain_to_agent(
        agent,
        [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}],
    )

    assert agent._fallback_chain == live
    assert agent._fallback_index == 1
    assert agent._fallback_activated is True




def test_load_fallback_model_static_unchanged_contract(tmp_path, monkeypatch):
    """_load_fallback_model remains a pure static reader used by refresh."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
        "fallback_model:\n"
        "  provider: nous\n"
        "  model: Hermes-4\n"
    )

    chain = GatewayRunner._load_fallback_model()
    assert chain == [
        {"provider": "deepseek", "model": "deepseek-v4-flash"},
        {"provider": "nous", "model": "Hermes-4"},
    ]


def _channel_runner(tmp_path, monkeypatch, overrides):
    from gateway.config import ChannelOverride, GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: openrouter\n"
        "    model: global/fallback\n"
    )
    runner = object.__new__(GatewayRunner)
    runner._fallback_model = None
    runner.config = GatewayConfig(platforms={Platform.WHATSAPP: PlatformConfig(
        enabled=True,
        channel_overrides={cid: ChannelOverride.from_dict(ov) for cid, ov in overrides.items()},
    )})
    return runner


def _source(chat_id):
    from gateway.config import Platform
    from gateway.session import SessionSource
    return SessionSource(platform=Platform.WHATSAPP, chat_id=chat_id, user_id="u1")


GLOBAL = [{"provider": "openrouter", "model": "global/fallback"}]
TERRA = [{"provider": "openai-codex", "model": "gpt-6.1-terra"}]


def test_channel_fallback_providers_replace_the_global_chain_for_that_chat_only(tmp_path, monkeypatch):
    runner = _channel_runner(tmp_path, monkeypatch, {
        "capped@g.us": {"model": "sonnet", "fallback_providers": [*TERRA, {"provider": "x"}]},
        "plain@g.us": {"model": "sonnet"},
        "off@g.us": {"fallback_providers": []},
    })

    # Normalised like the global key: the entry without a model is dropped.
    assert runner._fallback_chain_for(_source("capped@g.us")) == TERRA
    assert runner._fallback_chain_for(_source("plain@g.us")) == GLOBAL
    assert runner._fallback_chain_for(_source("dm@s.whatsapp.net")) == GLOBAL
    assert runner._fallback_chain_for(_source("off@g.us")) is None
    # The profile-wide slot stays the global chain; a channel chain never leaks into other chats.
    assert runner._fallback_model == GLOBAL


def test_cached_agent_refresh_keeps_the_channel_chain(tmp_path, monkeypatch):
    """The per-turn refresh of a reused agent must not swap a channel's chain for the global one."""
    from gateway.run import GatewayRunner

    runner = _channel_runner(tmp_path, monkeypatch, {"capped@g.us": {"fallback_providers": TERRA}})
    agent = SimpleNamespace(_fallback_chain=list(TERRA), _fallback_model=TERRA[0], _fallback_index=0,
                            _fallback_activated=False, _rate_limited_until=0)

    GatewayRunner._apply_fallback_chain_to_agent(agent, runner._fallback_chain_for(_source("capped@g.us")))

    assert agent._fallback_chain == TERRA
    assert agent._fallback_model == TERRA[0]
