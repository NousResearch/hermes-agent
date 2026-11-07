"""Fallback refresh keeps base semantics for invalid or null managed configuration."""
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("surface", ["cli", "tui", "gateway"])
@pytest.mark.parametrize("policy", ["fallback_model: [unterminated", "- invalid", "fallback_model:"])
def test_managed_policy_does_not_freeze_or_clear_fallback_refresh(
    tmp_path, monkeypatch, surface, policy
):
    from hermes_cli import config, config_effective, managed_scope
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin
    from gateway.run import GatewayRunner
    from tui_gateway import server

    home, managed = tmp_path / "home", tmp_path / "managed"
    home.mkdir()
    managed.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    monkeypatch.setattr("gateway.run._gateway_config_home", lambda: home)
    monkeypatch.setattr(server, "_active_config_path", lambda: home / "config.yaml")
    for cache in (config._LOAD_CONFIG_CACHE, config._RAW_CONFIG_CACHE,
                  config_effective._EFFECTIVE_CACHE, config_effective._LAST_GOOD_USER_RAW):
        cache.clear()
    managed_scope.invalidate_managed_cache()
    (managed / "config.yaml").write_text(policy)
    agent = SimpleNamespace(
        _fallback_chain=[], _fallback_model=None, _fallback_index=0,
        _fallback_activated=False, _rate_limited_until=0, _unavailable_fallback_keys=set(),
    )
    owner = SimpleNamespace(_fallback_model=[{"provider": "openrouter", "model": "other/home"}])
    for model in ("user/first", "user/edited"):
        (home / "config.yaml").write_text(
            f"fallback_model: {{provider: openrouter, model: {model}}}\n"
        )
        if surface == "cli":
            CLIChatTurnMixin._sync_fallback_chain_with_config(owner, agent)
            chain = agent._fallback_chain
        elif surface == "tui":
            server._sync_agent_fallback_with_config("synthetic", {"agent": agent})
            chain = agent._fallback_chain
        else:
            chain = GatewayRunner._refresh_fallback_model(owner)
        assert chain == [{"provider": "openrouter", "model": model}]
