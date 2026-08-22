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
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
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
    home_token = set_hermes_home_override(str(tmp_path))
    try:
        chain = bound()
        assert chain == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
        assert runner._fallback_model == chain

        cfg.write_text(
            "fallback_providers:\n"
            "  - provider: openrouter\n"
            "    model: anthropic/claude-sonnet-4.6\n"
        )
        updated = bound()
    finally:
        reset_hermes_home_override(home_token)

    assert updated == [
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}
    ]
    assert runner._fallback_model == updated


def test_refresh_fallback_model_reads_active_profile_home(tmp_path, monkeypatch):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    from gateway.run import GatewayRunner

    root_home = tmp_path / "root"
    profile_home = tmp_path / "profiles" / "secondary"
    root_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    (root_home / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: root-provider\n"
        "    model: root-model\n"
    )
    (profile_home / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: secondary-provider\n"
        "    model: secondary-model\n"
    )
    monkeypatch.setattr("gateway.run._hermes_home", root_home)

    runner = SimpleNamespace(_fallback_model=None)
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    home_token = set_hermes_home_override(str(profile_home))
    try:
        chain = bound()
    finally:
        reset_hermes_home_override(home_token)

    assert chain == [{"provider": "secondary-provider", "model": "secondary-model"}]
    assert runner._fallback_model == chain

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


def test_background_and_main_agent_paths_call_refresh():
    """Both AIAgent construction sites must pass a refreshed chain, not the
    startup snapshot, and the cached-agent reuse path must apply the refreshed
    chain. Source-level invariant for call sites that resist unit testing.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent.parent / "gateway" / "run.py"
    ).read_text(encoding="utf-8")
    # The agent-construction site inside TurnRunner.run_sync (extracted from
    # the old _run_agent_inner closure) references the runner as
    # ``self._runner``; the background-agent site still uses bare ``self``.
    _refresh_calls = (
        source.count("fallback_model=self._refresh_fallback_model()")
        + source.count("fallback_model=self._runner._refresh_fallback_model()")
    )
    assert _refresh_calls >= 2
    # The cached-agent reuse path (the load-bearing fix for a long-lived
    # session in a running gateway) must apply the refreshed chain.
    assert (
        "self._apply_fallback_chain_to_agent(" in source
        or "self._runner._apply_fallback_chain_to_agent(" in source
    )
    # The stale startup-snapshot form must not remain at create sites.
    assert "fallback_model=self._fallback_model," not in source
    assert "fallback_model=self._runner._fallback_model," not in source


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


def test_gateway_primary_auth_fallback_preserves_selected_reasoning_entry(monkeypatch):
    from gateway.run import _try_resolve_fallback_provider

    entry = {
        "provider": "secondary-provider",
        "model": "secondary-model",
        "reasoning_effort": "high",
        "api_mode": "codex_responses",
    }
    monkeypatch.setattr("gateway.run._load_gateway_runtime_config", lambda: {})
    monkeypatch.setattr("gateway.run.get_fallback_chain", lambda _cfg: [entry])
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key",
        lambda _entry: "test-key",
    )
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return {
            "provider": "secondary-provider",
            "requested_provider": "secondary-provider",
            "api_key": "test-key",
            "base_url": "https://secondary.invalid/v1",
            "api_mode": "chat_completions",
        }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider", resolve
    )

    resolved = _try_resolve_fallback_provider()

    assert resolved is not None
    assert resolved["model"] == "secondary-model"
    assert resolved["fallback_entry"] == entry
    assert resolved["api_mode"] == "codex_responses"
    assert calls[0]["target_model"] == "secondary-model"


def test_tui_primary_auth_fallback_preserves_selected_reasoning_entry(monkeypatch):
    from hermes_cli.auth import AuthError
    from tui_gateway.server import _resolve_runtime_with_fallback

    entry = {
        "provider": "secondary-provider",
        "model": "secondary-model",
        "reasoning_effort": "high",
        "api_mode": "codex_responses",
    }
    monkeypatch.setattr("tui_gateway.server._load_fallback_model", lambda: [entry])
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        if kwargs.get("requested") == "primary-provider":
            raise AuthError("primary unavailable", provider="primary-provider")
        return {
            "provider": "secondary-provider",
            "requested_provider": "secondary-provider",
            "api_key": "test-key",
            "base_url": "https://secondary.invalid/v1",
            "api_mode": "chat_completions",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)

    resolved = _resolve_runtime_with_fallback(
        {"requested": "primary-provider", "target_model": "primary-model"}
    )

    assert resolved.used_fallback is True
    assert resolved.selected_model == "secondary-model"
    assert resolved.selected_entry == entry
    assert resolved.runtime["api_mode"] == "codex_responses"
    assert calls[0]["target_model"] == "primary-model"
