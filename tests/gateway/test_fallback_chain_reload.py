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


def test_main_provider_policy_changes_bust_agent_cache_signature():
    from gateway.run import GatewayRunner

    enabled = {
        "main_provider_policies": {
            "openrouter": {
                "enabled": True,
                "provider_routing": {"data_collection": "deny"},
            }
        }
    }
    disabled = {
        "main_provider_policies": {
            "openrouter": {
                "enabled": False,
                "provider_routing": {"data_collection": "deny"},
            }
        }
    }
    runtime = {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1"}

    enabled_sig = GatewayRunner._agent_config_signature(
        "test/model",
        runtime,
        [],
        "",
        cache_keys=GatewayRunner._extract_cache_busting_config(enabled),
    )
    disabled_sig = GatewayRunner._agent_config_signature(
        "test/model",
        runtime,
        [],
        "",
        cache_keys=GatewayRunner._extract_cache_busting_config(disabled),
    )

    assert enabled_sig != disabled_sig


def test_refresh_fallback_model_resolves_live_main_policy(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "fallback_providers:\n"
        "  - provider: anthropic\n"
        "    model: base-model\n"
        "main_provider_policies:\n"
        "  openrouter:\n"
        "    fallback_providers:\n"
        "      - provider: openai-codex\n"
        "        model: policy-model\n",
        encoding="utf-8",
    )
    runner = object.__new__(GatewayRunner)
    runner._fallback_model = None
    bound = runner._refresh_fallback_model

    chain = bound("openrouter", "main-model")

    assert chain == [{"provider": "openai-codex", "model": "policy-model"}]
    assert runner._fallback_model == chain


def test_fallback_refresh_failure_does_not_leak_another_route(
    tmp_path, monkeypatch
):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "main_provider_policies:\n"
        "  openrouter:\n"
        "    fallback_providers:\n"
        "      - provider: openai-codex\n"
        "        model: policy-model\n",
        encoding="utf-8",
    )
    runner = object.__new__(GatewayRunner)
    runner._fallback_model = None
    bound = runner._refresh_fallback_model
    assert bound("openrouter", "main-model") == [
        {"provider": "openai-codex", "model": "policy-model"}
    ]

    cfg.write_text("main_provider_policies: [", encoding="utf-8")

    assert bound("anthropic", "claude-model") is None


def test_fallback_refresh_isolated_by_profile_home(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    profile_a = tmp_path / "profile-a"
    profile_b = tmp_path / "profile-b"
    profile_a.mkdir()
    profile_b.mkdir()
    (profile_a / "config.yaml").write_text(
        "main_provider_policies:\n"
        "  openrouter:\n"
        "    fallback_providers:\n"
        "      - provider: anthropic\n"
        "        model: profile-a\n",
        encoding="utf-8",
    )
    (profile_b / "config.yaml").write_text(
        "main_provider_policies:\n"
        "  openrouter:\n"
        "    fallback_providers:\n"
        "      - provider: openai-codex\n"
        "        model: profile-b\n",
        encoding="utf-8",
    )
    current = {"home": profile_a}
    monkeypatch.setattr("gateway.run._hermes_home", profile_a)
    monkeypatch.setattr(
        "gateway.run._gateway_config_home", lambda: current["home"]
    )
    runner = object.__new__(GatewayRunner)
    runner._fallback_model = None

    assert runner._refresh_fallback_model("openrouter", "main-model") == [
        {"provider": "anthropic", "model": "profile-a"}
    ]
    current["home"] = profile_b
    assert runner._refresh_fallback_model("openrouter", "main-model") == [
        {"provider": "openai-codex", "model": "profile-b"}
    ]

    (profile_a / "config.yaml").write_text("main_provider_policies: [")
    current["home"] = profile_a
    assert runner._refresh_fallback_model("openrouter", "main-model") == [
        {"provider": "anthropic", "model": "profile-a"}
    ]


def test_uncached_default_route_failure_does_not_borrow_named_route(
    tmp_path, monkeypatch
):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "main_provider_policies:\n"
        "  openrouter:\n"
        "    fallback_providers:\n"
        "      - provider: anthropic\n"
        "        model: policy-model\n",
        encoding="utf-8",
    )
    base_chain = [{"provider": "openai-codex", "model": "startup-base"}]
    runner = object.__new__(GatewayRunner)
    runner._fallback_model = base_chain

    assert runner._refresh_fallback_model("openrouter", "main-model") == [
        {"provider": "anthropic", "model": "policy-model"}
    ]
    cfg.write_text("main_provider_policies: [", encoding="utf-8")

    assert runner._refresh_fallback_model() == base_chain


def test_first_read_failure_uses_startup_base_for_default_profile_route(
    tmp_path, monkeypatch
):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    (tmp_path / "config.yaml").write_text(
        "main_provider_policies: [", encoding="utf-8"
    )
    base_chain = [{"provider": "openai-codex", "model": "startup-base"}]
    runner = object.__new__(GatewayRunner)
    runner._fallback_model = base_chain

    assert runner._refresh_fallback_model("openrouter", "main-model") == base_chain


def test_refresh_fallback_model_rereads_config(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = object.__new__(GatewayRunner)
    runner._fallback_model = None
    bound = runner._refresh_fallback_model
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
        source.count("fallback_model=self._refresh_fallback_model(")
        + source.count("fallback_model=self._runner._refresh_fallback_model(")
    )
    assert _refresh_calls >= 2
    # The cached-agent reuse path (the load-bearing fix for a long-lived
    # session in a running gateway) must apply the refreshed chain.
    assert (
        "self._apply_fallback_chain_to_agent(" in source
        or "self._runner._apply_fallback_chain_to_agent(" in source
    )
    reuse_start = source.index("if reused_cached_agent and agent is not None:")
    reuse_end = source.index(
        "# Lock released — now schedule cleanup", reuse_start
    )
    reuse_refresh = source[reuse_start:reuse_end]
    assert 'turn_route["runtime"].get("provider", "")' in reuse_refresh
    assert 'turn_route["model"]' in reuse_refresh
    assert 'getattr(agent, "provider", "")' not in reuse_refresh
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
