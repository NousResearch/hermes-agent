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


def test_refresh_fallback_model_clears_when_config_removed(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(
        _fallback_model=[{"provider": "stale", "model": "x"}],
    )
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    assert bound() is not None

    cfg.write_text("model:\n  provider: nvidia\n")
    assert bound() is None
    assert runner._fallback_model is None


def test_refresh_fallback_model_keeps_last_known_good_on_read_failure(
    tmp_path, monkeypatch,
):
    """A transient config.yaml read/parse failure (user mid-edit, non-atomic
    write) must NOT wipe the last known-good chain — only a successful read
    that genuinely lacks the key clears it."""
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "fallback_providers:\n"
        "  - provider: deepseek\n"
        "    model: deepseek-v4-flash\n"
    )

    runner = SimpleNamespace(_fallback_model=None)
    runner._load_fallback_model = GatewayRunner._load_fallback_model
    bound = GatewayRunner._refresh_fallback_model.__get__(runner)
    good = bound()
    assert good == [{"provider": "deepseek", "model": "deepseek-v4-flash"}]

    # Simulate a mid-edit torn write: invalid YAML.
    cfg.write_text("fallback_providers:\n  - provider: [unclosed\n")
    assert bound() == good
    assert runner._fallback_model == good


def test_apply_fallback_chain_updates_primary_agent():
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        _fallback_chain=[],
        _fallback_model=None,
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
    )
    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    GatewayRunner._apply_fallback_chain_to_agent(agent, chain)

    assert agent._fallback_chain == chain
    assert agent._fallback_model == chain[0]
    assert agent._fallback_index == 0


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


def test_apply_fallback_chain_updates_after_cooldown_expires():
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "deepseek", "model": "old"}],
        _fallback_model={"provider": "deepseek", "model": "old"},
        _fallback_index=1,
        _fallback_activated=True,
        _rate_limited_until=time.monotonic() - 1,
    )
    new_chain = [{"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"}]
    GatewayRunner._apply_fallback_chain_to_agent(agent, new_chain)

    assert agent._fallback_chain == new_chain
    assert agent._fallback_model == new_chain[0]
    # Activated agents keep their index; restore_primary_runtime owns reset.
    assert agent._fallback_index == 1


def test_apply_fallback_chain_clears_unavailable_memo_on_content_change():
    """A config edit must drop the session-scoped unavailability memo so a
    re-configured entry (credentials added mid-uptime) is retried instead of
    staying suppressed for the cached agent's lifetime."""
    from gateway.run import GatewayRunner

    agent = SimpleNamespace(
        _fallback_chain=[{"provider": "deepseek", "model": "old"}],
        _fallback_model={"provider": "deepseek", "model": "old"},
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys={("deepseek", "old", "")},
    )
    new_chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    GatewayRunner._apply_fallback_chain_to_agent(agent, new_chain)

    assert agent._fallback_chain == new_chain
    assert agent._unavailable_fallback_keys == set()


def test_apply_fallback_chain_keeps_unavailable_memo_when_unchanged():
    """The per-message no-op refresh must NOT clear the memo — it exists to
    rate-limit repeated activation attempts against dead entries."""
    from gateway.run import GatewayRunner

    chain = [{"provider": "deepseek", "model": "deepseek-v4-flash"}]
    memo = {("deepseek", "deepseek-v4-flash", "")}
    agent = SimpleNamespace(
        _fallback_chain=list(chain),
        _fallback_model=chain[0],
        _fallback_index=0,
        _fallback_activated=False,
        _rate_limited_until=0,
        _unavailable_fallback_keys=set(memo),
    )
    GatewayRunner._apply_fallback_chain_to_agent(agent, list(chain))

    assert agent._unavailable_fallback_keys == memo


def test_background_and_main_agent_paths_call_refresh():
    """Both AIAgent construction sites must pass a refreshed chain, not the
    startup snapshot, and the cached-agent reuse path must apply the refreshed
    chain. Source-level invariant for call sites that resist unit testing.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent.parent / "gateway" / "run.py"
    ).read_text(encoding="utf-8")
    assert "fallback_model=self._refresh_fallback_model()" in source
    assert source.count("fallback_model=self._refresh_fallback_model()") >= 2
    # The cached-agent reuse path (the load-bearing fix for a long-lived
    # session in a running gateway) must apply the refreshed chain.
    assert "self._apply_fallback_chain_to_agent(" in source
    # The stale startup-snapshot form must not remain at create sites.
    assert "fallback_model=self._fallback_model," not in source


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
