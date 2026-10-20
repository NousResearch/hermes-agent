"""Startup attribution for configured platforms that produce no adapter (#131974).

A platform can be enabled under ``gateway.platforms.<name>``, reach startup, and still get no
adapter because its plugin sits on ``plugins.disabled`` — ``gate_manifest`` honours the deny-list
before the bundled-platform deferral, so the registry never gains an entry. The old startup log
only said ``No adapter available for discord``, which points nowhere near the config cause (the
discovery skip is DEBUG-level). These tests pin the attribution: the warning must name the
deny-list entry and the command that undoes it, while a no-adapter platform with a clean
plugins config keeps the generic messages.
"""

import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner


@pytest.mark.asyncio
async def test_disabled_platform_warning_names_the_deny_list_entry(monkeypatch, caplog):
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True)})
    runner = GatewayRunner(config)
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)
    monkeypatch.setattr(
        "hermes_cli.plugins_cmd._get_disabled_set", lambda: {"platforms/discord"}
    )
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        aborted, count, _skipped, _pending = await runner._start_prefilter_platforms()
    assert (aborted, count) == (False, 1)
    messages = [r.getMessage() for r in caplog.records if r.name == "gateway.run"]
    assert any(
        "'platforms/discord' is on plugins.disabled" in m
        and "hermes plugins enable platforms/discord" in m
        for m in messages
    ), messages
    assert not any(m == "No adapter available for discord" for m in messages), messages


@pytest.mark.asyncio
async def test_missing_adapter_with_clean_plugins_config_keeps_generic_warning(
    monkeypatch, caplog
):
    config = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True)})
    runner = GatewayRunner(config)
    monkeypatch.setattr(runner, "_create_adapter", lambda platform, cfg: None)
    monkeypatch.setattr("hermes_cli.plugins_cmd._get_disabled_set", lambda: set())
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        aborted, count, _skipped, _pending = await runner._start_prefilter_platforms()
    assert (aborted, count) == (False, 1)
    messages = [r.getMessage() for r in caplog.records if r.name == "gateway.run"]
    assert any(m == "No adapter available for discord" for m in messages), messages
    assert not any("plugins.disabled" in m for m in messages), messages


def test_plugin_disabled_key_matches_canonical_key_and_bare_name(monkeypatch):
    from gateway.run_startup import _plugin_disabled_key

    monkeypatch.setattr(
        "hermes_cli.plugins_cmd._get_disabled_set",
        lambda: {"platforms/discord", "signal"},
    )
    assert _plugin_disabled_key("discord") == "platforms/discord"
    assert _plugin_disabled_key("signal") == "signal"
    assert _plugin_disabled_key("telegram") is None


def test_plugin_disabled_key_survives_config_read_failure(monkeypatch):
    """A broken plugins config must not turn the no-adapter warning path into a crash."""
    from gateway.run_startup import _plugin_disabled_key

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.plugins_cmd._get_disabled_set", _boom)
    assert _plugin_disabled_key("discord") is None
