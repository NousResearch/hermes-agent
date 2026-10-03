"""Legacy reset settings must not disappear silently at gateway startup."""
import json
import logging
from types import SimpleNamespace

import pytest

from gateway.run_startup import GatewayStartupMixin
from hermes_cli import session_reset_retirement


@pytest.mark.parametrize("plugin_enabled", [False, True])
@pytest.mark.parametrize("key,value", [
    ("reset_by_platform", {"msgraph_webhook": {"mode": "idle", "idle_minutes": 10}}),
    ("reset_by_type", {"dm": {"mode": "daily"}}),
    ("default_reset_policy", {"mode": "both"}),
])
def test_startup_names_ignored_legacy_reset(tmp_path, monkeypatch, caplog, plugin_enabled, key, value):
    home = tmp_path / "graph-profile"
    home.mkdir()
    legacy = home / "gateway.json"
    original = json.dumps({key: value})
    legacy.write_text(original, encoding="utf-8")
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: [("graph-profile", home)])
    monkeypatch.setattr(session_reset_retirement, "reset_plugin_enabled", lambda: plugin_enabled)
    runner = object.__new__(GatewayStartupMixin)
    runner.config = SimpleNamespace(multiplex_profiles=True)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._start_log_retired_session_reset()
    warnings = [r.getMessage() for r in caplog.records if "no longer applied" in r.getMessage()]
    assert any("graph-profile" in msg and f"gateway.json:{key}" in msg for msg in warnings)
    assert all("hermes plugins install" not in msg for msg in warnings)
    assert legacy.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("data", [
    {}, {"reset_by_platform": {}}, {"platforms": {}}, None, [], "broken",
    {"default_reset_policy": {"mode": "none"}},
    {"reset_by_type": {"dm": None, "group": {"mode": "none"}}},
])
def test_startup_without_legacy_reset_is_quiet(tmp_path, monkeypatch, caplog, data):
    (tmp_path / "gateway.json").write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: [("test", tmp_path)])
    runner = object.__new__(GatewayStartupMixin)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._start_log_retired_session_reset()
    assert not any("no longer applied" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("broken_file", ["gateway.json", "config.yaml"])
def test_one_broken_profile_does_not_hide_another_policy(tmp_path, monkeypatch, caplog, broken_file):
    bad, good = tmp_path / "bad", tmp_path / "good"
    bad.mkdir()
    good.mkdir()
    (bad / broken_file).write_text("{", encoding="utf-8")
    (good / "gateway.json").write_text(
        json.dumps({"reset_by_platform": {"msgraph_webhook": {"mode": "idle"}}}), encoding="utf-8",
    )
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: [("bad", bad), ("good", good)])
    runner = object.__new__(GatewayStartupMixin)
    runner.config = SimpleNamespace(multiplex_profiles=True)
    with caplog.at_level(logging.WARNING):
        runner._start_log_retired_session_reset()
    assert any("Profile good:" in r.getMessage() and "no longer applied" in r.getMessage() for r in caplog.records)

