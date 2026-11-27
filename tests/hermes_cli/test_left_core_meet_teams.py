"""Google Meet and the Teams meeting pipeline left core: what counts as a home that used them."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.left_core_migration as lcm
from hermes_cli import plugin_catalog


@pytest.fixture(autouse=True)
def _no_managed_overlay(tmp_path, monkeypatch):
    managed = tmp_path / "managed"
    managed.mkdir()
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    return managed


def _home(tmp_path: Path, *, env: str = "", config: str = "") -> Path:
    home = tmp_path / "home"
    home.mkdir()
    if env:
        (home / ".env").write_text(env, encoding="utf-8")
    if config:
        (home / "config.yaml").write_text(config, encoding="utf-8")
    return home


@pytest.mark.parametrize(("config", "in_use"), [
    ("plugins:\n  enabled: [google_meet]\n", True),                   # core's bundled key
    ("plugins:\n  enabled: [google_meet]\n  disabled: [google_meet]\n", False),   # disabled wins
    ("plugins:\n  enabled: [google_meet]\n  disabled: [google-meet]\n", False),   # under the new name too
    ("plugins:\n  enabled: [other]\n", False),
    ("plugins:\n  enabled: [google-meet]\n", False),                   # core's loader never matched it
    ("", False),                                                          # keys alone: it was opt-in
])
def test_google_meet_in_use_is_the_bundled_plugin_enabled(tmp_path, config, in_use):
    home = _home(tmp_path, env="HERMES_MEET_REALTIME_KEY=sk-x\nOPENAI_API_KEY=sk-y\n", config=config)
    assert lcm.google_meet_in_use(home) is in_use


@pytest.mark.parametrize(("config", "in_use"), [
    ("plugins:\n  enabled: [teams_pipeline]\n", True),
    ("plugins:\n  enabled: [teams-pipeline]\n", True),                # core's gateway wiring accepted it
    ("plugins:\n  enabled: [teams_pipeline]\n  disabled: [teams_pipeline]\n", False),
    ("platforms:\n  teams:\n    extra:\n      meeting_pipeline: {transcript_min_chars: 10}\n", False),
    ("", False),                                                          # Graph creds alone: it was opt-in
])
def test_teams_pipeline_in_use_is_the_bundled_plugin_enabled(tmp_path, config, in_use):
    env = "MSGRAPH_TENANT_ID=t\nMSGRAPH_CLIENT_ID=c\nMSGRAPH_CLIENT_SECRET=s\n"
    assert lcm.teams_pipeline_in_use(_home(tmp_path, env=env, config=config)) is in_use


def test_an_administrators_pinned_list_counts(tmp_path, _no_managed_overlay):
    (_no_managed_overlay / "config.yaml").write_text("plugins:\n  enabled: [teams_pipeline, google_meet]\n",
                                                     encoding="utf-8")
    home = _home(tmp_path)
    assert lcm.teams_pipeline_in_use(home) and lcm.google_meet_in_use(home)


def test_old_keys_stay_aliases_of_the_catalog_plugins():
    assert lcm.legacy_plugin_alias("google_meet") == "google-meet"
    assert lcm.legacy_plugin_alias("teams_pipeline") == "teams-pipeline"
    assert "teams_pipeline" in lcm.legacy_plugin_keys("teams-pipeline")


def test_the_old_bundled_name_finds_the_catalog_entry_but_saved_names_stay_exact(monkeypatch):
    """`hermes plugins install|info google_meet` accept the old name; a saved install record's name
    (update / repin) never drifts to another entry."""
    entries = [SimpleNamespace(name="google-meet"), SimpleNamespace(name="teams-pipeline")]
    monkeypatch.setattr(plugin_catalog, "load_catalog_live", lambda: entries)
    assert plugin_catalog.find_catalog_entry_for_user("google_meet") is entries[0]
    assert plugin_catalog.find_catalog_entry_for_user("teams_pipeline") is entries[1]
    assert plugin_catalog.find_catalog_entry_for_user("google-meet") is entries[0]
    assert plugin_catalog.get_live_catalog_entry("google_meet") is None


@pytest.mark.parametrize(("legacy", "canonical"), [("google_meet", "google-meet"), ("teams_pipeline", "teams-pipeline")])
def test_catalog_lookup_prefers_an_exact_entry_to_its_legacy_alias(monkeypatch, legacy, canonical):
    alias_entry = SimpleNamespace(name=canonical)
    exact_entry = SimpleNamespace(name=legacy)
    monkeypatch.setattr(plugin_catalog, "load_catalog_live", lambda: [alias_entry, exact_entry])
    assert plugin_catalog.find_catalog_entry_for_user(legacy) is exact_entry


@pytest.mark.parametrize("isolation", ["host", "in_process", None])
def test_a_host_isolated_home_is_told_what_the_catalog_plugins_lose(tmp_path, monkeypatch, isolation):
    """Core's bundled pipeline ran in-process even under ``plugins.isolation: host``; the catalog plugins
    run in the plugin host, where platform handlers and CLI commands do not. The migration says so and
    names the setting instead of silently losing the webhook handler; other homes hear nothing new."""
    import hermes_cli.memory_provider_migration as mpm
    monkeypatch.setattr(lcm, "_attempted", set())
    monkeypatch.setattr(mpm, "catalog_source", lambda name: name)
    config = "plugins:\n  enabled: [teams_pipeline, google_meet]\n"
    if isolation:
        config += f"  isolation: {isolation}\n"
    home = _home(tmp_path, config=config)
    said: list[str] = []

    def install(name):
        (home / "plugins" / name).mkdir(parents=True)
        return {"ok": True}

    assert sorted(lcm.migrate_home(home, install=install, say=said.append)) == ["google-meet", "teams-pipeline"]
    teams = next(line for line in said if "Teams meeting pipeline" in line)
    meet = next(line for line in said if "Google Meet" in line)
    if isolation == "host":
        assert "msgraph_webhook" in teams and "set plugins.isolation: in_process" in teams
        assert "`hermes meet` is not available" in meet and "set plugins.isolation: in_process" in meet
    else:
        assert "plugins.isolation" not in teams + meet


@pytest.mark.parametrize("isolation", ["host", "in_process"])
def test_a_missing_plugin_command_under_host_isolation_names_the_setting(tmp_path, monkeypatch, isolation):
    """The plugin host registers no CLI commands, so `hermes meet` is unknown there with the plugin installed and
    enabled; "enable it" would change nothing. In-process, the plugin is simply not loaded."""
    home = _home(tmp_path, config=f"plugins:\n  enabled: [google-meet]\n  isolation: {isolation}\n")
    (home / "plugins" / "google-meet").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    hint = lcm.moved_command_hint("hermes", "meet")
    if isolation == "host":
        assert "set plugins.isolation: in_process" in hint and "Enable it" not in hint
    else:
        assert "which is not loaded" in hint and "plugins.isolation" not in hint
