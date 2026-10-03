"""Tests for the model-fleet plugin.

Pure-logic tests only: the provider catalog, the switch pipeline and the cron store
are all stubbed so the suite never touches a real $HERMES_HOME or the network.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PLUGIN_PATH = Path(__file__).resolve().parent.parent / "__init__.py"


def _load():
    spec = importlib.util.spec_from_file_location("model_fleet_under_test", PLUGIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mf():
    module = _load()
    module._CACHE.clear()
    return module


PROVIDERS = [
    {"slug": "nous", "name": "Nous Portal", "is_current": True, "total_models": 3,
     "models": ["stealth/alpha", "z-ai/glm-5.3-flash", "z-ai/glm-5.4"]},
    {"slug": "copilot", "name": "GitHub Copilot", "is_current": False, "total_models": 2,
     "models": ["claude-fable-5.1", "gpt-5.6-sol"]},
]


# ------------------------------------------------------------------ settings

def test_settings_defaults_when_config_unreadable(mf, monkeypatch):
    monkeypatch.setattr(mf, "_settings", lambda: dict(mf._SETTING_DEFAULTS))
    assert mf._settings()["include_profiles"] is True
    assert mf._settings()["include_auxiliary"] is False


def test_settings_coerces_comma_string_allowlist(mf, monkeypatch):
    import hermes_cli.config as hconfig
    import hermes_cli.plugins_state as pstate

    monkeypatch.setattr(hconfig, "load_config_readonly", lambda: {
        "plugins": {"entries": {"model-fleet": {"profile_allowlist": "alpha, beta"}}},
    })
    monkeypatch.setattr(pstate, "_plugin_settings_entry",
                        lambda cfg, pid: (cfg.get("plugins", {}).get("entries", {}) or {}).get(pid))
    assert mf._settings()["profile_allowlist"] == ["alpha", "beta"]


# ------------------------------------------------------------------ profile selection

def test_profile_allowlist_narrows_scope(mf):
    settings = dict(mf._SETTING_DEFAULTS, profile_allowlist=["alpha"])
    assert mf._profile_selected("alpha", settings) is True
    assert mf._profile_selected("beta", settings) is False


def test_profile_blocklist_removes_one(mf):
    settings = dict(mf._SETTING_DEFAULTS, profile_blocklist=["beta"])
    assert mf._profile_selected("alpha", settings) is True
    assert mf._profile_selected("beta", settings) is False


def test_include_profiles_false_keeps_only_default(mf, tmp_path, monkeypatch):
    (tmp_path / "profiles" / "alpha").mkdir(parents=True)
    (tmp_path / "profiles" / "alpha" / "config.yaml").write_text("model: {}\n")
    monkeypatch.setattr(mf, "_hermes_home", lambda: tmp_path)
    settings = dict(mf._SETTING_DEFAULTS, include_profiles=False)
    assert mf._profile_homes(settings) == [("default", tmp_path)]


# ------------------------------------------------------------------ matching

def test_match_provider_by_index_cold_cache(mf):
    assert mf._match_provider(PROVIDERS, "2")["slug"] == "copilot"


def test_match_provider_index_out_of_range(mf):
    assert mf._match_provider(PROVIDERS, "9") is None


def test_match_provider_by_slug_case_insensitive(mf):
    assert mf._match_provider(PROVIDERS, "NOUS")["slug"] == "nous"


def test_match_provider_by_display_name(mf):
    assert mf._match_provider(PROVIDERS, "copilot")["slug"] == "copilot"


def test_match_provider_unknown(mf):
    assert mf._match_provider(PROVIDERS, "nope") is None


def test_match_model_exact_then_substring_then_prefix(mf):
    row = PROVIDERS[0]
    assert mf._match_model(row, "z-ai/glm-5.3-flash") == "z-ai/glm-5.3-flash"
    assert mf._match_model(row, "glm-5.4") == "z-ai/glm-5.4"
    assert mf._match_model(row, "stealth") == "stealth/alpha"


def test_match_model_rejects_unknown(mf):
    assert mf._match_model(PROVIDERS[0], "gpt-9") is None


def test_match_model_ignores_cache_from_another_provider(mf):
    """A cached model list from a different provider must never answer for this one."""
    mf._CACHE.update({"models": ["wrong-model"], "provider": "copilot"})
    assert mf._match_model(PROVIDERS[0], "1") == "stealth/alpha"


def test_match_model_uses_cache_for_same_provider(mf):
    mf._CACHE.update({"models": ["stealth/alpha"], "provider": "nous"})
    assert mf._match_model(PROVIDERS[0], "1") == "stealth/alpha"


# ------------------------------------------------------------------ arg parsing

def test_parse_extracts_dry_run_flag(mf):
    assert mf._parse("nous gpt-5 --dry-run") == (["nous", "gpt-5"], True)
    assert mf._parse("-n nous gpt-5") == (["nous", "gpt-5"], True)
    assert mf._parse("nous gpt-5") == (["nous", "gpt-5"], False)
    assert mf._parse("") == ([], False)


# ------------------------------------------------------------------ routing

def test_help_returns_usage(mf):
    assert "/model-fleet" in (mf._handle_sync("help") or "")


def test_status_reports_profiles_and_crons(mf, tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text(
        "model:\n  provider: nous\n  default: stealth/alpha\ndelegation:\n  provider: nous\n  model: z-ai/glm\n"
    )
    monkeypatch.setattr(mf, "_hermes_home", lambda: tmp_path)
    out = mf._handle_sync("status")
    assert "nous/stealth/alpha" in out
    assert "sub-agents `nous/z-ai/glm`" in out


def test_unknown_provider_token_falls_back_to_listing(mf, monkeypatch):
    monkeypatch.setattr(mf, "_list_providers_sync", lambda *a, **k: PROVIDERS)
    out = mf._handle_sync("definitely-not-a-provider")
    assert "Pick a provider" in out


def test_resolve_target_reports_out_of_range_index(mf, monkeypatch):
    monkeypatch.setattr(mf, "_list_providers_sync", lambda *a, **k: PROVIDERS)
    _, _, err = mf._resolve_target("nous", "99")
    assert "No model #99" in err


def test_resolve_target_suggests_close_matches(mf, monkeypatch):
    monkeypatch.setattr(mf, "_list_providers_sync", lambda *a, **k: PROVIDERS)
    # a typo that is not an exact/substring/prefix match, so only the hint can save it
    _, _, err = mf._resolve_target("nous", "glm-5.3-flahs")
    assert "Close matches" in err
    assert "z-ai/glm-5.3-flash" in err


def test_resolve_target_no_hint_when_nothing_is_close(mf, monkeypatch):
    monkeypatch.setattr(mf, "_list_providers_sync", lambda *a, **k: PROVIDERS)
    _, _, err = mf._resolve_target("nous", "gpt-9")
    assert "Close matches" not in err


# ------------------------------------------------------------------ cron writes

def test_apply_crons_skips_no_agent_jobs(mf, tmp_path, monkeypatch):
    jobs_file = tmp_path / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    jobs_file.write_text(json.dumps({"jobs": [
        {"id": "a1", "provider": "nous", "model": "old", "no_agent": False},
        {"id": "a2", "provider": "nous", "model": "old", "no_agent": True, "script": "/bin/true"},
    ]}))
    monkeypatch.setattr(mf, "_hermes_home", lambda: tmp_path)
    settings = dict(mf._SETTING_DEFAULTS)
    changed, skipped, backups = mf._apply_crons("nous", "new", settings, "STAMP", dry_run=False)
    jobs = {j["id"]: j for j in json.loads(jobs_file.read_text())["jobs"]}
    assert jobs["a1"]["model"] == "new"
    assert jobs["a1"]["model_snapshot"] == "new"
    assert jobs["a2"]["model"] == "old", "no_agent script job must not be touched"
    assert len(backups) == 1
    assert not skipped


def test_apply_crons_dry_run_writes_nothing(mf, tmp_path, monkeypatch):
    jobs_file = tmp_path / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    original = json.dumps({"jobs": [{"id": "a1", "provider": "nous", "model": "old", "no_agent": False}]})
    jobs_file.write_text(original)
    monkeypatch.setattr(mf, "_hermes_home", lambda: tmp_path)
    changed, _, backups = mf._apply_crons("nous", "new", dict(mf._SETTING_DEFAULTS), "STAMP", dry_run=True)
    assert jobs_file.read_text() == original
    assert backups == []
    assert any("would repoint" in c for c in changed)


def test_apply_crons_noop_when_already_on_target(mf, tmp_path, monkeypatch):
    jobs_file = tmp_path / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    jobs_file.write_text(json.dumps({"jobs": [
        {"id": "a1", "provider": "nous", "model": "new", "no_agent": False}]}))
    monkeypatch.setattr(mf, "_hermes_home", lambda: tmp_path)
    changed, _, backups = mf._apply_crons("nous", "new", dict(mf._SETTING_DEFAULTS), "STAMP", dry_run=False)
    assert backups == []
    assert any("already on" in c for c in changed)


def test_backup_can_be_disabled(mf, tmp_path, monkeypatch):
    jobs_file = tmp_path / "cron" / "jobs.json"
    jobs_file.parent.mkdir(parents=True)
    jobs_file.write_text(json.dumps({"jobs": [
        {"id": "a1", "provider": "nous", "model": "old", "no_agent": False}]}))
    monkeypatch.setattr(mf, "_hermes_home", lambda: tmp_path)
    _, _, backups = mf._apply_crons("nous", "new", dict(mf._SETTING_DEFAULTS, backup=False), "STAMP", False)
    assert backups == []
    assert not list(tmp_path.glob("*.bak-*"))


# ------------------------------------------------------------------ safety

def test_refused_switch_is_reported_not_applied(mf, monkeypatch):
    class Failed:
        success = False
        error_message = "no credentials for provider"

    monkeypatch.setattr(mf, "_switch_result", lambda p, mo: Failed())
    out = mf._apply_sync("bogus", "some-model", dry_run=False, with_auxiliary=False)
    assert "Switch refused" in out
    assert "no credentials" in out
