from pathlib import Path

import pytest

from hermes_cli.profile_slash import handle_profile_slash
from hermes_cli.profiles import create_profile, get_active_profile


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return default_home


def test_profile_slash_switch_sets_sticky_default(profile_env, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.create_wrapper_script", lambda name: None)
    create_profile("auditor", no_alias=True, no_skills=True)

    output = handle_profile_slash(["switch", "auditor"])

    assert "Default profile set to auditor" in output
    assert get_active_profile() == "auditor"


def test_profile_slash_lists_available_profiles(profile_env, monkeypatch):
    monkeypatch.setattr("hermes_cli.profiles.create_wrapper_script", lambda name: None)
    create_profile("auditor", no_alias=True, no_skills=True)

    output = handle_profile_slash(["list"])

    assert "default" in output
    assert "auditor" in output
