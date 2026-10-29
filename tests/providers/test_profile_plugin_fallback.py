"""A created profile with an empty plugin folder still sees the account's providers."""

from pathlib import Path

import providers


def test_empty_profile_plugins_fall_back_to_process_home(monkeypatch, tmp_path):
    account = tmp_path / "account"
    profile = tmp_path / "profiles" / "loki"
    (account / "plugins" / "claude").mkdir(parents=True)
    (profile / "plugins").mkdir(parents=True)
    monkeypatch.setattr(providers, "get_hermes_home", lambda: profile, raising=False)
    # The function imports these itself. Patch the module it imports from.
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: profile)
    monkeypatch.setattr(hermes_constants, "get_process_hermes_home", lambda: account)

    assert providers._installed_plugins_dir() == account / "plugins"


def test_profile_with_its_own_plugins_does_not_fall_back(monkeypatch, tmp_path):
    account = tmp_path / "account"
    profile = tmp_path / "profiles" / "loki"
    (account / "plugins" / "shared").mkdir(parents=True)
    (profile / "plugins" / "own").mkdir(parents=True)
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: profile)
    monkeypatch.setattr(hermes_constants, "get_process_hermes_home", lambda: account)

    assert providers._installed_plugins_dir() == profile / "plugins"
