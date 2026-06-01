"""Regression test for issue #36729.

The WebUI / standalone gateway processes start without ``HERMES_HOME``
being redirected into a profile directory. Before the fix,
``_load_gateway_config()`` only loaded ``~/.hermes/config.yaml`` and
silently ignored profile-level overrides like
``agent.disabled_toolsets`` from ``~/.hermes/profiles/<name>/config.yaml``.
This test verifies that the active profile's config is now deep-merged
on top of the global one.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def fake_hermes_home(tmp_path, monkeypatch):
    """Build a fake ~/.hermes layout and point HERMES_HOME at it."""
    home = tmp_path / "hermes"
    (home / "profiles" / "pm").mkdir(parents=True)
    # Global config: disabled_toolsets is empty (matches issue reporter's setup).
    (home / "config.yaml").write_text(
        yaml.safe_dump({
            "agent": {"disabled_toolsets": [], "model": "global-model"},
            "platforms": {"foo": True},
        })
    )
    # Profile config defines disabled_toolsets that should win.
    (home / "profiles" / "pm" / "config.yaml").write_text(
        yaml.safe_dump({
            "agent": {"disabled_toolsets": ["browser", "computer_use"]},
        })
    )
    # Sticky active_profile pointer (no HERMES_PROFILE env in this test).
    (home / "active_profile").write_text("pm\n")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    # Import / reload the relevant modules so the module-level
    # ``_hermes_home`` picks up the fake HERMES_HOME.
    import hermes_constants
    importlib.reload(hermes_constants)
    import hermes_cli.profiles as profiles
    importlib.reload(profiles)
    import gateway.run as gateway_run
    importlib.reload(gateway_run)

    # Also poke the module-level cached path the loader actually reads.
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    return home, gateway_run


def test_profile_disabled_toolsets_merged_into_gateway_config(fake_hermes_home):
    _, gateway_run = fake_hermes_home
    cfg = gateway_run._load_gateway_config()
    assert isinstance(cfg, dict)
    # Profile override wins for disabled_toolsets.
    assert cfg["agent"]["disabled_toolsets"] == ["browser", "computer_use"]
    # Other keys from the global config survive the deep-merge.
    assert cfg["agent"]["model"] == "global-model"
    assert cfg["platforms"]["foo"] is True


def test_no_active_profile_falls_back_to_global(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"disabled_toolsets": ["only_global"]}})
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_PROFILE", raising=False)

    import hermes_constants
    importlib.reload(hermes_constants)
    import hermes_cli.profiles as profiles
    importlib.reload(profiles)
    import gateway.run as gateway_run
    importlib.reload(gateway_run)
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    cfg = gateway_run._load_gateway_config()
    assert cfg["agent"]["disabled_toolsets"] == ["only_global"]


def test_hermes_profile_env_overrides_active_profile_file(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    (home / "profiles" / "coder").mkdir(parents=True)
    (home / "profiles" / "pm").mkdir(parents=True)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"disabled_toolsets": []}})
    )
    (home / "profiles" / "coder" / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"disabled_toolsets": ["from_coder"]}})
    )
    (home / "profiles" / "pm" / "config.yaml").write_text(
        yaml.safe_dump({"agent": {"disabled_toolsets": ["from_pm"]}})
    )
    (home / "active_profile").write_text("pm\n")

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "coder")

    import hermes_constants
    importlib.reload(hermes_constants)
    import hermes_cli.profiles as profiles
    importlib.reload(profiles)
    import gateway.run as gateway_run
    importlib.reload(gateway_run)
    monkeypatch.setattr(gateway_run, "_hermes_home", home)

    cfg = gateway_run._load_gateway_config()
    # HERMES_PROFILE env wins over the active_profile file.
    assert cfg["agent"]["disabled_toolsets"] == ["from_coder"]
