"""The holographic fact DB set up with the default path belongs to whichever profile opens it.

Setup used to offer, and store, the active profile's concrete path (``~/.hermes/memory_store.db``).
``profile create --clone`` copies config.yaml verbatim, so the clone opened the source profile's
facts; ``profile rename`` moves the directory, so the renamed profile opened an empty DB in a
re-created directory under its old name.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

import hermes_cli.memory_setup as memory_setup
from hermes_cli.profiles import create_profile, rename_profile
from plugins.memory.holographic import HolographicMemoryProvider, _load_plugin_config


@pytest.fixture()
def root(tmp_path, monkeypatch):
    # A ``~/`` path is displayed through Path.home() and expanded through HOME; both name the sandbox.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _setup_with_defaults(monkeypatch, home: Path) -> dict:
    """``hermes memory setup`` for holographic with Enter on every field, saved as ``cmd_setup`` does."""
    monkeypatch.setenv("HERMES_HOME", str(home))
    provider = HolographicMemoryProvider(config={"hrr_dim": 64})
    monkeypatch.setattr(memory_setup, "_curses_select", lambda *_a, default=0, **_k: default)
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n" * 10))
    answers: dict = {}
    assert memory_setup._prompt_schema_fields("holographic", provider.get_config_schema(), answers, {})
    provider.save_config(answers, str(home))
    return answers


def _opened_db(monkeypatch, home: Path) -> Path:
    """The DB a session started with ``hermes -p <profile>`` opens."""
    monkeypatch.setenv("HERMES_HOME", str(home))
    provider = HolographicMemoryProvider(config=_load_plugin_config())
    provider.initialize("s")
    try:
        return provider._store.db_path.resolve()
    finally:
        provider.shutdown()


@pytest.mark.parametrize("lifecycle", ["clone", "rename"])
def test_default_db_path_follows_the_profile_that_opens_it(root, monkeypatch, lifecycle):
    if lifecycle == "clone":
        answers = _setup_with_defaults(monkeypatch, root)
        monkeypatch.setenv("HERMES_HOME", str(root))
        profile = create_profile("work", clone_config=True, no_alias=True)
    else:
        source = create_profile("alpha", no_alias=True)
        answers = _setup_with_defaults(monkeypatch, source)
        monkeypatch.setenv("HERMES_HOME", str(root))
        with patch("hermes_cli.profiles.check_alias_collision", return_value="skip"):
            profile = rename_profile("alpha", "beta")

    assert _opened_db(monkeypatch, profile) == (profile / "memory_store.db").resolve()
    if lifecycle == "rename":
        assert not (root / "profiles" / "alpha").exists()
    # The prompt offers the value it stores, so what the user accepted is what config.yaml says.
    assert answers["db_path"] == "$HERMES_HOME/memory_store.db"


@pytest.mark.parametrize("spelling", ["display", "absolute", "elsewhere"])
def test_save_config_stores_only_this_profiles_default_path_as_the_placeholder(root, monkeypatch, spelling):
    # "display"/"absolute": the concrete value an older setup wrote, and the dashboard form re-submits on
    # its next save. "elsewhere": a deliberately shared store (#4726) is kept as given.
    profile = create_profile("work", no_alias=True)
    monkeypatch.setenv("HERMES_HOME", str(profile))
    given = {
        "display": "~/.hermes/profiles/work/memory_store.db",
        "absolute": str(profile / "memory_store.db"),
        "elsewhere": str(root / "shared" / "facts.db"),
    }[spelling]
    expected = given if spelling == "elsewhere" else "$HERMES_HOME/memory_store.db"

    HolographicMemoryProvider(config={"hrr_dim": 64}).save_config({"db_path": given, "hrr_dim": "64"}, str(profile))

    stored = yaml.safe_load((profile / "config.yaml").read_text(encoding="utf-8"))["plugins"]["hermes-memory-store"]
    assert stored == {"db_path": expected, "hrr_dim": "64"}
