"""Gateway startup names every served profile whose config still declares a retired timed reset."""
import contextlib
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.run_startup import GatewayStartupMixin
from hermes_cli import session_reset_retirement


@pytest.mark.parametrize("plugin_enabled", [False, True])
def test_startup_warns_on_timed_session_reset_unless_plugin_enabled(tmp_path, monkeypatch, caplog, plugin_enabled):
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("session_reset:\n  mode: idle\n  idle_minutes: 60\n", encoding="utf-8")
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda multiplex: [("default", home)])
    monkeypatch.setattr(session_reset_retirement, "reset_plugin_enabled", lambda: plugin_enabled)
    runner = object.__new__(GatewayStartupMixin)
    runner.config = SimpleNamespace(multiplex_profiles=False)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._start_log_retired_session_reset()

    warned = [r.getMessage() for r in caplog.records if "session_reset.mode: idle" in r.getMessage()]
    assert (warned == []) is plugin_enabled
    if not plugin_enabled:
        assert session_reset_retirement.PLUGIN_NAME in warned[0]


def test_only_timed_modes_count_as_retired_policy():
    find = session_reset_retirement.retired_reset_policy
    assert find({"session_reset": {"mode": "both"}}) == ("session_reset", "both")
    assert find({"gateway": {"session_reset": {"mode": "Daily"}}}) == ("gateway.session_reset", "daily")
    assert find({"session_reset": {"mode": "none"}}) is None
    assert find({"session_reset": {"idle_minutes": 60}}) is None
    assert find({}) is None


def test_startup_checks_the_plugin_in_each_served_profile_scope(tmp_path, monkeypatch, caplog):
    """#137010: a routed profile dispatches ``pre_gateway_dispatch`` through its OWN plugin manager,
    so a plugin enabled only on the host must not silence the warning for the profiles that lack it."""
    root, routed = tmp_path / "root", tmp_path / "profiles" / "b"
    for home in (root, routed):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("session_reset:\n  mode: both\n", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", root), ("b", routed)],
    )
    checked = []

    @contextlib.contextmanager
    def fake_scope(profile_home, *args, **kwargs):
        checked.append(Path(profile_home))
        yield

    monkeypatch.setattr("gateway.run._profile_runtime_scope", fake_scope)
    # The plugin is installed on the host profile only, never in the routed profile's plugin set.
    monkeypatch.setattr(
        session_reset_retirement, "reset_plugin_enabled",
        lambda: bool(checked) and checked[-1] == root,
    )
    runner = object.__new__(GatewayStartupMixin)
    runner.config = SimpleNamespace(multiplex_profiles=True)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._start_log_retired_session_reset()

    warned = [r.getMessage() for r in caplog.records if "session_reset.mode: both" in r.getMessage()]
    assert checked == [root, routed], "the plugin check must run in each profile's own scope"
    assert len(warned) == 1 and warned[0].startswith("Profile b: ")


def test_startup_does_not_name_a_profile_whose_own_plugin_set_has_it(tmp_path, monkeypatch, caplog):
    """Mirror case of #137010 (reporter's follow-up, 2026-10-16): the plugin installed on the
    *routed* profile honours that profile's resets, so the host's plugin set must not decide the
    notice for it — the false "install this plugin" line came from evaluating it in host scope."""
    root, routed = tmp_path / "root", tmp_path / "profiles" / "b"
    for home in (root, routed):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text("session_reset:\n  mode: both\n", encoding="utf-8")
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda multiplex: [("default", root), ("b", routed)],
    )
    checked = []

    @contextlib.contextmanager
    def fake_scope(profile_home, *args, **kwargs):
        checked.append(Path(profile_home))
        yield

    monkeypatch.setattr("gateway.run._profile_runtime_scope", fake_scope)
    # The plugin lives in the routed profile's scope only; the host has no copy.
    monkeypatch.setattr(
        session_reset_retirement, "reset_plugin_enabled",
        lambda: bool(checked) and checked[-1] == routed,
    )
    runner = object.__new__(GatewayStartupMixin)
    runner.config = SimpleNamespace(multiplex_profiles=True)

    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        runner._start_log_retired_session_reset()

    warned = [r.getMessage() for r in caplog.records if "session_reset.mode: both" in r.getMessage()]
    assert checked == [root, routed], "the plugin check must run in each profile's own scope"
    assert len(warned) == 1 and warned[0].startswith("Profile default: ")
