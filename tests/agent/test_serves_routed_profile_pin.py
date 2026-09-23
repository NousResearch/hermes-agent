"""A host that mirrors the turn's profile into HERMES_HOME must not flip any launch-home decision.

Hermes WebUI serves several profiles from one process and, for legacy readers, mirrors the active
turn's profile into ``os.environ["HERMES_HOME"]`` while also installing the context-local override.
Without a pinned process home the override then equals the "process" home and all four launch-home
decisions flip: serves_routed_profile(), _is_routed_home(), _is_process_home(), and
env_loader._process_hermes_home() all read get_process_hermes_home() which follows the env var.
"""
from __future__ import annotations

import pytest

import hermes_constants
from agent.secret_scope import serves_routed_profile, set_multiplex_active


@pytest.fixture(autouse=True)
def _reset_pin(monkeypatch):
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", None)
    yield
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HERMES_HOME", None)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    served = tmp_path / "profiles" / "served"
    launch.mkdir()
    served.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    return launch, served


def _with_override(path, fn):
    token = hermes_constants.set_hermes_home_override(path)
    try:
        return fn()
    finally:
        hermes_constants.reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# serves_routed_profile
# ---------------------------------------------------------------------------

def test_unpinned_behaviour_is_unchanged(homes, monkeypatch):
    launch, served = homes
    assert _with_override(served, serves_routed_profile) is True
    assert _with_override(launch, serves_routed_profile) is False
    # Mirroring the served home into HERMES_HOME makes it the process home (legacy semantics).
    monkeypatch.setenv("HERMES_HOME", str(served))
    assert _with_override(served, serves_routed_profile) is False


def test_pinned_home_survives_a_mirrored_hermes_home(homes, monkeypatch):
    launch, served = homes
    hermes_constants.pin_process_hermes_home(launch)
    monkeypatch.setenv("HERMES_HOME", str(served))  # the host's per-turn mirror
    assert _with_override(served, serves_routed_profile) is True
    assert _with_override(launch, serves_routed_profile) is False
    assert serves_routed_profile() is False  # no override: the process's own profile
    assert hermes_constants.get_process_hermes_home() == launch
    assert hermes_constants.get_routing_process_hermes_home() == launch  # alias


def test_clearing_the_pin_restores_hermes_home_semantics(homes, monkeypatch):
    launch, served = homes
    hermes_constants.pin_process_hermes_home(launch)
    hermes_constants.pin_process_hermes_home(None)
    monkeypatch.setenv("HERMES_HOME", str(served))
    assert hermes_constants.get_process_hermes_home() == served
    assert _with_override(served, serves_routed_profile) is False


def test_first_pin_wins(homes):
    launch, served = homes
    hermes_constants.pin_process_hermes_home(launch)
    hermes_constants.pin_process_hermes_home(served)  # ignored — first pin wins
    assert hermes_constants.get_process_hermes_home() == launch


def test_mcp_connection_key_is_profile_scoped_under_a_mirrored_home(homes, monkeypatch):
    from tools.mcp_tool_scope import _server_key
    from tools.registry import registry

    launch, served = homes
    hermes_constants.pin_process_hermes_home(launch)
    monkeypatch.setenv("HERMES_HOME", str(served))
    key = _with_override(served, lambda: _server_key("atlassian"))
    assert key == (hermes_constants.hermes_home_key(served), "atlassian")
    assert _with_override(served, registry.current_scope_key) == key[0]
    assert _with_override(launch, lambda: _server_key("atlassian")) == "atlassian"


# ---------------------------------------------------------------------------
# set_multiplex_active auto-pins the launch home
# ---------------------------------------------------------------------------

def test_set_multiplex_active_pins_launch_home(homes, monkeypatch):
    from agent.secret_scope import is_multiplex_active

    launch, served = homes  # HERMES_HOME == launch
    assert hermes_constants._PINNED_PROCESS_HERMES_HOME is None
    set_multiplex_active(True)
    try:
        assert hermes_constants._PINNED_PROCESS_HERMES_HOME == str(launch)
        # Now mirror served into HERMES_HOME — pin must hold
        monkeypatch.setenv("HERMES_HOME", str(served))
        assert hermes_constants.get_process_hermes_home() == launch
    finally:
        set_multiplex_active(False)


def test_set_multiplex_active_respects_explicit_pin(homes, monkeypatch):
    launch, served = homes  # HERMES_HOME == launch
    # Embedding host pins before multiplex is activated.
    hermes_constants.pin_process_hermes_home(launch)
    monkeypatch.setenv("HERMES_HOME", str(served))
    set_multiplex_active(True)
    try:
        # Pin should still be launch, not served.
        assert hermes_constants.get_process_hermes_home() == launch
    finally:
        set_multiplex_active(False)


# ---------------------------------------------------------------------------
# The three other launch-home decisions that were also broken (#119242)
# ---------------------------------------------------------------------------

def test_is_routed_home_respects_pin(homes, monkeypatch):
    from tools.environments.local import _is_routed_home

    launch, served = homes
    hermes_constants.pin_process_hermes_home(launch)
    monkeypatch.setenv("HERMES_HOME", str(served))
    assert _is_routed_home(served) is True   # served != pinned launch
    assert _is_routed_home(launch) is False  # launch == pinned launch


def test_is_process_home_respects_pin(homes, monkeypatch):
    from agent.secret_scope import _is_process_home

    launch, served = homes
    hermes_constants.pin_process_hermes_home(launch)
    monkeypatch.setenv("HERMES_HOME", str(served))
    assert _is_process_home(launch) is True   # launch == pinned
    assert _is_process_home(served) is False  # served != pinned


def test_env_loader_process_hermes_home_respects_pin(homes, monkeypatch):
    from hermes_cli.env_loader import _process_hermes_home

    launch, served = homes
    hermes_constants.pin_process_hermes_home(launch)
    monkeypatch.setenv("HERMES_HOME", str(served))
    assert _process_hermes_home() == launch
