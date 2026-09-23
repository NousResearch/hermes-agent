"""The launch home is frozen once the process serves several profiles (#119242).

Every "does this task serve a ROUTED home" decision (``agent.secret_scope.serves_routed_profile``,
``_is_process_home``, ``tools.environments.local._is_routed_home``,
``hermes_cli.env_loader._process_hermes_home``) compares the task's home override with
``get_process_hermes_home()``. That used to read ``os.environ["HERMES_HOME"]`` live, so a host that
mirrors the served profile into the env on every turn (Hermes WebUI does) made every served
profile look like the launch one: MCP connections fell back to bare cross-profile names, the launch
residue survived ``strip_launch_profile_env``, the launch profile's bridged grants seeded the
served scope. ``set_multiplex_active(True)`` now pins the launch home; a later env mutation cannot
re-label it. Standalone ``hermes -p x gateway run`` (multiplex inactive) keeps following the env.
"""
from __future__ import annotations

import pytest

import hermes_constants
from agent.secret_scope import _is_process_home, serves_routed_profile, set_multiplex_active
from hermes_cli.env_loader import _process_hermes_home
from hermes_constants import get_process_hermes_home, reset_hermes_home_override, set_hermes_home_override
from tools.environments.local import _is_routed_home


@pytest.fixture
def homes(tmp_path, monkeypatch):
    launch = tmp_path / "launch"
    served = tmp_path / "profiles" / "served"
    launch.mkdir()
    served.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(hermes_constants, "_PINNED_PROCESS_HOME", None, raising=False)
    return launch, served


def test_a_per_turn_env_mirror_cannot_relabel_the_launch_home_under_multiplex(homes, monkeypatch):
    launch, served = homes
    set_multiplex_active(True)
    monkeypatch.setenv("HERMES_HOME", str(served))  # the host's per-turn mirror
    token = set_hermes_home_override(served)
    try:
        assert get_process_hermes_home() == launch
        assert _process_hermes_home() == launch
        assert _is_routed_home(served) and not _is_routed_home(launch)
        assert not _is_process_home(served) and _is_process_home(launch)
        assert serves_routed_profile()
    finally:
        reset_hermes_home_override(token)
    set_multiplex_active(False)
    # Pin released with the mode: the env is authoritative again.
    assert get_process_hermes_home() == served


def test_a_standalone_profile_process_keeps_following_its_env(homes, monkeypatch):
    """T1 (``hermes -p x gateway run``): multiplex inactive, the env IS the profile — an override
    naming that same home is not routed, and a later env change is followed."""
    launch, served = homes
    monkeypatch.setenv("HERMES_HOME", str(served))
    token = set_hermes_home_override(served)
    try:
        assert get_process_hermes_home() == served
        assert not _is_routed_home(served)
        assert not serves_routed_profile()
    finally:
        reset_hermes_home_override(token)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    assert get_process_hermes_home() == launch
