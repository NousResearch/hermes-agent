"""Profile create/rename/delete ask the multiplexer to serve a profile only after the lease.

Serving a profile opens its state.db on the gateway's event loop, and a cold SessionDB open
takes that profile's lifecycle lease. Notifying while still holding the lease froze every
adapter until the rescan handler's 5 s wait gave up, and a rename's identity migration timed
out and printed a false "restart the gateway" warning (PR #93508 review).
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from hermes_cli import gateway_multiplex_served, profile_lifecycle, profiles


@pytest.fixture
def profile_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    # Never inspect or stop services and processes outside this disposable fixture.
    monkeypatch.setattr(profiles, "_check_gateway_running", lambda *_: False)
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service",
                 "_maybe_register_gateway_service", "_stop_profile_backends", "_stop_bot_desktop"):
        monkeypatch.setattr(profiles, name, lambda *_: None)
    monkeypatch.setattr(profile_lifecycle, "external_profile_file_holders", lambda *_: [])
    return home


def _lease_free(home: Path) -> bool:
    """Whether another thread (standing in for the gateway's cold SessionDB open) gets the lease now."""
    result: list[bool] = []

    def probe() -> None:
        try:
            with profile_lifecycle.profile_lifecycle_lease(home, timeout=0):
                result.append(True)
        except TimeoutError:
            result.append(False)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return result[0]


@pytest.fixture
def served(profile_env, monkeypatch):
    """Each multiplexer notification: (profile, whether its lease was free when it was sent)."""
    calls: list[tuple[str, bool]] = []

    def notify(name: str, **_kwargs):
        calls.append((name, _lease_free(profiles.get_profile_dir(name))))

    monkeypatch.setattr(gateway_multiplex_served, "notify_multiplexer_profiles_changed", notify)
    return calls


def test_create_notifies_the_multiplexer_after_releasing_the_lease(served):
    profiles.create_profile("coder", no_alias=True, no_skills=True)

    assert served == [("coder", True)]


def test_rename_migrates_identity_and_serves_the_new_name_after_releasing_the_lease(served, monkeypatch):
    from gateway import control_socket

    profiles.create_profile("coder", no_alias=True, no_skills=True)
    served.clear()
    monkeypatch.setattr(profiles, "_live_default_multiplexer", lambda: True)
    migrations: list[bool] = []

    def migrate(_root, old, new, **_kwargs):
        migrations.append(_lease_free(profiles.get_profile_dir(new)))
        return {"ok": True}

    monkeypatch.setattr(control_socket, "migrate_gateway_profile_identity", migrate)

    profiles.rename_profile("coder", "dev")

    assert migrations == [True]
    # The old name is unrouted under the lease (that needs no store open); the new one is
    # served after the identity migration, outside it.
    assert served[-1] == ("dev", True)


def test_a_refused_delete_reserves_the_profile_after_releasing_the_lease(served, monkeypatch):
    profile_dir = profiles.create_profile("coder", no_alias=True, no_skills=True)
    served.clear()

    def refuse(*_args):
        raise RuntimeError("a backend would not stop")

    monkeypatch.setattr(profiles, "_stop_profile_backends", refuse)

    with pytest.raises(RuntimeError):
        profiles.delete_profile("coder", yes=True)

    assert profile_dir.is_dir()
    assert served == [("coder", True)]
