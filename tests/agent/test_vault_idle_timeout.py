"""Profile-scoped, finite idle expiry without real manager credentials."""
from __future__ import annotations

import json

import pytest

from agent.vault_backends import unlock
from hermes_constants import get_hermes_home


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(unlock.time, "monotonic", lambda: now[0])
    unlock.lock()
    unlock.set_current_session_id(None)
    yield now
    unlock.lock()
    unlock.set_current_session_id(None)


def configure(value):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(json.dumps({
        "vault": {"bitwarden": {"idle_timeout_seconds": value}}
    }), encoding="utf-8")


def test_configured_timeout_expires_after_four_hours(clock):
    configure(14400)
    unlock.store_session_token("bitwarden", "synthetic")
    clock[0] += 1801
    assert unlock.is_unlocked("bitwarden")
    clock[0] += 12599
    assert unlock.is_unlocked("bitwarden")
    clock[0] += 1
    assert not unlock.is_unlocked("bitwarden")


def test_manager_use_refreshes_idle_but_status_does_not(clock):
    configure(14400)
    unlock.store_session_token("bitwarden", "synthetic")
    clock[0] += 14000
    assert unlock.get_session_token("bitwarden") == "synthetic"
    clock[0] += 14000
    assert unlock.is_unlocked("bitwarden")
    clock[0] += 401
    assert not unlock.is_unlocked("bitwarden")


@pytest.mark.parametrize("value", [None, 0, -1, True, False, "Never", "14400", 1.5, float("inf"), float("nan"), 86401, {}, []])
def test_invalid_timeout_keeps_finite_default(clock, value):
    configure(value)
    unlock.store_session_token("bitwarden", "synthetic")
    clock[0] += 1801
    assert not unlock.is_unlocked("bitwarden")


def test_other_backend_and_profile_keep_default(clock, monkeypatch, tmp_path):
    configure(14400)
    unlock.store_session_token("bitwarden", "synthetic")
    unlock.store_session_token("onepassword", "synthetic")
    clock[0] += 1801
    assert unlock.is_unlocked("bitwarden")
    assert not unlock.is_unlocked("onepassword")
    with monkeypatch.context() as scope:
        scope.setenv("HERMES_HOME", str(tmp_path / "other-profile"))
        unlock.store_session_token("bitwarden", "synthetic-other")
        clock[0] += 1801
        assert not unlock.is_unlocked("bitwarden")
    assert unlock.is_unlocked("bitwarden")


def test_manual_lock_and_owner_release_still_win(clock):
    configure(14400)
    unlock.set_current_session_id("owner")
    unlock.store_session_token("bitwarden", "synthetic")
    unlock.release_session("unrelated")
    assert unlock.is_unlocked("bitwarden")
    unlock.release_session("owner")
    assert not unlock.is_unlocked("bitwarden")
    unlock.store_session_token("bitwarden", "synthetic")
    unlock.lock("bitwarden")
    assert not unlock.is_unlocked("bitwarden")
