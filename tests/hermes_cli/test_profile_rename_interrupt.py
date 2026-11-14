"""An interrupted profile rename leaves exactly one identity: the old one before the move, the new one after.

Under a live multiplexer the rename tombstones and unroutes the old name, waits for the gateway to
release the home, then moves the directory. Ctrl-C in that wait used to skip the undo (it only ran on
``Exception``), leaving the untouched profile hidden behind its tombstone: absent from ``profile list``
and ``-p``, unserved. Once the move has landed, the undo must not run: it would re-serve and
re-register the old name, whose directory is gone.
"""
import os
from pathlib import Path

import pytest

from hermes_cli import profiles
from hermes_constants import named_profile_is_deleted


@pytest.fixture
def mux(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []
    for name in ("_cleanup_gateway_service", "_maybe_unregister_gateway_service", "_stop_gateway_process",
                 "_stop_bot_desktop"):
        monkeypatch.setattr(profiles, name, lambda *_a, **_k: None)
    monkeypatch.setattr(profiles, "_notify_multiplexer", lambda name: calls.append(("notify", name)))
    monkeypatch.setattr(profiles, "_maybe_register_gateway_service", lambda name: calls.append(("register", name)))
    monkeypatch.setattr(profiles, "check_alias_collision", lambda _name: "skip")
    old_dir = profiles.create_profile("before", no_alias=True)
    monkeypatch.setattr(profiles, "_live_default_multiplexer", lambda: True)
    calls.clear()
    return old_dir, calls


def _listed():
    return {p.name for p in profiles.list_profiles()} - {"default"}


def _interrupt(*_a, **_k):
    raise KeyboardInterrupt


def _racing_creator(old_dir, monkeypatch):
    """Another writer publishes a populated ``profiles/after`` just before the move, which then fails."""
    real_wait = profiles.wait_profile_unowned

    def wait(path, *a, **k):
        (old_dir.parent / "after").mkdir()
        (old_dir.parent / "after" / "config.yaml").write_text("{}\n", encoding="utf-8")
        return real_wait(path, *a, **k)

    monkeypatch.setattr(profiles, "wait_profile_unowned", wait)
    return OSError


@pytest.mark.parametrize("fault", ["interrupt_in_unroute_notify", "interrupt_in_owner_wait",
                                   "racing_creator_at_new_name"])
def test_rename_stopped_before_the_move_restores_the_old_profile(mux, monkeypatch, fault):
    old_dir, calls = mux
    expected = KeyboardInterrupt
    if fault == "interrupt_in_unroute_notify":  # the rescan request can wait up to 8 s on the gateway
        def notify(name):
            calls.append(("notify", name))
            if len(calls) == 1:
                raise KeyboardInterrupt
        monkeypatch.setattr(profiles, "_notify_multiplexer", notify)
    elif fault == "interrupt_in_owner_wait":
        monkeypatch.setattr(profiles, "wait_profile_unowned", _interrupt)
    else:
        expected = _racing_creator(old_dir, monkeypatch)
    with pytest.raises(expected):
        profiles.rename_profile("before", "after")
    assert old_dir.is_dir() and not named_profile_is_deleted(old_dir)
    assert "before" in _listed() and profiles.profile_exists("before")
    assert calls == [("notify", "before"), ("notify", "before"), ("register", "before")]  # unroute, re-serve


def test_interrupt_after_the_move_never_revives_the_old_name(mux, monkeypatch):
    old_dir, calls = mux

    def rename_then_interrupt(self, target):
        os.rename(self, target)
        raise KeyboardInterrupt  # Ctrl-C delivered as the move returns

    monkeypatch.setattr(Path, "rename", rename_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        profiles.rename_profile("before", "after")
    new_dir = old_dir.parent / "after"
    assert not old_dir.exists() and new_dir.is_dir()
    assert _listed() == {"after"}
    assert not named_profile_is_deleted(old_dir) and not named_profile_is_deleted(new_dir)
    assert calls == [("notify", "before")]  # the unroute only: no re-serve or re-register of "before"
