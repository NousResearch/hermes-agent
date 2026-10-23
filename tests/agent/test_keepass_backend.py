"""KeePass (.kdbx) login backend for the browser vault.

Contracts: a locked database lists nothing; the master password opens a real database built at test
time with pykeepass and never leaves the process (no child, no log line, no file); handles are
``kp:<entry uuid>``, never the title; ``resolve_password`` returns exactly the stored password; the
database relocks after ``relock_after_minutes`` without use; a configured database is wired through
the real registry and the browser_vault unlock wrapper.
"""

from __future__ import annotations

import json
import logging
import subprocess
from unittest.mock import patch

import pytest

pytest.importorskip("pykeepass")

from pykeepass import create_database

from agent.vault_backends import keepass as keepass_mod
from agent.vault_backends import unlock as unlock_mod
from agent.vault_backends.base import UnlockRequired
from agent.vault_backends.keepass import KeepassLoginBackend

MASTER = "correct horse battery staple"
SECRET = "plain sentence nobody would flag 7"


@pytest.fixture(autouse=True)
def _locked():
    unlock_mod.lock()
    keepass_mod._open.clear()
    yield
    unlock_mod.lock()
    keepass_mod._open.clear()


@pytest.fixture
def kdbx(tmp_path):
    path = tmp_path / "personal.kdbx"
    kp = create_database(str(path), password=MASTER)
    web = kp.add_entry(kp.root_group, "Example", "jane@example.com", SECRET, url="https://example.com/login")
    offline = kp.add_entry(kp.root_group, "Router", "admin", "router-pass")
    kp.add_entry(kp.root_group, "Bookmark only", "nobody", "", url="https://bookmark.example")
    kp.save()
    return path, str(web.uuid), str(offline.uuid)


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(keepass_mod.time, "monotonic", lambda: now[0])
    return now


def _backend(path, **cfg):
    return KeepassLoginBackend({"enabled": True, "database": str(path), **cfg})


def test_locked_database_lists_nothing_and_refuses_to_resolve(kdbx):
    path, web_id, _ = kdbx
    backend = _backend(path)
    assert backend.name == "keepass" and backend.prefix == "kp:" and backend.needs_unlock
    assert not backend.is_unlocked()
    assert backend.list_items() == []
    assert backend.get_meta(f"kp:{web_id}") is None
    with pytest.raises(UnlockRequired):
        backend.resolve_password(f"kp:{web_id}")


def test_wrong_master_raises_and_stays_locked(kdbx):
    path, _, _ = kdbx
    backend = _backend(path)
    with pytest.raises(RuntimeError, match="wrong master password") as exc:
        backend.unlock("not the password")
    assert "not the password" not in str(exc.value)
    assert not backend.is_unlocked()
    assert backend.list_items() == []


def test_unlock_lists_logins_by_uuid_and_resolves_the_exact_password(kdbx, tmp_path, caplog):
    path, web_id, offline_id = kdbx
    before = sorted(p.name for p in tmp_path.rglob("*"))
    caplog.set_level(logging.DEBUG)
    with patch.object(subprocess, "Popen", side_effect=AssertionError("no child process may be spawned")):
        _backend(path).unlock(MASTER)
        backend = _backend(path)  # a fresh instance per call, exactly as enabled_backends builds them
        assert backend.is_unlocked()
        metas = {m.id: m for m in backend.list_items()}

        assert set(metas) == {f"kp:{web_id}", f"kp:{offline_id}"}, "an entry with no password is never listed"
        web, offline = metas[f"kp:{web_id}"], metas[f"kp:{offline_id}"]
        assert (web.kind, web.label, web.origin, web.identifier, web.identifier_type) == (
            "login", "Example", "https://example.com", "jane@example.com", "email")
        assert web.allowed_origins == ("https://example.com",)
        assert (offline.label, offline.origin, offline.identifier_type) == ("Router", None, "username")
        assert backend.get_meta(f"kp:{web_id}") == web

        assert backend.resolve_password(f"kp:{web_id}") == SECRET
        assert SECRET not in json.dumps([m.to_dict() for m in metas.values()])
    assert MASTER not in caplog.text and SECRET not in caplog.text
    assert sorted(p.name for p in tmp_path.rglob("*")) == before, "nothing is written to disk"


def test_unknown_handle_raises(kdbx):
    path, _, _ = kdbx
    backend = _backend(path)
    backend.unlock(MASTER)
    for handle in ("kp:00000000-0000-0000-0000-000000000000", "kp:Example", "kp:"):
        with pytest.raises(RuntimeError, match="not found"):
            backend.resolve_password(handle)
        assert backend.get_meta(handle) is None


def test_relocks_after_the_configured_idle_window(kdbx, clock):
    path, web_id, _ = kdbx
    backend = _backend(path, relock_after_minutes=5)
    backend.unlock(MASTER)

    clock[0] += 4 * 60
    assert backend.resolve_password(f"kp:{web_id}") == SECRET  # use resets the window
    clock[0] += 4 * 60
    assert backend.is_unlocked()

    clock[0] += 2 * 60
    assert not backend.is_unlocked()
    assert backend.list_items() == []
    with pytest.raises(UnlockRequired):
        backend.resolve_password(f"kp:{web_id}")
    assert keepass_mod._open == {}, "the decrypted database is dropped, not just hidden"


def test_lock_from_the_shared_registry_drops_the_database(kdbx):
    path, _, _ = kdbx
    backend = _backend(path)
    backend.unlock(MASTER)
    unlock_mod.lock("keepass")  # what Settings → Lock and vault.source.set(enabled=False) call
    assert keepass_mod._open == {}, "a registry lock must drop the decrypted db now, not on next use"
    assert not backend.is_unlocked()
    assert backend.list_items() == []


def test_explicit_lock_drops_the_database_without_a_follow_up_access(kdbx):
    """The registry lock alone is lazy; the backend's own lock must release the decrypted db now."""
    path, _, _ = kdbx
    backend = _backend(path)
    backend.unlock(MASTER)
    assert keepass_mod._open != {}
    backend.lock()
    assert keepass_mod._open == {}, "lock() must drop the decrypted database, not defer it"


def test_entry_moved_to_the_recycle_bin_is_never_listed(tmp_path):
    path = tmp_path / "bin.kdbx"
    kp = create_database(str(path), password=MASTER)
    kp.add_entry(kp.root_group, "Live", "jane", SECRET, url="https://live.example")
    binned = kp.add_entry(kp.root_group, "Binned", "old", "stale-pass", url="https://binned.example")
    kp.trash_entry(binned)  # creates the recycle bin the first time
    kp.save()

    backend = _backend(path)
    backend.unlock(MASTER)
    labels = [m.label for m in backend.list_items()]
    assert labels == ["Live"]
    with pytest.raises(RuntimeError, match="not found"):
        backend.resolve_password(f"kp:{binned.uuid}")


def test_entry_without_a_scheme_lists_but_cannot_be_a_fill_target(tmp_path):
    path = tmp_path / "schemeless.kdbx"
    kp = create_database(str(path), password=MASTER)
    entry = kp.add_entry(kp.root_group, "No scheme", "jane", SECRET, url="example.com/login")
    kp.save()

    backend = _backend(path)
    backend.unlock(MASTER)
    [meta] = backend.list_items()
    assert meta.id == f"kp:{entry.uuid}"
    assert meta.origin is None and meta.allowed_origins == ()


def test_switching_the_configured_database_does_not_serve_the_previous_one(tmp_path):
    """A config change to another .kdbx must not keep resolving the old database's entries."""
    first = tmp_path / "first.kdbx"
    kp = create_database(str(first), password=MASTER)
    kp.add_entry(kp.root_group, "Old", "jane", SECRET, url="https://old.example")
    kp.save()

    second = tmp_path / "second.kdbx"
    kp2 = create_database(str(second), password=MASTER)
    new = kp2.add_entry(kp2.root_group, "New", "jane", "new-pass", url="https://new.example")
    kp2.save()

    backend = _backend(first)
    backend.unlock(MASTER)
    assert [m.label for m in backend.list_items()] == ["Old"]

    switched = _backend(second)  # same profile, different file, as enabled_backends rebuilds it
    assert switched.is_unlocked() is False
    assert switched.list_items() == []
    switched.unlock(MASTER)
    assert [m.label for m in switched.list_items()] == ["New"]
    assert switched.resolve_password(f"kp:{new.uuid}") == "new-pass"


def test_the_idle_window_drops_the_database_without_a_follow_up_call(kdbx):
    """The relock timer is the contract: no later call is needed for the decrypted db to go away."""
    from tests.waiting import wait_until

    path, _, _ = kdbx
    backend = _backend(path, relock_after_minutes=0.02)  # 1.2 s
    backend.unlock(MASTER)
    assert keepass_mod._open != {}
    wait_until(lambda: keepass_mod._open == {}, timeout=5.0,
               what="the relock timer to drop the decrypted database")
    assert unlock_mod.is_unlocked("keepass") is False


def test_the_timer_retires_only_the_profile_that_owns_the_database(kdbx, tmp_path, monkeypatch):
    """The relock callback runs on a bare thread with no HERMES_HOME: it must target its own slot."""
    from tests.waiting import wait_until

    path, _, _ = kdbx
    home_a, home_b = tmp_path / "a", tmp_path / "b"
    home_a.mkdir()
    home_b.mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    monkeypatch.setattr(keepass_mod, "_home", lambda: str(home_a))
    _backend(path, relock_after_minutes=0.02).unlock(MASTER)

    # By the time the timer fires this process looks like profile B.
    monkeypatch.setenv("HERMES_HOME", str(home_b))
    monkeypatch.setattr(keepass_mod, "_home", lambda: str(home_b))
    wait_until(lambda: keepass_mod._open == {}, timeout=5.0, what="the timer to drop the database")

    monkeypatch.setenv("HERMES_HOME", str(home_a))
    assert unlock_mod.is_unlocked("keepass") is False, "the owning profile's session must be retired"


def test_a_malformed_relock_knob_does_not_break_the_locked_state_or_other_backends(kdbx):
    path, _, _ = kdbx
    backend = _backend(path, relock_after_minutes="fifteen")
    assert backend.is_unlocked() is False, "a typo in an unused knob must not raise while locked"
    assert backend.list_items() == []
    backend.unlock(MASTER)
    assert backend.is_unlocked() is True, "the status probe must not raise for the other backends' sake"
    with pytest.raises(RuntimeError, match="relock_after_minutes"):
        backend.list_items()



def test_keyfile_only_database_opens_without_a_password(tmp_path):
    keyfile = tmp_path / "personal.keyx"
    keyfile.write_bytes(b"0123456789abcdef" * 4)
    path = tmp_path / "keyfile-only.kdbx"
    kp = create_database(str(path), keyfile=str(keyfile))
    kp.add_entry(kp.root_group, "Example", "jane", SECRET, url="https://example.com")
    kp.save()

    backend = _backend(path, keyfile=str(keyfile))
    backend.unlock("")
    [meta] = backend.list_items()
    assert backend.resolve_password(meta.id) == SECRET


def test_an_empty_master_reaches_a_keyfile_only_backend_through_the_unlock_wrapper(tmp_path, monkeypatch):
    """The shared wrapper used to read an empty prompt answer as a cancel, which made a
    keyfile-only database impossible to open from any surface."""
    from agent.vault_backends.base import enabled_backends
    from tools.browser_vault_tool import browser_vault_unlock

    keyfile = tmp_path / "only.keyx"
    keyfile.write_bytes(b"0123456789abcdef" * 4)
    path = tmp_path / "only.kdbx"
    kp = create_database(str(path), keyfile=str(keyfile))
    kp.add_entry(kp.root_group, "Example", "jane", SECRET, url="https://example.com")
    kp.save()

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        json.dumps({"vault": {"keepass": {"database": str(path), "keyfile": str(keyfile)}}}), encoding="utf-8")

    assert any(b.name == "keepass" and getattr(b, "allows_empty_master", False) for b in enabled_backends())
    unlock_mod.set_unlock_prompt_callback(lambda *_: "")
    try:
        with patch("tools.approval_context._no_user_can_answer", return_value=False):
            result = json.loads(browser_vault_unlock("keepass"))
    finally:
        unlock_mod.set_unlock_prompt_callback(None)
    assert result["success"] is True, result



def test_configured_database_is_a_source_and_a_wrong_master_reports_unlock_failed(kdbx, tmp_path, monkeypatch):
    """E2E through the real config + registry + browser_vault unlock wrapper (no backend patched)."""
    from agent.vault_backends.base import enabled_backends, external_backend_classes, is_enabled, is_installed
    from tools.browser_vault_tool import browser_vault_list, browser_vault_unlock

    path, web_id, _ = kdbx
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert KeepassLoginBackend in external_backend_classes()
    assert is_installed("keepass")
    assert not is_enabled("keepass"), "no database configured: not a source"

    (home / "config.yaml").write_text(json.dumps({"vault": {"keepass": {"database": str(path)}}}), encoding="utf-8")  # JSON is YAML
    assert is_enabled("keepass")
    assert any(b.name == "keepass" for b in enabled_backends())

    answers = iter(["hunter2-typo", MASTER])
    unlock_mod.set_unlock_prompt_callback(lambda *_: next(answers))
    try:
        with patch("tools.approval_context._no_user_can_answer", return_value=False):
            assert json.loads(browser_vault_list())["locked"][0]["backend"] == "keepass"
            failed = json.loads(browser_vault_unlock("keepass"))
            assert failed["error_type"] == "unlock_failed" and "hunter2-typo" not in failed["error"]
            assert json.loads(browser_vault_unlock("keepass"))["success"] is True
            listed = json.loads(browser_vault_list())
    finally:
        unlock_mod.set_unlock_prompt_callback(None)
    assert f"kp:{web_id}" in {item["handle"] for item in listed["items"]}
    assert SECRET not in json.dumps(listed)

    (home / "config.yaml").write_text(json.dumps({"vault": {"keepass": {"database": str(path), "enabled": False}}}),
                                      encoding="utf-8")
    assert not is_enabled("keepass")
