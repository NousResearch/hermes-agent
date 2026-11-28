"""A durable (gateway-stopped) profile rename rekeys the legacy sessions.json routing mirror.

Regression for #157580: ``_migrate_profile_identity`` rekeyed ``state.db`` but never touched
``sessions/sessions.json``, so the next gateway boot re-imported ``agent:<old>:*`` keys from the
mirror (one "no resolvable home" warning per key) and ``migrate-identity`` then collided with
its own rekeyed copies.
"""
import json
from pathlib import Path

import pytest

from hermes_cli.profile_identity import _migrate_profile_identity, _rekey_sessions_json_mirror
from persistence_assembly import open_persistence, profile_scope


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    with open_persistence(profile_scope(home=home), eager=True):
        pass
    return home


def _write_mirror(home, entries):
    sessions_dir = home / "sessions"
    sessions_dir.mkdir(exist_ok=True)
    mirror = sessions_dir / "sessions.json"
    mirror.write_text(json.dumps({"_README": "metadata", **entries}), encoding="utf-8")
    return mirror


def _entry(session_key, session_id, profile, transport=None):
    entry = {
        "session_key": session_key,
        "session_id": session_id,
        "created_at": "2026-11-18T00:00:00",
        "updated_at": "2026-11-18T00:00:00",
        "origin": {"platform": "feishu", "chat_id": "chatA", "profile": profile},
    }
    if transport is not None:
        entry["transport_profile"] = transport
    return entry


def test_durable_rename_rekeys_sessions_json_mirror(root):
    scope = str(root / "sessions")
    with open_persistence(profile_scope(home=root)) as store:
        store.routing.upsert_routing_entry(
            "agent:alpha:feishu:dm:chatA",
            json.dumps({"session_key": "agent:alpha:feishu:dm:chatA"}),
            routing_scope=scope)
    mirror = _write_mirror(root, {
        "agent:alpha:feishu:dm:chatA": _entry(
            "agent:alpha:feishu:dm:chatA", "sid-1", "alpha", transport="alpha-bot"),
        "agent:main:feishu:dm:root": _entry(
            "agent:main:feishu:dm:root", "sid-0", "default"),
    })

    assert _migrate_profile_identity("alpha", "beta", False) is True

    data = json.loads(mirror.read_text(encoding="utf-8"))
    assert set(data) == {"_README", "agent:beta:feishu:dm:chatA", "agent:main:feishu:dm:root"}
    moved = data["agent:beta:feishu:dm:chatA"]
    assert moved["session_key"] == "agent:beta:feishu:dm:chatA"
    assert moved["session_id"] == "sid-1"
    assert moved["origin"]["profile"] == "beta"
    # transport_profile names the receiving bot, not the runtime profile: it stays.
    assert moved["transport_profile"] == "alpha-bot"
    # Bystander (default profile) entry untouched.
    assert data["agent:main:feishu:dm:root"]["origin"]["profile"] == "default"
    with open_persistence(profile_scope(home=root)) as check:
        assert set(check.routing.load_routing_entries(routing_scope=scope)) == {
            "agent:beta:feishu:dm:chatA"}


def test_durable_rename_drops_mirror_copy_when_new_key_already_present(root):
    """An already-migrated mirror copy wins; the stale old-namespace copy is dropped, never
    overwriting the current entry (the rekeyed DB is authoritative)."""
    mirror = _write_mirror(root, {
        "agent:alpha:feishu:dm:chatA": _entry("agent:alpha:feishu:dm:chatA", "sid-old", "alpha"),
        "agent:beta:feishu:dm:chatA": _entry("agent:beta:feishu:dm:chatA", "sid-new", "beta"),
    })

    assert _rekey_sessions_json_mirror(root, "alpha", "beta") is True

    data = json.loads(mirror.read_text(encoding="utf-8"))
    assert set(data) == {"_README", "agent:beta:feishu:dm:chatA"}
    assert data["agent:beta:feishu:dm:chatA"]["session_id"] == "sid-new"


def test_durable_rename_rekeys_the_marked_main_namespace(root):
    """A profile literally named ``main`` lives in the ``agent:main~`` namespace, not the
    default profile's ``agent:main`` one — only the marked keys move."""
    mirror = _write_mirror(root, {
        "agent:main~:feishu:dm:named": _entry("agent:main~:feishu:dm:named", "sid-1", "main"),
        "agent:main:feishu:dm:root": _entry("agent:main:feishu:dm:root", "sid-0", "default"),
    })

    assert _rekey_sessions_json_mirror(root, "main", "renamed") is True

    data = json.loads(mirror.read_text(encoding="utf-8"))
    assert set(data) == {"_README", "agent:renamed:feishu:dm:named", "agent:main:feishu:dm:root"}
    assert data["agent:renamed:feishu:dm:named"]["origin"]["profile"] == "renamed"


def test_durable_rename_without_mirror_is_a_noop(root):
    assert _migrate_profile_identity("alpha", "beta", False) is True
    assert not (root / "sessions" / "sessions.json").exists()


def test_unreadable_mirror_reports_the_rename_as_pending(root, capsys):
    mirror = _write_mirror(root, {})
    mirror.write_text("{not json", encoding="utf-8")

    assert _migrate_profile_identity("alpha", "beta", False) is False
    assert "sessions.json routing mirror could not be rekeyed" in capsys.readouterr().err
