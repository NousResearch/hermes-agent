import os
import stat

from gateway import rich_sent_store


def _record_under_permissive_umask(chat_id, message_id, text):
    previous_umask = os.umask(0o022)
    try:
        rich_sent_store.record(chat_id, message_id, text)
    finally:
        os.umask(previous_umask)


def test_record_creates_private_index_under_permissive_umask(tmp_path, monkeypatch):
    profile_home = tmp_path / "tracker-home"
    store_path = profile_home / "state" / "rich_sent_index.json"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    _record_under_permissive_umask("chat-1", 42, "private preview")

    assert rich_sent_store.lookup("chat-1", 42) == "private preview"
    assert stat.S_IMODE(store_path.stat().st_mode) == 0o600


def test_record_replaces_broad_index_with_private_file(tmp_path, monkeypatch):
    profile_home = tmp_path / "tracker-home"
    store_path = profile_home / "state" / "rich_sent_index.json"
    store_path.parent.mkdir(parents=True)
    store_path.write_text("{}", encoding="utf-8")
    store_path.chmod(0o644)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    _record_under_permissive_umask("chat-2", 84, "new private preview")

    assert rich_sent_store.lookup("chat-2", 84) == "new private preview"
    assert stat.S_IMODE(store_path.stat().st_mode) == 0o600
