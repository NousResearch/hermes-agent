"""Official state backups must preserve image replay, not just transcript text."""

from argparse import Namespace
from pathlib import Path

import pytest

from agent.session_persistence import _db_flush_row
from hermes_cli import backup
from hermes_state import SessionDB


@pytest.mark.parametrize("kind", ["full", "pre_update", "pre_migration", "quick"])
def test_backup_restores_transcript_image(tmp_path, monkeypatch, kind):
    from hermes_cli import gateway, gateway_setup_service

    # Import may install a gateway service; only that host side effect is stubbed.
    monkeypatch.setattr(gateway_setup_service, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway, "_is_service_running", lambda: False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(source))
    data_url = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP4z8DwHwAFAAH/iZk9HQAAAABJRU5ErkJggg=="
    )
    content = [
        {"type": "text", "text": "Read this image"},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    with SessionDB(db_path=source / "state.db") as db:
        db.create_session("vision", "cli")
        row = _db_flush_row(Namespace(_session_db=db), {"role": "user", "content": content}, False)
        db.append_messages_batch("vision", [row])
        assert db.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == content

    snapshot_id = None
    archive = tmp_path / "backup.zip"
    if kind == "full":
        assert backup.run_backup(Namespace(output=str(archive)))
    elif kind == "quick":
        snapshot_id = backup.create_quick_snapshot(hermes_home=source)
        assert snapshot_id
    else:
        archive = getattr(backup, f"create_{kind}_backup")(hermes_home=source)
        assert archive is not None

    destination = tmp_path / "restored"
    destination.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(destination))
    if snapshot_id is not None:
        (source / "state-snapshots").rename(destination / "state-snapshots")
        assert backup.restore_quick_snapshot(snapshot_id, hermes_home=destination)
    else:
        assert backup.run_import(Namespace(zipfile=str(archive), force=True)) is None
    # The process lock is recreatable runtime state, not part of a portable backup.
    assert not (destination / "cache/.transcript_media.lock").exists()
    with SessionDB(db_path=destination / "state.db") as restored:
        assert restored.get_messages_as_conversation("vision", repair_alternation=True)[0]["content"] == content
