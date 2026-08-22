import copy
import importlib.util
import json
import os
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HERMES_ARCHIVE_SOURCE", str(ROOT))
os.environ.setdefault("HERMES_ARCHIVE_HOME", "/tmp/hermes-archive-regression-home")
os.environ.setdefault("HERMES_ARCHIVE_SYNC_DEVICE_ID", "test-device")

from hermes_cli.session_export_md import (  # noqa: E402
    append_manifest_entry,
    verify_export_file,
    write_session_markdown,
)

ARCHIVE_SCRIPT = ROOT / "scripts" / "archive_sessions.py"
_archive_spec = importlib.util.spec_from_file_location(
    "hermes_runtime_archive_sessions",
    ARCHIVE_SCRIPT,
)
assert _archive_spec is not None and _archive_spec.loader is not None
archive = importlib.util.module_from_spec(_archive_spec)
_archive_spec.loader.exec_module(archive)


SESSION_ID = "519dcf294078"


def _data(contents, *, source="webui"):
    messages = [
        {
            "id": index,
            "session_id": SESSION_ID,
            "role": "user" if index % 2 else "assistant",
            "content": content,
            "timestamp": 1_800_000_000 + index,
            "active": 1,
            "compacted": 0,
        }
        for index, content in enumerate(contents, start=1)
    ]
    segment = {
        "id": SESSION_ID,
        "source": source,
        "title": "session",
        "started_at": 1_799_999_000.0,
        "ended_at": 1_800_000_100.0,
        "messages": messages,
    }
    return {
        **segment,
        "segments": [segment],
        "lineage_session_ids": [SESSION_ID],
        "message_count": len(messages),
    }


class FakeDB:
    def __init__(self, snapshots):
        self.snapshots = [copy.deepcopy(item) for item in snapshots]
        self.export_calls = []
        self.deleted = []

    def export_session_lineage(self, session_id, *, include_compacted=False):
        self.export_calls.append((session_id, include_compacted))
        index = min(len(self.export_calls) - 1, len(self.snapshots) - 1)
        return copy.deepcopy(self.snapshots[index])

    def delete_session(self, session_id, sessions_dir=None):
        self.deleted.append(session_id)
        return True

    def get_session(self, session_id):
        return self.snapshots[-1]


def _isolate_archive(monkeypatch, tmp_path, *, active=False):
    vault = tmp_path / "vault"
    monkeypatch.setattr(archive, "VAULT", vault)
    monkeypatch.setattr(archive, "SESSION_ARCHIVE", vault / "10 Sessions")
    monkeypatch.setattr(archive, "trigger_scan", lambda _path: None)
    monkeypatch.setattr(archive, "wait_for_remote", lambda *_args: True)
    monkeypatch.setattr(archive, "append_event", lambda _event: None)
    monkeypatch.setattr(archive, "webui_session_is_active", lambda _sid: active)
    return vault


def test_active_webui_session_is_rejected_before_export(monkeypatch, tmp_path):
    vault = _isolate_archive(monkeypatch, tmp_path, active=True)
    db = FakeDB([_data(["still running"])])

    with pytest.raises(RuntimeError, match="refusing to export active WebUI"):
        archive.archive_one(
            db,
            SESSION_ID,
            delete_after_sync=True,
            timeout=1,
        )

    assert db.export_calls == [(SESSION_ID, True)]
    assert db.deleted == []
    assert not vault.exists()


def test_stale_valid_export_is_preserved_and_new_revision_is_verified(
    monkeypatch,
    tmp_path,
):
    vault = _isolate_archive(monkeypatch, tmp_path)
    old_data = _data(["old snapshot"])
    current_data = _data(["old snapshot", "continued turn"])
    output_dir = vault / "10 Sessions" / "2027" / "01"
    base_path = write_session_markdown(old_data, output_dir)
    append_manifest_entry(output_dir, old_data, base_path, fmt="md")
    old_bytes = base_path.read_bytes()
    db = FakeDB([current_data, current_data])

    result = archive.archive_one(
        db,
        SESSION_ID,
        delete_after_sync=False,
        timeout=1,
    )

    revision_path = vault / result["archive_path"]
    assert base_path.read_bytes() == old_bytes
    assert revision_path != base_path
    assert "-rev-" in revision_path.name
    assert verify_export_file(revision_path, current_data) == (True, "ok")
    assert result["archive_message_count"] == 2
    assert db.export_calls == [(SESSION_ID, True), (SESSION_ID, True)]

    entries = [
        json.loads(line)
        for line in (output_dir / "manifest.jsonl").read_text().splitlines()
    ]
    revision_entry = [entry for entry in entries if entry["path"] == str(revision_path)][-1]
    assert revision_entry["source_fingerprint"] == result["source_fingerprint"]


def test_corrupt_existing_export_is_not_bypassed_by_revision(monkeypatch, tmp_path):
    vault = _isolate_archive(monkeypatch, tmp_path)
    data = _data(["original snapshot"])
    output_dir = vault / "10 Sessions" / "2027" / "01"
    base_path = write_session_markdown(data, output_dir)
    base_path.write_text(
        base_path.read_text(encoding="utf-8") + "\ntampered\n",
        encoding="utf-8",
    )
    db = FakeDB([data])

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        archive.archive_one(
            db,
            SESSION_ID,
            delete_after_sync=False,
            timeout=1,
        )

    assert list(output_dir.glob("*-rev-*.md")) == []
    assert db.deleted == []


def test_session_change_during_sync_prevents_delete(monkeypatch, tmp_path):
    _isolate_archive(monkeypatch, tmp_path)
    before = _data(["first snapshot"])
    after = _data(["first snapshot", "late message"])
    db = FakeDB([before, after])

    with pytest.raises(RuntimeError, match="session changed during archive"):
        archive.archive_one(
            db,
            SESSION_ID,
            delete_after_sync=True,
            timeout=1,
        )

    assert db.deleted == []


def test_webui_reactivation_after_sync_prevents_delete(monkeypatch, tmp_path):
    _isolate_archive(monkeypatch, tmp_path)
    active_checks = iter([False, True])
    monkeypatch.setattr(
        archive,
        "webui_session_is_active",
        lambda _sid: next(active_checks),
    )
    data = _data(["stable snapshot"])
    db = FakeDB([data, data])

    with pytest.raises(RuntimeError, match="refusing to delete active WebUI"):
        archive.archive_one(
            db,
            SESSION_ID,
            delete_after_sync=True,
            timeout=1,
        )

    assert db.deleted == []


class CandidateDB:
    def __init__(self, last_active):
        self.last_active = last_active

    def list_sessions_rich(self, **_kwargs):
        return [
            {
                "id": SESSION_ID,
                "last_active": self.last_active,
                "started_at": self.last_active - 100,
            }
        ]

    def get_compression_lineage(self, session_id):
        return [session_id]

    def get_session(self, session_id):
        return {
            "id": session_id,
            "ended_at": self.last_active - 500,
        }


def test_ended_candidate_grace_uses_latest_activity_not_stale_ended_at():
    db = CandidateDB(last_active=time.time() - 5)

    assert archive.ended_candidates(db, None, min_age=60) == []


def test_ended_candidate_becomes_eligible_after_latest_activity_grace():
    db = CandidateDB(last_active=time.time() - 120)

    assert archive.ended_candidates(db, None, min_age=60) == [SESSION_ID]


def test_syncthing_device_id_must_be_configured(monkeypatch):
    monkeypatch.setattr(archive, "MAC_DEVICE_ID", "")

    with pytest.raises(
        RuntimeError,
        match="HERMES_ARCHIVE_SYNC_DEVICE_ID is not configured",
    ):
        archive.syncthing_remote_device_id()
