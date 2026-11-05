"""`hermes sessions export` single-file formats accept a directory as OUTPUT."""

import json
import sys

import hermes_cli.main as main_mod
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


def _export(monkeypatch, *argv):
    # One stored session, found by its id prefix.
    with open_persistence(profile_scope()) as store:
        seed_session(store, "sess-123", "cli")
        seed_messages(store, "sess-123", {"role": "user", "content": "hi"})
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "export", "--session-id", "sess", *argv])
    main_mod.main()


def test_jsonl_export_into_directory_writes_default_named_file(monkeypatch, tmp_path):
    out_dir = tmp_path / "saved"
    out_dir.mkdir()

    _export(monkeypatch, f"{out_dir}/")

    written = out_dir / "hermes_session_sess-123.jsonl"
    assert json.loads(written.read_text(encoding="utf-8"))["id"] == "sess-123"


def test_jsonl_export_to_file_path_is_unchanged(monkeypatch, tmp_path):
    target = tmp_path / "one.jsonl"

    _export(monkeypatch, str(target))

    assert json.loads(target.read_text(encoding="utf-8"))["id"] == "sess-123"
    assert target.is_file() and not (tmp_path / "hermes_session_sess-123.jsonl").exists()
