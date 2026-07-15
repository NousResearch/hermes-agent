from __future__ import annotations

import sqlite3
import os
from pathlib import Path

import pytest

from gateway import production_discord_journal_bootstrap as bootstrap


INTENT_SHA256 = "a" * 64


def _state_path(tmp_path: Path, name: str, monkeypatch) -> Path:
    parent = tmp_path / name
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    monkeypatch.setattr(bootstrap.os, "getegid", lambda: parent.stat().st_gid)
    return parent / "journal.sqlite3"


def test_connector_bootstrap_is_explicit_clean_create_only_and_resumable(
    tmp_path,
    monkeypatch,
):
    journal = _state_path(tmp_path, "connector", monkeypatch)
    monkeypatch.setattr(bootstrap, "CONNECTOR_JOURNAL_PATH", journal)

    first = bootstrap.ensure_clean_journal(
        "connector", intent_sha256=INTENT_SHA256
    )
    second = bootstrap.ensure_clean_journal(
        "connector", intent_sha256=INTENT_SHA256
    )

    assert first["schema"] == bootstrap.BOOTSTRAP_SCHEMA
    assert first["created"] is True
    assert second["created"] is False
    assert first["clean"] is second["clean"] is True
    assert first["clean_row_counts"] == {
        "connector_events_v1": 0,
        "connector_sends_v1": 0,
    }
    assert first["secret_material_recorded"] is False
    assert first["secret_digest_recorded"] is False

    with sqlite3.connect(journal) as connection:
        connection.execute(
            """
            INSERT INTO connector_sends_v1(
                idempotency_key,request_sha256,state,result_json,updated_at_unix_ms
            ) VALUES ('one','aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
                      'prepared',NULL,1)
            """
        )
    with pytest.raises(
        bootstrap.ProductionDiscordJournalBootstrapError,
        match="journal_not_clean",
    ):
        bootstrap.ensure_clean_journal(
            "connector", intent_sha256=INTENT_SHA256
        )


def test_routeback_bootstrap_requires_atomic_database_and_marker_and_stays_clean(
    tmp_path,
    monkeypatch,
):
    journal = _state_path(tmp_path, "routeback", monkeypatch)
    marker = Path(f"{journal}.initialized")
    monkeypatch.setattr(bootstrap, "ROUTEBACK_JOURNAL_PATH", journal)

    first = bootstrap.ensure_clean_journal(
        "routeback", intent_sha256=INTENT_SHA256
    )
    second = bootstrap.ensure_clean_journal(
        "routeback", intent_sha256=INTENT_SHA256
    )

    assert first["created"] is True
    assert second["created"] is False
    assert first["marker_path"] == str(marker)
    assert first["clean_row_counts"] == {
        "discord_edge_idempotency_v1": 0,
        "discord_edge_receipt_history_v1": 0,
    }

    marker.unlink()
    recovered = bootstrap.ensure_clean_journal(
        "routeback", intent_sha256=INTENT_SHA256
    )
    assert recovered["recovered_partial"] is True
    assert marker.is_file()


def test_connector_partial_schema_is_reset_only_when_exact_and_empty(
    tmp_path,
    monkeypatch,
):
    journal = _state_path(tmp_path, "connector-partial", monkeypatch)
    monkeypatch.setattr(bootstrap, "CONNECTOR_JOURNAL_PATH", journal)
    previous_umask = os.umask(0o077)
    try:
        with sqlite3.connect(journal) as connection:
            connection.execute(
                "CREATE TABLE connector_meta_v1(schema_version TEXT PRIMARY KEY) STRICT"
            )
    finally:
        os.umask(previous_umask)
    journal.chmod(0o600)

    receipt = bootstrap.ensure_clean_journal(
        "connector", intent_sha256=INTENT_SHA256
    )
    assert receipt["created"] is True
    assert receipt["clean"] is True


def test_routeback_database_only_crash_is_recovered_from_exact_meta_uuid(
    tmp_path,
    monkeypatch,
):
    journal = _state_path(tmp_path, "routeback-db-only", monkeypatch)
    monkeypatch.setattr(bootstrap, "ROUTEBACK_JOURNAL_PATH", journal)

    def crash_before_marker(_self, _marker_id):
        raise RuntimeError("injected_before_marker")

    monkeypatch.setattr(
        bootstrap.DurableDiscordEdgeJournal,
        "_create_marker",
        crash_before_marker,
    )
    with pytest.raises(RuntimeError, match="injected_before_marker"):
        bootstrap.DurableDiscordEdgeJournal.bootstrap(journal)
    monkeypatch.undo()
    monkeypatch.setattr(bootstrap, "ROUTEBACK_JOURNAL_PATH", journal)
    monkeypatch.setattr(bootstrap.os, "getegid", lambda: journal.parent.stat().st_gid)

    receipt = bootstrap.ensure_clean_journal(
        "routeback", intent_sha256=INTENT_SHA256
    )
    assert receipt["recovered_partial"] is True
    assert Path(f"{journal}.initialized").is_file()


def test_bootstrap_rejects_unpinned_path_and_insecure_parent(
    tmp_path,
    monkeypatch,
):
    journal = _state_path(tmp_path, "connector", monkeypatch)
    monkeypatch.setattr(bootstrap, "CONNECTOR_JOURNAL_PATH", journal)

    with pytest.raises(
        bootstrap.ProductionDiscordJournalBootstrapError,
        match="journal_path_invalid",
    ):
        bootstrap.ensure_clean_journal(
            "connector",
            intent_sha256=INTENT_SHA256,
            path=tmp_path / "different.sqlite3",
        )

    journal.parent.chmod(0o750)
    with pytest.raises(
        bootstrap.ProductionDiscordJournalBootstrapError,
        match="journal_parent_invalid",
    ):
        bootstrap.ensure_clean_journal(
            "connector", intent_sha256=INTENT_SHA256
        )
