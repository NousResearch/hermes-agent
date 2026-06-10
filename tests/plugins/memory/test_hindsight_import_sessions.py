import argparse
import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from plugins.memory.hindsight import import_sessions as importer
from plugins.memory.hindsight.import_sessions import (
    HindsightImportClient,
    ImportOptions,
    ImportSessionsError,
    build_retain_item,
    build_transcript,
    load_sessions,
    run_import,
)


def _ts(date: str) -> float:
    return datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def _make_db(tmp_path, sessions):
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                started_at REAL NOT NULL,
                archived INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                timestamp REAL NOT NULL,
                active INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        for session in sessions:
            conn.execute(
                "INSERT INTO sessions (id, title, started_at, archived) VALUES (?, ?, ?, ?)",
                (
                    session["id"],
                    session.get("title", ""),
                    _ts(session.get("date", "2026-01-01")),
                    session.get("archived", 0),
                ),
            )
            for index, message in enumerate(session.get("messages", [])):
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, timestamp, active) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        session["id"],
                        message["role"],
                        message.get("content", ""),
                        _ts(session.get("date", "2026-01-01")) + index,
                        message.get("active", 1),
                    ),
                )
    return db_path


class FakeClient:
    def __init__(self, existing=None, fail_docs=None):
        self.existing = set(existing or [])
        self.fail_docs = set(fail_docs or [])
        self.calls = []
        self.closed = False

    def list_document_ids(self, *, bank_id, doc_id_prefix=""):
        return set(self.existing)

    def retain(self, *, bank_id, item, document_id):
        self.calls.append((bank_id, item, document_id))
        if document_id in self.fail_docs:
            raise RuntimeError("transient failure")
        return SimpleNamespace(ok=True)

    def close(self):
        self.closed = True


def test_load_sessions_builds_transcript_with_configured_prefixes(tmp_path):
    _make_db(
        tmp_path,
        [
            {
                "id": "s1",
                "title": "Prefix test",
                "messages": [
                    {"role": "system", "content": "rules"},
                    {"role": "user", "content": "hello there"},
                    {"role": "assistant", "content": "general kenobi"},
                ],
            }
        ],
    )

    sessions, skipped = load_sessions(
        tmp_path / "state.db",
        config={"retain_user_prefix": "Human", "retain_assistant_prefix": "Bot"},
        options=ImportOptions(),
    )

    assert skipped == 0
    assert sessions[0].transcript == "System: rules\n\nHuman: hello there\n\nBot: general kenobi"
    assert sessions[0].turn_count == 1


def test_date_filters_since_until_and_days(tmp_path, monkeypatch):
    _make_db(
        tmp_path,
        [
            {"id": "old", "date": "2026-01-01", "messages": [{"role": "user", "content": "old enough content"}]},
            {"id": "mid", "date": "2026-01-10", "messages": [{"role": "user", "content": "middle enough content"}]},
            {"id": "new", "date": "2026-01-20", "messages": [{"role": "user", "content": "new enough content"}]},
        ],
    )
    sessions, _ = load_sessions(
        tmp_path / "state.db",
        config={},
        options=ImportOptions(since="2026-01-05", until="2026-01-15"),
    )
    assert [s.session_id for s in sessions] == ["mid"]

    class FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 21, tzinfo=timezone.utc)

    monkeypatch.setattr(importer, "datetime", FakeDateTime)
    sessions, _ = load_sessions(
        tmp_path / "state.db",
        config={},
        options=ImportOptions(days=5),
    )
    assert [s.session_id for s in sessions] == ["new"]


def test_invalid_date_raises_import_sessions_error(tmp_path):
    _make_db(tmp_path, [])
    with pytest.raises(ImportSessionsError, match="YYYY-MM-DD"):
        load_sessions(
            tmp_path / "state.db",
            config={},
            options=ImportOptions(since="01-05-2026"),
        )


def test_skips_short_empty_and_artifact_only_sessions(tmp_path):
    _make_db(
        tmp_path,
        [
            {"id": "short", "messages": [{"role": "user", "content": "tiny"}]},
            {"id": "empty", "messages": [{"role": "system", "content": "[SYSTEM: note]"}]},
            {"id": "ok", "messages": [{"role": "user", "content": "this session has enough useful content"}]},
        ],
    )

    sessions, skipped = load_sessions(tmp_path / "state.db", config={}, options=ImportOptions())

    assert [s.session_id for s in sessions] == ["ok"]
    assert skipped == 2


def test_build_transcript_strips_context_compaction_and_system_notes():
    transcript, _ = build_transcript(
        [
            {"role": "user", "content": "[SYSTEM: temporary note]\nkeep this   text"},
            {
                "role": "assistant",
                "content": "before\n[CONTEXT COMPACTION START]\nold context\n\nfinal answer",
            },
        ],
        config={},
    )

    assert "[SYSTEM:" not in transcript
    assert "CONTEXT COMPACTION" not in transcript
    assert "old context" not in transcript
    assert "keep this text" in transcript


def test_retain_item_doc_id_tags_timestamp_and_string_metadata():
    session = importer.SessionCandidate(
        session_id="abc123",
        title="A Session",
        started_at=_ts("2026-02-03"),
        transcript="User: useful historical transcript",
        turn_count=2,
    )

    item = build_retain_item(session, options=ImportOptions(extra_tags="hermes-backfill,extra"))

    assert item["document_id"] == "abc123"
    assert item["update_mode"] == "replace"
    assert item["tags"] == ["session:abc123", "hermes-backfill", "extra"]
    # Memories are dated at session time, not import time.
    assert item["timestamp"] == session.iso_date
    # Hindsight metadata values must all be strings.
    assert all(isinstance(value, str) for value in item["metadata"].values())
    assert item["metadata"]["session_id"] == "abc123"
    assert item["metadata"]["turn_count"] == "2"
    assert item["metadata"]["source"] == "hermes-backfill"
    assert item["metadata"]["imported_via"] == "hermes hindsight import-sessions"
    assert json.dumps(item["metadata"])  # JSON-serializable


def test_doc_id_prefix_is_applied():
    session = importer.SessionCandidate("s1", "", _ts("2026-01-01"), "long enough transcript", 1)
    item = build_retain_item(session, options=ImportOptions(doc_id_prefix="hist-"))
    assert item["document_id"] == "hist-s1"


def test_dry_run_never_calls_hindsight(tmp_path):
    _make_db(
        tmp_path,
        [{"id": "s1", "messages": [{"role": "user", "content": "long enough session content"}]}],
    )
    client = FakeClient()

    summary = run_import(
        ImportOptions(dry_run=True),
        hermes_home=tmp_path,
        client=client,
        config={"bank_id": "bank"},
    )

    assert summary.candidates == 1
    assert summary.bank_id == "bank"
    assert client.calls == []


def test_skip_existing_skips_matching_document_ids(tmp_path):
    _make_db(
        tmp_path,
        [
            {"id": "s1", "messages": [{"role": "user", "content": "first long enough session"}]},
            {"id": "s2", "messages": [{"role": "user", "content": "second long enough session"}]},
        ],
    )
    client = FakeClient(existing={"s1"})

    summary = run_import(
        ImportOptions(skip_existing=True, yes=True),
        hermes_home=tmp_path,
        client=client,
        config={"bank_id": "bank"},
    )

    assert summary.imported == 1
    assert summary.skipped_existing == 1
    assert client.calls[0][2] == "s2"


def test_per_session_failure_isolation_imports_unaffected_sessions(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "time", SimpleNamespace(sleep=lambda _seconds: None))
    _make_db(
        tmp_path,
        [
            {"id": "s1", "messages": [{"role": "user", "content": "first long enough session"}]},
            {"id": "s2", "messages": [{"role": "user", "content": "second long enough session"}]},
            {"id": "s3", "messages": [{"role": "user", "content": "third long enough session"}]},
        ],
    )
    client = FakeClient(fail_docs={"s2"})

    summary = run_import(
        ImportOptions(yes=True),
        hermes_home=tmp_path,
        client=client,
        config={"bank_id": "bank"},
    )

    assert summary.imported == 2
    assert summary.failed == 1
    assert summary.failed_session_ids == ["s2"]
    # s1 and s3 retained once each; s2 retried 3 times before giving up.
    assert len(client.calls) == 5


def test_local_embedded_client_creation_is_mockable(monkeypatch):
    captured = {}

    class FakeEmbedded:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(importer, "_check_local_runtime", lambda: (True, ""))
    monkeypatch.setitem(__import__("sys").modules, "hindsight", SimpleNamespace(HindsightEmbedded=FakeEmbedded))

    client = HindsightImportClient(
        {
            "mode": "local_embedded",
            "profile": "hermes",
            "llm_provider": "openrouter",
            "llmApiKey": "key",
            "llm_model": "model",
        },
        timeout=10,
    )

    assert isinstance(client._get_client(), FakeEmbedded)
    assert captured["profile"] == "hermes"
    assert captured["llm_provider"] == "openai"
    assert captured["idle_timeout"] > 0


def test_list_document_ids_paginates_and_filters_by_prefix(monkeypatch):
    pages = {
        0: SimpleNamespace(items=[{"id": f"doc-{i}"} for i in range(500)]),
        500: SimpleNamespace(items=[{"id": "doc-500"}]),
    }
    calls = []

    async def fake_list_documents(bank_id, q=None, limit=None, offset=None):
        calls.append((bank_id, q, limit, offset))
        return pages[offset]

    monkeypatch.setattr(importer, "_run_sync", lambda coro, timeout=None: asyncio.run(coro))
    client = HindsightImportClient({"mode": "cloud"}, timeout=10)
    client._client = SimpleNamespace(
        documents=SimpleNamespace(list_documents=fake_list_documents)
    )

    ids = client.list_document_ids(bank_id="bank", doc_id_prefix="hist-")

    assert len(ids) == 501
    assert calls[0] == ("bank", "hist-", 500, 0)
    assert calls[1] == ("bank", "hist-", 500, 500)


def test_list_document_ids_errors_when_client_cannot_list():
    client = HindsightImportClient({"mode": "cloud"}, timeout=10)
    client._client = SimpleNamespace()  # no documents API

    with pytest.raises(ImportSessionsError, match="--skip-existing"):
        client.list_document_ids(bank_id="bank")


def test_confirmation_eof_cancels_without_importing(monkeypatch):
    monkeypatch.setattr(importer, "_load_config", lambda: {})
    monkeypatch.setattr(
        importer, "run_import", lambda *a, **k: pytest.fail("run_import must not be called")
    )

    def _raise_eof(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise_eof)

    with pytest.raises(SystemExit) as exc:
        importer.handle_import_sessions_command(argparse.Namespace(dry_run=False, yes=False))

    assert exc.value.code == 1


def test_confirmation_declined_cancels_without_importing(monkeypatch, capsys):
    monkeypatch.setattr(importer, "_load_config", lambda: {})
    monkeypatch.setattr(
        importer, "run_import", lambda *a, **k: pytest.fail("run_import must not be called")
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "no")

    importer.handle_import_sessions_command(argparse.Namespace(dry_run=False, yes=False))

    assert "Cancelled" in capsys.readouterr().out


def test_bank_template_error_is_user_facing(monkeypatch, capsys):
    monkeypatch.setattr(importer, "_load_config", lambda: {"bank_id_template": "x-{session}"})

    with pytest.raises(SystemExit) as exc:
        importer.handle_import_sessions_command(argparse.Namespace(dry_run=True))

    assert exc.value.code == 1
    assert "--bank-id" in capsys.readouterr().out


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"bank_id": "direct"}, "direct"),
        ({"banks": {"hermes": {"bankId": "legacy"}}}, "legacy"),
        ({}, "hermes"),
    ],
)
def test_resolve_bank_id_static(config, expected):
    assert importer.resolve_bank_id(config) == expected


def test_resolve_bank_id_override_wins():
    config = {"bank_id": "cfg", "bank_id_template": "x-{session}"}
    assert importer.resolve_bank_id(config, override="forced") == "forced"


def test_resolve_bank_id_profile_template_matches_live_retention(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: "Work Profile"
    )
    assert importer.resolve_bank_id({"bank_id_template": "hermes-{profile}"}) == "hermes-Work-Profile"


def test_resolve_bank_id_per_session_template_requires_override():
    with pytest.raises(ImportSessionsError, match="--bank-id"):
        importer.resolve_bank_id({"bank_id_template": "hermes-{platform}-{user}"})
