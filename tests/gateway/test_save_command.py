"""Regression tests for gateway /save (``_handle_save_command``).

Two independent bugs found in the same function:

1. ``self.get_adapter(source.platform)`` calls a method that does not exist
   anywhere on ``GatewayRunner`` (the established pattern used everywhere
   else in this file is ``self.adapters.get(platform)``) — every invocation
   raised ``AttributeError``, caught by the broad ``except Exception`` and
   surfaced as "Error exporting session: ...", so the command never
   delivered a file.
2. Redaction of the exported session was opt-in only (a trailing ``redact``
   token) even though the export is always sent as a document into the
   invoking chat — which may be a group chat visible to other participants
   and retained on the platform's own servers. ``hermes sessions export
   --format trace`` treats "leaves the machine" as reason enough to redact
   by default; /save did not.

This file drives the REAL ``_handle_save_command`` against a REAL
SessionStore + SessionDB (SQLite in tmp_path), mirroring the harness in
``tests/gateway/test_branch_routing_columns.py``.
"""

from __future__ import annotations

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource, SessionStore
from hermes_state import AsyncSessionDB

_FAKE_GH_TOKEN = "ghp_" + "F" * 36


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Real SessionStore backed by a real SessionDB (SQLite in tmp_path)."""
    import hermes_state

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", tmp_path / "state.db")
    config = GatewayConfig()
    return SessionStore(sessions_dir=tmp_path, config=config)


def _make_source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="170829464",
        chat_id="170829464",
        chat_type="dm",
        thread_id=None,
    )


def _make_event(text: str) -> MessageEvent:
    return MessageEvent(text=text, source=_make_source(), message_id="m1")


class _RecordingAdapter:
    """Captures send_document's file content before the caller's finally
    block deletes the temp file."""

    def __init__(self):
        self.calls: list[dict] = []
        self.captured_content: str | None = None

    async def send_document(self, *, chat_id, file_path, caption, file_name):
        with open(file_path, encoding="utf-8") as f:
            self.captured_content = f.read()
        self.calls.append(
            {"chat_id": chat_id, "file_path": file_path, "caption": caption, "file_name": file_name}
        )


def _make_save_runner(store: SessionStore, adapter):
    """Minimal GatewayRunner stub wired to a REAL session_store/session_db."""
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner.session_store = store
    runner._session_db = AsyncSessionDB(store._db)
    return runner


class TestSaveCommandDeliversDocument:
    @pytest.mark.asyncio
    async def test_save_delivers_document_via_platform_adapter(self, store):
        """Regression: /save must actually call adapter.send_document, not
        crash on a nonexistent self.get_adapter() before ever reaching it."""
        source = _make_source()
        entry = store.get_or_create_session(source)
        store._db.append_message(entry.session_id, role="user", content="hello")
        store._db.append_message(entry.session_id, role="assistant", content="world")

        adapter = _RecordingAdapter()
        runner = _make_save_runner(store, adapter)

        result = await runner._handle_save_command(_make_event("/save json"))

        assert result == "Export complete."
        assert len(adapter.calls) == 1
        assert adapter.captured_content is not None


class TestSaveCommandRedaction:
    @pytest.mark.asyncio
    async def test_save_redacts_secrets_by_default(self, store):
        source = _make_source()
        entry = store.get_or_create_session(source)
        store._db.append_message(
            entry.session_id, role="user", content=f"export GITHUB_TOKEN={_FAKE_GH_TOKEN}"
        )

        adapter = _RecordingAdapter()
        runner = _make_save_runner(store, adapter)

        result = await runner._handle_save_command(_make_event("/save json"))

        assert result == "Export complete."
        assert adapter.captured_content is not None
        assert _FAKE_GH_TOKEN not in adapter.captured_content, (
            "raw secret leaked into the default gateway /save export"
        )

    @pytest.mark.asyncio
    async def test_save_noredact_skips_redaction(self, store):
        source = _make_source()
        entry = store.get_or_create_session(source)
        store._db.append_message(
            entry.session_id, role="user", content=f"export GITHUB_TOKEN={_FAKE_GH_TOKEN}"
        )

        adapter = _RecordingAdapter()
        runner = _make_save_runner(store, adapter)

        result = await runner._handle_save_command(_make_event("/save json noredact"))

        assert result == "Export complete."
        assert adapter.captured_content is not None
        assert _FAKE_GH_TOKEN in adapter.captured_content

    @pytest.mark.asyncio
    async def test_save_no_adapter_reports_error_instead_of_crashing(self, store):
        """No adapter registered for the platform must return the documented
        'not found' message, not an unhandled AttributeError."""
        source = _make_source()
        store.get_or_create_session(source)

        runner = _make_save_runner(store, adapter=None)
        runner.adapters = {}

        result = await runner._handle_save_command(_make_event("/save json"))

        assert result == "Platform adapter not found to send the document."
