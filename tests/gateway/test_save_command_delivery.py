"""Regression tests for #107746 — /save must resolve its delivery adapter via ``_adapter_for_source``.

``_handle_save_command`` rendered the export to a temp file and then called
``self.get_adapter(source.platform)`` — a method ``GatewayRunner`` does not have (leftover from
a refactor). Every ``/save`` therefore crashed with ``AttributeError`` at the delivery step
and replied "Error exporting session: 'GatewayRunner' object has no attribute 'get_adapter'".

The fix resolves the adapter through ``_adapter_for_source`` (``GatewayAuthorizationMixin``),
which is profile-aware: a multiplexed secondary-profile chat delivers through the profile's own
adapter, and a source with no live adapter fails closed instead of falling back to the default
bot. These tests exercise the real handler end to end: real ``SessionStore``/``SessionDB``
(export + render), the real authz resolution, and a recording adapter.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource, SessionStore


class _RecordingAdapter:
    """Async adapter double: records every ``send_document`` call and, like a real adapter,
    consumes the document while it exists (the handler removes the temp file afterwards)."""

    def __init__(self, name: str = "adapter") -> None:
        self.name = name
        self.calls: list[dict] = []
        self.delivered_contents: list[str] = []

    async def send_document(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))
        file_path = kwargs.get("file_path")
        if file_path and os.path.exists(file_path):
            with open(file_path, encoding="utf-8") as fh:
                self.delivered_contents.append(fh.read())


def _make_event(text: str = "/save json", platform=Platform.TELEGRAM,
                user_id: str = "12345", chat_id: str = "67890",
                profile: str | None = None) -> MessageEvent:
    source = SessionSource(
        platform=platform,
        user_id=user_id,
        chat_id=chat_id,
        user_name="testuser",
        profile=profile,
    )
    return MessageEvent(text=text, source=source)


def _setup_store_and_db(tmp_path, source: SessionSource):
    """Real store + session DB with one session holding a user message."""
    store = SessionStore(
        sessions_dir=tmp_path / "sessions",
        config=GatewayConfig(),
    )
    entry = store.get_or_create_session(source)
    db = store._db
    db.create_session(
        entry.session_id, "telegram", session_key=entry.session_key,
        user_id=source.user_id, chat_id=source.chat_id,
    )
    db.append_message(entry.session_id, "user", "hello world")
    return store, db, entry


def _make_runner(store, db, *, adapters=None, profile_adapters=None):
    """Bare ``GatewayRunner`` (``object.__new__``, the repo's established harness pattern)."""
    from gateway.run import GatewayRunner
    from gateway.session import AsyncSessionStore
    from hermes_state import AsyncSessionDB

    runner = object.__new__(GatewayRunner)
    runner.adapters = adapters or {}
    runner._profile_adapters = profile_adapters or {}
    runner.config = SimpleNamespace(platforms={})
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    runner._is_user_authorized = lambda _source: True
    return runner


class TestSaveDeliversThroughResolvedAdapter:
    """Layer 1 + 4: the delivery step resolves the adapter and sends the rendered file."""

    @pytest.mark.asyncio
    async def test_save_exports_and_delivers_document(self, tmp_path):
        event = _make_event(text="/save json")
        store, db, _entry = _setup_store_and_db(tmp_path, event.source)
        adapter = _RecordingAdapter()
        runner = _make_runner(store, db, adapters={Platform.TELEGRAM: adapter})

        result = await runner._handle_save_command(event)
        db.close()

        assert result == "Export complete."
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call["chat_id"] == "67890"
        assert call["file_name"].endswith(".json")
        assert call["caption"] == f"Session export: {call['file_name']}"
        assert adapter.delivered_contents
        assert "hello world" in adapter.delivered_contents[0]

    @pytest.mark.asyncio
    async def test_save_temp_file_is_cleaned_up_after_delivery(self, tmp_path):
        event = _make_event(text="/save json")
        store, db, _entry = _setup_store_and_db(tmp_path, event.source)
        adapter = _RecordingAdapter()
        runner = _make_runner(store, db, adapters={Platform.TELEGRAM: adapter})

        result = await runner._handle_save_command(event)
        db.close()

        assert result == "Export complete."
        assert len(adapter.calls) == 1
        delivered_path = adapter.calls[0]["file_path"]
        assert not os.path.exists(delivered_path)


class TestSaveProfileAwareRouting:
    """Layer 2: a secondary-profile source delivers through its own adapter, never the default."""

    @pytest.mark.asyncio
    async def test_save_delivers_through_secondary_profile_adapter(self, tmp_path):
        event = _make_event(text="/save json", profile="work")
        store, db, _entry = _setup_store_and_db(tmp_path, event.source)
        primary = _RecordingAdapter("primary")
        work = _RecordingAdapter("work")
        runner = _make_runner(
            store, db,
            adapters={Platform.TELEGRAM: primary},
            profile_adapters={"work": {Platform.TELEGRAM: work}},
        )

        result = await runner._handle_save_command(event)
        db.close()

        assert result == "Export complete."
        assert len(work.calls) == 1
        assert work.calls[0]["chat_id"] == "67890"
        assert primary.calls == []


class TestSaveFailsClosedWithoutAdapter:
    """Layer 3: no live adapter → clean message, no crash, no cross-profile fallback."""

    @pytest.mark.asyncio
    async def test_save_with_no_adapter_returns_not_found_message(self, tmp_path):
        event = _make_event(text="/save json")
        store, db, _entry = _setup_store_and_db(tmp_path, event.source)
        runner = _make_runner(store, db, adapters={})

        result = await runner._handle_save_command(event)
        db.close()

        assert result == "Platform adapter not found to send the document."

    @pytest.mark.asyncio
    async def test_save_secondary_profile_without_adapter_does_not_use_primary(self, tmp_path):
        # Fail-closed contract: a secondary profile whose adapter is missing must NOT
        # deliver out the default profile's bot.
        event = _make_event(text="/save json", profile="ghost")
        store, db, _entry = _setup_store_and_db(tmp_path, event.source)
        primary = _RecordingAdapter("primary")
        runner = _make_runner(store, db, adapters={Platform.TELEGRAM: primary})

        result = await runner._handle_save_command(event)
        db.close()

        assert result == "Platform adapter not found to send the document."
        assert primary.calls == []


class TestSaveRedactionPath:
    """Layer 5: the redact flag reaches delivery, and the payload is actually redacted."""

    @pytest.mark.asyncio
    async def test_save_redact_delivers_redacted_export(self, tmp_path):
        event = _make_event(text="/save json redact")
        store, db, entry = _setup_store_and_db(tmp_path, event.source)
        db.append_message(entry.session_id, "user", "my token is sk-test-secret-4f9c")
        adapter = _RecordingAdapter()
        runner = _make_runner(store, db, adapters={Platform.TELEGRAM: adapter})

        result = await runner._handle_save_command(event)
        db.close()

        assert result == "Export complete."
        assert len(adapter.calls) == 1
        assert adapter.delivered_contents
        assert "sk-test-secret-4f9c" not in adapter.delivered_contents[0]

    @pytest.mark.asyncio
    async def test_save_without_redact_keeps_raw_content(self, tmp_path):
        # Baseline proving the secret above really was present pre-redaction.
        event = _make_event(text="/save json")
        store, db, entry = _setup_store_and_db(tmp_path, event.source)
        db.append_message(entry.session_id, "user", "my token is sk-test-secret-4f9c")
        adapter = _RecordingAdapter()
        runner = _make_runner(store, db, adapters={Platform.TELEGRAM: adapter})

        result = await runner._handle_save_command(event)
        db.close()

        assert result == "Export complete."
        assert len(adapter.calls) == 1
        assert adapter.delivered_contents
        assert "sk-test-secret-4f9c" in adapter.delivered_contents[0]


class TestSaveMissingSession:
    """Existing behavior guard: an export with no stored session returns the clear message."""

    @pytest.mark.asyncio
    async def test_save_without_stored_session_returns_clear_error(self, tmp_path):
        from hermes_state import SessionDB

        event = _make_event(text="/save json")
        store, _db, _entry = _setup_store_and_db(tmp_path, event.source)
        _db.close()
        # A separate DB that never saw this session: export_session() returns None.
        empty_db = SessionDB(db_path=tmp_path / "empty.db")
        adapter = _RecordingAdapter()
        runner = _make_runner(store, empty_db, adapters={Platform.TELEGRAM: adapter})

        result = await runner._handle_save_command(event)
        empty_db.close()

        assert result.startswith("No stored messages found for this session")
        assert adapter.calls == []
