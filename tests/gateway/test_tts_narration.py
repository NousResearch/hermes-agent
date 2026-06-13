"""Long-form TTS narration job pipeline tests."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, SessionSource, SendResult


def _source(chat_id="123", thread_id=None):
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        user_id="user1",
        chat_type="group",
        thread_id=thread_id,
    )


def _event(text="hello", chat_id="123", thread_id=None):
    ev = MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(chat_id=chat_id, thread_id=thread_id),
    )
    ev.message_id = "msg42"
    return ev


class TestNarrationChunking:
    def test_short_text_produces_one_chunk_without_labels(self):
        from gateway.tts_narration import chunk_narration_text

        chunks = chunk_narration_text("A short paragraph for reading aloud.")

        assert chunks == ["A short paragraph for reading aloud."]
        assert not any("part 1" in c.lower() for c in chunks)

    def test_long_text_prefers_sentence_boundaries_and_hard_limit(self):
        from gateway.tts_narration import chunk_narration_text

        paragraph = " ".join([f"Sentence {i} has a calm ordinary shape." for i in range(1, 90)])
        chunks = chunk_narration_text(paragraph, target_chars=220, max_chars=260)

        assert len(chunks) > 1
        assert all(0 < len(chunk) <= 260 for chunk in chunks)
        assert "".join(chunk.replace(" ", "") for chunk in chunks) == paragraph.replace(" ", "")
        assert all(chunk.endswith(".") for chunk in chunks[:-1])

    def test_empty_text_produces_no_chunks(self):
        from gateway.tts_narration import chunk_narration_text

        assert chunk_narration_text("   \n\n  ") == []


class TestNarrationProviderMetadata:
    def test_defaults_to_regular_tts_provider_path_without_override(self, monkeypatch):
        from gateway import tts_narration

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"tts": {"provider": "edge", "providers": {"edge": {"voice": "en-US"}}}},
        )

        assert tts_narration.provider_metadata_from_config() == {
            "provider": None,
            "model": None,
            "voice": None,
        }

    def test_tts_narration_override_is_explicit_and_optional(self, monkeypatch):
        from gateway import tts_narration

        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {
                "tts": {
                    "provider": "edge",
                    "narration": {
                        "provider": "openai",
                        "model": "gpt-4o-mini-tts",
                        "voice": "coral",
                    },
                }
            },
        )

        assert tts_narration.provider_metadata_from_config() == {
            "provider": "openai",
            "model": "gpt-4o-mini-tts",
            "voice": "coral",
        }


class TestNarrationStore:
    def test_enqueue_persists_job_and_chunks_without_full_text_in_job_row(self, tmp_path):
        from gateway.tts_narration import NarrationJobStore, chunk_narration_text

        store = NarrationJobStore(tmp_path / "narration.sqlite")
        text = "First sentence. " * 80
        chunks = chunk_narration_text(text, target_chars=120, max_chars=160)

        job = store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id="1495",
            reply_to_message_id="msg42",
            idempotency_key="turn-1",
            text=text,
            chunks=chunks,
            provider="openrouter-coral",
            model="openai/gpt-audio-mini",
            voice="coral",
            scope_key="telegram:123:1495",
            policy={"target_chars": 120, "max_chars": 160},
        )

        loaded = store.get_job(job.job_id)
        assert loaded is not None
        assert loaded["text_sha256"]
        assert "First sentence" not in json.dumps(loaded)
        assert len(store.list_chunks(job.job_id)) == len(chunks)

    def test_enqueue_allows_default_provider_resolution_with_no_override(self, tmp_path):
        from gateway.tts_narration import NarrationJobStore

        store = NarrationJobStore(tmp_path / "narration.sqlite")

        job = store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id=None,
            reply_to_message_id="msg42",
            idempotency_key="default-provider",
            text="Use configured default provider.",
            chunks=["Use configured default provider."],
            provider=None,
            model=None,
            voice=None,
            scope_key="telegram:123",
            policy={"target_chars": 1000, "max_chars": 1200},
        )

        loaded = store.get_job(job.job_id)
        assert loaded["provider"] is None

    def test_init_migrates_legacy_provider_not_null_schema(self, tmp_path):
        from gateway.tts_narration import NarrationJobStore

        db_path = tmp_path / "legacy-narration.sqlite"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE tts_narration_jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE NOT NULL,
                    platform TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    thread_id TEXT,
                    reply_to_message_id TEXT,
                    provider TEXT NOT NULL,
                    model TEXT,
                    voice TEXT,
                    scope_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    text_chars INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    policy_json TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    last_error TEXT
                );
                CREATE TABLE tts_narration_chunks (
                    job_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    chunk_total INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    text_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    audio_path TEXT,
                    sent_message_id TEXT,
                    telegram_file_id TEXT,
                    last_error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, chunk_index),
                    FOREIGN KEY (job_id) REFERENCES tts_narration_jobs(job_id)
                );
                """
            )

        store = NarrationJobStore(db_path)
        with sqlite3.connect(db_path) as conn:
            provider_column = [row for row in conn.execute("PRAGMA table_info(tts_narration_jobs)") if row[1] == "provider"][0]
            assert provider_column[3] == 0
            fk = conn.execute("PRAGMA foreign_key_list(tts_narration_chunks)").fetchall()
            assert fk[0][2] == "tts_narration_jobs"

        job = store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id=None,
            reply_to_message_id="msg42",
            idempotency_key="migrated-default-provider",
            text="Use configured default provider after migration.",
            chunks=["Use configured default provider after migration."],
            provider=None,
            model=None,
            voice=None,
            scope_key="telegram:123",
            policy={"target_chars": 1000, "max_chars": 1200},
        )

        loaded = store.get_job(job.job_id)
        assert loaded is not None
        assert loaded["provider"] is None

    def test_completed_job_redacts_sent_chunk_text_but_keeps_hashes(self, tmp_path):
        from gateway.tts_narration import NarrationJobStore

        store = NarrationJobStore(tmp_path / "narration.sqlite")
        job = store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id=None,
            reply_to_message_id="msg42",
            idempotency_key="redact-complete",
            text="Private sentence.",
            chunks=["Private sentence."],
            provider=None,
            model=None,
            voice=None,
            scope_key="telegram:123",
            policy={"target_chars": 1000, "max_chars": 1200},
        )

        store.update_chunk(job.job_id, 1, status="sent", sent_message_id="sent-1")
        store.update_job_status(job.job_id, "complete")

        chunks = store.list_chunks(job.job_id)
        assert chunks[0]["text"] == ""
        assert chunks[0]["text_sha256"]
        assert "Private sentence" not in json.dumps(chunks)

    def test_idempotency_returns_existing_job_without_duplicate_chunks(self, tmp_path):
        from gateway.tts_narration import NarrationJobStore

        store = NarrationJobStore(tmp_path / "narration.sqlite")
        kwargs = dict(
            platform="telegram",
            chat_id="123",
            thread_id=None,
            reply_to_message_id="msg42",
            idempotency_key="same-turn",
            text="Read this once.",
            chunks=["Read this once."],
            provider="openrouter-coral",
            model="openai/gpt-audio-mini",
            voice="coral",
            scope_key="telegram:123",
            policy={"target_chars": 1000, "max_chars": 1200},
        )

        first = store.enqueue_job(**kwargs)
        second = store.enqueue_job(**kwargs)

        assert first.job_id == second.job_id
        assert len(store.list_jobs()) == 1
        assert len(store.list_chunks(first.job_id)) == 1

    def test_recover_stale_processing_marks_job_and_chunk_retryable(self, tmp_path):
        from gateway.tts_narration import NarrationJobStore

        store = NarrationJobStore(tmp_path / "narration.sqlite")
        job = store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id=None,
            reply_to_message_id="msg42",
            idempotency_key="stale-processing",
            text="First. Second.",
            chunks=["First.", "Second."],
            provider=None,
            model=None,
            voice=None,
            scope_key="telegram:123",
            policy={"target_chars": 1000, "max_chars": 1200},
        )
        assert store.claim_job(job.job_id) is True
        assert store.claim_chunk(job.job_id, 1) is True

        with sqlite3.connect(tmp_path / "narration.sqlite") as conn:
            conn.execute(
                "UPDATE tts_narration_jobs SET started_at = '2000-01-01T00:00:00+00:00' WHERE job_id = ?",
                (job.job_id,),
            )
            conn.execute(
                "UPDATE tts_narration_chunks SET updated_at = '2000-01-01T00:00:00+00:00' WHERE job_id = ? AND chunk_index = 1",
                (job.job_id,),
            )

        recovered = store.recover_stale_processing(older_than_seconds=900)

        assert recovered == 2
        loaded = store.get_job(job.job_id)
        assert loaded is not None
        assert loaded["status"] == "failed"
        chunks = store.list_chunks(job.job_id)
        assert chunks[0]["status"] == "failed"
        assert chunks[1]["status"] == "queued"
        assert "stale narration processing" in chunks[0]["last_error"]


class TestGatewayNarrationMode:
    @pytest.fixture
    def runner(self, tmp_path):
        from gateway.run import GatewayRunner
        from gateway.tts_narration import NarrationJobStore

        runner = object.__new__(GatewayRunner)
        runner.adapters = {}
        runner._voice_mode = {}
        runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
        runner._TTS_NARRATION_DB_PATH = tmp_path / "narration.sqlite"
        runner._tts_narration_store = NarrationJobStore(runner._TTS_NARRATION_DB_PATH)
        runner._session_key_for_source = lambda source: "session-key"
        runner._reply_anchor_for_event = lambda event: getattr(event, "message_id", None)
        runner._thread_metadata_for_source = lambda source, reply_anchor=None: {
            "thread_id": source.thread_id,
            "reply_to_message_id": reply_anchor,
        }
        return runner

    @pytest.mark.asyncio
    async def test_voice_narrate_sets_topic_scope_when_thread_present(self, runner):
        event = _event("/voice narrate", thread_id="1495")

        result = await runner._handle_voice_command(event)

        assert "narration" in result.lower()
        assert runner._voice_mode["telegram:123:1495"] == "narration"
        assert "telegram:123" not in runner._voice_mode

    @pytest.mark.asyncio
    async def test_voice_narrate_chat_sets_chat_scope(self, runner):
        event = _event("/voice narrate chat", thread_id="1495")

        await runner._handle_voice_command(event)

        assert runner._voice_mode["telegram:123"] == "narration"

    @pytest.mark.asyncio
    async def test_voice_off_in_topic_disables_topic_narration_override(self, runner):
        event = _event("/voice narrate", thread_id="1495")
        await runner._handle_voice_command(event)

        off_event = _event("/voice off", thread_id="1495")
        await runner._handle_voice_command(off_event)

        mode, scope_key = runner._resolve_voice_mode(_source(thread_id="1495"))
        assert mode == "off"
        assert scope_key == "telegram:123:1495"

    def test_topic_narration_overrides_chat_off(self, runner):
        runner._voice_mode["telegram:123"] = "off"
        runner._voice_mode["telegram:123:1495"] = "narration"

        mode, scope_key = runner._resolve_voice_mode(_source(thread_id="1495"))

        assert mode == "narration"
        assert scope_key == "telegram:123:1495"

    def test_load_voice_modes_normalizes_short_lived_thread_prefix_key(self, runner):
        runner._VOICE_MODE_PATH.write_text(json.dumps({
            "telegram:123:1495": "all",
            "telegram:123:thread:1495": "narration",
        }))

        loaded = runner._load_voice_modes()
        runner._voice_mode = loaded
        mode, scope_key = runner._resolve_voice_mode(_source(thread_id="1495"))

        assert loaded["telegram:123:1495"] == "narration"
        assert "telegram:123:thread:1495" not in loaded
        assert mode == "narration"
        assert scope_key == "telegram:123:1495"

    def test_load_voice_modes_keeps_canonical_explicit_topic_mode_over_stale_legacy(self, runner):
        runner._VOICE_MODE_PATH.write_text(json.dumps({
            "telegram:123:1495": "off",
            "telegram:123:thread:1495": "narration",
        }))

        loaded = runner._load_voice_modes()

        assert loaded["telegram:123:1495"] == "off"

    def test_load_voice_modes_only_normalizes_telegram_thread_prefix_keys(self, runner):
        runner._VOICE_MODE_PATH.write_text(json.dumps({
            "matrix:!room:thread:1495": "narration",
        }))

        loaded = runner._load_voice_modes()

        assert loaded["matrix:!room:thread:1495"] == "narration"

    def test_narration_mode_does_not_use_synchronous_voice_reply(self, runner):
        runner._voice_mode["telegram:123:1495"] = "narration"

        assert runner._should_send_voice_reply(_event(thread_id="1495"), "hello", []) is False
        assert runner._should_enqueue_narration(_event(thread_id="1495"), "hello", []) is True

    def test_topic_narration_suppresses_base_auto_tts_but_still_enqueues_narration(self, runner, monkeypatch):
        from gateway.config import Platform
        from gateway.platforms.base import BasePlatformAdapter

        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"voice": {"auto_tts": True}})
        runner._voice_mode["telegram:123:1495"] = "narration"
        adapter = SimpleNamespace(
            _auto_tts_default=False,
            _auto_tts_disabled_chats=set(),
            _auto_tts_enabled_chats=set(),
            platform=Platform.TELEGRAM,
        )
        runner._sync_voice_mode_state_to_adapter(adapter)
        event = _event(thread_id="1495")
        event.message_type = MessageType.VOICE

        assert BasePlatformAdapter._should_auto_tts_for_chat(adapter, event.source.chat_id, event.source) is False
        assert runner._should_enqueue_narration(event, "text first, then narrated", []) is True

    @pytest.mark.asyncio
    async def test_deferred_narration_registers_post_delivery_callback(self, runner):
        adapter = SimpleNamespace(
            register_post_delivery_callback=MagicMock(),
            _active_sessions={"session-key": SimpleNamespace(_hermes_run_generation=7)},
        )
        runner.adapters[Platform.TELEGRAM] = adapter
        event = _event(thread_id="1495")
        runner._voice_mode["telegram:123:1495"] = "narration"

        await runner._defer_narration_after_delivery(event, "Narrate this.")

        adapter.register_post_delivery_callback.assert_called_once()
        call = adapter.register_post_delivery_callback.call_args
        assert call.args[0] == "session-key"
        assert call.kwargs["generation"] == 7

    @pytest.mark.asyncio
    async def test_process_narration_job_sends_chunks_in_order_and_records_success(self, runner, tmp_path, monkeypatch):
        from gateway.tts_narration import chunk_narration_text

        audio_paths = []
        providers = []

        def fake_tts(text, output_path=None, provider=None):
            providers.append(provider)
            path = tmp_path / f"{len(audio_paths)}.ogg"
            path.write_bytes(b"OggS fake")
            audio_paths.append(str(path))
            return json.dumps({"success": True, "file_path": str(path), "provider": provider})

        monkeypatch.setattr("gateway.tts_narration.text_to_speech_tool", fake_tts)
        adapter = SimpleNamespace(send_voice=AsyncMock(return_value=SendResult(success=True, message_id="sent")))
        runner.adapters[Platform.TELEGRAM] = adapter
        event = _event(thread_id="1495")
        runner._voice_mode["telegram:123:1495"] = "narration"
        text = "One sentence. " * 120
        chunks = chunk_narration_text(text, target_chars=100, max_chars=130)

        job = runner._tts_narration_store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id="1495",
            reply_to_message_id="msg42",
            idempotency_key="turn-1",
            text=text,
            chunks=chunks,
            provider="openrouter-coral",
            model="openai/gpt-audio-mini",
            voice="coral",
            scope_key="telegram:123:1495",
            policy={"target_chars": 100, "max_chars": 130},
        )

        await runner._process_narration_job(job.job_id)

        stored_chunks = runner._tts_narration_store.list_chunks(job.job_id)
        assert [c["text"] for c in stored_chunks] == [""] * len(chunks)
        assert [c["text_sha256"] for c in stored_chunks] == [__import__("hashlib").sha256(chunk.encode("utf-8")).hexdigest() for chunk in chunks]
        assert [c["status"] for c in stored_chunks] == ["sent"] * len(chunks)
        assert adapter.send_voice.await_count == len(chunks)
        assert providers == ["openrouter-coral"] * len(chunks)
        assert runner._tts_narration_store.get_job(job.job_id)["status"] == "complete"

    @pytest.mark.asyncio
    async def test_process_narration_job_sends_exact_chunk_text_to_tts(self, runner, tmp_path, monkeypatch):
        seen = []

        def fake_tts(text, output_path=None, provider=None):
            seen.append(text)
            path = tmp_path / f"{len(seen)}.ogg"
            path.write_bytes(b"OggS fake")
            return json.dumps({"success": True, "file_path": str(path), "provider": provider})

        monkeypatch.setattr("gateway.tts_narration.text_to_speech_tool", fake_tts)
        adapter = SimpleNamespace(send_voice=AsyncMock(return_value=SendResult(success=True, message_id="sent")))
        runner.adapters[Platform.TELEGRAM] = adapter
        job = runner._tts_narration_store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id=None,
            reply_to_message_id="msg42",
            idempotency_key="turn-exact",
            text="First. Second.",
            chunks=["First.", "Second."],
            provider="openrouter-coral",
            model="openai/gpt-audio-mini",
            voice="coral",
            scope_key="telegram:123",
            policy={"target_chars": 1000, "max_chars": 1200},
        )

        await runner._process_narration_job(job.job_id)

        assert seen == ["First.", "Second."]

    @pytest.mark.asyncio
    async def test_process_narration_job_redacts_skipped_media_only_chunk_on_completion(self, runner, monkeypatch):
        calls = []

        def fake_tts(text, output_path=None, provider=None):
            calls.append(text)
            return json.dumps({"success": False, "error": "should not run"})

        monkeypatch.setattr("gateway.tts_narration.text_to_speech_tool", fake_tts)
        adapter = SimpleNamespace(send_voice=AsyncMock(return_value=SendResult(success=True, message_id="sent")))
        runner.adapters[Platform.TELEGRAM] = adapter
        private_media_text = "MEDIA:/tmp/private-file.png"
        job = runner._tts_narration_store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id=None,
            reply_to_message_id="msg42",
            idempotency_key="turn-media-only",
            text=private_media_text,
            chunks=[private_media_text],
            provider="openrouter-coral",
            model="openai/gpt-audio-mini",
            voice="coral",
            scope_key="telegram:123",
            policy={"target_chars": 1000, "max_chars": 1200},
        )

        await runner._process_narration_job(job.job_id)

        chunks = runner._tts_narration_store.list_chunks(job.job_id)
        assert runner._tts_narration_store.get_job(job.job_id)["status"] == "complete"
        assert chunks[0]["status"] == "skipped"
        assert chunks[0]["text"] == ""
        assert chunks[0]["text_sha256"] == __import__("hashlib").sha256(private_media_text.encode("utf-8")).hexdigest()
        assert calls == []
        adapter.send_voice.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_narration_job_preserves_stored_telegram_topic_metadata(self, runner, tmp_path, monkeypatch):
        def fake_tts(text, output_path=None, provider=None):
            path = tmp_path / "chunk.ogg"
            path.write_bytes(b"OggS fake")
            return json.dumps({"success": True, "file_path": str(path)})

        monkeypatch.setattr("gateway.tts_narration.text_to_speech_tool", fake_tts)
        adapter = SimpleNamespace(send_voice=AsyncMock(return_value=SendResult(success=True, message_id="sent")))
        runner.adapters[Platform.TELEGRAM] = adapter
        metadata = {
            "thread_id": "1495",
            "direct_messages_topic_id": "1495",
            "telegram_reply_to_message_id": "msg42",
        }
        job = runner._tts_narration_store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id="1495",
            reply_to_message_id="msg42",
            idempotency_key="turn-metadata",
            text="First.",
            chunks=["First."],
            provider="openrouter-coral",
            model="openai/gpt-audio-mini",
            voice="coral",
            scope_key="telegram:123:1495",
            policy={"target_chars": 1000, "max_chars": 1200},
            metadata=metadata,
        )

        await runner._process_narration_job(job.job_id)

        sent_metadata = adapter.send_voice.await_args.kwargs["metadata"]
        assert sent_metadata["thread_id"] == "1495"
        assert sent_metadata["direct_messages_topic_id"] == "1495"
        assert sent_metadata["telegram_reply_to_message_id"] == "msg42"
        assert sent_metadata["notify"] is True

    @pytest.mark.asyncio
    async def test_process_narration_job_stops_on_send_failure_without_path_leak(self, runner, tmp_path, monkeypatch):
        def fake_tts(text, output_path=None, provider=None):
            path = tmp_path / "chunk.ogg"
            path.write_bytes(b"OggS fake")
            return json.dumps({"success": True, "file_path": str(path)})

        monkeypatch.setattr("gateway.tts_narration.text_to_speech_tool", fake_tts)
        adapter = SimpleNamespace(send_voice=AsyncMock(return_value=SendResult(success=False, error="telegram failed")))
        runner.adapters[Platform.TELEGRAM] = adapter
        job = runner._tts_narration_store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id=None,
            reply_to_message_id="msg42",
            idempotency_key="turn-2",
            text="First. Second.",
            chunks=["First.", "Second."],
            provider="openrouter-coral",
            model="openai/gpt-audio-mini",
            voice="coral",
            scope_key="telegram:123",
            policy={"target_chars": 1000, "max_chars": 1200},
        )

        await runner._process_narration_job(job.job_id)

        chunks = runner._tts_narration_store.list_chunks(job.job_id)
        assert chunks[0]["status"] == "failed"
        assert chunks[1]["status"] == "queued"
        assert runner._tts_narration_store.get_job(job.job_id)["status"] == "failed"
        assert "/" not in (runner._tts_narration_store.get_job(job.job_id)["last_error"] or "")

    @pytest.mark.asyncio
    async def test_failed_job_retry_retries_failed_chunk_before_later_chunks(self, runner, tmp_path, monkeypatch):
        calls = []

        def fake_tts(text, output_path=None, provider=None):
            calls.append(text)
            path = tmp_path / f"{len(calls)}.ogg"
            path.write_bytes(b"OggS fake")
            return json.dumps({"success": True, "file_path": str(path)})

        monkeypatch.setattr("gateway.tts_narration.text_to_speech_tool", fake_tts)
        send_results = [
            SendResult(success=False, error="telegram failed"),
            SendResult(success=True, message_id="sent-1"),
            SendResult(success=True, message_id="sent-2"),
        ]
        adapter = SimpleNamespace(send_voice=AsyncMock(side_effect=send_results))
        runner.adapters[Platform.TELEGRAM] = adapter
        job = runner._tts_narration_store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id=None,
            reply_to_message_id="msg42",
            idempotency_key="turn-retry",
            text="First. Second.",
            chunks=["First.", "Second."],
            provider="openrouter-coral",
            model="openai/gpt-audio-mini",
            voice="coral",
            scope_key="telegram:123",
            policy={"target_chars": 1000, "max_chars": 1200},
        )

        await runner._process_narration_job(job.job_id)
        assert [c["status"] for c in runner._tts_narration_store.list_chunks(job.job_id)] == ["failed", "queued"]

        await runner._process_narration_job(job.job_id)

        assert calls == ["First.", "First.", "Second."]
        assert [c["status"] for c in runner._tts_narration_store.list_chunks(job.job_id)] == ["sent", "sent"]
        assert runner._tts_narration_store.get_job(job.job_id)["status"] == "complete"

    @pytest.mark.asyncio
    async def test_cancelled_narration_marks_active_chunk_failed_and_retry_resumes_in_order(self, runner, tmp_path, monkeypatch):
        calls = []
        cancelled = {"done": False}

        def flaky_tts(text, output_path=None, provider=None):
            calls.append(text)
            if text == "Fourth." and not cancelled["done"]:
                cancelled["done"] = True
                raise asyncio.CancelledError()
            path = tmp_path / f"{len(calls)}.ogg"
            path.write_bytes(b"OggS fake")
            return json.dumps({"success": True, "file_path": str(path)})

        monkeypatch.setattr("gateway.tts_narration.text_to_speech_tool", flaky_tts)
        adapter = SimpleNamespace(send_voice=AsyncMock(return_value=SendResult(success=True, message_id="sent")))
        runner.adapters[Platform.TELEGRAM] = adapter
        job = runner._tts_narration_store.enqueue_job(
            platform="telegram",
            chat_id="123",
            thread_id="1495",
            reply_to_message_id="msg42",
            idempotency_key="turn-cancelled",
            text="First. Second. Third. Fourth. Fifth.",
            chunks=["First.", "Second.", "Third.", "Fourth.", "Fifth."],
            provider="openrouter-coral",
            model="openai/gpt-audio-mini",
            voice="coral",
            scope_key="telegram:123:1495",
            policy={"target_chars": 1000, "max_chars": 1200},
        )

        with pytest.raises(asyncio.CancelledError):
            await runner._process_narration_job(job.job_id)

        chunks = runner._tts_narration_store.list_chunks(job.job_id)
        assert [c["status"] for c in chunks] == ["sent", "sent", "sent", "failed", "queued"]
        assert "cancelled" in (chunks[3]["last_error"] or "")
        assert runner._tts_narration_store.get_job(job.job_id)["status"] == "failed"

        await runner._process_narration_job(job.job_id)

        assert calls == ["First.", "Second.", "Third.", "Fourth.", "Fourth.", "Fifth."]
        assert [c["status"] for c in runner._tts_narration_store.list_chunks(job.job_id)] == [
            "sent",
            "sent",
            "sent",
            "sent",
            "sent",
        ]
        assert runner._tts_narration_store.get_job(job.job_id)["status"] == "complete"
