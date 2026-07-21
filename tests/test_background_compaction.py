"""Behavior tests for non-blocking gateway transcript compaction."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.background_compaction import (
    BackgroundCompactionSettings,
    GatewayBackgroundCompactionMixin,
    select_compaction_prefix,
)
from hermes_state import AsyncSessionDB, SessionDB
from agent.context_compressor import ContextCompressor
from gateway.turn_lease import SessionTurnLeaseRegistry


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session("session-1", source="discord", model="test-model")
    yield session_db
    session_db.close()


def _append_exchange(db, label):
    db.append_message("session-1", role="user", content=f"user {label}")
    db.append_message("session-1", role="assistant", content=f"assistant {label}")


def test_snapshot_commit_preserves_turns_appended_during_summary(db):
    _append_exchange(db, "one")
    _append_exchange(db, "two")
    _append_exchange(db, "three")
    snapshot = db.get_compaction_snapshot("session-1")
    prefix_ids = snapshot["active_message_ids"][:4]

    # This exchange lands while the model is supposedly generating a summary.
    _append_exchange(db, "four")

    result = db.commit_compaction_snapshot(
        "session-1",
        prefix_ids,
        [{"role": "user", "content": "Summary of exchanges one and two."}],
    )

    assert result == {
        "status": "committed",
        "session_id": "session-1",
        "active_count": 5,
        "archived_prefix_count": 4,
        "preserved_tail_count": 4,
    }
    active = db.get_messages_as_conversation("session-1")
    assert [message["content"] for message in active] == [
        "Summary of exchanges one and two.",
        "user three",
        "assistant three",
        "user four",
        "assistant four",
    ]
    archived = db._conn.execute(
        "SELECT content, compacted FROM messages "
        "WHERE session_id = ? AND active = 0 ORDER BY id",
        ("session-1",),
    ).fetchall()
    assert [row["compacted"] for row in archived[:4]] == [1, 1, 1, 1]
    # Superseded copies of the unchanged tail are recoverable but hidden from
    # search to avoid duplicate results.
    assert [row["compacted"] for row in archived[4:]] == [0, 0, 0, 0]


def test_snapshot_commit_discards_result_after_history_rewrite(db):
    _append_exchange(db, "one")
    _append_exchange(db, "two")
    snapshot = db.get_compaction_snapshot("session-1")

    replacement = [
        {"role": "user", "content": "replacement user"},
        {"role": "assistant", "content": "replacement assistant"},
    ]
    db.replace_messages("session-1", replacement, active_only=True)
    result = db.commit_compaction_snapshot(
        "session-1",
        snapshot["active_message_ids"][:2],
        [{"role": "user", "content": "stale summary"}],
    )

    assert result["status"] == "stale"
    assert result["reason"] == "prefix_changed"
    active = db.get_messages_as_conversation("session-1")
    assert [(message["role"], message["content"]) for message in active] == [
        ("user", "replacement user"),
        ("assistant", "replacement assistant"),
    ]


def test_snapshot_commit_discards_result_after_session_end(db):
    _append_exchange(db, "one")
    snapshot = db.get_compaction_snapshot("session-1")
    db.end_session("session-1", "reset")

    result = db.commit_compaction_snapshot(
        "session-1",
        snapshot["active_message_ids"],
        [{"role": "user", "content": "stale summary"}],
    )

    assert result["status"] == "stale"
    assert result["reason"] == "ended_session"
    assert [m["content"] for m in db.get_messages_as_conversation("session-1")] == [
        "user one",
        "assistant one",
    ]


def test_snapshot_ids_are_internal_to_snapshot_projection(db):
    _append_exchange(db, "one")
    snapshot = db.get_compaction_snapshot("session-1")

    assert all("_state_message_id" in message for message in snapshot["messages"])
    assert all(
        "_state_message_id" not in message
        for message in db.get_messages_as_conversation("session-1")
    )


def test_prefix_selection_stops_at_clean_turn_boundary_and_keeps_tail():
    messages = []
    for index in range(8):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append(
            {
                "role": role,
                "content": f"{role}-{index} " + ("x" * 500),
                "_state_message_id": index + 1,
            }
        )

    prefix = select_compaction_prefix(
        messages,
        target_tokens=1_000,
        protect_tail_messages=4,
    )

    assert prefix is not None
    assert len(prefix.messages) == 4
    assert prefix.messages[-1]["role"] == "assistant"
    assert messages[len(prefix.messages)]["role"] == "user"
    assert prefix.last_state_message_id == 4
    assert all("_state_message_id" not in message for message in prefix.messages)


def test_prefix_selection_never_drops_unrecognised_covered_roles():
    messages = [
        {
            "role": "system",
            "content": "legacy persisted system prompt",
            "_state_message_id": 1,
        }
    ]
    for index in range(8):
        role = "user" if index % 2 == 0 else "assistant"
        messages.append(
            {
                "role": role,
                "content": f"{role}-{index} " + ("x" * 500),
                "_state_message_id": index + 2,
            }
        )

    assert (
        select_compaction_prefix(
            messages,
            target_tokens=1_000,
            protect_tail_messages=4,
        )
        is None
    )


def test_background_settings_are_safe_and_use_main_model_by_default():
    settings = BackgroundCompactionSettings.from_config(
        {
            "compression": {
                "background_threshold": 9,
                "background_chunk_tokens": 10,
            }
        }
    )

    assert settings.threshold == 0.84
    assert settings.chunk_tokens == 16_000
    assert settings.use_main_model is True


def test_background_summary_force_routes_to_active_chat_runtime():
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=200_000,
    ):
        compressor = ContextCompressor(
            model="grok-chat-model",
            provider="xai-oauth",
            base_url="https://example.invalid/v1",
            api_key="fake-key",
            quiet_mode=True,
        )
    compressor.force_main_runtime = True
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Summary body"))]
    )

    with patch("agent.context_compressor.call_llm", return_value=response) as call:
        summary = compressor._generate_summary(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        )

    assert "Summary body" in summary
    kwargs = call.call_args.kwargs
    assert kwargs["task"] == "compression"
    assert kwargs["provider"] == "xai-oauth"
    assert kwargs["model"] == "grok-chat-model"
    assert kwargs["base_url"] == "https://example.invalid/v1"
    assert kwargs["api_key"] == "fake-key"


class _SchedulerHarness(GatewayBackgroundCompactionMixin):
    def __init__(self):
        self._session_db = object()
        self._background_tasks = set()
        self._init_background_compaction()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def _background_compaction_settings(self):
        return BackgroundCompactionSettings(), {}

    async def _run_background_compaction(self, **kwargs):
        self.started.set()
        await self.release.wait()


@pytest.mark.asyncio
async def test_scheduler_deduplicates_one_job_per_durable_session():
    runner = _SchedulerHarness()
    first = runner._maybe_schedule_background_compaction(
        source=object(),
        session_key="discord:chat",
        session_id="session-1",
        observed_prompt_tokens=70,
        context_length=100,
    )
    await runner.started.wait()
    second = runner._maybe_schedule_background_compaction(
        source=object(),
        session_key="discord:alias",
        session_id="session-1",
        observed_prompt_tokens=80,
        context_length=100,
    )

    assert first is True
    assert second is False
    assert len(runner._background_compaction_tasks) == 1
    runner.release.set()
    await asyncio.gather(*runner._background_tasks)


class _AsyncSessionStoreStub:
    def __init__(self):
        self.updates = []

    async def update_session(self, session_key, last_prompt_tokens=None):
        self.updates.append((session_key, last_prompt_tokens))


class _BackgroundRunHarness(GatewayBackgroundCompactionMixin):
    def __init__(self, db):
        self._session_db = AsyncSessionDB(db)
        self._turn_leases = SessionTurnLeaseRegistry()
        self.async_session_store = _AsyncSessionStoreStub()
        self.evicted = []

    def _resolve_session_agent_runtime(self, **kwargs):
        return "test-model", {
            "provider": "custom",
            "base_url": "https://example.invalid/v1",
            "api_key": "fake-key",
            "api_mode": "chat_completions",
            "max_tokens": None,
        }

    async def _run_in_executor_with_context(self, func, *args):
        return await asyncio.to_thread(func, *args)

    def _evict_cached_agent(self, session_key):
        self.evicted.append(session_key)


@pytest.mark.asyncio
async def test_background_run_allows_append_during_model_call_and_commits(db, monkeypatch):
    for index in range(15):
        db.append_message(
            "session-1",
            role="user",
            content=f"old user {index} " + ("u" * 1_000),
        )
        db.append_message(
            "session-1",
            role="assistant",
            content=f"old assistant {index} " + ("a" * 1_000),
        )

    summary_started = threading.Event()
    release_summary = threading.Event()

    def _blocking_summary(self, messages, **kwargs):
        summary_started.set()
        assert release_summary.wait(timeout=5)
        return [{"role": "user", "content": "background summary"}]

    monkeypatch.setattr(ContextCompressor, "compress", _blocking_summary)
    runner = _BackgroundRunHarness(db)
    task = asyncio.create_task(
        runner._run_background_compaction(
            source=object(),
            session_key="discord:chat",
            session_id="session-1",
            observed_prompt_tokens=70_000,
            context_length=100_000,
            settings=BackgroundCompactionSettings(
                chunk_tokens=16_000,
                protect_tail_messages=4,
            ),
            user_config={"context": {"engine": "compressor"}},
            generation=1,
        )
    )
    assert await asyncio.to_thread(summary_started.wait, 5)

    # The model call is still blocked, but the canonical DB accepts the next
    # exchange. Prefix-CAS publication must rebase this tail, not discard it.
    await asyncio.to_thread(_append_exchange, db, "during summary")
    release_summary.set()
    await task

    active = db.get_messages_as_conversation("session-1")
    assert active[0]["content"] == "background summary"
    assert [message["content"] for message in active[-2:]] == [
        "user during summary",
        "assistant during summary",
    ]
    assert runner.async_session_store.updates == [("discord:chat", 0)]
    assert runner.evicted == ["discord:chat"]
