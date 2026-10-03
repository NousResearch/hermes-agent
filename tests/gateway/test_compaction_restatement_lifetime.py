"""PR #130909: replay provenance must survive a persisted proactive prune.

Extends @andrexibiza's real-compressor/SQLite lifetime reproduction; only the
summarizer response is substituted, and no provider request is sent.
"""
from types import SimpleNamespace
from unittest.mock import patch
import time

import pytest

from agent.context_compressor import (
    ContextCompressor,
    INFLIGHT_TASK_REPLAY_METADATA_KEY,
    _INFLIGHT_TASK_REPLAY_HEADER,
    _SUMMARY_END_MARKER,
)
from agent.turn_context import build_api_messages
from agent.transports.chat_completions import ChatCompletionsTransport
from gateway.config import GatewayConfig
from gateway.message_timestamps import render_user_content_with_timestamp
from gateway.run import _build_gateway_agent_history
from gateway.session import SessionStore
from hermes_state import SessionDB
from hermes_time import get_timezone


def _append_read_exchange(messages, prefix):
    for i in range(40):
        call_id = f"{prefix}-{i}"
        messages.extend([
            {"role": "assistant", "content": f"step {call_id}", "tool_calls": [
                {"id": call_id, "type": "function", "function": {
                    "name": "read_file", "arguments": "{}",
                }},
            ]},
            {"role": "tool", "tool_call_id": call_id, "content": "fixture details " * 200},
        ])


def _produce(merged):
    now = time.time()
    messages = [
        {"role": "system", "content": "Stable system"},
        {"role": "user", "content": render_user_content_with_timestamp("Set project scope", now),
         "timestamp": now},
        {"role": "assistant", "content": "Scope noted"},
        {"role": "user", "content": render_user_content_with_timestamp("Summarize project files", now),
         "timestamp": now},
    ]
    _append_read_exchange(messages, "first")
    compressor = ContextCompressor(
        model="test", quiet_mode=True, config_context_length=100_000,
        protect_first_n=4, protect_last_n=2,
    )
    compressor.tail_token_budget = 500
    if merged:
        # Later compactions decay the protected head and merge the restatement
        # onto a user-role carrier instead of appending a standalone user row.
        compressor.compression_count = 1
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="## Historical Task Snapshot\nReview project files.\n## Summary\nFiles read.",
    ))])
    with patch("agent.context_compressor.call_llm", return_value=response):
        out = compressor.compress(messages, current_tokens=200_000, force=True)
    out.append({"role": "assistant", "content": "Done"})
    return out


def _task(messages):
    return next(row for row in messages if row["role"] == "user"
                and _INFLIGHT_TASK_REPLAY_HEADER in str(row.get("content")))


def _reload(tmp_path, path):
    db = SessionDB(db_path=path)
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    store._db = db
    try:
        loaded = store.load_transcript("review")
        on, _ = _build_gateway_agent_history(loaded, inject_timestamps=True)
        off, _ = _build_gateway_agent_history(loaded, inject_timestamps=False)
        return loaded, on, off
    finally:
        db.close()
        store.close_all_db_handles()


class _SendAgent:
    api_mode = "chat_completions"
    ephemeral_system_prompt = None
    _compression_warning = None
    _current_turn_timestamp = 1_700_000_000.0

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        pass

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


def _wire(rows):
    request, _ = build_api_messages(
        _SendAgent(), rows, current_turn_user_idx=None, ext_prefetch_cache="",
        plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    assert all("display_metadata" not in row for row in request)
    wire = ChatCompletionsTransport().convert_messages(request)
    assert all("display_metadata" not in row for row in wire)
    return wire


@pytest.mark.parametrize("merged", [False, True], ids=["standalone", "merged"])
def test_restatement_survives_gateway_persisted_prune_and_reload(tmp_path, merged):
    produced = _produce(merged)
    original = _task(produced)["content"]
    assert bool(_task(produced).get("_compressed_summary")) is merged
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session("review", source="telegram")
        db.archive_and_compact("review", produced)
    finally:
        db.close()
    loaded, replay, off = _reload(tmp_path, path)
    assert _task(loaded)["display_metadata"][INFLIGHT_TASK_REPLAY_METADATA_KEY] is True
    assert _task(replay)["content"] == _task(off)["content"] == original
    assert _task(_wire(produced))["content"] == _task(_wire(replay))["content"] == original

    # Persist the RETURNED history, not the original DB rows or a harness repair.
    replay.append({"role": "user", "content": "Count project files", "timestamp": time.time()})
    _append_read_exchange(replay, "second")
    compressor = ContextCompressor(
        model="test", quiet_mode=True, config_context_length=100_000,
        protect_last_n=2, proactive_prune_tokens=1_000, proactive_prune_min_reclaim_tokens=1,
    )
    db = SessionDB(db_path=path)
    compressor._session_db, compressor._session_id = db, "review"
    try:
        pruned, count = compressor.prune_tool_results_only(replay, current_tokens=60_000)
        assert count > 0
    finally:
        db.close()
    assert _task(pruned)["content"] == original
    assert bool(_task(pruned).get("_compressed_summary")) is merged
    loaded2, replay2, off2 = _reload(tmp_path, path)
    assert _task(off2)["content"] == original
    assert _task(_wire(pruned))["content"] == original
    assert _task(_wire(replay2))["content"] == original
    for rows in (replay, pruned, loaded2, replay2, off2):
        assert _task(rows)["display_metadata"][INFLIGHT_TASK_REPLAY_METADATA_KEY] is True


@pytest.mark.parametrize("prefix", ["", "The log includes: ", _SUMMARY_END_MARKER + "\n\n"])
def test_human_header_copy_keeps_timestamp_after_repeated_projection(tmp_path, prefix):
    timestamp = 1_700_000_000.0
    raw = prefix + _INFLIGHT_TASK_REPLAY_HEADER + "\nThis is a pasted log"
    messages = [
        {"role": "user", "content": raw, "timestamp": timestamp,
         "display_metadata": {"unrelated_display_hint": True}},
        {"role": "assistant", "content": "Noted"},
    ]
    expected = render_user_content_with_timestamp(raw, timestamp, tz=get_timezone())
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    try:
        db.create_session("review", source="telegram")
        db.archive_and_compact("review", messages)
    finally:
        db.close()
    _, replay, off = _reload(tmp_path, path)
    assert replay[0]["content"] == expected
    assert off[0]["content"] == raw
    assert not replay[0].get("display_metadata", {}).get(INFLIGHT_TASK_REPLAY_METADATA_KEY)
    db = SessionDB(db_path=path)
    try:
        db.archive_and_compact("review", replay)
    finally:
        db.close()
    _, replay2, _ = _reload(tmp_path, path)
    assert replay2[0]["content"] == expected
    assert not replay2[0].get("display_metadata", {}).get(INFLIGHT_TASK_REPLAY_METADATA_KEY)
