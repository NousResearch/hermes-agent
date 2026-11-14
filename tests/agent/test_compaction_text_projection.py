"""Compaction writes the same text-only rows as the per-tool flush.

The flush stores a multimodal message as its text projection (``session_persistence._durable_content``:
text parts kept, each image part a ``[screenshot]`` line at its position). The compaction writers handed
SessionDB the live dicts instead, so every generation they wrote stored the base64 image again: one session
held 51 MB of ``vision_analyze`` payload in its active rows, which every request rebuilt from the store sends
again (#148146), and each compaction stored it once more (#137540). The live dicts keep their parts, so the
model still sees the image for the rest of the process.

The checks that recognise a carried row compare its copy with the stored row, so they also need the copy in
the stored representation: otherwise the image row looks edited and is stored twice.
"""

from __future__ import annotations

import os
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY, ContextCompressor, _fresh_compaction_message_copy,
)
from agent.session_persistence import _durable_content
from hermes_state import SessionDB

_IMAGE_URL = "data:image/png;base64," + "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB" * 64
_IMAGE_RESULT = [
    {"type": "text", "text": "Image loaded into your context."},
    {"type": "image_url", "image_url": {"url": _IMAGE_URL}},
]
_SID = "vision-session"


def _call(call_id: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "vision_analyze", "arguments": "{}"}}]}


def _result(call_id: str, content) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": "vision_analyze", "tool_name": "vision_analyze",
            "content": content}


def _vision_turn() -> list[dict]:
    return [
        {"role": "user", "content": "an earlier question " * 400},
        {"role": "assistant", "content": "an earlier answer " * 400},
        {"role": "user", "content": "what is on this label?"},
        _call("call_1"),
        _result("call_1", list(_IMAGE_RESULT)),
        {"role": "assistant", "content": "The label says 42."},
    ]


def _tool_history() -> list[dict]:
    """Three large text results the prune summarizes, then five image results: all but the newest three
    images are retired, the rest stay live."""
    messages: list[dict] = [{"role": "user", "content": "start"}]
    for index in range(8):
        messages.append(_call(f"call_{index}"))
        messages.append(_result(f"call_{index}", chr(65 + index) * 24_000 if index < 3 else list(_IMAGE_RESULT)))
    return messages


def _agent(db: SessionDB):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent(api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
                       quiet_mode=True, session_db=db, session_id=_SID, skip_context_files=True, skip_memory=True)


def _rows_with_base64(db: SessionDB) -> int:
    """Rows of any state holding a data URL; as a BLOB, so the search reads past a ``\\x00json:`` prefix."""
    return db._conn.execute("SELECT COUNT(*) FROM messages WHERE instr(CAST(content AS BLOB), "
                            "CAST(';base64,' AS BLOB)) > 0").fetchone()[0]


def _compaction(db: SessionDB, *, in_place: bool, messages: list[dict] | None = None):
    """A vision turn the agent flushed as usual, then one compaction that carries the image row: ``compress()``
    returns a summary of the earlier exchange and marker-swept copies of the kept tail."""
    agent = _agent(db)
    compressor = MagicMock()
    compressor.compress.side_effect = lambda incoming, **_kw: [
        {"role": "user", "content": "[CONTEXT COMPACTION] the user asked about a label",
         COMPRESSED_SUMMARY_METADATA_KEY: True},
        *({**_fresh_compaction_message_copy(m), "_compaction_tail": True} for m in incoming[2:])]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = compressor.last_completion_tokens = 0
    compressor._last_summary_error = compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    compressor._last_compress_aborted = False
    agent.context_compressor, agent.compression_in_place = compressor, in_place
    agent._compression_feasibility_checked = True  # the stub never calls an auxiliary model
    messages = messages or _vision_turn()
    agent._flush_messages_to_session_db(messages, [])
    compressed, _prompt = agent._compress_context(messages, "sys", approx_tokens=120_000)
    # The commit happened: an in-place generation, or a published child.
    assert agent._last_compaction_in_place if in_place else agent.session_id != _SID
    return agent, compressed


def _micro_compaction(db: SessionDB):
    """Micro-compaction names the originals of the rows it carries by row identity (#118481)."""
    messages = _vision_turn()
    _agent(db)._flush_messages_to_session_db(messages, [])
    compressor = ContextCompressor.__new__(ContextCompressor)
    compressor._session_db, compressor._session_id = db, _SID
    marker = {"role": "assistant", "content": "[micro summary] an earlier answer",
              COMPRESSED_SUMMARY_METADATA_KEY: True}
    compacted = [messages[0], marker, *messages[2:]]
    assert compressor._sync_micro_compact_to_db(compacted)
    return _SID, compacted


def _pruning_agent(db: SessionDB):
    agent = _agent(db)
    compressor = agent.context_compressor
    compressor.proactive_prune_tokens, compressor.proactive_prune_min_result_chars = 48_000, 8_000
    compressor.proactive_prune_min_reclaim_tokens = 4_096
    compressor.protect_first_n, compressor.protect_last_n = 2, 4
    return agent


def _prune(db: SessionDB, *, full_rewrite: bool):
    """The proactive prune rewrites the rows it retires in place, or through ``archive_and_compact`` when a row
    cannot be named."""
    from hermes_state_errors import PruneRowUnresolvedError

    agent, messages = _pruning_agent(db), _tool_history()
    agent._flush_messages_to_session_db(messages, [])
    refuse = PruneRowUnresolvedError("no single row") if full_rewrite else None
    with patch.object(db, "rewrite_pruned_rows", wraps=db.rewrite_pruned_rows, side_effect=refuse), \
            patch.object(db, "archive_and_compact", wraps=db.archive_and_compact) as rewrite_all:
        pruned, count = agent.context_compressor.prune_tool_results_only(messages, current_tokens=120_000)
    assert count >= 1 and rewrite_all.call_count == full_rewrite
    return _SID, pruned


def _in_place_or_rotation(db: SessionDB, *, in_place: bool):
    agent, compressed = _compaction(db, in_place=in_place)
    return agent.session_id, compressed


def _acp_save(db: SessionDB):
    """The ACP save of an agent that does not persist itself replaces the session's rows with its history."""
    from acp_adapter.session import SessionManager, SessionState

    live = _vision_turn()
    SessionManager(db=db)._persist(SessionState(session_id=_SID, agent=SimpleNamespace(), history=live))
    return _SID, live


def _tui_branch(db: SessionDB):
    from tui_gateway import server

    db.create_session("parent", source="desktop", model="m")
    live = _vision_turn()
    server._persist_branch(db, "child", "parent", "Branch", live, source="desktop", cwd=".", profile_name="default",
                           model="m")
    return "child", live


def _tui_seed(db: SessionDB):
    """A seeded TUI session (a branch's copied parent, a client's opening turns) persists its history once."""
    from tui_gateway import server

    db.create_session("seeded", source="desktop", model="m")
    live = _vision_turn()
    session = {"session_key": "seeded", "seeded": True, "history_lock": threading.Lock(), "history": live,
               "profile_home": str(Path(db.db_path).parent)}
    server._persist_branch_seed(session)
    assert session.get("_branch_seed_persisted")
    return "seeded", live


_WRITERS = {
    "in_place": partial(_in_place_or_rotation, in_place=True),
    "rotation": partial(_in_place_or_rotation, in_place=False),
    "micro_compaction": _micro_compaction,
    "prune": partial(_prune, full_rewrite=False),
    "prune_full_rewrite": partial(_prune, full_rewrite=True),
    "acp_save": _acp_save,
    "tui_branch": _tui_branch,
    "tui_seed": _tui_seed,
}


@pytest.mark.parametrize("writer", list(_WRITERS))
def test_a_new_generation_stores_the_flush_text_while_the_live_dicts_keep_the_image(tmp_path: Path, writer: str):
    """Every writer that stores message dicts as rows (a compaction, a rewrite, a copy into another session)."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        session_id, live = _WRITERS[writer](db)
        stored = db.get_messages(session_id)
        assert len(stored) == len(live)
        assert not any(isinstance(message["content"], list) for message in stored)
        images = [index for index, message in enumerate(live) if message["content"] == _IMAGE_RESULT]
        assert images and all(stored[index]["content"] == _durable_content(_IMAGE_RESULT) for index in images)
        assert _rows_with_base64(db) == 0
    finally:
        db.close()


def test_micro_compaction_supersedes_the_image_row_it_carries(tmp_path: Path) -> None:
    """The image row's original is superseded by its copy, not kept as summarized history beside it: the commit
    names it by comparing the carried copy with the stored row."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        _micro_compaction(db)
        assert [row[0] for row in db._conn.execute(
            "SELECT compacted FROM messages WHERE role = 'tool' AND tool_call_id = 'call_1' AND active = 0")] == [0]
    finally:
        db.close()


def test_a_restored_copy_of_a_carried_image_row_is_not_stored_again(tmp_path: Path) -> None:
    """Copies of the carried vision turn that crossed a wire strip (no row id, digest or marker; their
    ``message_uid`` kept) are the re-insert the flush's dedupe skips. It compares the flush projection
    with the stored row, so a compaction that stored the raw parts made the image row look edited and
    inserted it again."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        agent, compressed = _compaction(db, in_place=True)
        active_before = len(db.get_messages(_SID))
        restored = [{k: v for k, v in m.items() if not k.startswith("_")} for m in compressed
                    if not m.get(COMPRESSED_SUMMARY_METADATA_KEY)]
        agent._flush_messages_to_session_db(restored, None)
        assert len(db.get_messages(_SID)) == active_before
    finally:
        db.close()


def test_the_prune_still_rewrites_an_image_row_an_older_build_stored_with_its_parts(tmp_path: Path) -> None:
    """An older build's compaction stored the parts themselves. A row-addressed resume holds them without a row
    digest, so the prune names the row by what it holds and still rewrites it, now as text, instead of reading
    the row as rewritten by another writer and skipping the pass."""
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session(_SID, source="cli")
        db.append_messages_batch(_SID, _tool_history())  # the shape an older build's compaction wrote
        compressor = _pruning_agent(db).context_compressor
        messages = db.get_messages_as_conversation(_SID, include_row_ids=True)
        with patch.object(db, "archive_and_compact", wraps=db.archive_and_compact) as rewrite_all:
            pruned, count = compressor.prune_tool_results_only(messages, current_tokens=120_000)
        assert count >= 1 and rewrite_all.call_count == 0
        retired = [i for i, (old, new) in enumerate(zip(messages, pruned))
                   if isinstance(old["content"], list) and old["content"] != new["content"]]
        stored = db.get_messages(_SID)
        assert retired and all(stored[i]["content"] == _durable_content(pruned[i]["content"]) for i in retired)
    finally:
        db.close()


def test_a_turn_compacted_in_place_before_its_context_backfill_keeps_its_image(tmp_path: Path) -> None:
    """In-place preflight compaction stores the image turn, then the prologue backfills its plugin context into that
    row. The row holds text and the live dict owns that write: a later re-flush (the surrogate repair's, which drops
    the persist marker) that read the row as another writer's would adopt its text over the live image."""
    import types

    from agent.context_compressor import _DB_PERSISTED_MARKER
    from agent.turn_context import build_turn_context

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        agent = _agent(db)
        turn = [{"type": "text", "text": "what is on this label?"}, _IMAGE_RESULT[1]]
        compressor = MagicMock(protect_first_n=0, protect_last_n=0, threshold_tokens=1, context_length=1000,
                               last_prompt_tokens=-1, compression_count=1, _last_summary_error=None,
                               _last_compress_aborted=False, _last_summary_auth_failure=False,
                               _last_aux_model_failure_model=None, _last_aux_model_failure_error=None)
        compressor.should_compress.side_effect = [True] + [False] * 8
        compressor.should_defer_preflight_to_real_usage.return_value = False
        compressor.get_active_compression_failure_cooldown.return_value = None
        compressor.compress.side_effect = lambda incoming, **_kw: [
            {"role": "assistant", "content": "[CONTEXT COMPACTION] summary", COMPRESSED_SUMMARY_METADATA_KEY: True},
            {"role": "user", "content": list(turn)}]
        agent.context_compressor, agent.compression_in_place = compressor, True
        agent._cached_system_prompt, agent._compression_feasibility_checked = "SYSTEM", True
        with patch("agent.auxiliary_client.set_runtime_main", lambda *a, **k: None), \
                patch("hermes_cli.plugins.invoke_hook", return_value=[{"context": "PLUGIN-CTX"}]):
            ctx = build_turn_context(
                agent=agent, user_message=list(turn), system_message=None, task_id=None, stream_callback=None,
                conversation_history=[{"role": "user", "content": "x" * 4000}, {"role": "assistant", "content": "y" * 4000}],
                persist_user_message=None, restore_or_build_system_prompt=lambda *a, **k: None,
                install_safe_stdio=lambda: None, sanitize_surrogates=lambda text: text,
                summarize_user_message_for_log=lambda _m: "[image]", set_session_context=lambda _sid: None,
                set_current_write_origin=lambda _o: None,
                ra=lambda: types.SimpleNamespace(_set_interrupt=lambda *a, **k: None))
        assert agent._last_compaction_in_place
        live = ctx.messages[ctx.current_turn_user_idx]
        live.pop(_DB_PERSISTED_MARKER)
        agent._flush_messages_to_session_db(ctx.messages, None)
        assert isinstance(live["content"], list) and _IMAGE_RESULT[1] in live["content"], live["content"]
        stored = db.get_messages(agent.session_id)[-1]["content"]
        assert "PLUGIN-CTX" in stored and stored == _durable_content(live["content"])
    finally:
        db.close()
