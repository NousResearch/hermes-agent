"""Tests for in-place context compaction (config: compression.in_place, #38763).

When ``compression.in_place`` is True, ``compress_context()`` rewrites the
message list and rebuilds the system prompt but keeps the SAME ``session_id``:
no ``end_session``, no ``parent_session_id`` child row, no ``name #N`` title
renumber, no flush-cursor reset. This eliminates the session-rotation bug
cluster (#33618 /goal loss, #14238 lost response, #33907 orphans, #45117 search
gaps, #42228 null cwd). When the flag is False (default), rotation behaves
exactly as before.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest


def _make_agent(session_db, session_id, *, in_place):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.compression_in_place = in_place
    # Mock the compressor to return a deterministic shrunk transcript so the
    # test exercises the DB-mutation path, not summarization quality.
    def _fake_compress(messages, current_tokens=None, focus_topic=None, force=False):
        return [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary of prior turns"},
            {"role": "assistant", "content": "recent reply"},
        ]

    agent.context_compressor.compress = _fake_compress
    agent.context_compressor._last_compress_aborted = False
    agent.context_compressor._last_summary_error = None
    agent.context_compressor.compression_count = 1
    return agent


def _seed(db, sid, title, n=8):
    db.create_session(sid, "cli", model="test/model")
    db.set_session_title(sid, title)
    for i in range(n):
        db.append_message(
            session_id=sid,
            role="user" if i % 2 == 0 else "assistant",
            content=f"msg {i}",
        )


class TestInPlaceCompaction:
    def test_multimodal_merged_summary_persists_checkpoint(self):
        """A summary merged into list content remains compactable and durable."""
        from agent.conversation_compression import (
            _COMPRESSION_CHECKPOINT_META_PREFIX,
            compress_context,
        )
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_115500_multimodal"
            _seed(db, sid, "multimodal", n=9)
            agent = _make_agent(db, sid, in_place=True)
            compressor = getattr(agent, "context_compressor")
            del compressor.__dict__["compress"]
            compressor.protect_first_n = 2
            compressor.protect_last_n = 3
            compressor._previous_summary = None
            compressor.compression_count = 0
            messages = [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "msg 1"},
                {"role": "assistant", "content": "msg 2"},
                {"role": "user", "content": "msg 3"},
                {"role": "assistant", "content": "msg 4"},
                {"role": "user", "content": "msg 5"},
                {"role": "user", "content": [{"type": "text", "text": "msg 6"}]},
                {"role": "assistant", "content": "msg 7"},
                {"role": "user", "content": "msg 8"},
            ]

            with patch.object(compressor, "_generate_summary", return_value=(
                "## Blocked\nNone.\n\n"
                "## Key Decisions\n"
                "- Decision: retain multimodal content | Rationale: preserve provider input | "
                "Rejected: flattening blocks | Scope: compressed transcript"
            )):
                compressed, _ = compress_context(
                    agent,
                    messages,
                    approx_tokens=100_000,
                    system_message="sys",
                )

            assert compressed is not messages
            merged = next(
                message for message in compressed
                if message.get("_compressed_summary")
            )
            assert isinstance(merged["content"], list)
            rendered = "\n".join(
                block if isinstance(block, str) else str(block.get("text") or "")
                for block in merged["content"]
            )
            assert "<!-- hermes:compression-checkpoint:v1 -->" in rendered
            assert rendered.index("<!-- /hermes:compression-checkpoint -->") < rendered.index(
                "--- END OF CONTEXT SUMMARY"
            )
            checkpoint = json.loads(
                db.get_meta(_COMPRESSION_CHECKPOINT_META_PREFIX + sid) or "{}"
            )
            assert checkpoint["decisions"][0]["decision"] == "retain multimodal content"

    def test_static_fallback_with_tool_error_commits_opaque_checkpoint(self):
        from agent.conversation_compression import (
            _COMPRESSION_CHECKPOINT_META_PREFIX,
            _COMPRESSION_CHECKPOINT_START,
            compress_context,
        )
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260719_opaque_fallback"
            _seed(db, sid, "opaque-fallback", n=11)
            agent = _make_agent(db, sid, in_place=True)
            compressor = getattr(agent, "context_compressor")
            del compressor.__dict__["compress"]
            compressor.protect_first_n = 2
            compressor.protect_last_n = 3
            compressor._previous_summary = None
            compressor.compression_count = 0
            compressor._last_summary_auth_failure = False
            compressor._last_summary_network_failure = False
            compressor.abort_on_summary_failure = False
            messages = [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "inspect the project"},
                {"role": "assistant", "content": "I will inspect it."},
                {"role": "user", "content": "run the failing command"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-error",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": json.dumps({"command": "false"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-error",
                    "name": "terminal",
                    "content": "fatal error: command failed with exit code 1",
                },
                {"role": "assistant", "content": "The command failed."},
                {"role": "user", "content": "record that and continue"},
                {"role": "assistant", "content": "Recorded."},
                {"role": "user", "content": "what is the current state?"},
                {"role": "assistant", "content": "Checking now."},
            ]

            with patch.object(compressor, "_generate_summary", return_value=None):
                compressed, _ = compress_context(
                    agent,
                    messages,
                    approx_tokens=100_000,
                    system_message="sys",
                )

            assert compressed is not messages
            assert len(compressed) < len(messages)
            checkpoint = json.loads(
                db.get_meta(_COMPRESSION_CHECKPOINT_META_PREFIX + sid) or "{}"
            )
            assert checkpoint == {
                "version": 1,
                "session_id": sid,
                "decisions": [],
                "blockers": [],
                "checkpoint_quality": "fallback_opaque",
            }
            rendered = "\n".join(
                str(message.get("content") or "") for message in compressed
            )
            assert _COMPRESSION_CHECKPOINT_START in rendered
            assert "structured decision and blocker state is opaque" in rendered.casefold()
            assert "## Blocked" not in rendered
            assert "## Key Decisions" not in rendered
            assert "fatal error: command failed with exit code 1" in rendered
            assert getattr(agent, "_last_compression_checkpoint") == checkpoint
            assert db.get_compression_fallback_streak(sid) == 1

    def test_checkpoint_failure_restores_real_compressor_durable_state(self):
        from agent.conversation_compression import (
            _COMPRESSION_CHECKPOINT_META_PREFIX,
            compress_context,
        )
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_115600_rollback"
            _seed(db, sid, "rollback", n=9)
            agent = _make_agent(db, sid, in_place=True)
            compressor = getattr(agent, "context_compressor")
            del compressor.__dict__["compress"]
            compressor.protect_first_n = 2
            compressor.protect_last_n = 3
            compressor._previous_summary = None
            compressor.compression_count = 0
            cooldown_until = time.time() + 600
            db.record_compression_failure_cooldown(sid, cooldown_until, "before")
            compressor.get_active_compression_failure_cooldown()
            before = db.get_compression_failure_cooldown(sid)
            assert before is not None
            active_before = db.get_messages_as_conversation(sid)
            messages = [
                {"role": "system", "content": "system prompt"},
                *[
                    {
                        "role": "user" if index % 2 else "assistant",
                        "content": f"msg {index}",
                    }
                    for index in range(1, 9)
                ],
            ]
            summary = (
                "## Blocked\nNone.\n\n## Key Decisions\n"
                "- Decision: preserve rollback | Rationale: atomic boundary | "
                "Rejected: partial commit | Scope: checkpoint failure"
            )

            def _generate_and_clear(*_args, **_kwargs):
                compressor._previous_summary = summary
                compressor._clear_compression_failure_cooldown()
                return summary

            def _fail_message_insert(*_args, **_kwargs):
                raise OSError("disk full")

            with (
                patch.object(compressor, "_generate_summary", _generate_and_clear),
                patch.object(db, "_insert_message_rows", _fail_message_insert),
            ):
                returned, _ = compress_context(
                    agent,
                    messages,
                    approx_tokens=100_000,
                    system_message="sys",
                )

            after = db.get_compression_failure_cooldown(sid)
            assert returned is messages
            assert after is not None
            assert after["error"] == before["error"] == "before"
            assert after["cooldown_until"] == pytest.approx(
                before["cooldown_until"], abs=0.01
            )
            assert db.get_meta(_COMPRESSION_CHECKPOINT_META_PREFIX + sid) is None
            assert db.get_messages_as_conversation(sid) == active_before

    def test_persisted_checkpoint_is_replaced_after_reload_and_recompression(self):
        from agent.context_compressor import _SUMMARY_END_MARKER
        from agent.conversation_compression import (
            _COMPRESSION_CHECKPOINT_END,
            _COMPRESSION_CHECKPOINT_META_PREFIX,
            _COMPRESSION_CHECKPOINT_START,
            compress_context,
        )
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_115700_recompress"
            db.create_session(sid, "cli", model="test/model")
            old_checkpoint = json.dumps(
                {
                    "version": 1,
                    "session_id": sid,
                    "decisions": [{"decision": "obsolete choice"}],
                    "blockers": [{"blocker": "old blocker", "status": "unresolved"}],
                }
            )
            checkpoint_start_cut = len(_COMPRESSION_CHECKPOINT_START) // 2
            checkpoint_end_cut = len(_COMPRESSION_CHECKPOINT_END) // 2
            old_summary = [
                {
                    "type": "text",
                    "text": (
                        "[CONTEXT COMPACTION — REFERENCE ONLY]\n"
                        "## Historical Task Snapshot\nkeep historical fact\n\n"
                        "## Blo"
                    ),
                },
                {"type": "text", "text": "cked\n- Blocker: old blocker\n\n"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,old"}},
                {"type": "text", "text": "## Key Deci"},
                {"type": "text", "text": "sions\n- Decision: obsolete choice\n\n"},
                {
                    "type": "text",
                    "text": _COMPRESSION_CHECKPOINT_START[:checkpoint_start_cut],
                },
                {
                    "type": "text",
                    "text": _COMPRESSION_CHECKPOINT_START[checkpoint_start_cut:]
                    + "\n```json\n"
                    + old_checkpoint
                    + "\n```\n"
                    + _COMPRESSION_CHECKPOINT_END[:checkpoint_end_cut],
                },
                {
                    "type": "text",
                    "text": _COMPRESSION_CHECKPOINT_END[checkpoint_end_cut:]
                    + "\n\n"
                    + _SUMMARY_END_MARKER,
                },
            ]
            persisted = [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": old_summary, "_compressed_summary": True},
                {"role": "assistant", "content": "msg 2"},
                {"role": "user", "content": "msg 3"},
                {"role": "assistant", "content": "msg 4"},
                {"role": "user", "content": "msg 5"},
                {"role": "user", "content": "msg 6"},
                {"role": "assistant", "content": "msg 7"},
                {"role": "user", "content": "msg 8"},
            ]
            db.replace_messages(sid, persisted)
            reloaded = db.get_messages_as_conversation(sid)
            assert all("_compressed_summary" not in message for message in reloaded)

            agent = _make_agent(db, sid, in_place=True)
            compressor = getattr(agent, "context_compressor")
            del compressor.__dict__["compress"]
            compressor.protect_first_n = 2
            compressor.protect_last_n = 3
            compressor.tail_token_budget = 1
            compressor._previous_summary = None
            compressor.compression_count = 0
            with patch.object(
                compressor,
                "_generate_summary",
                return_value=(
                    "## Blocked\nNone.\n\n"
                    "## Key Decisions\n- Decision: current choice | Rationale: current | "
                    "Rejected: prior approach | Scope: recompression"
                ),
            ):
                compressed, _ = compress_context(
                    agent,
                    reloaded,
                    approx_tokens=100_000,
                    system_message="sys",
                )

            rendered = "\n".join(str(message.get("content") or "") for message in compressed)
            assert rendered.count(_COMPRESSION_CHECKPOINT_START) == 1
            assert "obsolete choice" not in rendered
            assert "old blocker" not in rendered
            assert "keep historical fact" in rendered
            assert "data:image/png;base64,old" in rendered
            assert "current choice" in rendered
            checkpoint = json.loads(
                db.get_meta(_COMPRESSION_CHECKPOINT_META_PREFIX + sid) or "{}"
            )
            assert checkpoint["decisions"][0]["decision"] == "current choice"

            persisted_again = "\n".join(
                str(message.get("content") or "")
                for message in db.get_messages_as_conversation(sid)
            )
            assert persisted_again.count(_COMPRESSION_CHECKPOINT_START) == 1
            assert "obsolete choice" not in persisted_again

    def test_in_place_keeps_same_session_id(self):
        """In-place mode: id unchanged, no child row, no rename, history kept."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_120000_aaaaaa"
            _seed(db, sid, "my-research")
            agent = _make_agent(db, sid, in_place=True)
            agent._last_flushed_db_idx = 5

            messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
            compressed, _sp = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )

            # Identity never moved.
            assert agent.session_id == sid
            # No continuation row forked.
            child = db._conn.execute(
                "SELECT id FROM sessions WHERE parent_session_id = ?", (sid,)
            ).fetchall()
            assert child == []
            # Session not ended; title untouched (no "#2").
            row = db.get_session(sid)
            assert row["end_reason"] is None
            assert row["title"] == "my-research"
            # DURABLE, NON-DESTRUCTIVE compaction (the core invariant, per
            # Teknium's review): the LIVE context is the compacted set, but the
            # pre-compaction turns are PRESERVED on disk (active=0), not deleted
            # — searchable + recoverable under the SAME id. A resume reloads the
            # compacted set so compaction actually shrinks the live session and
            # doesn't immediately re-compact (#38763).
            reloaded = db.get_messages_as_conversation(sid)
            assert len(reloaded) == 2
            summary = reloaded[0].get("content") or ""
            assert summary.startswith("[CONTEXT COMPACTION] summary of prior turns")
            assert "<!-- hermes:compression-checkpoint:v1 -->" in summary
            assert '"blockers":[]' in summary
            assert '"decisions":[]' in summary
            assert reloaded[1].get("content") == "recent reply"
            assert row["message_count"] == 2  # live (active) count
            # NON-DESTRUCTIVE: the 8 seeded originals survive at active=0
            # alongside the 2 compacted rows — nothing was DELETEd.
            all_rows = db.get_messages(sid, include_inactive=True)
            assert len(all_rows) == 10
            archived = [m for m in all_rows if not m.get("active", 1)]
            assert len(archived) == 8
            # The originals remain FTS-searchable (active=0 is a content-
            # preserving UPDATE; the fts triggers don't key on active).
            hit = db._conn.execute(
                "SELECT 1 FROM messages_fts f JOIN messages m ON m.id = f.rowid "
                "WHERE m.session_id = ? AND messages_fts MATCH 'msg' AND m.active = 0 "
                "LIMIT 1",
                (sid,),
            ).fetchone()
            assert hit is not None
            # Flush identity/cursor reset so next-turn appends diff against the
            # compacted transcript (rebuilds the identity set on next flush).
            assert agent._last_flushed_db_idx == 0
            assert agent._flushed_db_message_ids == set()
            # Rotation-independent in-place signal set for the gateway.
            assert agent._last_compaction_in_place is True
            # Live transcript actually shrank.
            assert len(compressed) == 2

    def test_in_place_alternation_preserved(self):
        """The compacted list must not introduce consecutive same-role messages."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_120500_cccccc"
            _seed(db, sid, "alt")
            agent = _make_agent(db, sid, in_place=True)
            messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
            compressed, _ = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )
            roles = [m["role"] for m in compressed if m.get("role") != "system"]
            assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))

    def test_in_place_skips_redundant_preflush(self):
        """In-place must NOT pre-flush current-turn messages: replace_messages
        rewrites the whole row, so a flush would INSERT rows it immediately
        deletes (wasted writes). The current-turn tail survives via the
        compressor's `compressed` output, not the flush."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            _seed(db, "ip_flush", "f")
            agent = _make_agent(db, "ip_flush", in_place=True)
            calls = {"n": 0}
            agent._flush_messages_to_session_db = lambda *a, **k: calls.__setitem__(
                "n", calls["n"] + 1
            )
            compress_context(
                agent, [{"role": "user", "content": "x"}] * 8,
                approx_tokens=100_000, system_message="sys",
            )
            assert calls["n"] == 0

    def test_rotation_still_preflushes(self):
        """Rotation MUST pre-flush so current-turn messages survive in the
        preserved old (parent) session before it is ended (#47202)."""
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            _seed(db, "rot_flush", "f")
            agent = _make_agent(db, "rot_flush", in_place=False)
            calls = {"n": 0}
            agent._flush_messages_to_session_db = lambda *a, **k: calls.__setitem__(
                "n", calls["n"] + 1
            )
            compress_context(
                agent, [{"role": "user", "content": "x"}] * 8,
                approx_tokens=100_000, system_message="sys",
            )
            assert calls["n"] == 1


class TestRotationFallbackWhenFlagOff:
    def test_rotation_when_flag_off(self):
        """Rotation is now the OPT-OUT fallback (default flipped to in-place in
        #38763). With in_place=False explicitly set, legacy rotation is
        unchanged — forks a renamed continuation session."""
        from hermes_state import SessionDB
        from agent.conversation_compression import (
            _COMPRESSION_CHECKPOINT_META_PREFIX,
            compress_context,
        )

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_130000_bbbbbb"
            _seed(db, sid, "my-research")
            agent = _make_agent(db, sid, in_place=False)
            agent._last_flushed_db_idx = 5

            messages = [{"role": "user", "content": f"m{i}"} for i in range(8)]
            compressed, _ = compress_context(
                agent, messages, approx_tokens=100_000, system_message="sys"
            )

            # Identity rotated to a fresh id.
            assert agent.session_id != sid
            # Old session ended via compression; continuation forked + renamed.
            assert db.get_session(sid)["end_reason"] == "compression"
            child = db._conn.execute(
                "SELECT id, title FROM sessions WHERE parent_session_id = ?", (sid,)
            ).fetchall()
            assert len(child) == 1
            assert child[0]["title"] == "my-research #2"
            # The compacted child is persisted atomically at the rotation
            # boundary, so a headless process killed before finalization can
            # still resume it without duplicating the two handoff messages.
            assert agent._last_flushed_db_idx == 2
            child_id = child[0]["id"]
            assert db.get_meta(_COMPRESSION_CHECKPOINT_META_PREFIX + sid) is None
            child_checkpoint = json.loads(
                db.get_meta(_COMPRESSION_CHECKPOINT_META_PREFIX + child_id) or "{}"
            )
            assert child_checkpoint["session_id"] == child_id
            assert getattr(agent, "_last_compression_checkpoint")["session_id"] == child_id
            assert child_id in "\n".join(
                str(message.get("content") or "") for message in compressed
            )
            persisted = db.get_messages_as_conversation(child_id)
            assert len(persisted) == 2
            persisted_summary = str(persisted[0].get("content") or "")
            assert persisted_summary.startswith(
                "[CONTEXT COMPACTION] summary of prior turns"
            )
            assert "<!-- hermes:compression-checkpoint:v1 -->" in persisted_summary
            assert child_id in persisted_summary
            assert persisted[1].get("content") == "recent reply"
            # Rotation mode does NOT set the in-place signal.
            assert getattr(agent, "_last_compaction_in_place", False) is False


class TestInPlaceSignalForGateway:
    """compress_context must expose a rotation-independent flag the gateway can
    read (instead of an id-change diff) to re-baseline transcript handling."""

    def test_signal_set_on_in_place_unset_on_rotation(self):
        from hermes_state import SessionDB
        from agent.conversation_compression import compress_context

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            # in-place → flag True
            _seed(db, "s_ip", "ip")
            a_ip = _make_agent(db, "s_ip", in_place=True)
            compress_context(
                a_ip, [{"role": "user", "content": "x"}] * 8,
                approx_tokens=100_000, system_message="sys",
            )
            assert a_ip._last_compaction_in_place is True

            # rotation → flag False
            _seed(db, "s_rot", "rot")
            a_rot = _make_agent(db, "s_rot", in_place=False)
            compress_context(
                a_rot, [{"role": "user", "content": "x"}] * 8,
                approx_tokens=100_000, system_message="sys",
            )
            assert a_rot._last_compaction_in_place is False


class TestInPlaceConfigDefault:
    def test_flag_defaults_on(self):
        """In-place is the default as of #38763 (rotation is now opt-out via
        compression.in_place: false)."""
        from hermes_cli.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["compression"].get("in_place") is True


class TestCompactedTurnsStaySearchable:
    """Teknium's review hinges on the pre-compaction transcript staying
    DISCOVERABLE after in-place compaction. Compaction-archived rows
    (active=0, compacted=1) must surface in session_search by default, while
    rewind/undo rows (active=0, compacted=0) must stay hidden. The two share
    the active flag but are distinguished by the compacted flag."""

    def test_compacted_turns_found_by_default_search(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_search"
            db.create_session(sid, "cli", model="test/model")
            for r, c in [
                ("user", "configure the HMAC secret"),
                ("assistant", "set it in config.yaml"),
                ("user", "deploy returns 403"),
                ("assistant", "rotate the HMAC"),
                ("user", "works now"),
                ("assistant", "great"),
            ]:
                db.append_message(session_id=sid, role=r, content=c)

            before = db.search_messages("HMAC", role_filter=["user", "assistant"])
            assert len(before) == 2

            db.archive_and_compact(
                sid,
                [
                    {"role": "user", "content": "[SUMMARY] earlier setup"},
                    {"role": "assistant", "content": "ok"},
                ],
            )

            # The archived originals (active=0, compacted=1) are still found by
            # the DEFAULT search — this is the durability requirement.
            after = db.search_messages("HMAC", role_filter=["user", "assistant"])
            assert {m["id"] for m in after} == {1, 4}
            # Live context still excludes them.
            assert len(db.get_messages_as_conversation(sid)) == 2

    def test_rewound_turns_stay_hidden(self):
        """Rewind/undo (active=0, compacted=0) must NOT leak into default
        search — the distinction the compacted flag preserves."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db = SessionDB(db_path=Path(tmp) / "t.db")
            sid = "20260619_undo"
            db.create_session(sid, "cli", model="test/model")
            db.append_message(session_id=sid, role="user", content="ZEBRAWORD remember this")
            db.append_message(session_id=sid, role="assistant", content="noted")
            db.rewind_to_message(sid, db.get_messages(sid)[0]["id"])

            assert db.search_messages("ZEBRAWORD", role_filter=["user", "assistant"]) == []
            recovered = db.search_messages(
                "ZEBRAWORD", role_filter=["user", "assistant"], include_inactive=True
            )
            assert len(recovered) == 1
