"""Regression coverage: the compaction summary role must alternate against
TEMPLATE-VISIBLE neighbours, not literal list neighbours.

Mistral-family chat templates (Devstral, Mistral Small 3.x, Magistral)
enforce user/assistant alternation at render time but exempt the tool flow
from the check: ``tool`` results and assistant messages carrying
``tool_calls`` are skipped. The summary-role selection previously keyed off
the LITERAL last head message (``compressed[-1]``), producing this captured
failure (Hermes Desktop v0.19.0 against llama.cpp-served Devstral):

    [0] system
    [1] user            <- original first user turn, protected
    [2] assistant       (tool_calls, exempt)
    [3] tool            (exempt)
    [4] user            <- COMPACTION SUMMARY, pinned to "user" because the
                           literal previous role was "tool"
    [5..] assistant(tool_calls)/tool pairs

The template counts only [1] and [4]: user -> user, and the backend rejects
the ENTIRE request with a Jinja alternation error (HTTP 500):

    "After the optional system message, conversation roles must alternate
     user and assistant roles except for tool calls and results"

Because the summary persists in the stored conversation, every retry
replays the same poisoned history: the session is permanently broken, on
EVERY compaction, overflow or not.

These tests pin the fix: neighbour roles for summary-role selection are
computed by ``_template_visible_role`` (skipping the exempt tool flow), and
the assembled output always satisfies the Mistral alternation check. The
existing #52160 / #58753 forced-user guards must keep winning: their forced
shapes (summary-user followed only by exempt messages) are alternation-safe.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    SUMMARY_PREFIX,
    _template_visible_role,
)
import itertools


@pytest.fixture()
def compressor():
    from agent.context_compressor import ContextCompressor

    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=100_000,
    ):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.50,
            protect_first_n=3,
            protect_last_n=20,
            quiet_mode=True,
        )
        c.tail_token_budget = 40
        return c


def _tool_turns(start: int, n: int, payload: str = "x" * 300) -> list[dict]:
    out: list[dict] = []
    for i in range(start, start + n):
        out.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            }
        )
        out.append({"role": "tool", "content": payload, "tool_call_id": f"c{i}"})
    return out


def _summary_rows(messages: list[dict]) -> list[dict]:
    return [
        m
        for m in messages
        if isinstance(m, dict) and m.get(COMPRESSED_SUMMARY_METADATA_KEY)
    ]


def _mistral_alternation_ok(messages: list[dict]) -> bool:
    """Replay the Mistral template's pre-flight alternation check: count
    only user and assistant-without-tool_calls messages; the counted
    sequence must go user, assistant, user, assistant, ..."""
    expect = "user"
    for m in messages:
        role = _template_visible_role(m)
        if role in (None, "system"):
            continue
        if role != expect:
            return False
        expect = "assistant" if expect == "user" else "user"
    return True


class TestTemplateVisibleRoleHelper:
    def test_tool_flow_is_exempt(self):
        assert _template_visible_role({"role": "tool", "content": "r"}) is None
        assert (
            _template_visible_role(
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c"}]}
            )
            is None
        )

    def test_counted_roles_pass_through(self):
        assert _template_visible_role({"role": "user", "content": "q"}) == "user"
        assert (
            _template_visible_role({"role": "assistant", "content": "a"})
            == "assistant"
        )
        assert _template_visible_role({"role": "system", "content": "s"}) == "system"

    def test_non_dict_is_exempt(self):
        assert _template_visible_role(None) is None
        assert _template_visible_role("not a message") is None


class TestSummaryRoleAlternatesAgainstVisibleNeighbours:
    def test_captured_devstral_shape_emits_assistant_summary(self, compressor):
        """The byte-captured poisoning shape: protected head ends
        ``[user, assistant(tool_calls), tool]``, tail is all tool flow.
        The literal previous role is ``tool`` (which used to pin the
        summary to "user"); the template-visible previous role is
        ``user``, so the summary must be emitted as ``assistant``."""
        c = compressor
        messages = [{"role": "user", "content": "run a full systems diagnostic"}]
        messages += _tool_turns(0, 30)

        mocked = f"{SUMMARY_PREFIX}\nrolled-up summary of the tool work"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        rows = _summary_rows(out)
        assert len(rows) == 1
        assert rows[0].get("role") == "assistant", (
            "REGRESSION: compaction summary emitted as role=user directly "
            "after a template-visible user turn (only exempt tool-flow "
            "messages between). Mistral-strict templates reject the whole "
            "request with a Jinja alternation 500 and the stored session is "
            f"poisoned permanently. Got role={rows[0].get('role')!r}."
        )

    def test_captured_shape_passes_mistral_alternation(self, compressor):
        c = compressor
        messages = [{"role": "user", "content": "run a full systems diagnostic"}]
        messages += _tool_turns(0, 30)

        mocked = f"{SUMMARY_PREFIX}\nrolled-up summary of the tool work"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        assert _mistral_alternation_ok(out), (
            "Compressed transcript fails the Mistral template alternation "
            "check. Visible-role sequence: "
            f"{[_template_visible_role(m) for m in out if _template_visible_role(m)]}"
        )

    def test_visible_head_assistant_visible_tail_user_merges(self, compressor):
        """When the visible head ends ``assistant`` and the visible tail
        opens ``user``, NO standalone role can satisfy alternation (user
        collides with the tail, assistant with the head -- and the
        intervening tool flow is template-exempt, so literal separation
        does not help). The existing merge-into-tail fallback must fire.
        The literal-role logic used to emit a standalone role="user"
        summary here (literal previous role was ``tool``), which is a
        second poisoning shape: visible user(summary) -> user(tail)."""
        c = compressor
        messages = [
            {"role": "user", "content": "question " + "q" * 200},
            {"role": "assistant", "content": "answer " + "a" * 200},
        ]
        messages += _tool_turns(0, 30)
        # A recent user turn preserved in the tail keeps the zero-user
        # guard out of the picture.
        messages += [
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "on it"},
        ]

        mocked = f"{SUMMARY_PREFIX}\nsummary body"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        rows = _summary_rows(out)
        assert len(rows) == 1
        # Merged into a template-exempt tail message: invisible to the
        # alternation check, summary content still delivered.
        assert _template_visible_role(rows[0]) is None
        assert _mistral_alternation_ok(out)


class TestForcedUserGuardsStillWin:
    def test_zero_user_guard_still_forces_user(self, compressor):
        """#58753: when no genuine user turn survives, the summary must
        still be pinned to role=user (and that shape is alternation-safe
        because everything after it is template-exempt)."""
        c = compressor
        c.compression_count = 1  # protect_first_n decays -> no head
        messages = [{"role": "user", "content": "work kanban task 42"}]
        messages += _tool_turns(0, 12)

        mocked = f"{SUMMARY_PREFIX}\nsummary body"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        rows = _summary_rows(out)
        assert len(rows) == 1
        assert rows[0].get("role") == "user"
        assert _mistral_alternation_ok(out)

    def test_no_literal_consecutive_user_roles(self, compressor):
        """The pre-existing literal invariant still holds alongside the
        template-visible one."""
        c = compressor
        messages = [{"role": "user", "content": "run diagnostics"}]
        messages += _tool_turns(0, 30)

        mocked = f"{SUMMARY_PREFIX}\nsummary body"
        with patch.object(c, "_generate_summary", return_value=mocked):
            out = c.compress(messages, current_tokens=90_000)

        for prev, cur in itertools.pairwise(out):
            assert not (
                prev.get("role") == "user" and cur.get("role") == "user"
            ), "compression introduced literal consecutive user-role messages"

    def test_publish_compression_child_folds_assistant_summary_seam_before_sqlite_insert(self, tmp_path):
        """#CF-96: When ContextCompressor places an assistant summary before a carried
        assistant tool-call turn (because the protected head ends in role=user),
        publish_compression_child must fold the seam before _insert_message_rows so
        state.db persists ONE merged assistant row instead of two consecutive assistant rows."""
        from contextlib import closing
        from hermes_state import SessionDB
        from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

        with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
            parent_sid = "20261114_070000_parent"
            child_sid = "20261114_071129_child"
            db.create_session(parent_sid, "cli", model="test/model")
            u_id = db.append_message(session_id=parent_sid, role="user", content="initial ask")
            db.append_message(session_id=parent_sid, role="user", content="middle turn")
            db.append_message(session_id=parent_sid, role="assistant", content="middle reply")
            tail_tc = [
                {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}},
                {"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"b.py"}'}},
            ]
            a_id = db.append_message(session_id=parent_sid, role="assistant", content="", tool_calls=tail_tc)
            t1_id = db.append_message(session_id=parent_sid, role="tool", content="res1", tool_call_id="call_1", tool_name="read_file")
            t2_id = db.append_message(session_id=parent_sid, role="tool", content="res2", tool_call_id="call_2", tool_name="read_file")

            parent_rows = {
                r["id"]: dict(r)
                for r in db._conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY id", (parent_sid,)).fetchall()
            }
            compressed_messages = [
                {"role": "user", "content": "initial ask", "_row_id": u_id, "message_uid": parent_rows[u_id]["message_uid"]},
                {
                    "role": "assistant",
                    "content": f"{SUMMARY_PREFIX}\ncompacted summary",
                    COMPRESSED_SUMMARY_METADATA_KEY: True,
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tail_tc,
                    "_row_id": a_id,
                    "message_uid": parent_rows[a_id]["message_uid"],
                },
                {"role": "tool", "content": "res1", "tool_call_id": "call_1", "tool_name": "read_file", "_row_id": t1_id, "message_uid": parent_rows[t1_id]["message_uid"]},
                {"role": "tool", "content": "res2", "tool_call_id": "call_2", "tool_name": "read_file", "_row_id": t2_id, "message_uid": parent_rows[t2_id]["message_uid"]},
            ]
            assert db.publish_compression_child(
                parent_session_id=parent_sid,
                child_session_id=child_sid,
                source="cli",
                model="test/model",
                messages=compressed_messages,
                require_compression_lease=False,
            ) is None
            child_rows = [
                dict(r)
                for r in db._conn.execute(
                    "SELECT id, role, content, tool_calls, message_uid, absorbed_message_uids FROM messages WHERE session_id = ? AND active = 1 ORDER BY id",
                    (child_sid,),
                ).fetchall()
            ]
            roles = [r["role"] for r in child_rows]
            assert roles == ["user", "assistant", "tool", "tool"], f"Expected 1 folded assistant row in SQLite, got {roles}"
            assert child_rows[1]["tool_calls"] is not None
            assert parent_rows[a_id]["message_uid"] in (child_rows[1]["absorbed_message_uids"] or "")

    def test_replace_session_messages_folds_assistant_summary_seam_before_sqlite_insert(self, tmp_path):
        """#CF-96: In-place replace_session_messages also folds the assistant summary +
        assistant tool-call seam before _insert_message_rows so the stored summary row
        carries the merged tool_calls in SQLite."""
        from contextlib import closing
        from hermes_state import SessionDB
        from agent.context_compressor import COMPRESSED_SUMMARY_METADATA_KEY

        with closing(SessionDB(db_path=tmp_path / "state.db")) as db:
            sid = "20261114_070500_inplace"
            db.create_session(sid, "cli", model="test/model")
            u_id = db.append_message(session_id=sid, role="user", content="initial ask")
            db.append_message(session_id=sid, role="user", content="middle turn")
            db.append_message(session_id=sid, role="assistant", content="middle reply")
            tail_tc = [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"a.py"}'}}]
            a_id = db.append_message(session_id=sid, role="assistant", content="", tool_calls=tail_tc)
            t1_id = db.append_message(session_id=sid, role="tool", content="res1", tool_call_id="call_1", tool_name="read_file")

            rows = {
                r["id"]: dict(r)
                for r in db._conn.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY id", (sid,)).fetchall()
            }
            compressed_messages = [
                {"role": "user", "content": "initial ask", "_row_id": u_id, "message_uid": rows[u_id]["message_uid"]},
                {
                    "role": "assistant",
                    "content": f"{SUMMARY_PREFIX}\ncompacted summary",
                    COMPRESSED_SUMMARY_METADATA_KEY: True,
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tail_tc,
                    "_row_id": a_id,
                    "message_uid": rows[a_id]["message_uid"],
                },
                {"role": "tool", "content": "res1", "tool_call_id": "call_1", "tool_name": "read_file", "_row_id": t1_id, "message_uid": rows[t1_id]["message_uid"]},
            ]
            assert db.archive_and_compact(sid, compressed_messages) == 3
            active_rows = [
                dict(r)
                for r in db._conn.execute(
                    "SELECT id, role, content, tool_calls, absorbed_message_uids FROM messages WHERE session_id = ? AND active = 1 ORDER BY id",
                    (sid,),
                ).fetchall()
            ]
            assert [r["role"] for r in active_rows] == ["user", "assistant", "tool"]
            assert active_rows[1]["tool_calls"] is not None
            assert rows[a_id]["message_uid"] in (active_rows[1]["absorbed_message_uids"] or "")

    def test_merge_consecutive_assistants_idempotent_when_second_uid_already_absorbed(self):
        """#CF-96: Historical child sessions where row 1 already absorbed row 2
        (uid2 in row1._absorbed_message_uids and identical tool_calls) must fold
        idempotently on reload without duplicating content or doubling tool_calls."""
        from agent.agent_runtime_helpers import _merge_consecutive_assistants
        from agent.message_metadata import ABSORBED_MESSAGE_UIDS, TOOL_CALL_UIDS

        tc = [
            {"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"x.py"}'}},
            {"id": "call_b", "type": "function", "function": {"name": "read_file", "arguments": '{"path":"y.py"}'}},
        ]
        row1 = {
            "role": "assistant",
            "content": f"{SUMMARY_PREFIX}\nsummary text\n\ntail text",
            "reasoning": "thought 1",
            "tool_calls": list(tc),
            "message_uid": "11111111111111111111111111111111",
            ABSORBED_MESSAGE_UIDS: ["22222222222222222222222222222222"],
            TOOL_CALL_UIDS: {"call_a": "aaaa", "call_b": "bbbb"},
        }
        row2 = {
            "role": "assistant",
            "content": "tail text",
            "reasoning": "thought 1",
            "tool_calls": list(tc),
            "message_uid": "22222222222222222222222222222222",
            TOOL_CALL_UIDS: {"call_a": "aaaa", "call_b": "bbbb"},
        }
        repaired, count = _merge_consecutive_assistants([row1, row2])
        assert count == 1
        assert len(repaired) == 1
        merged = repaired[0]
        assert len(merged["tool_calls"]) == 2
        assert merged["content"].count("tail text") == 1
        assert merged["reasoning"].count("thought 1") == 1
        assert merged[TOOL_CALL_UIDS] == {"call_a": "aaaa", "call_b": "bbbb"}

