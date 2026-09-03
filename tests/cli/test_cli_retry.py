"""Regression tests for CLI /retry and carrier-aware rewind semantics."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    _SUMMARY_END_MARKER,
)
from hermes_state import SessionDB

from tests.cli.test_cli_init import _make_cli


def _composite_carrier(ask="REAL ASK"):
    return {
        "role": "user",
        "content": (
            f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nold task\n\n"
            f"{_SUMMARY_END_MARKER}\n\n{ask}"
        ),
    }


def _message_rows(db, session_id):
    rows = db._conn.execute(
        "SELECT id, content, active FROM messages "
        "WHERE session_id = ? ORDER BY id",
        (session_id,),
    ).fetchall()
    return [tuple(row) for row in rows]


def test_retry_last_truncates_history_before_requeueing_message():
    cli = _make_cli()
    cli._session_db = None
    cli.conversation_history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "retry me"},
        {"role": "assistant", "content": "old answer"},
    ]

    retry_msg = cli.retry_last()

    assert retry_msg == "retry me"
    assert cli.conversation_history == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
    ]

    cli.conversation_history.append({"role": "user", "content": retry_msg})
    cli.conversation_history.append({"role": "assistant", "content": "new answer"})

    assert [m["content"] for m in cli.conversation_history if m["role"] == "user"] == [
        "first",
        "retry me",
    ]


def test_process_command_retry_requeues_original_message_not_retry_command():
    cli = _make_cli()
    cli._session_db = None
    queued = []

    class _Queue:
        def put(self, value):
            queued.append(value)

    cli._pending_input = _Queue()
    cli.conversation_history = [
        {"role": "user", "content": "retry me"},
        {"role": "assistant", "content": "old answer"},
    ]

    cli.process_command("/retry")

    assert queued == ["retry me"]
    assert cli.conversation_history == []


def test_retry_fails_closed_when_warm_and_durable_targets_differ(tmp_path):
    cli = _make_cli()
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db = db
    cli.session_id = "cli-target-mismatch"
    db.create_session(cli.session_id, source="cli")
    db.append_message(cli.session_id, "user", "DURABLE ASK")
    db.append_message(cli.session_id, "assistant", "old answer")

    history = [
        {"role": "user", "content": "WARM ASK"},
        {"role": "assistant", "content": "old answer"},
    ]
    cli.conversation_history = history
    cli._pending_input = MagicMock()
    before_rows = _message_rows(db, cli.session_id)

    cli.process_command("/retry")

    cli._pending_input.put.assert_not_called()
    assert cli.conversation_history is history
    assert _message_rows(db, cli.session_id) == before_rows
    db.close()


def test_retry_fails_closed_when_transcript_changes_after_snapshot(
    tmp_path, monkeypatch
):
    cli = _make_cli()
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    sibling = SessionDB(db_path=db.db_path)
    cli._session_db = db
    cli.session_id = "cli-cas-race"
    db.create_session(cli.session_id, source="cli")
    db.append_message(cli.session_id, "user", "RETRY ME")
    db.append_message(cli.session_id, "assistant", "failed answer")
    history = db.get_messages_as_conversation(cli.session_id)
    cli.conversation_history = history
    original_rewind = db.rewind_to_message

    def _append_then_rewind(*args, **kwargs):
        sibling.append_message(cli.session_id, "assistant", "concurrent tail")
        return original_rewind(*args, **kwargs)

    monkeypatch.setattr(db, "rewind_to_message", _append_then_rewind)

    assert cli.retry_last() is None

    assert cli.conversation_history is history
    rows = db._conn.execute(
        "SELECT content, active FROM messages "
        "WHERE session_id = ? ORDER BY id",
        (cli.session_id,),
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        ("RETRY ME", 1),
        ("failed answer", 1),
        ("concurrent tail", 1),
    ]
    sibling.close()
    db.close()


@pytest.mark.parametrize("command", ["retry", "undo"])
def test_rewind_matches_warm_raw_carrier_to_durable_sanitized_sidecar(
    tmp_path, command
):
    from agent.memory_manager import sanitize_context

    cli = _make_cli()
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db = db
    cli.session_id = f"cli-sanitized-{command}"
    db.create_session(cli.session_id, source="cli")
    raw_carrier = _composite_carrier(
        "  REAL ASK\n\n<memory-context>\nprivate\n</memory-context>  "
    )["content"]
    db.append_message(
        cli.session_id,
        "user",
        sanitize_context(raw_carrier).strip(),
        api_content=raw_carrier,
    )
    db.append_message(cli.session_id, "assistant", "failed answer")
    durable = db.get_messages_as_conversation(cli.session_id)
    cli.conversation_history = [
        {"role": "user", "content": raw_carrier},
        durable[1],
    ]
    cli._pending_input = MagicMock()
    cli._prefill_input_buffer = MagicMock()

    if command == "retry":
        cli.process_command("/retry")
        cli._pending_input.put.assert_called_once_with("REAL ASK")
    else:
        cli.undo_last()
        cli._prefill_input_buffer.assert_called_once_with("REAL ASK")

    assert len(cli.conversation_history) == 1
    scaffold = cli.conversation_history[0]
    assert scaffold["display_kind"] == "hidden"
    assert "REAL ASK" not in scaffold["content"]
    active = db.get_messages_as_conversation(cli.session_id, include_row_ids=True)
    assert active[0]["_row_id"] == scaffold["_row_id"]
    assert active[0]["content"] == scaffold["content"]
    db.close()


@pytest.mark.parametrize("command", ["retry", "undo"])
@pytest.mark.parametrize("prefix_kind", ["buried_ephemeral", "old_media"])
def test_rewind_keeps_the_richer_warm_prefix_after_validating_the_target(
    tmp_path, command, prefix_kind
):
    cli = _make_cli()
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db = db
    cli.session_id = f"cli-projection-{prefix_kind}-{command}"
    db.create_session(cli.session_id, source="cli")

    if prefix_kind == "buried_ephemeral":
        history = [
            {"role": "user", "content": "OLDER ASK"},
            {"role": "assistant", "content": "candidate answer"},
            {
                "role": "user",
                "content": "[System: verify before stopping]",
                "_verification_stop_synthetic": True,
            },
            {"role": "assistant", "content": "verified answer"},
            {"role": "user", "content": "PLAIN TARGET"},
            {"role": "assistant", "content": "failed answer"},
        ]
        durable_prefix = [
            ("user", "OLDER ASK"),
            ("assistant", "candidate answer"),
            ("assistant", "verified answer"),
        ]
        expected_prefix = ["OLDER ASK", "candidate answer", "verified answer"]
        expected_active = [1, 1, 1, 0, 0]
    else:
        media_content = [
            {"type": "text", "text": "OLDER ASK"},
            {"type": "image_url", "image_url": {"url": "data:image/png,AA"}},
        ]
        history = [
            {"role": "user", "content": media_content},
            {"role": "assistant", "content": "older answer"},
            {"role": "user", "content": "PLAIN TARGET"},
            {"role": "assistant", "content": "failed answer"},
        ]
        durable_prefix = [
            ("user", "OLDER ASK\n[screenshot]"),
            ("assistant", "older answer"),
        ]
        expected_prefix = [media_content, "older answer"]
        expected_active = [1, 1, 0, 0]

    for role, content in durable_prefix:
        db.append_message(cli.session_id, role, content)
    db.append_message(cli.session_id, "user", "PLAIN TARGET")
    db.append_message(cli.session_id, "assistant", "failed answer")
    cli.conversation_history = history
    cli._pending_input = MagicMock()
    cli._prefill_input_buffer = MagicMock()

    if command == "retry":
        cli.process_command("/retry")
        cli._pending_input.put.assert_called_once_with("PLAIN TARGET")
    else:
        cli.undo_last()
        cli._prefill_input_buffer.assert_called_once_with("PLAIN TARGET")

    assert [message.get("content") for message in cli.conversation_history] == (
        expected_prefix
    )
    assert [row[2] for row in _message_rows(db, cli.session_id)] == expected_active
    db.close()


def test_retry_last_durably_preserves_composite_carrier_scaffold(tmp_path):
    cli = _make_cli()
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db = db
    cli.session_id = "cli-carrier-retry"
    db.create_session(cli.session_id, source="cli")
    db.append_message(cli.session_id, "user", _composite_carrier()["content"])
    db.append_message(cli.session_id, "assistant", "failed answer")
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)
    old_history = cli.conversation_history
    cli.agent = SimpleNamespace(
        _session_messages=old_history,
        _last_flushed_db_idx=len(old_history),
        _db_flush_scan_prefix=list(old_history),
    )

    retry_msg = cli.retry_last()

    assert retry_msg == "REAL ASK"
    assert len(cli.conversation_history) == 1
    scaffold = cli.conversation_history[0]
    assert scaffold["display_kind"] == "hidden"
    assert "REAL ASK" not in scaffold["content"]
    assert scaffold["_db_persisted"] is True
    active = db.get_messages_as_conversation(cli.session_id, include_row_ids=True)
    assert len(active) == 1
    assert active[0]["content"] == scaffold["content"]
    assert active[0]["_row_id"] == scaffold["_row_id"]
    assert cli.agent._session_messages is cli.conversation_history
    assert cli.agent._last_flushed_db_idx == 1
    assert cli.agent._db_flush_scan_prefix == cli.conversation_history
    db.close()


def test_retry_last_rejects_media_before_db_or_memory_mutation():
    cli = _make_cli()
    db = MagicMock()
    cli._session_db = db
    history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look again"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        },
        {"role": "assistant", "content": "old answer"},
    ]
    cli.conversation_history = history

    assert cli.retry_last() is None
    assert cli.conversation_history is history
    db.get_messages_as_conversation.assert_not_called()
    db.rewind_to_message.assert_not_called()


def test_retry_last_db_failure_leaves_warm_history_unchanged():
    cli = _make_cli()
    db = MagicMock()
    db.get_messages_as_conversation.side_effect = OSError("db unavailable")
    cli._session_db = db
    history = [
        {"role": "user", "content": "retry me"},
        {"role": "assistant", "content": "old answer"},
    ]
    cli.conversation_history = history

    assert cli.retry_last() is None
    assert cli.conversation_history is history


def test_undo_last_prefills_live_text_and_retains_durable_scaffold(tmp_path):
    cli = _make_cli()
    cli._session_db.close()
    db = SessionDB(db_path=tmp_path / "state.db")
    cli._session_db = db
    cli.session_id = "cli-carrier-undo"
    db.create_session(cli.session_id, source="cli")
    db.append_message(cli.session_id, "user", "older ask")
    db.append_message(cli.session_id, "assistant", "older answer")
    db.append_message(cli.session_id, "user", _composite_carrier()["content"])
    db.append_message(cli.session_id, "assistant", "failed answer")
    cli.conversation_history = db.get_messages_as_conversation(cli.session_id)
    cli._prefill_input_buffer = MagicMock()
    cli.agent = SimpleNamespace(
        _session_messages=cli.conversation_history,
        _last_flushed_db_idx=len(cli.conversation_history),
        _db_flush_scan_prefix=list(cli.conversation_history),
        _invalidate_system_prompt=MagicMock(),
        _memory_manager=None,
    )

    cli.undo_last()

    cli._prefill_input_buffer.assert_called_once_with("REAL ASK")
    assert [m.get("content") for m in cli.conversation_history[:2]] == [
        "older ask",
        "older answer",
    ]
    scaffold = cli.conversation_history[2]
    assert scaffold["display_kind"] == "hidden"
    assert "REAL ASK" not in scaffold["content"]
    assert scaffold["_db_persisted"] is True
    active = db.get_messages_as_conversation(cli.session_id, include_row_ids=True)
    assert active[2]["_row_id"] == scaffold["_row_id"]
    assert active[2]["content"] == scaffold["content"]
    assert cli.agent._session_messages is cli.conversation_history
    assert cli.agent._last_flushed_db_idx == 3
    db.close()
