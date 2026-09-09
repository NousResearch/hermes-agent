"""CLI command for immediate native Responses compaction."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB
from tests.cli.test_cli_init import _make_cli


def test_native_compact_persists_checkpoint_projection_without_rewriting_history(capsys):
    """The command keeps the readable transcript while persisting the new sidecar."""
    shell = _make_cli()
    history: list[dict[str, object]] = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
    ]
    projected = [dict(message) for message in history]
    projected[1]["codex_reasoning_items"] = [
        {"type": "compaction", "encrypted_content": "opaque"}
    ]
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent._session_db = MagicMock()

    with patch(
        "agent.native_compaction.manual_native_responses_compaction",
        return_value=SimpleNamespace(messages=projected, checkpoint_count=1),
    ) as compact:
        shell._handle_native_compact_command("/native-compact")

    compact.assert_called_once_with(shell.agent, history)
    assert shell.conversation_history == projected
    assert shell.agent._session_messages == projected
    shell.agent._session_db.replace_messages.assert_called_once_with(
        shell.session_id,
        projected,
        active_only=True,
        reject_active_turn_lease=True,
    )
    assert "Native Codex context compacted" in capsys.readouterr().out


def test_native_compact_checkpoint_survives_a_session_db_reload(tmp_path):
    """The sidecar must be durable; a resume cannot fall back to the pre-compact wire."""
    session_id = "native-compact"
    history: list[dict[str, object]] = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
    ]
    projected = [dict(message) for message in history]
    projected[1]["codex_reasoning_items"] = [
        {"type": "compaction", "encrypted_content": "opaque"}
    ]
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id, "cli")
    db.append_messages_batch(session_id, history)
    shell = _make_cli()
    shell.session_id = session_id
    shell.conversation_history = history
    shell.agent = MagicMock()
    shell.agent._session_db = db

    with patch(
        "agent.native_compaction.manual_native_responses_compaction",
        return_value=SimpleNamespace(messages=projected, checkpoint_count=1),
    ):
        shell._handle_native_compact_command("/native-compact")

    restored = db.get_messages_as_conversation(session_id)
    assert [message["content"] for message in restored] == [
        "old user", "old assistant", "recent user", "recent assistant"
    ]
    assert restored[1]["codex_reasoning_items"] == [
        {"type": "compaction", "encrypted_content": "opaque"}
    ]
