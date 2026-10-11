"""Expiry uses producer-owned reference/live boundaries, not action vocabulary."""
import copy

import pytest

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
from agent.replay_cleanup import (
    _EXPIRED_CONFIRMATION_SENTINEL,
    canonicalize_replay_history,
)
from hermes_state import SessionDB
from agent.turn_context import build_api_messages
from agent.transports.chat_completions import ChatCompletionsTransport


class _SendAgent:
    api_mode = "chat_completions"
    ephemeral_system_prompt = None
    _compression_warning = None
    _current_turn_timestamp = 1_120.0

    @staticmethod
    def _copy_reasoning_content_for_api(_source, _target):
        return None

    @staticmethod
    def _should_sanitize_tool_calls():
        return False


DESTRUCTIVE = [
    "confirm reboot. Reboot it now.",
    "確認強制重開機。今すぐ本番サーバーを再起動してください。",
    "confirm reboot. Please reboot the production host now; I authorize it.",
    "confirm reboot. On the production host, reboot it now.",
    "Reboot the production host now; confirm reboot.",
    "Reference: confirm reboot. Live request: Reboot it now.",
]


@pytest.mark.parametrize("content", DESTRUCTIVE)
def test_unsegmented_confirmation_expires_whole_row(content):
    row = {"role": "user", "content": content, "api_content": content, "timestamp": 1_000.0}
    frozen = copy.deepcopy(row)
    out = canonicalize_replay_history([row], now=1_120.0)[0]
    assert out["content"] == _EXPIRED_CONFIRMATION_SENTINEL
    assert "api_content" not in out
    assert row == frozen
    assert canonicalize_replay_history([out], now=1_120.0) == [out]


def _carrier(live, force_user_leading):
    compressor = object.__new__(ContextCompressor)
    compressor._summary_has_user_turn = True
    row = {"role": "user", "content": live, "timestamp": 1_000.0}
    compressor._merge_summary_into_tail_row(
        row,
        SUMMARY_PREFIX + "Documentation mentions confirm reboot in restart.md. Budget is 25000.",
        "user", force_user_leading,
    )
    return row


@pytest.mark.parametrize("force_user_leading", [False, True])
@pytest.mark.parametrize("live", DESTRUCTIVE[:5])
def test_merged_carrier_expires_live_authorization_but_keeps_reference(live, force_user_leading):
    row = _carrier(live, force_user_leading)
    row["api_content"] = row["content"]
    frozen = copy.deepcopy(row)
    out = canonicalize_replay_history([row], now=1_120.0)[0]
    assert "Budget is 25000." in out["content"]
    assert "restart.md" in out["content"]
    assert "confirm reboot" not in out["content"]
    assert "Reboot it now" not in out["content"]
    assert "production host" not in out["content"]
    assert "本番サーバー" not in out["content"]
    assert "api_content" not in out
    assert row == frozen


@pytest.mark.parametrize("force_user_leading", [False, True])
@pytest.mark.parametrize("live", [
    "inspect the logs", "explain the Delete key", "do not restart anything; summarize the logs",
    *DESTRUCTIVE[:2],
])
@pytest.mark.parametrize("inject_timestamps", [False, True])
def test_real_carrier_survives_db_and_gateway(tmp_path, monkeypatch, live, force_user_leading, inject_timestamps):
    from gateway.run import _build_gateway_agent_history

    row = _carrier(live, force_user_leading)
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session(session_id="s", source="cli")
    db.append_message("s", role="user", content=row["content"], timestamp=row["timestamp"],
                      _compressed_summary=True, api_content=row["content"])
    db.close()
    db = SessionDB(db_path=path)
    try:
        history = db.get_messages_as_conversation("s", repair_alternation=True)
    finally:
        db.close()
    monkeypatch.setattr("agent.replay_cleanup.time.time", lambda: 1_120.0)
    replay, _ = _build_gateway_agent_history(history, inject_timestamps=inject_timestamps)
    assert replay[0]["_compressed_summary"] is True
    if live in DESTRUCTIVE:
        assert "Reboot it now" not in replay[0]["content"]
        assert "本番サーバー" not in replay[0]["content"]
    else:
        assert live in replay[0]["content"]
    assert "Budget is 25000." in replay[0]["content"]
    assert "confirm reboot" not in replay[0]["content"]
    assert "api_content" not in replay[0]
    assert canonicalize_replay_history(replay, now=1_120.0) == replay
    # The normal send path canonicalizes the same persisted prefix, independent
    # of the gateway's opt-in presentation switch. Compare provider-visible text.
    send_history = history + [{"role": "user", "content": "continue", "timestamp": 1_120.0}]
    request, _ = build_api_messages(
        _SendAgent(), send_history, current_turn_user_idx=1,
        ext_prefetch_cache="", plugin_user_context="", moa_config=None, active_system_prompt="",
    )
    wire = ChatCompletionsTransport().convert_messages(request)
    assert wire[0]["content"] == replay[0]["content"]
