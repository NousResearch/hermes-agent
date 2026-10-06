"""Tests for ``gateway.run._build_replay_entry``.

The gateway rebuilds ``agent_history`` from the persisted transcript on every
turn (unlike the CLI, which keeps the live in-memory message list).  When a
pure-text assistant turn (no ``tool_calls``) is replayed, the simple-text
branch in ``run_sync`` used to whitelist only three reasoning fields:
``reasoning``, ``reasoning_details``, ``codex_reasoning_items``.

That whitelist predated three fields the DB now persists:
``reasoning_content``, ``codex_message_items``, and ``finish_reason``.  The
unrecovered drop of ``codex_message_items`` in particular kills prefix-cache
hits for OpenAI Codex Responses API users — OpenAI's docs require the
``phase`` field be replayed on every assistant message.

These tests pin the expanded whitelist so it doesn't regress.
"""
from __future__ import annotations


from gateway.run import _build_replay_entry


class TestBuildReplayEntry:
    def test_user_message_has_only_role_and_content(self):
        entry = _build_replay_entry(
            "user",
            "hello",
            {"role": "user", "content": "hello", "reasoning": "leak", "extra": "drop"},
        )
        assert entry == {"role": "user", "content": "hello"}

    def test_tool_message_has_only_role_and_content(self):
        # Tool messages aren't routed through this helper in production
        # (they take the rich-passthrough branch), but the helper itself
        # must not leak reasoning fields onto non-assistant roles even if
        # someone calls it incorrectly.
        entry = _build_replay_entry(
            "tool",
            "result",
            {"role": "tool", "content": "result", "reasoning": "leak"},
        )
        assert entry == {"role": "tool", "content": "result"}


    def test_assistant_drops_falsy_reasoning(self):
        """Empty/None reasoning fields stay dropped (matching PR #2974
        behaviour) — empty strings/lists for these fields carry no info."""
        msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning": "",
            "reasoning_details": [],
            "codex_reasoning_items": [],
            "codex_message_items": [],
            "finish_reason": "",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert entry == {"role": "assistant", "content": "answer"}


    def test_assistant_preserves_all_six_fields_together(self):
        details = [{"type": "reasoning.summary", "summary": "s"}]
        codex_items = [{"type": "reasoning", "encrypted_content": "b"}]
        msg_items = [
            {
                "type": "message",
                "role": "assistant",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "x"}],
            }
        ]
        msg = {
            "role": "assistant",
            "content": "answer",
            "reasoning": "thinking",
            "reasoning_content": "structured",
            "reasoning_details": details,
            "codex_reasoning_items": codex_items,
            "codex_message_items": msg_items,
            "finish_reason": "stop",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert entry["reasoning"] == "thinking"
        assert entry["reasoning_content"] == "structured"
        assert entry["reasoning_details"] == details
        assert entry["codex_reasoning_items"] == codex_items
        assert entry["codex_message_items"] == msg_items
        assert entry["finish_reason"] == "stop"


    def test_assistant_carries_display_kind_marker(self):
        """A projected background-result row must stay identifiable through
        this rebuild or the continuity digest cannot exclude it — the digest
        re-presenting the agent's own delivered answer is the
        double-presentation pathology the SDK background lane fixes."""
        msg = {
            "role": "assistant",
            "content": "report",
            "display_kind": "sdk_background_result",
        }
        entry = _build_replay_entry("assistant", "report", msg)
        assert entry["display_kind"] == "sdk_background_result"
        # Non-assistant rows keep the minimal shape.
        user = _build_replay_entry(
            "user", "hi",
            {"role": "user", "content": "hi", "display_kind": "hidden"},
        )
        assert "display_kind" not in user


    def test_unrelated_keys_are_ignored(self):
        """Random keys on the message must not leak into the replay entry."""
        msg = {
            "role": "assistant",
            "content": "answer",
            "timestamp": 12345.6,
            "internal_marker": "should not flow",
            "tool_call_id": "should not be set on simple-text branch",
        }
        entry = _build_replay_entry("assistant", "answer", msg)
        assert "timestamp" not in entry
        assert "internal_marker" not in entry
        assert "tool_call_id" not in entry


class TestReplayEntryApiContentSidecar:
    """The api_content sidecar (persist-what-you-send) must survive the
    gateway's transcript→agent_history rebuild, or the whole prompt-cache
    fix is inert on gateway platforms — but only when this pipeline did not
    rewrite the content (a rewrite means different bytes must replay)."""

    def test_user_forwards_api_content_when_content_unchanged(self):
        msg = {"role": "user", "content": "hi", "api_content": "hi\n\nCTX"}
        entry = _build_replay_entry("user", "hi", msg)
        assert entry["api_content"] == "hi\n\nCTX"
        assert entry["content"] == "hi"

    def test_assistant_forwards_api_content_when_content_unchanged(self):
        msg = {"role": "assistant", "content": "a", "api_content": "a <memory-context>"}
        entry = _build_replay_entry("assistant", "a", msg)
        assert entry["api_content"] == "a <memory-context>"


class TestGatewayHistoryBuildForwardsSidecar:
    def test_end_to_end_history_build_keeps_sidecar(self):
        from gateway.run import _build_gateway_agent_history

        history = [
            {"role": "user", "content": "hi", "api_content": "hi\n\nCTX", "timestamp": 123.0},
            {"role": "assistant", "content": "hello"},
        ]
        agent_history, _obs = _build_gateway_agent_history(history)
        assert agent_history[0]["api_content"] == "hi\n\nCTX"



def test_gateway_history_keeps_sidecar_only_assistant_row():
    """A reasoning-only clean stop persists content="" with the promoted reply in ``api_content``
    (agent/turn_final_response.py); the gateway rebuild must replay it, not drop it (user->user)."""
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "", "reasoning": "The answer is 4.", "api_content": "The answer is 4."},
        {"role": "user", "content": "thanks"},
    ]
    agent_history, _ = _build_gateway_agent_history(history)
    assert [m["role"] for m in agent_history] == ["user", "assistant", "user"]
    assert agent_history[1]["api_content"] == "The answer is 4."
    assert agent_history[1]["reasoning"] == "The answer is 4."


def test_replay_rebuilds_the_same_conversation_so_every_role_keeps_its_identity():
    """Gateway replay turns the stored transcript back into the SAME session's history for its next turn:
    user, assistant, tool-calling and tool rows all keep ``message_uid`` and the merge witness."""
    from gateway.run import _build_gateway_agent_history

    call = {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
    history = [
        {"role": "user", "content": "a\n\nb", "message_uid": "1" * 32, "_absorbed_message_uids": ["b" * 32],
         "timestamp": 1_700_000_000.0},
        {"role": "assistant", "content": "", "tool_calls": [call], "message_uid": "2" * 32},
        {"role": "tool", "content": "r", "tool_call_id": "call_1", "message_uid": "3" * 32},
        {"role": "assistant", "content": "done", "message_uid": "4" * 32, "_absorbed_message_uids": ["c" * 32]},
    ]
    replayed, _ = _build_gateway_agent_history(history)
    assert [(m["role"], m.get("message_uid"), m.get("_absorbed_message_uids")) for m in replayed] == [
        (m["role"], m["message_uid"], m.get("_absorbed_message_uids")) for m in history]


def test_a_plain_replay_row_does_not_carry_a_tool_call_uid_map_without_its_calls():
    """Repair can prune an assistant's unanswered tool_calls and leave its ``_tool_call_uids`` behind: the
    replayed plain row keeps its own uid but not a map naming calls it no longer carries."""
    from gateway.run import _build_gateway_agent_history

    history = [
        {"role": "user", "content": "q", "message_uid": "1" * 32},
        {"role": "assistant", "content": "done", "message_uid": "2" * 32, "_tool_call_uids": {"call_1": "5" * 32}},
    ]
    replayed, _ = _build_gateway_agent_history(history)
    assert replayed[1]["message_uid"] == "2" * 32
    assert "_tool_call_uids" not in replayed[1] and "_tool_call_uid" not in replayed[1]
