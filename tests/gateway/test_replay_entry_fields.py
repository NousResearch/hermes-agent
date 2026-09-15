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


from gateway.run import (
    _ASSISTANT_REPLAY_FIELDS,
    _build_gateway_agent_history,
    _build_replay_entry,
)


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


    def test_replay_fields_constant_is_stable(self):
        """Pin the whitelist explicitly so accidental renames are caught."""
        assert _ASSISTANT_REPLAY_FIELDS == (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "codex_reasoning_items",
            "codex_message_items",
            "finish_reason",
        )

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


def _assistant_rows(history):
    """Rebuild replay history via the PRODUCTION builder and return the
    assistant rows.

    These tests deliberately exercise ``_build_gateway_agent_history``
    itself rather than a local copy of its filtering logic: the dispatch
    under test lives in the builder, so a replica would pass no matter what
    the builder does.
    """
    agent_history, _observed = _build_gateway_agent_history(history)
    return [m for m in agent_history if m.get("role") == "assistant"]


class TestReasoningOnlyAssistantTurnsSurviveReload:
    """An assistant turn whose only payload is reasoning state must survive
    the transcript -> agent_history rebuild.

    The builder's row dispatch keeps rich rows (tool_calls / tool results)
    and rows with truthy ``content``.  A row with empty ``content`` carrying
    only the fields named by ``_ASSISTANT_REPLAY_FIELDS`` matched neither
    arm and was dropped outright, losing multi-turn thinking fidelity on
    reload.
    """

    def test_reasoning_content_only_turn_is_preserved(self):
        """DeepSeek/Kimi thinking-mode shape: content empty, reasoning_content set."""
        assistant = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": "step by step"},
        ])
        assert len(assistant) == 1
        assert assistant[0]["reasoning_content"] == "step by step"
        assert assistant[0]["content"] == ""

    def test_reasoning_only_turn_with_none_content_is_preserved(self):
        """A dead stream persists ``content=None``; the reasoning still replays."""
        assistant = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "reasoning": "thinking"},
        ])
        assert len(assistant) == 1
        assert assistant[0]["reasoning"] == "thinking"
        # `content` is normalized to "" -- the shape the API builders expect.
        assert assistant[0]["content"] == ""

    def test_reasoning_details_only_turn_is_preserved(self):
        """``content`` may be absent entirely, not merely falsy."""
        details = [{"type": "reasoning.summary", "summary": "s"}]
        assistant = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "reasoning_details": details},
        ])
        assert len(assistant) == 1
        assert assistant[0]["reasoning_details"] == details

    def test_reasoning_only_turn_keeps_position_between_user_turns(self):
        """The recovered row must replay in transcript order, not be appended."""
        agent_history, _observed = _build_gateway_agent_history([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "", "reasoning_content": "mulling"},
            {"role": "user", "content": "second"},
        ])
        assert [m["role"] for m in agent_history] == ["user", "assistant", "user"]
        assert agent_history[1]["reasoning_content"] == "mulling"

    def test_assistant_timestamp_is_still_dropped(self):
        """Assistant timestamps are deliberately not replayed; recovering the
        row must not start leaking them."""
        assistant = _assistant_rows([
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "thinking",
                "timestamp": 12345.6,
            },
        ])
        assert len(assistant) == 1
        assert "timestamp" not in assistant[0]

    def test_empty_list_content_keeps_its_type(self):
        """Only ``None``/absent normalizes to ``""``.

        An empty multimodal block list is falsy but is not a missing value,
        so it must survive as a list rather than being rewritten to a string
        -- downstream consumers branch on `isinstance(content, list)`.
        """
        assistant = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [], "reasoning_content": "thinking"},
        ])
        assert len(assistant) == 1
        assert assistant[0]["content"] == []

    def test_truthy_content_turn_is_unaffected(self):
        """Control: the pre-existing hot path is unchanged."""
        assistant = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "answer", "reasoning_content": "thinking"},
        ])
        assert assistant == [
            {"role": "assistant", "content": "answer", "reasoning_content": "thinking"}
        ]


class TestCodexReasoningOnlyTurnsSurviveReload:
    """Codex Responses item carriers are the case where an empty ``content``
    is not a degenerate row but the designed representation.

    A commentary-phase assistant turn persists with ``content: ""`` because
    its text lives in ``codex_message_items``; OpenAI's docs require those
    items be replayed on every assistant message for prefix-cache hits.
    Dropping the row on reload discards them.
    """

    def test_codex_message_items_only_turn_is_preserved(self):
        msg_items = [
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "working on it"}],
            }
        ]
        assistant = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "codex_message_items": msg_items},
        ])
        assert len(assistant) == 1
        assert assistant[0]["codex_message_items"] == msg_items

    def test_codex_reasoning_items_only_turn_is_preserved(self):
        codex_items = [{"type": "reasoning", "encrypted_content": "blob"}]
        assistant = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "codex_reasoning_items": codex_items},
        ])
        assert len(assistant) == 1
        assert assistant[0]["codex_reasoning_items"] == codex_items


class TestEmptyAssistantShellsStillDrop:
    """Negative cases: the recovery must not resurrect rows that carry no
    replay state at all.

    These already dropped before the change and must keep dropping -- they
    guard the fix against over-reaching, which is the failure mode that
    would turn it into a source of empty assistant turns.
    """

    def test_truly_empty_assistant_shell_still_drops(self):
        assert _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ]) == []

    def test_assistant_shell_with_none_content_still_drops(self):
        assert _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
        ]) == []

    def test_assistant_shell_with_all_falsy_replay_fields_still_drops(self):
        """Every contract field present but empty carries no information --
        matching `_build_replay_entry`, which drops them individually."""
        assert _assistant_rows([
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "",
                "reasoning_details": [],
                "codex_reasoning_items": [],
                "codex_message_items": [],
                "finish_reason": "",
            },
        ]) == []

    def test_non_assistant_empty_row_still_drops(self):
        """The new arm is assistant-scoped; an empty user row is untouched
        and must not pick up reasoning fields."""
        agent_history, _observed = _build_gateway_agent_history([
            {"role": "user", "content": "", "reasoning": "not mine"},
            {"role": "assistant", "content": "answer"},
        ])
        assert [m["role"] for m in agent_history] == ["assistant"]


class TestEmptyStringReasoningContentSentinelSurvivesReload:
    """``reasoning_content == ""`` is a meaningful sentinel, not an empty
    value, and the rebuild must not collapse it.

    ``_build_replay_entry`` singles this field out: every other replay field
    is dropped when falsy, but an empty-string ``reasoning_content`` is kept
    because thinking-mode replay upgrades it to a single space. If the row
    is dropped on reload instead, the next request carries no
    ``reasoning_content`` at all, which strict thinking providers reject
    with HTTP 400.

    This is also the case a plain truthiness test gets wrong, so it pins the
    ``None``-vs-``""`` distinction rather than just the round-trip.
    """

    def test_empty_string_sentinel_row_is_preserved(self):
        assistant = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": ""},
        ])
        assert len(assistant) == 1
        assert assistant[0]["reasoning_content"] == ""

    def test_sentinel_is_distinguished_from_absent_reasoning_content(self):
        """An empty-string sentinel is payload; an absent key is not."""
        with_sentinel = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": ""},
        ])
        without_key = _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
        ])
        assert len(with_sentinel) == 1
        assert without_key == []

    def test_explicit_none_reasoning_content_is_not_payload(self):
        """``None`` is the genuine no-value case and must not resurrect the
        row -- matching `_build_replay_entry`, which skips it on ``None``."""
        assert _assistant_rows([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "reasoning_content": None},
        ]) == []

    def test_sentinel_survives_alongside_a_later_user_turn(self):
        """The sentinel must still be there when it is no longer the tail."""
        agent_history, _observed = _build_gateway_agent_history([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "", "reasoning_content": ""},
            {"role": "user", "content": "second"},
        ])
        assert [m["role"] for m in agent_history] == ["user", "assistant", "user"]
        assert agent_history[1]["reasoning_content"] == ""


class TestGatewayHistoryBuildForwardsSidecar:
    def test_end_to_end_history_build_keeps_sidecar(self):
        history = [
            {"role": "user", "content": "hi", "api_content": "hi\n\nCTX", "timestamp": 123.0},
            {"role": "assistant", "content": "hello"},
        ]
        agent_history, _obs = _build_gateway_agent_history(history)
        assert agent_history[0]["api_content"] == "hi\n\nCTX"

