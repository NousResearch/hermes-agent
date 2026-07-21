"""Tests for agent/system_reminder_engine.py.

Covers:
- Engine construction and defaults
- Cadence counting (every N tool calls triggers injection)
- No injection when disabled or no blocks
- Rotation through blocks
- Counter reset
- ``build_reminder()`` action shapes for each position
- ``agent_inject_system_reminder()`` with realistic messages
- Edge cases: zero-cadence guard, single block, empty messages
"""

from __future__ import annotations

import copy

import pytest

from agent.system_reminder_engine import (
    DEFAULT_REMINDER_BLOCKS,
    SystemReminderEngine,
    agent_inject_system_reminder,
)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def engine() -> SystemReminderEngine:
    """Default engine: enabled, cadence=5, default blocks."""
    return SystemReminderEngine(enabled=True, cadence=5)


@pytest.fixture
def agent_stub():
    """Minimal agent with a ``_system_reminder_engine`` attribute."""
    eng = SystemReminderEngine(enabled=True, cadence=3)
    return type("AgentStub", (), {"_system_reminder_engine": eng})


@pytest.fixture
def agent_no_engine():
    """Agent stub with no engine attribute at all."""
    return type("AgentStub", (), {})


# ======================================================================
# Engine construction
# ======================================================================


class TestConstruction:
    def test_defaults(self):
        eng = SystemReminderEngine()
        assert eng.enabled is True
        assert eng.cadence == 5
        assert len(eng.blocks) == len(DEFAULT_REMINDER_BLOCKS)
        assert eng.position == "last_user"
        assert eng._counter == 0
        assert eng._block_index == 0

    def test_custom_config(self):
        blocks = [{"id": "b1", "content": "one"}, {"id": "b2", "content": "two"}]
        eng = SystemReminderEngine(
            enabled=True, cadence=2, blocks=blocks, position="new_user"
        )
        assert eng.cadence == 2
        assert eng.blocks == blocks
        assert eng.position == "new_user"

    def test_zero_cadence_is_clamped(self):
        """Cadence of 0 is clamped to 1 so we don't infinite-loop."""
        eng = SystemReminderEngine(enabled=True, cadence=0)
        assert eng.cadence == 1

    def test_negative_cadence_is_clamped(self):
        eng = SystemReminderEngine(enabled=True, cadence=-5)
        assert eng.cadence == 1

    def test_disabled_reports_false(self):
        eng = SystemReminderEngine(enabled=False)
        assert eng.should_inject() is False
        assert eng.build_reminder() is None

    def test_no_blocks_reports_false(self):
        eng = SystemReminderEngine(enabled=True, blocks=[])
        assert eng.should_inject() is False
        assert eng.build_reminder() is None


# ======================================================================
# Cadence counting (should_inject)
# ======================================================================


class TestCadence:
    def test_does_not_inject_before_threshold(self, engine):
        for _ in range(4):
            assert engine.should_inject() is False

    def test_injects_at_threshold(self, engine):
        for _ in range(4):
            engine.should_inject()  # advance to threshold
        assert engine.should_inject() is True

    def test_counter_resets_after_injection(self):
        """After hitting the threshold, counter is 0 again."""
        eng = SystemReminderEngine(enabled=True, cadence=3)
        for _ in range(2):
            eng.should_inject()
        # third call hits threshold
        assert eng.should_inject() is True
        # next call should be 1/3: False
        assert eng.should_inject() is False
        assert eng.should_inject() is False  # 2/3
        assert eng.should_inject() is True   # 3/3 → hit

    def test_explicit_reset(self, engine):
        for _ in range(4):
            engine.should_inject()  # 4 calls, not yet 5
        engine.reset_counter()
        for _ in range(4):
            assert engine.should_inject() is False
        # now on 5th call after reset, should inject
        assert engine.should_inject() is True

    def test_disabled_never_injects(self):
        eng = SystemReminderEngine(enabled=False)
        for _ in range(100):
            assert eng.should_inject() is False

    def test_blocks_empty_never_injects(self):
        eng = SystemReminderEngine(enabled=True, blocks=[])
        for _ in range(100):
            assert eng.should_inject() is False

    def test_cadence_one_injects_every_time(self):
        eng = SystemReminderEngine(enabled=True, cadence=1)
        for _ in range(5):
            assert eng.should_inject() is True


# ======================================================================
# Block rotation
# ======================================================================


class TestBlockRotation:
    def test_single_block_repeats(self):
        blocks = [{"id": "only", "content": "just this one"}]
        eng = SystemReminderEngine(enabled=True, cadence=1, blocks=blocks)
        r1 = eng.build_reminder()
        r2 = eng.build_reminder()
        assert r1 is not None and r2 is not None
        # both contain the same content
        assert r1["content"] == r2["content"]

    def test_cycles_through_multiple_blocks(self):
        blocks = [
            {"id": "a", "content": "alpha"},
            {"id": "b", "content": "beta"},
            {"id": "c", "content": "gamma"},
        ]
        eng = SystemReminderEngine(enabled=True, cadence=1, blocks=blocks)
        seen = []
        for _ in range(5):
            r = eng.build_reminder()
            assert r is not None
            seen.append(r["content"].lstrip("\n"))
        assert seen == ["alpha", "beta", "gamma", "alpha", "beta"]

    def test_block_index_increments_on_each_build(self):
        blocks = [{"id": "x", "content": "X"}, {"id": "y", "content": "Y"}]
        eng = SystemReminderEngine(enabled=True, cadence=1, blocks=blocks)
        assert eng._block_index == 0
        eng.build_reminder()
        assert eng._block_index == 1
        eng.build_reminder()
        assert eng._block_index == 2


# ======================================================================
# build_reminder action shapes
# ======================================================================


class TestReminderShape:
    def test_last_user_position(self):
        blocks = [{"id": "t", "content": "test"}]
        eng = SystemReminderEngine(
            enabled=True, cadence=1, blocks=blocks, position="last_user"
        )
        reminder = eng.build_reminder()
        assert reminder is not None
        assert reminder["action"] == "append_to_last_user"
        assert "test" in reminder["content"]

    def test_new_user_position(self):
        blocks = [{"id": "t", "content": "test"}]
        eng = SystemReminderEngine(
            enabled=True, cadence=1, blocks=blocks, position="new_user"
        )
        reminder = eng.build_reminder()
        assert reminder is not None
        assert reminder["role"] == "user"
        assert reminder["content"] == "test"

    def test_assistant_prefix_position(self):
        blocks = [{"id": "t", "content": "test"}]
        eng = SystemReminderEngine(
            enabled=True, cadence=1, blocks=blocks, position="assistant_prefix"
        )
        reminder = eng.build_reminder()
        assert reminder is not None
        assert reminder["action"] == "assistant_prefix"
        assert reminder["content"] == "test"


# ======================================================================
# maybe_inject convenience
# ======================================================================


class TestMaybeInject:
    def test_returns_none_when_no_injection(self, engine):
        assert engine.maybe_inject() is None

    def test_returns_reminder_when_due(self):
        eng = SystemReminderEngine(enabled=True, cadence=1)
        r = eng.maybe_inject()
        assert r is not None
        assert "content" in r

    def test_returns_none_when_disabled(self):
        eng = SystemReminderEngine(enabled=False)
        assert eng.maybe_inject() is None

    def test_returns_none_when_no_blocks(self):
        eng = SystemReminderEngine(enabled=True, blocks=[])
        assert eng.maybe_inject() is None


# ======================================================================
# agent_inject_system_reminder (integration helper)
# ======================================================================


class TestAgentInject:
    def test_skips_when_no_tool_msgs(self, agent_stub):
        messages = [{"role": "user", "content": "hello"}]
        before = copy.deepcopy(messages)
        agent_inject_system_reminder(agent_stub, messages, num_tool_msgs=0)
        assert messages == before  # unchanged

    def test_skips_when_empty_messages(self, agent_stub):
        messages = []
        agent_inject_system_reminder(agent_stub, messages, num_tool_msgs=1)
        assert messages == []

    def test_skips_when_no_engine_attr(self, agent_no_engine):
        messages = [{"role": "tool", "content": "result"}]
        before = copy.deepcopy(messages)
        agent_inject_system_reminder(agent_no_engine, messages, num_tool_msgs=1)
        assert messages == before

    def test_injects_into_last_tool_message(self):
        """When cadence is 1, expects injection after each tool msg."""
        eng = SystemReminderEngine(enabled=True, cadence=1)
        agent = type("A", (), {"_system_reminder_engine": eng})
        messages = [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": "ok", "tool_calls": [{"id": "tc1"}]},
            {"role": "tool", "tool_call_id": "tc1", "content": "done"},
        ]
        original_last = messages[-1]["content"]
        agent_inject_system_reminder(agent, messages, num_tool_msgs=1)
        assert len(messages) == 3  # no new messages
        assert messages[-1]["role"] == "tool"
        assert messages[-1]["content"] != original_last  # appended
        assert "<system-reminder>" in messages[-1]["content"]

    def test_injects_into_correct_tool_in_batch(self):
        """With cadence 3, third tool result in batch gets the injection."""
        eng = SystemReminderEngine(enabled=True, cadence=3)
        agent = type("A", (), {"_system_reminder_engine": eng})

        # First tool call
        msgs1 = [
            {"role": "user", "content": "do three things"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "tool_call_id": "tc1", "content": "result 1"},
        ]
        agent_inject_system_reminder(agent, msgs1, num_tool_msgs=1)
        # Counter at 1, not yet 3 — no injection
        assert "<system-reminder>" not in (msgs1[-1]["content"] or "")

        # Second tool call
        msgs2 = msgs1 + [
            {"role": "tool", "tool_call_id": "tc2", "content": "result 2"}
        ]
        agent_inject_system_reminder(agent, msgs2, num_tool_msgs=1)
        # Counter at 2 — no injection
        assert "<system-reminder>" not in (msgs2[-1]["content"] or "")

        # Third tool call — cadence hit
        msgs3 = msgs2 + [
            {"role": "tool", "tool_call_id": "tc3", "content": "result 3"}
        ]
        agent_inject_system_reminder(agent, msgs3, num_tool_msgs=1)
        assert "<system-reminder>" in msgs3[-1]["content"]

    def test_multimodal_content_list(self):
        """Injection handles Anthropic-style content list tool results."""
        eng = SystemReminderEngine(enabled=True, cadence=1)
        agent = type("A", (), {"_system_reminder_engine": eng})
        messages = [
            {"role": "user", "content": "do it"},
            {
                "role": "tool",
                "tool_call_id": "tc1",
                "content": [
                    {"type": "text", "text": "Here's the image result."},
                    {"type": "image_url", "image_url": {"url": "data:img/png;base64,abc"}},
                ],
            },
        ]
        agent_inject_system_reminder(agent, messages, num_tool_msgs=1)
        content = messages[-1]["content"]
        assert isinstance(content, list)
        assert any(
            isinstance(b, dict) and b.get("type") == "text"
            and "<system-reminder>" in b.get("text", "")
            for b in content
        )

    def test_none_content_on_tool_msg(self):
        """Edge case: tool message with content=None is handled safely."""
        eng = SystemReminderEngine(enabled=True, cadence=1)
        agent = type("A", (), {"_system_reminder_engine": eng})
        messages = [
            {"role": "tool", "tool_call_id": "tc1", "content": None},
        ]
        agent_inject_system_reminder(agent, messages, num_tool_msgs=1)
        assert messages[-1]["role"] == "tool"
        # content=None becomes a list (multimodal fallback) with a text block
        content = messages[-1]["content"]
        assert isinstance(content, list)
        assert any(
            isinstance(b, dict) and b.get("type") == "text"
            and "<system-reminder>" in b.get("text", "")
            for b in content
        )

    def test_no_tool_role_in_messages_does_not_crash(self):
        """num_tool_msgs > 0 but no tool-role message in the tail."""
        eng = SystemReminderEngine(enabled=True, cadence=1)
        agent = type("A", (), {"_system_reminder_engine": eng})
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        # This should not crash — the helper skips gracefully
        agent_inject_system_reminder(agent, messages, num_tool_msgs=1)
        assert len(messages) == 2  # unchanged


# ======================================================================
# Default blocks
# ======================================================================


class TestDefaultBlocks:
    def test_all_defaults_have_required_keys(self):
        for block in DEFAULT_REMINDER_BLOCKS:
            assert "id" in block, f"Missing 'id' in {block}"
            assert "content" in block, f"Missing 'content' in {block}"
            assert isinstance(block["id"], str)
            assert isinstance(block["content"], str)

    def test_all_defaults_contain_system_reminder_tag(self):
        for block in DEFAULT_REMINDER_BLOCKS:
            assert "<system-reminder>" in block["content"]

    def test_ids_are_unique(self):
        ids = [b["id"] for b in DEFAULT_REMINDER_BLOCKS]
        assert len(ids) == len(set(ids)), f"Duplicate ids: {ids}"


# ======================================================================
# State snapshot
# ======================================================================


class TestState:
    def test_state_contains_diagnostics(self, engine):
        state = engine.state
        assert state["enabled"] is True
        assert state["cadence"] == 5
        assert state["num_blocks"] == len(DEFAULT_REMINDER_BLOCKS)
        assert state["position"] == "last_user"
        assert state["counter"] == 0
        assert state["block_index"] == 0
        assert state["next_block_id"] == DEFAULT_REMINDER_BLOCKS[0]["id"]

    def test_state_updates_as_engine_runs(self):
        eng = SystemReminderEngine(enabled=True, cadence=3, blocks=[{"id": "x", "content": "X"}])
        for _ in range(3):
            eng.should_inject()
        state = eng.state
        assert state["counter"] == 0  # reset after injection

    def test_disabled_state(self):
        eng = SystemReminderEngine(enabled=False)
        state = eng.state
        assert state["enabled"] is False
