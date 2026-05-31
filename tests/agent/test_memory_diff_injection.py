"""Unit tests for diff-only memory injection.

Covers memory extraction, diff formatting, turn-based prompt construction,
and conversation loop integration/restore paths.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.system_prompt import build_system_prompt
from tools.memory_tool import MemoryStore


def _make_test_agent():
    """Construct a mock agent with all required attributes pre-configured."""
    agent = MagicMock()
    agent._cached_system_prompt = None
    agent.session_id = "test-session"
    agent.model = "test-model"
    agent.provider = "test-provider"
    agent.platform = "cli"
    agent._memory_store = None
    agent._memory_enabled = False
    agent._user_profile_enabled = False
    agent.skip_context_files = True
    agent.load_soul_identity = False
    agent.valid_tool_names = []
    agent.pass_session_id = False
    agent._kanban_worker_guidance = None
    agent._task_completion_guidance = False
    agent._tool_use_enforcement = False
    agent._memory_manager = None
    agent._user_turn_count = 0
    return agent


def test_parse_entries_from_prompt():
    """Verify that parse_entries_from_prompt extracts entries correctly."""
    prompt = (
        "══════════════════════════════════════════════\n"
        "MEMORY (your personal notes) [10% — 100/2200 chars]\n"
        "══════════════════════════════════════════════\n"
        "First memory entry\n"
        "§\n"
        "Second memory entry\n"
        "══════════════════════════════════════════════\n"
        "USER PROFILE (who the user is) [5% — 50/1375 chars]\n"
        "══════════════════════════════════════════════\n"
        "User profile entry 1\n"
        "§\n"
        "User profile entry 2"
    )

    mem_entries = MemoryStore.parse_entries_from_prompt(prompt, "memory")
    user_entries = MemoryStore.parse_entries_from_prompt(prompt, "user")

    assert mem_entries == ["First memory entry", "Second memory entry"]
    assert user_entries == ["User profile entry 1", "User profile entry 2"]


def test_load_initial_memories_from_prompt():
    """Verify that load_initial_memories_from_prompt hydrates memory store correctly."""
    prompt = (
        "══════════════════════════════════════════════\n"
        "MEMORY (your personal notes) [10% — 100/2200 chars]\n"
        "══════════════════════════════════════════════\n"
        "Initial Memory A\n"
        "══════════════════════════════════════════════\n"
        "USER PROFILE (who the user is) [5% — 50/1375 chars]\n"
        "══════════════════════════════════════════════\n"
        "Initial User Profile A"
    )

    store = MemoryStore()
    store.load_initial_memories_from_prompt(prompt)

    assert store._initial_memory_entries == ["Initial Memory A"]
    assert store._initial_user_entries == ["Initial User Profile A"]


def test_format_diff_for_system_prompt():
    """Verify that format_diff_for_system_prompt returns correct +/- diff blocks."""
    store = MemoryStore()

    # 1. No change -> returns None
    store._initial_memory_entries = ["A", "B"]
    store.memory_entries = ["A", "B"]
    assert store.format_diff_for_system_prompt("memory") is None

    # 2. Addition
    store._initial_memory_entries = ["A"]
    store.memory_entries = ["A", "B"]
    diff = store.format_diff_for_system_prompt("memory")
    assert diff is not None
    assert "MEMORY UPDATES" in diff
    assert "+ B" in diff
    assert "- A" not in diff

    # 3. Deletion
    store._initial_memory_entries = ["A", "B"]
    store.memory_entries = ["B"]
    diff = store.format_diff_for_system_prompt("memory")
    assert diff is not None
    assert "- A" in diff
    assert "+ B" not in diff

    # 4. Mix
    store._initial_memory_entries = ["A", "B"]
    store.memory_entries = ["B", "C"]
    diff = store.format_diff_for_system_prompt("memory")
    assert diff is not None
    assert "- A" in diff
    assert "+ C" in diff


def test_build_system_prompt_turn_based():
    """Verify that system prompt formatting depends on agent._user_turn_count."""
    agent = _make_test_agent()
    agent._memory_store = MemoryStore()
    agent._memory_enabled = True
    agent._user_profile_enabled = True

    agent._memory_store.memory_entries = ["Current Mem"]
    agent._memory_store.user_entries = ["Current User"]
    # Populate the system prompt snapshot which is returned on Turn 1
    agent._memory_store._system_prompt_snapshot = {
        "memory": agent._memory_store._render_block("memory", agent._memory_store.memory_entries),
        "user": agent._memory_store._render_block("user", agent._memory_store.user_entries),
    }

    # --- Turn 1 ---
    agent._user_turn_count = 1
    prompt_t1 = build_system_prompt(agent)
    assert "MEMORY (your personal notes)" in prompt_t1
    assert "USER PROFILE (who the user is)" in prompt_t1
    assert "Current Mem" in prompt_t1
    assert "Current User" in prompt_t1
    assert agent._system_prompt_is_diff is False

    # --- Turn 2 (No Change) ---
    agent._user_turn_count = 2
    agent._memory_store._initial_memory_entries = ["Current Mem"]
    agent._memory_store._initial_user_entries = ["Current User"]
    prompt_t2_no_change = build_system_prompt(agent)
    # Full headers and entries should be absent
    assert "MEMORY (your personal notes)" not in prompt_t2_no_change
    assert "USER PROFILE (who the user is)" not in prompt_t2_no_change
    assert "MEMORY UPDATES" not in prompt_t2_no_change
    assert "USER PROFILE UPDATES" not in prompt_t2_no_change
    assert agent._system_prompt_is_diff is True

    # --- Turn 2 (With Changes) ---
    agent._memory_store.memory_entries = ["Current Mem", "Added Mem"]
    agent._memory_store.user_entries = []  # Removed Current User
    prompt_t2_with_change = build_system_prompt(agent)
    assert "MEMORY UPDATES" in prompt_t2_with_change
    assert "+ Added Mem" in prompt_t2_with_change
    assert "USER PROFILE UPDATES" in prompt_t2_with_change
    assert "- Current User" in prompt_t2_with_change


def test_restore_or_build_system_prompt_subsequent_turn():
    """Verify restore logic on subsequent turns preserves DB state and updates memories."""
    stored_t1_prompt = (
        "══════════════════════════════════════════════\n"
        "MEMORY (your personal notes) [10% — 100/2200 chars]\n"
        "══════════════════════════════════════════════\n"
        "Turn 1 Mem\n"
        "══════════════════════════════════════════════\n"
        "USER PROFILE (who the user is) [5% — 50/1375 chars]\n"
        "══════════════════════════════════════════════\n"
        "Turn 1 User"
    )

    db = MagicMock()
    db.get_session.return_value = {"system_prompt": stored_t1_prompt}

    agent = _make_test_agent()
    agent._session_db = db
    agent._user_turn_count = 2
    agent._cached_system_prompt = None

    # Memory store setup
    store = MemoryStore()
    store.memory_entries = ["Turn 1 Mem", "Added Turn 2 Mem"]
    store.user_entries = ["Turn 1 User"]
    agent._memory_store = store
    agent._memory_enabled = True
    agent._user_profile_enabled = True

    # Build prompt mock
    agent._build_system_prompt = lambda system_message: build_system_prompt(agent, system_message)

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hello"}])

    # 1. Check that initial memories were loaded from stored_t1_prompt
    assert store._initial_memory_entries == ["Turn 1 Mem"]
    assert store._initial_user_entries == ["Turn 1 User"]

    # 2. Check that the rebuilt prompt is cached but contains the Turn 2 diff
    rebuilt = agent._cached_system_prompt
    assert rebuilt is not None
    assert "MEMORY UPDATES" in rebuilt
    assert "+ Added Turn 2 Mem" in rebuilt
    assert "USER PROFILE UPDATES" not in rebuilt  # Unchanged

    # 3. Check that the DB was NOT updated (we keep Turn 1 prompt in DB)
    db.update_system_prompt.assert_not_called()
