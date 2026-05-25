"""Tests for live present-state memory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.present_state_memory import PresentStateMemory, get_present_state_path
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from run_agent import AIAgent


@pytest.fixture()
def hermes_home(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        yield tmp_path
    finally:
        reset_hermes_home_override(token)


def test_initialize_uses_profile_scoped_state_file(hermes_home: Path):
    memory = PresentStateMemory()
    memory.initialize("session-1")

    path = get_present_state_path()
    assert path == hermes_home / "memories" / "PRESENT_STATE.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["active_session_id"] == "session-1"
    assert data["schema_version"] == 1


def test_memory_write_is_available_in_same_session_context(hermes_home: Path):
    memory = PresentStateMemory()
    memory.initialize("session-1")

    memory.capture_memory_write(
        "add",
        "user",
        "Volmarr prefers direct, implementation-first answers.",
        session_id="session-1",
    )

    rendered = memory.render_context(session_id="session-1")
    assert "ACTIVE PRESENT STATE" in rendered
    assert "Volmarr prefers direct" in rendered


def test_completed_turn_extracts_profile_and_current_session_facts(hermes_home: Path):
    memory = PresentStateMemory()
    memory.initialize("session-1")

    memory.capture_turn(
        "Call me Volmarr. I prefer concise updates. We are going to improve Hermes memory.",
        "Implemented the first slice.",
        session_id="session-1",
    )

    rendered = memory.render_context(session_id="session-1")
    assert "User wants to be called Volmarr" in rendered
    assert "User prefers concise updates" in rendered
    assert "Current goal: improve Hermes memory" in rendered
    assert "Latest assistant outcome: Implemented the first slice." in rendered


def test_session_switch_carries_lineage_but_reset_starts_fresh_session(hermes_home: Path):
    memory = PresentStateMemory()
    memory.initialize("session-1")
    memory.capture_turn(
        "We are going to improve Hermes memory.",
        "Implemented the first slice.",
        session_id="session-1",
    )

    memory.on_session_switch("session-2", parent_session_id="session-1", reset=False)
    assert "Current goal: improve Hermes memory" in memory.render_context(session_id="session-2")

    memory.on_session_switch("session-3", parent_session_id="session-2", reset=True)
    assert "Current goal: improve Hermes memory" not in memory.render_context(session_id="session-3")


def test_corrupt_state_file_falls_back_without_crashing(hermes_home: Path):
    path = get_present_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")

    memory = PresentStateMemory()
    memory.initialize("session-1")

    rendered = memory.render_context(session_id="session-1")
    assert rendered == ""


def test_agent_bridge_captures_memory_write(hermes_home: Path):
    memory = PresentStateMemory()
    memory.initialize("session-1")
    agent = type("AgentStub", (), {})()
    agent._present_state_memory = memory
    agent.session_id = "session-1"

    AIAgent._capture_present_state_memory_write(
        agent,
        action="add",
        target="memory",
        content="Hermes runs in a profile-scoped HERMES_HOME.",
    )

    rendered = memory.render_context(session_id="session-1")
    assert "Hermes runs in a profile-scoped HERMES_HOME" in rendered


def test_agent_bridge_captures_completed_turn(hermes_home: Path):
    memory = PresentStateMemory()
    memory.initialize("session-1")
    agent = type("AgentStub", (), {})()
    agent._present_state_memory = memory
    agent.session_id = "session-1"

    AIAgent._sync_present_state_for_turn(
        agent,
        original_user_message="Remember that Hermes should keep live present state.",
        final_response="Done.",
        interrupted=False,
    )

    rendered = memory.render_context(session_id="session-1")
    assert "Hermes should keep live present state" in rendered


def test_agent_bridge_skips_interrupted_turns(hermes_home: Path):
    memory = PresentStateMemory()
    memory.initialize("session-1")
    agent = type("AgentStub", (), {})()
    agent._present_state_memory = memory
    agent.session_id = "session-1"

    AIAgent._sync_present_state_for_turn(
        agent,
        original_user_message="Remember that this should not persist.",
        final_response="Partial.",
        interrupted=True,
    )

    assert "this should not persist" not in memory.render_context(session_id="session-1")
