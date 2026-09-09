"""
Tests for Hermes Core Engine, ModelManager, AgentOrchestrator, SecurityManager, and EventBus.
"""

import pytest
from core.models import ModelManager, ModelStatus
from core.orchestration import AgentOrchestrator, SubAgentType, TaskStatus
from core.security import SecurityManager, SecurityLevel, PermissionState
from core.events import EventBus, EventType, AgentEvent


def test_model_manager_hardware_detection():
    mm = ModelManager()
    assert mm.hardware.cpu_count >= 1
    assert mm.hardware.ram_gb > 0
    recommended = mm.recommend_models()
    assert isinstance(recommended, list)


def test_agent_orchestrator_task_lifecycle():
    orchestrator = AgentOrchestrator()
    task = orchestrator.create_task(
        session_id="sess_test",
        title="Test Task",
        description="Verify orchestration",
        agent_type=SubAgentType.CODING,
    )
    assert task.task_id.startswith("task_")
    assert task.status == TaskStatus.QUEUED
    assert task.agent_type == SubAgentType.CODING


def test_security_manager_permissions():
    sec = SecurityManager(SecurityLevel.STANDARD)
    read_perm = sec.validate_action("file_tool", "read_files")
    assert read_perm == PermissionState.ALLOW

    write_perm = sec.validate_action("file_tool", "write_files")
    assert write_perm == PermissionState.ASK


def test_event_bus_pub_sub():
    bus = EventBus.get_instance()
    bus.clear()
    received = []

    def handler(evt: AgentEvent):
        received.append(evt)

    bus.subscribe(EventType.AGENT_STARTED, handler)
    bus.publish(AgentEvent(EventType.AGENT_STARTED, "sess_1", {"msg": "hello"}))

    assert len(received) == 1
    assert received[0].session_id == "sess_1"
    assert received[0].payload["msg"] == "hello"
