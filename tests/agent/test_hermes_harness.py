import pytest

from agent import harness_control_plane, system_prompt
from agent.harness import ControlPlaneHarness, HermesHarness, SystemPromptHarness


class _Agent:
    name = "agent"


def test_hermes_harness_groups_system_prompt_and_control_plane(monkeypatch):
    agent = _Agent()

    monkeypatch.setattr(
        system_prompt,
        "build_system_prompt_parts",
        lambda actual, system_message=None: {
            "stable": actual.name,
            "context": system_message or "",
            "volatile": "",
        },
    )
    monkeypatch.setattr(
        system_prompt,
        "build_system_prompt",
        lambda actual, system_message=None: f"{actual.name}:{system_message}",
    )
    monkeypatch.setattr(
        system_prompt,
        "format_tools_for_system_message",
        lambda actual: f"tools:{actual.name}",
    )

    harness = HermesHarness.for_agent(agent)

    assert isinstance(harness.system_prompt, SystemPromptHarness)
    assert isinstance(harness.control_plane, ControlPlaneHarness)
    assert harness.system_prompt.build_parts("context") == {
        "stable": "agent",
        "context": "context",
        "volatile": "",
    }
    assert harness.system_prompt.build("context") == "agent:context"
    assert harness.system_prompt.format_tools() == "tools:agent"


def test_system_prompt_harness_requires_agent():
    with pytest.raises(ValueError, match="requires an agent"):
        HermesHarness().system_prompt.build()


def test_system_prompt_harness_invalidates_agent_cache(monkeypatch):
    agent = _Agent()
    called = []

    def fake_invalidate(actual):
        called.append(actual)

    monkeypatch.setattr(system_prompt, "invalidate_system_prompt", fake_invalidate)

    HermesHarness.for_agent(agent).system_prompt.invalidate()

    assert called == [agent]


def test_control_plane_harness_keeps_default_call_shape(monkeypatch):
    calls = []

    def fake_run_core_harness(case_ids=None):
        calls.append({"case_ids": case_ids})
        return {"status": "passed", "case_ids": case_ids}

    monkeypatch.setattr(
        harness_control_plane,
        "run_core_harness",
        fake_run_core_harness,
    )

    result = HermesHarness().control_plane.run_core(case_ids=["harness-event-safety"])

    assert result == {"status": "passed", "case_ids": ["harness-event-safety"]}
    assert calls == [{"case_ids": ["harness-event-safety"]}]


def test_control_plane_harness_passes_profile_when_set(monkeypatch):
    monkeypatch.setattr(
        harness_control_plane,
        "core_harness_status",
        lambda profile=None: {"profile": profile},
    )

    status = ControlPlaneHarness(profile="founder").core_status()

    assert status == {"profile": "founder"}
