from importlib import import_module
from types import SimpleNamespace

import pytest

import agent.archetypes as archetypes
from agent.prompt_builder import build_wave1_overlay_prompt_from_normalized, normalize_wave1_overlay_inputs

if not hasattr(archetypes, "resolve_specialist_mapping"):
    archetypes.resolve_specialist_mapping = lambda value: None
if not hasattr(archetypes, "resolve_specialist_defaults"):
    archetypes.resolve_specialist_defaults = lambda value: {}

run_agent = import_module("run_agent")
AIAgent = run_agent.AIAgent


def _sample_task_contract() -> dict:
    return {
        "task": "Implement the delegated change",
        "expected_outcome": "A passing implementation with verification evidence",
        "required_skills": ["python", "testing"],
        "required_tools": ["read_file", "patch", "terminal"],
        "must_do": ["inspect repo patterns before editing"],
        "must_not_do": {"forbidden_files": ["tools/delegate_tool.py"]},
        "context": {"repo": "/root/.hermes/hermes-agent"},
    }


def _make_bare_agent(delegate_depth: int = 0) -> AIAgent:
    agent = AIAgent.__new__(AIAgent)
    agent._delegate_depth = delegate_depth
    agent._delegate_resolution = {}
    return agent


def _sample_named_workflow() -> dict:
    return {
        "schema": "hermes/named-workflow",
        "schema_version": "1.0",
        "workflow_name": "planner",
        "mode": "plan",
        "objective": "Plan the implementation before execution.",
        "plan": ["inspect", "plan", "handoff"],
        "acceptance": ["machine-readable artifact present"],
        "taxonomy": {
            "named_workflow": "planner",
            "workflow": "planner",
            "specialist": "planner",
            "archetype": "generalist",
            "route_category": "unspecified_low",
            "runtime_mode": "default",
            "delegation_profile": "general",
        },
        "execution_task_contract": None,
        "consumption": {"downstream_role": "deep_worker"},
    }


def test_parent_runtime_activation_uses_canonical_overlay_contract(monkeypatch):
    contract = _sample_task_contract()
    monkeypatch.setattr(
        run_agent,
        "preclassify_intent",
        lambda _message: SimpleNamespace(
            inferred_specialist=None,
            inferred_archetype="implementer",
            inferred_category="deep",
            inferred_route_category="deep",
            inferred_runtime_mode="execution_supervisor",
            inferred_delegation_profile="verification",
            activation_reason="parent parity test",
            inference_source="wave2_intent_preclassifier",
            task_contract=contract,
        ),
    )

    agent = _make_bare_agent(delegate_depth=0)
    state = agent._resolve_runtime_activation_state("verify parity")
    normalized = normalize_wave1_overlay_inputs(
        archetype_name="implementer",
        route_category="deep",
        delegation_profile="verification",
        runtime_mode="execution_supervisor",
        skills=archetypes.resolve_archetype("implementer").default_skills,
        task_contract=contract,
    )

    assert state["archetype"] == normalized["archetype"]
    assert state["route_category"] == normalized["route_category"]
    assert state["delegation_profile"] == normalized["delegation_profile"]
    assert state["runtime_mode"] == normalized["runtime_mode"]
    assert state["task_contract"] == normalized["task_contract"]
    assert state["wave1_overlay_prompt"] == build_wave1_overlay_prompt_from_normalized(normalized)
    assert state["wave1_overlay_prompt"] in state["activation_note"]
    assert state["activation_note"].startswith("<wave2-runtime-activation>")
    assert state["activation_note"].endswith("</wave2-runtime-activation>")


def test_delegated_child_preserves_passed_through_wave1_state_without_reset():
    contract = _sample_task_contract()
    agent = _make_bare_agent(delegate_depth=1)
    agent._delegate_resolution = {
        "archetype": "implementer",
        "category": "deep",
        "route_category": "deep",
        "route_category_definition": {"name": "deep", "summary": "Deep implementation lane", "intensity": "high"},
        "delegation_profile": "verification",
        "runtime_mode": "execution_supervisor",
        "runtime_mode_definition": {
            "name": "execution_supervisor",
            "description": "Supervise delegated execution and reconcile results.",
            "operating_posture": "oversight_and_coordination",
            "kind": "runtime_mode",
        },
        "skills": ["testing", "python"],
        "task_contract": contract,
        "orchestration_hints": {"permission_preset": "inherit", "fallback_policy": "legacy_default_mapping"},
    }

    state = agent._resolve_runtime_activation_state("ignored in delegated child")
    normalized = normalize_wave1_overlay_inputs(
        archetype_name=agent._delegate_resolution["archetype"],
        route_category=agent._delegate_resolution["route_category_definition"],
        delegation_profile=agent._delegate_resolution["delegation_profile"],
        runtime_mode=agent._delegate_resolution["runtime_mode_definition"],
        skills=agent._delegate_resolution["skills"],
        task_contract=agent._delegate_resolution["task_contract"],
        orchestration_hints=agent._delegate_resolution["orchestration_hints"],
    )

    assert state["inference_source"] == "delegate_passthrough"
    assert state["archetype"] == normalized["archetype"]
    assert state["category"] == normalized["category"]
    assert state["route_category"] == normalized["route_category"]
    assert state["delegation_profile"] == normalized["delegation_profile"]
    assert state["runtime_mode"] == normalized["runtime_mode"]
    assert state["task_contract"] == normalized["task_contract"]
    assert state["wave1_overlay_prompt"] == build_wave1_overlay_prompt_from_normalized(normalized)
    assert state["wave1_overlay_prompt"] in state["activation_note"]
    assert "category: deep" in state["activation_note"]
    assert "route_category: deep" in state["activation_note"]
    assert "delegation_profile: verification" in state["activation_note"]
    assert "## Archetype\nname: implementer" in state["wave1_overlay_prompt"]
    assert "## Archetype\nname: generalist" not in state["wave1_overlay_prompt"]


def test_delegated_child_preserves_literal_category_when_distinct_from_route_and_profile():
    agent = _make_bare_agent(delegate_depth=1)
    agent._delegate_resolution = {
        "specialist": "multimodal_specialist",
        "archetype": "researcher",
        "category": "visual-engineering",
        "route_category": "visual",
        "delegation_profile": "research",
        "runtime_mode": "default",
        "skills": ["vision", "research"],
    }

    state = agent._resolve_runtime_activation_state("ignored in delegated child")
    snapshot = agent._build_runtime_activation_snapshot_entry(state)

    assert state["specialist"] == "multimodal_specialist"
    assert state["archetype"] == "researcher"
    assert state["category"] == "visual-engineering"
    assert state["route_category"] == "visual"
    assert state["delegation_profile"] == "research"
    assert state["runtime_mode"] == "default"
    assert state["inference_source"] == "delegate_passthrough"
    assert "specialist=multimodal_specialist" in state["activation_reason"]
    assert "archetype=researcher" in state["activation_reason"]
    assert "category=visual-engineering" in state["activation_reason"]
    assert "route_category=visual" in state["activation_reason"]
    assert "delegation_profile=research" in state["activation_reason"]
    assert "runtime_mode=default" in state["activation_reason"]
    assert "category: visual-engineering" in state["activation_note"]
    assert "route_category: visual" in state["activation_note"]
    assert "delegation_profile: research" in state["activation_note"]
    assert "## Category\nname: visual-engineering" in state["wave1_overlay_prompt"]
    assert "## Route Category\nname: visual" in state["wave1_overlay_prompt"]
    assert "## Delegation Profile\nname: research" in state["wave1_overlay_prompt"]
    assert snapshot["category"] == "visual-engineering"
    assert snapshot["route_category"] == "visual"
    assert snapshot["delegation_profile"] == "research"


def test_delegated_child_preserves_named_agent_identity_in_runtime_state_and_snapshot():
    agent = _make_bare_agent(delegate_depth=1)
    agent._delegate_resolution = {
        "agent": "oracle",
        "named_agent": "oracle",
        "specialist": "consultant",
        "archetype": "researcher",
        "route_category": "deep",
        "delegation_profile": "research",
        "runtime_mode": "default",
    }

    state = agent._resolve_runtime_activation_state("ignored in delegated child")
    snapshot = agent._build_runtime_activation_snapshot_entry(state)

    assert state["named_agent"] == "oracle"
    assert state["category"] == "deep"
    assert "named_agent: oracle" in state["activation_note"]
    assert "category: deep" in state["activation_note"]
    assert "route_category: deep" in state["activation_note"]
    assert snapshot["named_agent"] == "oracle"
    assert snapshot["category"] == "deep"
    assert snapshot["activation_identity"] == "oracle"


def test_parent_runtime_activation_proves_qa_guard_quick_overlay_end_to_end():
    agent = _make_bare_agent(delegate_depth=0)
    state = agent._resolve_runtime_activation_state("qa regression validate the fix")

    assert state["specialist"] == "qa_guard"
    assert state["archetype"] == "verifier"
    assert state["category"] == "quick"
    assert state["route_category"] == "quick"
    assert state["delegation_profile"] == "verification"
    assert state["runtime_mode"] == "default"
    assert state["inference_source"] == "wave2_intent_preclassifier"

    assert "keyword-derived specialist: qa_guard" in state["activation_reason"]
    assert "archetype=verifier" in state["activation_reason"]
    assert "specialist=qa_guard" in state["activation_reason"]
    assert "category=quick" in state["activation_reason"]
    assert "route_category=quick" in state["activation_reason"]
    assert "delegation_profile=verification" in state["activation_reason"]
    assert "runtime_mode=default" in state["activation_reason"]

    assert state["activation_note"].startswith("<wave2-runtime-activation>")
    assert state["activation_note"].endswith("</wave2-runtime-activation>")
    assert "category: quick" in state["activation_note"]
    assert "route_category: quick" in state["activation_note"]
    assert "delegation_profile: verification" in state["activation_note"]
    assert "runtime_mode: default" in state["activation_note"]
    assert state["wave1_overlay_prompt"] in state["activation_note"]

    assert "## Archetype\nname: verifier" in state["wave1_overlay_prompt"]
    assert "default_route_category: quick" in state["wave1_overlay_prompt"]
    assert "default_delegation_profile: verification" in state["wave1_overlay_prompt"]
    assert "## Category\nname: quick" in state["wave1_overlay_prompt"]
    assert "fallback_semantics: inherits_mapped_route_category" in state["wave1_overlay_prompt"]
    assert "## Route Category\nname: quick" in state["wave1_overlay_prompt"]
    assert "summary: Fast-path routing lane for lighter-weight execution." in state["wave1_overlay_prompt"]
    assert "intensity: low" in state["wave1_overlay_prompt"]
    assert "## Delegation Profile\nname: verification" in state["wave1_overlay_prompt"]
    assert "## Runtime Mode\nname: default" in state["wave1_overlay_prompt"]
    assert "operating_posture: balanced_general_operation" in state["wave1_overlay_prompt"]
    assert "## Skills" in state["wave1_overlay_prompt"]
    assert "- verification" in state["wave1_overlay_prompt"]
    assert "- testing" in state["wave1_overlay_prompt"]
    assert "- review" in state["wave1_overlay_prompt"]

    assert "## Route Category\nname: ultrabrain" not in state["wave1_overlay_prompt"]


def test_parent_runtime_activation_proves_builder_deep_default_overlay_end_to_end():
    agent = _make_bare_agent(delegate_depth=0)
    state = agent._resolve_runtime_activation_state("implement the delegated change")

    assert state["specialist"] == "builder"
    assert state["archetype"] == "implementer"
    assert state["category"] == "deep"
    assert state["route_category"] == "deep"
    assert state["delegation_profile"] == "implementation"
    assert state["runtime_mode"] == "default"
    assert state["inference_source"] == "wave2_intent_preclassifier"

    assert "keyword-derived specialist: builder" in state["activation_reason"]
    assert "archetype=implementer" in state["activation_reason"]
    assert "specialist=builder" in state["activation_reason"]
    assert "category=deep" in state["activation_reason"]
    assert "route_category=deep" in state["activation_reason"]
    assert "delegation_profile=implementation" in state["activation_reason"]
    assert "runtime_mode=default" in state["activation_reason"]

    assert state["activation_note"].startswith("<wave2-runtime-activation>")
    assert state["activation_note"].endswith("</wave2-runtime-activation>")
    assert "category: deep" in state["activation_note"]
    assert "route_category: deep" in state["activation_note"]
    assert "delegation_profile: implementation" in state["activation_note"]
    assert "runtime_mode: default" in state["activation_note"]
    assert state["wave1_overlay_prompt"] in state["activation_note"]

    assert "## Archetype\nname: implementer" in state["wave1_overlay_prompt"]
    assert "default_route_category: deep" in state["wave1_overlay_prompt"]
    assert "default_delegation_profile: implementation" in state["wave1_overlay_prompt"]
    assert "## Category\nname: deep" in state["wave1_overlay_prompt"]
    assert "## Route Category\nname: deep" in state["wave1_overlay_prompt"]
    assert "## Delegation Profile\nname: implementation" in state["wave1_overlay_prompt"]
    assert "## Runtime Mode\nname: default" in state["wave1_overlay_prompt"]


def test_parent_runtime_activation_proves_mixed_visual_research_prompt_keeps_literal_category_distinct():
    agent = _make_bare_agent(delegate_depth=0)
    agent.valid_tool_names = {"vision_analyze"}
    state = agent._resolve_runtime_activation_state(
        "Inspect this PDF diagram, investigate the architecture, and gather evidence for the visual flow."
    )

    assert state["specialist"] == "multimodal_specialist"
    assert state["archetype"] == "researcher"
    assert state["category"] == "visual-engineering"
    assert state["route_category"] == "visual"
    assert state["delegation_profile"] == "research"
    assert state["runtime_mode"] == "default"
    assert state["inference_source"] == "wave2_intent_preclassifier"

    assert "keyword-derived specialist: multimodal_specialist" in state["activation_reason"]
    assert "archetype=researcher" in state["activation_reason"]
    assert "specialist=multimodal_specialist" in state["activation_reason"]
    assert "category=visual-engineering" in state["activation_reason"]
    assert "route_category=visual" in state["activation_reason"]
    assert "delegation_profile=research" in state["activation_reason"]
    assert "runtime_mode=default" in state["activation_reason"]

    assert "category: visual-engineering" in state["activation_note"]
    assert "route_category: visual" in state["activation_note"]
    assert "delegation_profile: research" in state["activation_note"]
    assert "runtime_mode: default" in state["activation_note"]
    assert "## Category\nname: visual-engineering" in state["wave1_overlay_prompt"]
    assert "## Route Category\nname: visual" in state["wave1_overlay_prompt"]
    assert "## Delegation Profile\nname: research" in state["wave1_overlay_prompt"]
    assert "## Runtime Mode\nname: default" in state["wave1_overlay_prompt"]


def test_parent_runtime_activation_explicit_overrides_keep_activation_reason_honest():
    agent = _make_bare_agent(delegate_depth=0)
    agent.enabled_toolsets = ["file"]
    agent.disabled_toolsets = []

    state = agent._resolve_runtime_activation_state(
        {
            "message": "do something",
            "category": "visual-engineering",
            "route_category": "deep",
            "runtime_mode": "execution_supervisor",
        }
    )

    assert state["category"] == "visual-engineering"
    assert state["route_category"] == "deep"
    assert state["delegation_profile"] == "general"
    assert state["runtime_mode"] == "execution_supervisor"
    assert "category=visual-engineering" in state["activation_reason"]
    assert "route_category=deep" in state["activation_reason"]
    assert "delegation_profile=general" in state["activation_reason"]
    assert "runtime_mode=execution_supervisor" in state["activation_reason"]
    assert "category=unspecified-low" not in state["activation_reason"]
    assert "route_category=unspecified_low" not in state["activation_reason"]
    assert "runtime_mode=default" not in state["activation_reason"]


def test_parent_runtime_activation_baseline_reason_still_reports_final_resolved_state():
    agent = _make_bare_agent(delegate_depth=0)

    state = agent._resolve_runtime_activation_state("implement the delegated change")

    assert "keyword-derived specialist: builder" in state["activation_reason"]
    assert "specialist=builder" in state["activation_reason"]
    assert "archetype=implementer" in state["activation_reason"]
    assert "category=deep" in state["activation_reason"]
    assert "route_category=deep" in state["activation_reason"]
    assert "delegation_profile=implementation" in state["activation_reason"]
    assert "runtime_mode=default" in state["activation_reason"]


def test_named_workflow_alone_triggers_runtime_activation_injection():
    named_workflow = _sample_named_workflow()
    state = {
        "specialist": None,
        "archetype": "generalist",
        "route_category": "unspecified_low",
        "delegation_profile": "general",
        "runtime_mode": "default",
        "task_contract": None,
        "named_workflow": named_workflow,
        "wave1_overlay_prompt": "",
        "activation_reason": "named workflow only",
        "inference_source": "test",
    }
    agent = _make_bare_agent(delegate_depth=0)

    assert agent._should_apply_runtime_activation_state(state) is True

    activation_note = agent._build_runtime_activation_note(state)
    assert activation_note.startswith("<wave2-runtime-activation>")
    assert "<named-workflow>" in activation_note
    assert '"workflow_name": "planner"' in activation_note
    assert activation_note.endswith("</wave2-runtime-activation>")


def test_delegated_child_ignores_invalid_named_workflow_passthrough():
    contract = _sample_task_contract()
    agent = _make_bare_agent(delegate_depth=1)
    agent._delegate_resolution = {
        "archetype": "implementer",
        "route_category": "deep",
        "delegation_profile": "implementation",
        "runtime_mode": "default",
        "task_contract": contract,
        "named_workflow": {"workflow_name": "unknown-workflow", "mode": "execute"},
    }

    state = agent._resolve_runtime_activation_state("ignored in delegated child")

    assert state["named_workflow"] is None
    assert "<named-workflow>" not in state["activation_note"]


def test_parent_runtime_activation_mixed_review_and_bug_prompt_keeps_keyword_collision_behavior_explicit():
    agent = _make_bare_agent(delegate_depth=0)
    state = agent._resolve_runtime_activation_state(
        "Review this patch, reproduce the bug, and call out regressions with verification evidence."
    )

    assert state["specialist"] == "bug_hunter"
    assert state["archetype"] == "implementer"
    assert state["route_category"] == "deep"
    assert state["delegation_profile"] == "implementation"
    assert state["runtime_mode"] == "default"
    assert state["inference_source"] == "wave2_intent_preclassifier"
    assert "keyword-derived specialist: bug_hunter" in state["activation_reason"]
    assert "specialist=bug_hunter" in state["activation_reason"]
    assert "specialist=code_reviewer" not in state["activation_reason"]
    assert "specialist: bug_hunter" in state["activation_note"]
    assert "## Archetype\nname: implementer" in state["wave1_overlay_prompt"]
    assert "## Route Category\nname: deep" in state["wave1_overlay_prompt"]
    assert "## Delegation Profile\nname: implementation" in state["wave1_overlay_prompt"]


@pytest.mark.parametrize(
    ("prompt", "specialist", "archetype", "archetype_default_route_category", "route_category", "delegation_profile"),
    [
        (
            "Review this patch, verify the risky changes, and call out regressions.",
            "code_reviewer",
            "verifier",
            "quick",
            "quick",
            "verification",
        ),
        (
            "qa regression validate the fix",
            "qa_guard",
            "verifier",
            "quick",
            "quick",
            "verification",
        ),
        (
            "Plan the rollout, decompose the work, and sequence the execution plan.",
            "planner",
            "generalist",
            "unspecified_low",
            "deep",
            "general",
        ),
        (
            "Reproduce the bug, trace the failure, and identify the root cause.",
            "bug_hunter",
            "implementer",
            "deep",
            "deep",
            "implementation",
        ),
        (
            "Analyze the API behavior, compare sources, and synthesize the findings.",
            "analyst",
            "researcher",
            "deep",
            "deep",
            "research",
        ),
        (
            "Investigate the architecture, gather evidence, and triage the issue.",
            "investigator",
            "researcher",
            "deep",
            "deep",
            "research",
        ),
    ],
    ids=["code_reviewer", "qa_guard", "planner", "bug_hunter", "analyst", "investigator"],
)
def test_parent_runtime_activation_specialist_overlay_matrix_preserves_explicit_taxonomy(
    prompt: str,
    specialist: str,
    archetype: str,
    archetype_default_route_category: str,
    route_category: str,
    delegation_profile: str,
):
    agent = _make_bare_agent(delegate_depth=0)
    state = agent._resolve_runtime_activation_state(prompt)

    assert state["specialist"] == specialist
    assert state["archetype"] == archetype
    assert state["route_category"] == route_category
    assert state["delegation_profile"] == delegation_profile
    assert state["runtime_mode"] == "default"
    assert state["inference_source"] == "wave2_intent_preclassifier"

    assert f"keyword-derived specialist: {specialist}" in state["activation_reason"]
    assert f"archetype={archetype}" in state["activation_reason"]
    assert f"specialist={specialist}" in state["activation_reason"]
    assert f"route_category={route_category}" in state["activation_reason"]
    assert f"delegation_profile={delegation_profile}" in state["activation_reason"]
    assert "runtime_mode=default" in state["activation_reason"]

    assert state["activation_note"].startswith("<wave2-runtime-activation>")
    assert state["activation_note"].endswith("</wave2-runtime-activation>")
    assert f"specialist: {specialist}" in state["activation_note"]
    assert state["wave1_overlay_prompt"] in state["activation_note"]

    assert f"## Archetype\nname: {archetype}" in state["wave1_overlay_prompt"]
    assert f"default_route_category: {archetype_default_route_category}" in state["wave1_overlay_prompt"]
    assert f"default_delegation_profile: {delegation_profile}" in state["wave1_overlay_prompt"]
    assert f"## Route Category\nname: {route_category}" in state["wave1_overlay_prompt"]
    assert f"## Delegation Profile\nname: {delegation_profile}" in state["wave1_overlay_prompt"]
    assert "## Runtime Mode\nname: default" in state["wave1_overlay_prompt"]
    assert "operating_posture: balanced_general_operation" in state["wave1_overlay_prompt"]
    assert "## Skills" in state["wave1_overlay_prompt"]
