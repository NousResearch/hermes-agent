from __future__ import annotations

import pytest


def test_ecc_skills_are_represented_as_metadata_not_wholesale_content():
    from hermes_cli.capabilities.skills import ECC_SKILLS_INSPECTED, load_skill_registry

    registry = load_skill_registry()
    inspected = [skill for skill in registry.skills if skill.source == "ecc-inspected-metadata"]
    assert registry.ecc_commit == "34faa39bd3cd496a0aece0245f2b7e38b7923abc"
    assert len(inspected) == ECC_SKILLS_INSPECTED == 271
    assert all(skill.ecc_reference for skill in inspected)
    assert all(skill.status in {"review", "rejected"} for skill in inspected)
    assert not any("full prompt" in skill.description.lower() for skill in inspected)


def test_skill_metadata_has_required_fields_and_safe_defaults():
    from hermes_cli.capabilities.skills import HARD_FORBIDDEN_ACTIONS, load_skill_registry

    registry = load_skill_registry()
    assert registry.skills
    for skill in registry.skills:
        assert skill.name
        assert skill.category
        assert skill.description
        assert skill.source
        assert skill.risk in {"low", "medium", "high"}
        assert skill.status in {"rejected", "review", "approved", "active"}
        assert skill.activation_conditions
        assert isinstance(skill.required_tools, tuple)
        assert skill.forbidden_actions
        assert skill.context_budget_weight > 0
        assert set(HARD_FORBIDDEN_ACTIONS).issubset(set(skill.forbidden_actions))
    assert not any(skill.status == "active" for skill in registry.skills)


def test_design_quality_pack_contains_required_buidl_review_skills():
    from hermes_cli.capabilities.skills import load_skill_registry

    registry = load_skill_registry()
    pack = registry.pack("Design Quality Pack")
    assert pack is not None
    expected = {
        "visual-hierarchy-review",
        "landing-page-structure",
        "conversion-focused-ux",
        "saas-dashboard-ui",
        "responsive-design-qa",
        "accessibility-checks",
        "spacing-typography-layout-critique",
        "brand-consistency",
        "copy-clarity",
        "empty-loading-error-states",
        "preview-iframe-quality",
        "generated-website-quality-grading",
        "anti-template-fallback-detection",
        "prompt-domain-fit-evaluation",
    }
    assert expected <= set(pack.skill_names)
    assert "Product QA Agent" in pack.agent_roles
    assert "Code Reviewer Agent" in pack.agent_roles


def test_generated_website_verifier_gets_design_quality_without_known_prompt_tuning():
    from hermes_cli.capabilities.skills import load_skill_registry

    plan = load_skill_registry().activate_pack_for_goal(
        "generated website quality review for Buidl",
        agent_role="Verifier Agent",
        max_context_budget=14,
    )
    names = {skill.name for skill in plan.skills}
    assert plan.pack_name == "Design Quality Pack"
    assert {
        "prompt-domain-fit-evaluation",
        "generated-website-quality-grading",
        "responsive-design-qa",
        "accessibility-checks",
        "anti-template-fallback-detection",
        "preview-iframe-quality",
    } <= names
    assert not any("known prompt" in skill.description.lower() for skill in plan.skills)
    assert plan.total_context_budget <= 14


def test_skill_packs_activate_by_goal_type_not_globally():
    from hermes_cli.capabilities.skills import load_skill_registry

    registry = load_skill_registry()
    security = registry.activate_pack_for_goal("security secret scan", agent_role="Security Reviewer Agent")
    memory = registry.activate_pack_for_goal("lesson memory update", agent_role="Memory Curator")
    assert security.pack_name == "Security and Safety Pack"
    assert memory.pack_name == "Memory and Learning Pack"
    assert {skill.name for skill in security.skills} != {skill.name for skill in memory.skills}
    with pytest.raises(ValueError, match="entire Hermes Agent Harness skills library"):
        registry.refuse_global_activation()


def test_risky_skills_hooks_mcps_installers_and_secret_touching_remain_disabled():
    from hermes_cli.capabilities.skills import RISKY_SKILL_PATTERNS, load_skill_registry

    registry = load_skill_registry()
    risky = [skill for skill in registry.skills if skill.risk == "high" or skill.status == "rejected"]
    assert risky
    assert all(skill.status in {"review", "rejected"} for skill in risky)
    assert any(pattern in skill.forbidden_actions for skill in risky for pattern in RISKY_SKILL_PATTERNS)
    for pack in registry.packs:
        plan = registry.activate_pack_for_goal(pack.goal_types[0], agent_role=pack.agent_roles[0])
        assert not any(skill.risk == "high" or skill.status == "rejected" for skill in plan.skills)


def test_required_buidl_skill_packs_exist():
    from hermes_cli.capabilities.skills import load_skill_registry

    registry = load_skill_registry()
    packs = {pack.name: pack for pack in registry.packs}
    assert {
        "Design Quality Pack",
        "Security and Safety Pack",
        "Verification Pack",
        "Memory and Learning Pack",
        "Agentic Build Pack",
        "Buidl Skill Pack",
    } <= set(packs)
    universal = {
        "Design Quality Pack",
        "Security and Safety Pack",
        "Verification Pack",
        "Memory and Learning Pack",
        "Agentic Build Pack",
    }
    assert all(packs[name].project_scope == "universal" for name in universal)
    assert packs["Buidl Skill Pack"].project_scope == "buidl"


def test_buidl_is_specialized_pack_on_shared_harness():
    from hermes_cli.capabilities.skills import load_skill_registry

    registry = load_skill_registry()
    plan = registry.activate_pack_for_goal("buidl provider-safe route shell", agent_role="Builder Agent")
    assert plan.pack_name == "Buidl Skill Pack"
    names = {skill.name for skill in plan.skills}
    assert {"builder", "verifier", "hard-approval-gate-check"} <= names
    assert plan.total_context_budget <= 12