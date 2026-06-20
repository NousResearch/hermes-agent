"""Hermes Agent Harness skills capability layer.

The shared Hermes Agent Harness keeps ECC-style skills as metadata, not copied
instructions. Skill packs are selected by project, goal type and agent role so
the full library never enters one context. Buidl is the first specialized
project pack. Asvoria and future Niko projects can add their own packs without
turning the harness into a Buidl-only system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ECC_COMMIT_INSPECTED = "34faa39bd3cd496a0aece0245f2b7e38b7923abc"
ECC_SKILLS_INSPECTED = 271
SKILL_CONTEXT_BUDGET_LIMIT = 24

SkillRisk = Literal["low", "medium", "high"]
SkillStatus = Literal["rejected", "review", "approved", "active"]

HARD_FORBIDDEN_ACTIONS = (
    "real provider calls without explicit Niko approval",
    "real prompt execution without explicit Niko approval",
    "image generation without explicit Niko approval",
    "production deploy without explicit Niko approval",
    "DNS changes without explicit Niko approval",
    "DB migrations without explicit Niko approval",
    "billing, credits or payments changes without explicit Niko approval",
    "secret, credential, provider key or env value access",
    "worker enablement without explicit Niko approval",
    "merge to main without explicit Niko approval",
    "printing Basic Auth plaintext, Caddy hashes or env values",
)

RISKY_SKILL_PATTERNS = (
    "shell-executing hook",
    "installer",
    "mcp enablement",
    "secret-touching integration",
    "provider runtime connector",
    "memory persistence hook",
    "broad docker/root automation",
)


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    category: str
    description: str
    source: str
    risk: SkillRisk
    status: SkillStatus
    activation_conditions: tuple[str, ...]
    required_tools: tuple[str, ...]
    forbidden_actions: tuple[str, ...]
    context_budget_weight: int
    ecc_reference: str | None = None
    packs: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillPack:
    name: str
    goal_types: tuple[str, ...]
    agent_roles: tuple[str, ...]
    skill_names: tuple[str, ...]
    project_scope: str = "universal"
    max_context_budget: int = 12


@dataclass(frozen=True)
class SkillActivationPlan:
    pack_name: str
    goal_type: str
    agent_role: str
    skills: tuple[SkillMetadata, ...]
    total_context_budget: int
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True)
class SkillRegistry:
    skills: tuple[SkillMetadata, ...]
    packs: tuple[SkillPack, ...]
    ecc_commit: str = ECC_COMMIT_INSPECTED
    inspected_skill_count: int = ECC_SKILLS_INSPECTED
    context_budget_limit: int = SKILL_CONTEXT_BUDGET_LIMIT
    rejected_reason_counts: dict[str, int] = field(default_factory=dict)

    def by_name(self, name: str) -> SkillMetadata | None:
        lowered = name.lower()
        return next((skill for skill in self.skills if skill.name.lower() == lowered), None)

    def by_status(self, status: SkillStatus) -> tuple[SkillMetadata, ...]:
        return tuple(skill for skill in self.skills if skill.status == status)

    def pack(self, name: str) -> SkillPack | None:
        lowered = name.lower()
        return next((pack for pack in self.packs if pack.name.lower() == lowered), None)

    def activate_pack_for_goal(
        self,
        goal_type: str,
        *,
        agent_role: str = "Verifier Agent",
        max_context_budget: int | None = None,
    ) -> SkillActivationPlan:
        goal_lower = goal_type.lower()
        role_lower = agent_role.lower()
        matching_pack = next(
            (
                pack
                for pack in self.packs
                if any(token in goal_lower for token in pack.goal_types)
                and any(role_lower == role.lower() for role in pack.agent_roles)
            ),
            None,
        )
        if matching_pack is None:
            matching_pack = self.pack("Verification Pack") or self.packs[0]

        budget = min(max_context_budget or matching_pack.max_context_budget, self.context_budget_limit)
        selected: list[SkillMetadata] = []
        skipped: list[str] = []
        used = 0
        for skill_name in matching_pack.skill_names:
            skill = self.by_name(skill_name)
            if skill is None:
                skipped.append(f"missing:{skill_name}")
                continue
            if skill.status not in {"approved", "active"}:
                skipped.append(f"not-approved:{skill.name}")
                continue
            if skill.risk == "high":
                skipped.append(f"high-risk:{skill.name}")
                continue
            if used + skill.context_budget_weight > budget:
                skipped.append(f"budget:{skill.name}")
                continue
            selected.append(skill)
            used += skill.context_budget_weight
        return SkillActivationPlan(
            pack_name=matching_pack.name,
            goal_type=goal_type,
            agent_role=agent_role,
            skills=tuple(selected),
            total_context_budget=used,
            skipped=tuple(skipped),
        )

    def refuse_global_activation(self) -> None:
        raise ValueError("context bloat guard refused to load the entire Hermes Agent Harness skills library")


def _skill(
    name: str,
    category: str,
    description: str,
    *,
    source: str = "buidl-curated",
    risk: SkillRisk = "low",
    status: SkillStatus = "approved",
    activation_conditions: tuple[str, ...] = (),
    required_tools: tuple[str, ...] = (),
    forbidden_actions: tuple[str, ...] = HARD_FORBIDDEN_ACTIONS,
    context_budget_weight: int = 1,
    ecc_reference: str | None = None,
    packs: tuple[str, ...] = (),
) -> SkillMetadata:
    return SkillMetadata(
        name=name,
        category=category,
        description=description,
        source=source,
        risk=risk,
        status=status,
        activation_conditions=activation_conditions or (category, name),
        required_tools=required_tools,
        forbidden_actions=forbidden_actions,
        context_budget_weight=context_budget_weight,
        ecc_reference=ecc_reference,
        packs=packs,
    )


def _inspected_ecc_skill(index: int) -> SkillMetadata:
    categories = ("design", "security", "verification", "memory", "agentic-build", "general")
    category = categories[(index - 1) % len(categories)]
    risky = index in {17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255}
    status: SkillStatus = "rejected" if risky else "review"
    risk: SkillRisk = "high" if risky else "medium"
    description = (
        "Inspected ECC skill metadata placeholder. Instructions are not copied. "
        "Requires review before promotion into a Buidl pack."
    )
    if risky:
        description = "Rejected ECC skill metadata placeholder for risky shell, MCP, installer, provider or secret behavior."
    return _skill(
        name=f"ecc-inspected-skill-{index:03d}",
        category=category,
        description=description,
        source="ecc-inspected-metadata",
        risk=risk,
        status=status,
        activation_conditions=("manual review", category),
        required_tools=(),
        forbidden_actions=HARD_FORBIDDEN_ACTIONS + RISKY_SKILL_PATTERNS,
        context_budget_weight=2,
        ecc_reference=f"skills/inspected/{index:03d}",
        packs=(),
    )


def _curated_pack_skills() -> tuple[SkillMetadata, ...]:
    design_pack = "Design Quality Pack"
    security_pack = "Security and Safety Pack"
    verification_pack = "Verification Pack"
    memory_pack = "Memory and Learning Pack"
    agentic_pack = "Agentic Build Pack"
    return (
        _skill("visual-hierarchy-review", "design", "Review contrast, hierarchy, scan path and section emphasis.", packs=(design_pack,)),
        _skill("landing-page-structure", "design", "Check hero, proof, offer, sections, CTA sequence and conversion flow.", packs=(design_pack,)),
        _skill("conversion-focused-ux", "design", "Critique CTAs, friction, clarity, trust cues and conversion intent.", packs=(design_pack,)),
        _skill("saas-dashboard-ui", "design", "Review SaaS dashboard density, navigation, cards, tables and cockpit surfaces.", packs=(design_pack,)),
        _skill("responsive-design-qa", "design", "Check mobile, tablet and desktop states for overflow, spacing and usability.", required_tools=("browser",), packs=(design_pack,)),
        _skill("accessibility-checks", "design", "Check semantic basics, keyboard risk, contrast and alt or label gaps.", required_tools=("browser",), packs=(design_pack,)),
        _skill("spacing-typography-layout-critique", "design", "Review spacing scale, type hierarchy, alignment and layout rhythm.", packs=(design_pack,)),
        _skill("brand-consistency", "design", "Check tone, visual system and domain-appropriate brand consistency.", packs=(design_pack,)),
        _skill("copy-clarity", "design", "Check clear, specific, non-generic copy and visible user outcome.", packs=(design_pack,)),
        _skill("empty-loading-error-states", "design", "Check empty states, loading states and error states for product polish.", packs=(design_pack,)),
        _skill("preview-iframe-quality", "design", "Check iframe preview sizing, chrome, honest state and visible artifact quality.", required_tools=("browser",), packs=(design_pack,)),
        _skill("generated-website-quality-grading", "design", "Grade generated websites for premium feel, layout, CTA and prompt fit.", required_tools=("browser",), packs=(design_pack,)),
        _skill("anti-template-fallback-detection", "design", "Detect generic template output, Local Gym-like fallback and fake previews.", packs=(design_pack,)),
        _skill("prompt-domain-fit-evaluation", "design", "Evaluate whether the result matches the prompt domain without storing blind prompts.", packs=(design_pack,)),
        _skill("secret-scanning", "security", "Scan changed files for secret-shaped values before commit.", required_tools=("terminal",), packs=(security_pack,)),
        _skill("env-value-redaction", "security", "Ensure reports and logs redact env values and secret-shaped content.", packs=(security_pack,)),
        _skill("basic-auth-safety", "security", "Prevent Basic Auth plaintext printing and unsafe auth weakening.", packs=(security_pack,)),
        _skill("caddy-hash-non-printing", "security", "Prevent Caddy hash output or storage in logs, reports or notes.", packs=(security_pack,)),
        _skill("provider-key-safety", "security", "Keep provider keys, credentials and API key values out of agent context.", packs=(security_pack,)),
        _skill("hard-approval-gate-check", "security", "Block production, DNS, DB, billing, credits, payments, providers, prompts and image generation without approval.", packs=(security_pack,)),
        _skill("unsafe-sudo-docker-root-block", "security", "Reject unsafe sudo, broad Docker/root capability and arbitrary /tmp script execution.", risk="medium", status="review", packs=(security_pack,)),
        _skill("test-planning", "verification", "Select focused tests before full verification.", required_tools=("terminal",), packs=(verification_pack,)),
        _skill("focused-test-selection", "verification", "Run smallest meaningful tests for changed surfaces first.", required_tools=("terminal",), packs=(verification_pack,)),
        _skill("full-suite-gating", "verification", "Require full suite evidence before done when broad code changed.", required_tools=("terminal",), packs=(verification_pack,)),
        _skill("lint-typecheck-build-gating", "verification", "Gate completion on lint, typecheck and build where available.", required_tools=("terminal",), packs=(verification_pack,)),
        _skill("deploy-marker-verification", "verification", "Verify deploy markers only when deployment scope is explicitly approved.", status="review", required_tools=("terminal",), packs=(verification_pack,)),
        _skill("ci-workflow-monitoring", "verification", "Monitor PR checks and classify failures without asking Niko to operate.", required_tools=("terminal", "gh"), packs=(verification_pack,)),
        _skill("browser-acceptance-criteria", "verification", "Use browser evidence for visible acceptance criteria when relevant.", required_tools=("browser",), packs=(verification_pack,)),
        _skill("regression-detection", "verification", "Compare changed behavior against known blockers and route regressions.", packs=(verification_pack,)),
        _skill("no-builder-self-report-done", "verification", "Do not claim done from builder self-report alone. Require observed evidence.", packs=(verification_pack,)),
        _skill("session-summary-extraction", "memory", "Extract safe session summaries without storing secrets or blind prompts.", packs=(memory_pack,)),
        _skill("lesson-candidate-creation", "memory", "Create sanitized lesson candidates for review.", packs=(memory_pack,)),
        _skill("lesson-review-before-promotion", "memory", "Keep lessons in review before promotion to durable skills.", packs=(memory_pack,)),
        _skill("obsidian-full-read-before-write", "memory", "Read relevant Obsidian context before project note writes.", required_tools=("file",), packs=(memory_pack,)),
        _skill("blind-prompt-never-stored", "memory", "Never ask for, print, store or hardcode live blind prompts.", packs=(memory_pack,)),
        _skill("repeated-blocker-detection", "memory", "Detect recurring blockers from summaries before asking Niko again.", packs=(memory_pack,)),
        _skill("skill-promotion-rejection-logs", "memory", "Record why skill candidates are promoted or rejected.", packs=(memory_pack,)),
        _skill("planner", "agentic-build", "Decompose goals into cards with acceptance criteria and gates.", packs=(agentic_pack,)),
        _skill("architect", "agentic-build", "Check system boundaries, route hygiene and Buidl 2.0 chrome.", packs=(agentic_pack,)),
        _skill("builder", "agentic-build", "Implement scoped feature branch code changes.", required_tools=("terminal", "file"), packs=(agentic_pack,)),
        _skill("reviewer", "agentic-build", "Review diffs for correctness, UX, safety and maintainability.", required_tools=("terminal",), packs=(agentic_pack,)),
        _skill("verifier", "agentic-build", "Verify tests, builds, browser evidence and PR checks before done.", required_tools=("terminal",), packs=(agentic_pack,)),
        _skill("ops-staging", "agentic-build", "Respect staging wrapper and deployment gates without deploying by default.", status="review", packs=(agentic_pack,)),
        _skill("product-qa", "agentic-build", "Check product behavior, fake preview risk and user-facing quality.", required_tools=("browser",), packs=(agentic_pack, design_pack)),
        _skill("debugging", "agentic-build", "Use root-cause debugging before patching failures.", packs=(agentic_pack,)),
        _skill("tdd", "agentic-build", "Prefer tests before fixes for regression-prone logic.", required_tools=("terminal",), packs=(agentic_pack,)),
        _skill("build-error-resolver", "agentic-build", "Resolve lint, typecheck, build and CI errors with evidence.", required_tools=("terminal",), packs=(agentic_pack,)),
        _skill("pr-review", "agentic-build", "Prepare PR summaries, monitor checks and keep main merge gated.", required_tools=("terminal", "gh"), packs=(agentic_pack,)),
        _skill("goal-os-card-decomposition", "agentic-build", "Use Goal OS cards for durable plan, execute, verify and learn flow.", packs=(agentic_pack,)),
    )


def _packs() -> tuple[SkillPack, ...]:
    return (
        SkillPack(
            name="Design Quality Pack",
            goal_types=("buidl", "generated website", "website", "design", "preview", "product qa"),
            agent_roles=("Product QA Agent", "Code Reviewer Agent", "Reviewer Agent", "Verifier Agent"),
            skill_names=(
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
            ),
            max_context_budget=14,
        ),
        SkillPack(
            name="Security and Safety Pack",
            goal_types=("security", "safety", "secret", "deploy", "provider", "staging"),
            agent_roles=("Security Reviewer Agent", "Verifier Agent", "Ops/Staging Agent", "Code Reviewer Agent"),
            skill_names=(
                "secret-scanning",
                "env-value-redaction",
                "basic-auth-safety",
                "caddy-hash-non-printing",
                "provider-key-safety",
                "hard-approval-gate-check",
                "unsafe-sudo-docker-root-block",
            ),
            max_context_budget=8,
        ),
        SkillPack(
            name="Verification Pack",
            goal_types=("verify", "test", "ci", "build", "done", "generated website"),
            agent_roles=("Verifier Agent", "Test Runner Agent", "Build Error Resolver", "Code Reviewer Agent"),
            skill_names=(
                "test-planning",
                "focused-test-selection",
                "full-suite-gating",
                "lint-typecheck-build-gating",
                "deploy-marker-verification",
                "ci-workflow-monitoring",
                "browser-acceptance-criteria",
                "regression-detection",
                "no-builder-self-report-done",
            ),
            max_context_budget=10,
        ),
        SkillPack(
            name="Memory and Learning Pack",
            goal_types=("learn", "memory", "lesson", "obsidian", "summary"),
            agent_roles=("Memory Curator", "Clio Orchestrator", "Verifier Agent"),
            skill_names=(
                "session-summary-extraction",
                "lesson-candidate-creation",
                "lesson-review-before-promotion",
                "obsidian-full-read-before-write",
                "blind-prompt-never-stored",
                "repeated-blocker-detection",
                "skill-promotion-rejection-logs",
            ),
            max_context_budget=7,
        ),
        SkillPack(
            name="Agentic Build Pack",
            goal_types=("build", "agentic", "mvp", "goal os"),
            agent_roles=(
                "Clio Orchestrator",
                "Planner Agent",
                "Architect Agent",
                "Builder Agent",
                "Code Reviewer Agent",
                "Verifier Agent",
                "Product QA Agent",
                "Build Error Resolver",
            ),
            skill_names=(
                "planner",
                "architect",
                "builder",
                "reviewer",
                "verifier",
                "ops-staging",
                "product-qa",
                "debugging",
                "tdd",
                "build-error-resolver",
                "pr-review",
                "goal-os-card-decomposition",
            ),
            max_context_budget=12,
        ),
        SkillPack(
            name="Buidl Skill Pack",
            goal_types=("buidl", "buidl 2.0", "provider-safe route shell", "generated website"),
            agent_roles=(
                "Clio Orchestrator",
                "Planner Agent",
                "Architect Agent",
                "Builder Agent",
                "Reviewer Agent",
                "Verifier Agent",
                "Product QA Agent",
            ),
            skill_names=(
                "architect",
                "builder",
                "reviewer",
                "verifier",
                "product-qa",
                "preview-iframe-quality",
                "generated-website-quality-grading",
                "anti-template-fallback-detection",
                "prompt-domain-fit-evaluation",
                "hard-approval-gate-check",
            ),
            project_scope="buidl",
            max_context_budget=12,
        ),
    )


def load_skill_registry() -> SkillRegistry:
    curated = _curated_pack_skills()
    inspected = tuple(_inspected_ecc_skill(index) for index in range(1, ECC_SKILLS_INSPECTED + 1))
    return SkillRegistry(
        skills=curated + inspected,
        packs=_packs(),
        rejected_reason_counts={
            "shell_executing_hooks": 4,
            "mcp_configs": 1,
            "installers_or_bootstrap": 6,
            "provider_or_secret_touching": 4,
        },
    )


__all__ = [
    "ECC_COMMIT_INSPECTED",
    "ECC_SKILLS_INSPECTED",
    "HARD_FORBIDDEN_ACTIONS",
    "RISKY_SKILL_PATTERNS",
    "SKILL_CONTEXT_BUDGET_LIMIT",
    "SkillActivationPlan",
    "SkillMetadata",
    "SkillPack",
    "SkillRegistry",
    "load_skill_registry",
]
