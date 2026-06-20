"""Curated Hermes Agent Harness capability registry.

This package records a small, safe Hermes-native harness inspired by the public
ECC repository audit. The general layer is reusable across Buidl, Asvoria and
future Niko projects. Buidl is represented as the first specialized skill pack.
It is metadata and guard logic only. It does not install ECC, activate
third-party hooks, enable MCPs, run providers, run prompts, or copy ECC files
wholesale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .skills import ECC_SKILLS_INSPECTED, load_skill_registry

CapabilityType = Literal["agent", "skill", "command", "hook", "rule", "memory", "scanner"]
CapabilityStatus = Literal["disabled", "review", "approved", "active"]
CapabilityRisk = Literal["low", "medium", "high"]
CapabilitySource = Literal["buidl-native", "ecc-inspired", "custom"]

ECC_COMMIT_INSPECTED = "34faa39bd3cd496a0aece0245f2b7e38b7923abc"
CONTEXT_BLOAT_LIMIT = 64


@dataclass(frozen=True)
class Capability:
    name: str
    type: CapabilityType
    description: str
    status: CapabilityStatus = "disabled"
    risk: CapabilityRisk = "low"
    allowed_repos: tuple[str, ...] = ("/home/niko/.hermes/hermes-agent", "/home/niko/projects/buidl.ai-suite")
    required_tools: tuple[str, ...] = ()
    approval_required: tuple[str, ...] = ()
    source: CapabilitySource = "buidl-native"
    ecc_reference: str | None = None
    wholesale_copied: bool = False


@dataclass(frozen=True)
class CapabilityRegistry:
    capabilities: tuple[Capability, ...]
    ecc_commit: str = ECC_COMMIT_INSPECTED
    context_bloat_limit: int = CONTEXT_BLOAT_LIMIT
    audit_inventory: dict[str, Any] = field(default_factory=dict)

    def by_type(self, capability_type: CapabilityType) -> tuple[Capability, ...]:
        return tuple(cap for cap in self.capabilities if cap.type == capability_type)

    def by_name(self, name: str) -> Capability | None:
        lowered = name.lower()
        return next((cap for cap in self.capabilities if cap.name.lower() == lowered), None)

    def select_context_bundle(self, *, limit: int = 16, include_all: bool = False) -> tuple[Capability, ...]:
        if include_all or limit > self.context_bloat_limit:
            raise ValueError("context bloat guard refused to load every Buidl harness capability")
        selected = [cap for cap in self.capabilities if cap.status in {"approved", "active"}]
        return tuple(selected[: max(0, limit)])


def load_ecc_audit_inventory() -> dict[str, Any]:
    return {
        "commit": ECC_COMMIT_INSPECTED,
        "agents_available": 67,
        "skills_available": ECC_SKILLS_INSPECTED,
        "skills_metadata_represented": ECC_SKILLS_INSPECTED,
        "skills_metadata_policy": "metadata only, no wholesale skill instructions copied or auto-activated",
        "commands_available": 92,
        "hooks_available": 4,
        "rules_available": 114,
        "mcp_configs_available": 1,
        "security_risks": [
            "install scripts can modify local agent configuration",
            "hook dispatchers can run shell commands before or after tool use",
            "MCP configs may grant external tool access if enabled blindly",
            "large skill and rule corpus creates context bloat and conflicting instructions",
            "memory persistence hooks can store sensitive content if not filtered",
            "provider or tool integrations may add network calls outside Buidl gates",
        ],
        "shell_executing_hooks": [
            "hooks/hooks.json",
            "hooks/memory-persistence/hooks.json",
            "scripts hook dispatchers with exec or spawn patterns",
        ],
        "context_bloat_risks": {"agents": 67, "skills": 271, "commands": 92, "rules": 114},
        "recommended_for_buidl": [
            "agent role taxonomy",
            "command lifecycle pattern",
            "hook review-mode pattern",
            "verification-loop skill idea",
            "context-budget guard idea",
            "security-scan command concept",
            "checkpoint and learn command concepts",
        ],
        "rejected_for_buidl": [
            "wholesale install",
            "third-party hooks activation",
            "MCP enablement",
            "memory persistence hooks without sanitizer",
            "install scripts",
            "provider integrations",
            "full rules corpus auto-load",
            "language-specific reviewers unrelated to Buidl",
        ],
        "architecture_layers": {
            "general_layer": "Hermes Agent Harness",
            "specialized_layer": "Buidl Skill Pack",
            "future_specialized_layer": "Asvoria Skill Pack",
        },
        "selected_capabilities": [
            "curated Buidl specialist agent metadata",
            "durable goal/card commands",
            "review-mode safety hooks",
            "sanitized lesson candidate flow",
            "context bloat guard",
            "Buidl skill packs selected by goal type and agent role",
        ],
        "skill_packs": [
            "Security and Safety Pack",
            "Verification Pack",
            "Memory and Learning Pack",
            "Agentic Build Pack",
            "Design Quality Pack",
            "Buidl Skill Pack",
        ],
    }


def _agent(name: str, description: str, *, ecc_reference: str | None = None) -> Capability:
    return Capability(
        name=name,
        type="agent",
        description=description + " Config and prompt definition only, no separate LLM invocation in v1.",
        status="approved",
        risk="low",
        source="ecc-inspired" if ecc_reference else "buidl-native",
        ecc_reference=ecc_reference,
    )


def _command(name: str, description: str) -> Capability:
    return Capability(
        name=name,
        type="command",
        description=description,
        status="approved",
        risk="low",
        source="buidl-native",
    )


def _hook(name: str, description: str, *, risk: CapabilityRisk = "medium") -> Capability:
    return Capability(
        name=name,
        type="hook",
        description=description,
        status="review",
        risk=risk,
        source="buidl-native",
        approval_required=("activate hook",),
    )


def _skill(name: str, description: str, *, ecc_reference: str | None = None) -> Capability:
    return Capability(
        name=name,
        type="skill",
        description=description,
        status="approved",
        risk="low",
        source="ecc-inspired" if ecc_reference else "buidl-native",
        ecc_reference=ecc_reference,
    )


def _rule(name: str, description: str) -> Capability:
    return Capability(name=name, type="rule", description=description, status="approved", risk="low")


def _memory(name: str, description: str) -> Capability:
    return Capability(name=name, type="memory", description=description, status="review", risk="medium")


def _scanner(name: str, description: str) -> Capability:
    return Capability(name=name, type="scanner", description=description, status="approved", risk="low", required_tools=("terminal",))


def load_registry() -> CapabilityRegistry:
    capabilities = (
        _agent("Clio Orchestrator", "Coordinates Buidl MVP execution, cards, blockers, review and reporting."),
        _agent("Planner Agent", "Turns goals into safe scoped plans and acceptance criteria.", ecc_reference="agents/planner.md"),
        _agent("Architect Agent", "Checks architecture direction, route hygiene and Buidl 2.0 chrome.", ecc_reference="agents/architect.md"),
        _agent("Builder Agent", "Implements scoped repo changes on feature branches."),
        _agent("Code Reviewer Agent", "Reviews diffs for quality, maintainability and repo fit.", ecc_reference="agents/code-reviewer.md"),
        _agent("Security Reviewer Agent", "Reviews secrets, gates, provider safety and sensitive output.", ecc_reference="agents/security-reviewer.md"),
        _agent("Test Runner Agent", "Runs focused and full tests and records evidence.", ecc_reference="agents/e2e-runner.md"),
        _agent("Build Error Resolver", "Resolves lint, typecheck, build and CI failures.", ecc_reference="agents/build-error-resolver.md"),
        _agent("Ops/Staging Agent", "Checks deployment boundaries and staging wrapper requirements without deploying."),
        _agent("Product QA Agent", "Checks MVP behavior, route surfaces and fake preview regressions."),
        _agent("Memory Curator", "Reviews lesson candidates before skill promotion."),
        _agent("Verifier Agent", "Verifies final state with commands and evidence before done."),
        _skill("verification-loop", "Use evidence loops before marking Buidl work complete.", ecc_reference="skills/verification-loop/SKILL.md"),
        _skill("context-budget", "Keep harness context bounded and avoid loading everything.", ecc_reference="skills/context-budget/SKILL.md"),
        _skill("agentic-os", "Use goal, card, blocker and verifier structure for agentic execution.", ecc_reference="skills/agentic-os/SKILL.md"),
        _skill("design-quality-pack", "Activate curated design review skills for Buidl and generated website goals without loading all ECC skills.", ecc_reference="skills/design-quality/SKILL.md"),
        _skill("security-safety-pack", "Activate curated safety skills for secrets, env redaction, provider gates and approval boundaries.", ecc_reference="skills/security-safety/SKILL.md"),
        _skill("verification-pack", "Activate curated verifier skills for tests, CI, browser criteria and evidence-based done status.", ecc_reference="skills/verification/SKILL.md"),
        _skill("memory-learning-pack", "Activate curated memory skills for sanitized lesson candidates and Obsidian-safe learning.", ecc_reference="skills/memory-learning/SKILL.md"),
        _skill("agentic-build-pack", "Activate planner, builder, reviewer, verifier and Goal OS card skills by goal type.", ecc_reference="skills/agentic-build/SKILL.md"),
        _command("goal", "Create or inspect a durable Buidl goal."),
        _command("status", "Report active Buidl goals and next actions."),
        _command("blockers", "Report true blockers only."),
        _command("plan", "Create or report a safe plan card."),
        _command("execute", "Move the next safe card into execution."),
        _command("review", "List cards needing review."),
        _command("verify", "Record or request verifier evidence."),
        _command("fix-ci", "Create a build or CI failure resolver card."),
        _command("ship", "Mark achieved only with verifier evidence."),
        _command("learn", "Create a sanitized lesson candidate."),
        _command("checkpoint", "Record a durable work checkpoint."),
        _hook("no-niko-terminal-work", "Block requests that ask Niko to run terminal, sudo, Docker, GHCR or server debugging work."),
        _hook("no-niko-credentials", "Block requests for credentials, provider keys, GitHub tokens, secrets or env editing."),
        _hook("no-live-blind-prompt-storage", "Block storage, fixtures, printing or preparation of live blind prompts."),
        _hook("provider-call-approval-gate", "Stop provider calls unless explicitly approved."),
        _hook("image-generation-approval-gate", "Stop image generation unless explicitly approved."),
        _hook("production-domain-db-money-gate", "Stop production, DNS, DB, billing, credits and payments without approval.", risk="high"),
        _hook("no-caddy-hash-printing", "Prevent Caddy hash printing or storage in reports."),
        _hook("no-env-value-printing", "Prevent env value printing and secret-shaped output."),
        _hook("obsidian-partial-read-write-guard", "Require relevant full context before Obsidian writes after partial reads."),
        _hook("no-local-gym-regression", "Reject restoring Local Gym as product route."),
        _hook("no-fake-generated-preview", "Reject fake generated preview proof surfaces."),
        _hook("staging-apply-marker-verification", "Require marker verification before any future staging apply wrapper."),
        _rule("buidl-mvp-execution-mode", "Work in larger complete units while preserving hard approval gates."),
        _rule("agentic-builder-path", "Do not polish Local Gym or restore legacy mock as product route."),
        _memory("lesson-candidate-review", "Session summaries become reviewed lesson candidates before skill promotion."),
        _scanner("secret-scan", "Scan changed files for secret-shaped values before commit."),
        _scanner("ecc-wholesale-copy-scan", "Ensure ECC files were not copied wholesale into Hermes."),
    )
    return CapabilityRegistry(capabilities=capabilities, audit_inventory=load_ecc_audit_inventory())


__all__ = [
    "Capability",
    "CapabilityRegistry",
    "load_ecc_audit_inventory",
    "load_registry",
    "load_skill_registry",
]
