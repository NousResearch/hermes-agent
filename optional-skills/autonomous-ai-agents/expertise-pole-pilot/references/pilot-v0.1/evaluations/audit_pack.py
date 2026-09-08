"""Deterministic structural audit for the reusable-support / SEO pilot pack.

This verifier is deliberately conservative: it proves required artefacts,
explicit document controls, workflow/skill contract markers, declared
traceability, and the absence of unsupported execution claims.  It does not
claim to validate SEO outcomes or external sources.
"""
from __future__ import annotations

import pathlib
import re
from collections import Counter
from typing import Any

SUPPORT_FILES = (
    "README.md",
    "AGENTS.md",
    "GOVERNANCE.md",
    "DOCUMENT-TAXONOMY.md",
    "NAMING-AND-STRUCTURE.md",
    "LIFECYCLE.md",
    "QUALITY-GATES.md",
    "ORCHESTRATION.md",
    "RESEARCH-PROTOCOL.md",
    "SECURITY-AND-PERMISSIONS.md",
    "EVALUATION-FRAMEWORK.md",
    "CHANGELOG.md",
    "templates/domain-template.md",
    "templates/skill-template.md",
    "templates/agent-profile-template.md",
    "templates/contradictory-review-template.md",
)
SEO_FILES = (
    "README.md",
    "CAPABILITIES.md",
    "EXPERTS.md",
    "CONTRACTS.md",
    "EVALUATION.md",
    "SOURCES.md",
    "SHOULD.md",
    "AGENTS.md",
    "WORKFLOW.md",
)
CONTROLLED = tuple(f"support-pole/{name}" for name in SUPPORT_FILES[:12]) + tuple(
    f"domains/seo/{name}" for name in SEO_FILES
)
CONTROL_MARKERS = (
    "## Document control",
    "- **Scope:**",
    "- **Owner:**",
    "- **Maturity:**",
    "- **Last reviewed:**",
    "- **Freshness rule:**",
    "- **Linked documents:**",
    "- **Evidence requirement:**",
    "- **Exit / validation criteria:**",
    "- **Risks / limits / escalation:**",
    "- **Change history:**",
    "- **Exclusive responsibility:**",
)
CAPABILITIES = tuple(f"SEO-C0{n}" for n in range(1, 7))
EVALUATION_CASES = (
    "01-nominal.md",
    "02-incomplete-data.md",
    "03-contradictory-signals.md",
    "04-risky-recommendation.md",
    "05-obsolete-source.md",
    "06-injected-error.md",
)
SKILL_MARKERS = (
    "## Trigger",
    "## Prerequisites",
    "## Procedure",
    "## Pitfalls",
    "## Verification evidence",
    "## Do not use when",
)
WORKFLOW_MARKERS = (
    "## Trigger",
    "## Output",
    "## States and failure paths",
    "## Handoffs",
)
PROFILE_MARKERS = (
    "## Mission",
    "## Competencies",
    "## Scope / anti-scope",
    "## Inputs / outputs",
    "## Authorized skills and tools",
    "## Handoff rule",
    "## Autonomy levels",
    "## Escalation conditions",
    "## Verification method",
)
REQUIRED_PROFILES = (
    "technical-seo-expert.md",
    "evidence-source-analyst.md",
    "seo-strategist-prioritizer.md",
    "seo-quality-controller.md",
    "workflow-orchestrator.md",
    "independent-seo-critic.md",
)
CASE_RUNS = tuple(f"CASE-0{n}.md" for n in range(1, 7))


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _relative(paths: list[pathlib.Path], root: pathlib.Path) -> list[str]:
    return [str(path.relative_to(root)) for path in sorted(paths)]


def audit(root: pathlib.Path) -> dict[str, Any]:
    """Return machine-readable structural findings; an empty list is a pass."""
    root = root.resolve()
    required = [root / "support-pole" / name for name in SUPPORT_FILES]
    required += [root / "domains" / "seo" / name for name in SEO_FILES]
    missing_required = _relative([path for path in required if not path.is_file()], root)

    missing_controls: list[str] = []
    responsibility_values: list[str] = []
    for rel in CONTROLLED:
        path = root / rel
        if not path.is_file():
            continue
        text = _read(path)
        absent = [marker for marker in CONTROL_MARKERS if marker not in text]
        if absent:
            missing_controls.append(f"{rel}: {', '.join(absent)}")
        match = re.search(r"^- \*\*Exclusive responsibility:\*\*\s*(.+)$", text, re.M)
        if match:
            responsibility_values.append(match.group(1).strip().casefold())
    duplicate_responsibilities = sorted(
        value for value, count in Counter(responsibility_values).items() if count > 1
    )

    workflow_contract_gaps: list[str] = []
    workflow_paths = [root / "domains" / "seo" / "WORKFLOW.md"]
    workflow_paths += list((root / "domains" / "seo" / "workflows").glob("*.md"))
    for path in workflow_paths:
        if not path.is_file():
            workflow_contract_gaps.append(str(path.relative_to(root)))
            continue
        absent = [marker for marker in WORKFLOW_MARKERS if marker not in _read(path)]
        if absent:
            workflow_contract_gaps.append(f"{path.relative_to(root)}: {', '.join(absent)}")

    skill_contract_gaps: list[str] = []
    skill_paths = list((root / "domains" / "seo" / "skills").glob("*/SKILL-CONTRACT.md"))
    if not skill_paths:
        skill_contract_gaps.append("domains/seo/skills: no reusable skill")
    for path in skill_paths:
        absent = [marker for marker in SKILL_MARKERS if marker not in _read(path)]
        if absent:
            skill_contract_gaps.append(f"{path.relative_to(root)}: {', '.join(absent)}")

    capability_gaps: list[str] = []
    coverage_paths = (
        root / "domains" / "seo" / "CAPABILITIES.md",
        root / "domains" / "seo" / "EXPERTS.md",
        root / "domains" / "seo" / "CONTRACTS.md",
        root / "domains" / "seo" / "EVALUATION.md",
    )
    for capability in CAPABILITIES:
        absent_from = [
            str(path.relative_to(root))
            for path in coverage_paths
            if not path.is_file() or capability not in _read(path)
        ]
        if absent_from:
            capability_gaps.append(f"{capability}: missing from {', '.join(absent_from)}")

    evaluation_gaps = _relative(
        [
            root / "domains" / "seo" / "evaluations" / "cases" / name
            for name in EVALUATION_CASES
            if not (root / "domains" / "seo" / "evaluations" / "cases" / name).is_file()
        ],
        root,
    )
    run = root / "domains" / "seo" / "evaluations" / "runs" / "RUN-001.md"
    if not run.is_file() or "RESULT: PASS" not in _read(run):
        evaluation_gaps.append("domains/seo/evaluations/runs/RUN-001.md: absent or not passing")

    safety_claim_gaps: list[str] = []
    execution_log = root / "reports" / "EXECUTION-LOG.md"
    inventory = root / "reports" / "PHASE-0-INVENTORY.md"
    sources = root / "domains" / "seo" / "SOURCES.md"
    required_safety_text = "No external, billable, or irreversible action was executed."
    if not execution_log.is_file() or required_safety_text not in _read(execution_log):
        safety_claim_gaps.append("reports/EXECUTION-LOG.md lacks the required safety declaration")
    if not inventory.is_file() or "NO-SPEND GUARD" not in _read(inventory):
        safety_claim_gaps.append("reports/PHASE-0-INVENTORY.md lacks research-blocker disclosure")
    if not sources.is_file() or "BLOCKED BY NO-SPEND GUARD" not in _read(sources):
        safety_claim_gaps.append("domains/seo/SOURCES.md lacks source-retrieval status")

    profile_contract_gaps: list[str] = []
    profile_dir = root / "domains" / "seo" / "agents"
    for name in REQUIRED_PROFILES:
        path = profile_dir / name
        if not path.is_file():
            profile_contract_gaps.append(f"domains/seo/agents/{name}: missing")
            continue
        absent = [marker for marker in PROFILE_MARKERS if marker not in _read(path)]
        if absent:
            profile_contract_gaps.append(
                f"domains/seo/agents/{name}: {', '.join(absent)}"
            )

    case_execution_gaps: list[str] = []
    for name in CASE_RUNS:
        path = root / "domains" / "seo" / "evaluations" / "runs" / name
        if not path.is_file():
            case_execution_gaps.append(f"domains/seo/evaluations/runs/{name}: missing")
            continue
        text = _read(path)
        if "No external action:** yes" not in text or "Result" not in text:
            case_execution_gaps.append(
                f"domains/seo/evaluations/runs/{name}: missing safe execution markers"
            )

    research_report = root / "reports" / "RESEARCH-COMPARISON-STATUS.md"
    research_status = (
        "blocked-no-spend"
        if sources.is_file()
        and research_report.is_file()
        and "BLOCKED BY NO-SPEND GUARD" in _read(sources)
        and "n'a **pas** pu être menée" in _read(research_report)
        else "inconsistent"
    )
    independent_review = root / "reports" / "INDEPENDENT-REVIEW.md"
    independent_review_status = (
        "blocked-no-spend"
        if independent_review.is_file()
        and "BLOCKED — no independent reviewer executed" in _read(independent_review)
        and "NO-SPEND GUARD" in _read(independent_review)
        else "inconsistent"
    )

    return {
        "missing_required": missing_required,
        "missing_controls": missing_controls,
        "duplicate_responsibilities": duplicate_responsibilities,
        "workflow_contract_gaps": workflow_contract_gaps,
        "skill_contract_gaps": skill_contract_gaps,
        "capability_gaps": capability_gaps,
        "evaluation_gaps": evaluation_gaps,
        "safety_claim_gaps": safety_claim_gaps,
        "profile_contract_gaps": profile_contract_gaps,
        "case_execution_gaps": case_execution_gaps,
        "research_status": research_status,
        "independent_review_status": independent_review_status,
    }
