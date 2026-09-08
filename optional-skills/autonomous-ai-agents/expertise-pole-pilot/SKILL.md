---
name: expertise-pole-pilot
description: Build auditable expert-domain pilots with evidence gates.
version: 0.1.0
author: Vincent HERON, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Agents, Skills, Workflows, Governance, Evaluation]
    related_skills: []
---

# Expertise-Pole Pilot Skill

Build a bounded, auditable domain-expertise pilot: capabilities, explicit role contracts, reusable skills, workflows, evidence gates, and regression checks. This optional skill ships a **French, experimental SEO pilot** as a reference pack; it does not make SEO claims, retrieve sources, or permit external actions by itself.

## When to Use

Use when a project needs to turn a recurring expert domain into a governed system of agents, skills, rules, workflows, contracts, and evaluation artifacts.

- Establishing a first vertical pilot before extracting reusable support-pole conventions.
- Reviewing an agentic process that has unclear ownership, evidence, escalation, or quality gates.
- Designing a test-and-learn experiment where promotion is blocked until evidence and an independent review exist.

Don't use for a one-off request, live SEO changes, unauthorised crawling, source research without a source budget/permission, or an implementation that skips human approval gates.

## Prerequisites

- A named human steward, bounded scope, non-objectives, and explicit acceptance criteria.
- Permission for any external retrieval, credential use, publication, account creation, production change, or billable action; absence of permission means stop and record a gap.
- A local workspace in which documentation and tests can be reviewed and versioned.
- Read the reference pack before adapting it: `references/pilot-v0.1/README.md`.

## How to Run

Use `terminal` for the deterministic structural audit:

```python
terminal(
  command=(
    "python -c \"from pathlib import Path; from importlib.util import "
    "spec_from_file_location, module_from_spec; p=Path('optional-skills/"
    "autonomous-ai-agents/expertise-pole-pilot/references/pilot-v0.1/"
    "evaluations/audit_pack.py'); s=spec_from_file_location('audit_pack', p); "
    "m=module_from_spec(s); s.loader.exec_module(m); print(m.audit(p.parent.parent))\""
  ),
  workdir="<hermes-agent-repo>",
  timeout=120,
)
```

Use `read_file` for individual contracts and `search_files` to trace a capability across experts, skills, workflows, and evaluations. Use `write_file` or `patch` only after the relevant evaluation contract has been updated.

## Quick Reference

| Need | Reference artifact |
|---|---|
| Support-pole governance and lifecycle | `references/pilot-v0.1/support-pole/` |
| SEO scope and capability map | `references/pilot-v0.1/domains/seo/CAPABILITIES.md` |
| Role boundaries and contracts | `references/pilot-v0.1/domains/seo/EXPERTS.md`, `CONTRACTS.md` |
| Source hierarchy and freshness | `references/pilot-v0.1/domains/seo/SOURCES.md` |
| Workflow states and handoffs | `references/pilot-v0.1/domains/seo/WORKFLOW.md` |
| Scenarios, rubrics, and runs | `references/pilot-v0.1/domains/seo/EVALUATION.md`, `evaluations/` |
| Open decisions and maturity gate | `references/pilot-v0.1/reports/OPEN-DECISIONS.md`, `MATURITY-DECISION.md` |

## Procedure

1. **Record the boundary.** Create an inventory, scope, non-objectives, risks, and unresolved decisions. Completion: each in-scope capability has an observable output; ambiguity is an open decision, not an invented convention.
2. **Map the domain.** Separate capability, expert role, skill, workflow, rule, and test. Completion: every priority capability has one accountable owner, a contract, and an evaluation strategy.
3. **Instantiate the support-pole.** Start from the templates in `references/pilot-v0.1/support-pole/templates/`. Completion: document precedence, maturity, review cadence, evidence requirements, human approvals, and handoff format without duplicating canonical responsibilities.
4. **Build a vertical pilot.** Keep the first workflow narrow: evidence → prioritisation → non-executing action plan → QA. Completion: the workflow has a trigger, state model, failure states, human escalation points, and named handoff owners.
5. **Add reusable skills only where recurrence warrants them.** Each skill needs a precise trigger, prerequisites, executable procedure, pitfalls, end proof, and a non-use condition. Completion: a one-off or vague task has not been turned into a skill.
6. **Exercise adversarial fixtures.** Cover nominal input, missing data, conflicting signals, risky recommendations, stale sources, and an injected deliverable error. Completion: the regression log records expected and actual outcomes plus correction status.
7. **Hold promotion.** Run the audit and a role-distinct contradictory review; compare against an unstructured baseline if claimed. Completion: anything without primary-source provenance, independent review, and evidence stays experimental or quarantined.

## Pitfalls

- Treating a written document as a standard before an experiment, critique, correction, and regression run.
- Renaming a capability, role, skill, workflow, rule, or test into another category to hide missing ownership.
- Promoting a vertical-specific convention to the support-pole after only one domain.
- Calling an integrator self-check an independent review.
- Turning incomplete or blocked research into an implied recommendation.
- Making a production SEO, crawl, publication, credential, or spend action because a workflow describes it.
- Reading the SEO reference as authoritative SEO guidance: its source gate is intentionally blocked pending authorised primary-source research.

## Verification

Run the targeted no-network regression test through `terminal`:

```python
terminal(
  command="scripts/run_tests.sh tests/skills/test_expertise_pole_pilot_skill.py -q",
  workdir="<hermes-agent-repo>",
  timeout=300,
)
```

If the repository test wrapper lacks a pytest-enabled virtual environment, record that infrastructure blocker verbatim and run the same test with the available standard-library runner only as supplementary evidence. Green means the pack’s structural contracts hold and its blocked research/review gates are disclosed; it does **not** mean the SEO pilot is promoted or production-ready.
