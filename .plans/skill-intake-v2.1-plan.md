# skill_intake V2.1 Plan

> **For Hermes:** Execute after V2 baseline is considered complete. Keep scope tight and bias toward realism, provenance, and operational hardening.

**Goal:** Improve confidence, realism, and operational rigor of `skill_intake` without redesigning the pipeline.

**Architecture:** Preserve the current 4-stage pipeline (`audit`, `convert`, `re-audit`, `admit`) and harden around it using richer fixtures, stronger provenance, and better evaluation support.

**Tech Stack:** Python 3, pytest, Hermes tool registry, local fixtures, optional remote metadata resolution.

---

## Priorities

1. richer provenance
2. larger / more realistic fixtures
3. eval planning
4. sandbox / CI hardening

---

## Task 1: Provenance enrichment

**Objective:** Add stronger provenance for remote and converted sources.

**Files:**
- Modify: `tools/skill_intake_tool.py`
- Extend: `tests/tools/test_skill_intake_tool.py`

**Targets:**
- repo owner / repo name when reliably parsed
- ref/branch if discoverable
- raw URL normalization trace
- conversion timestamp / origin metadata in generated outputs

**Verification:**
- unit tests assert added provenance fields without breaking current output contracts

---

## Task 2: Larger realistic fixtures

**Objective:** Improve confidence with more realistic source samples.

**Files:**
- Create: `tests/tools/fixtures/skill_intake/` (if desired)
- Extend: `tests/tools/test_skill_intake_tool_integration.py`

**Targets:**
- larger CLAUDE-style workflow
- larger generic workflow markdown
- richer OpenClaw-like package
- at least one risky candidate with preserved findings

**Verification:**
- integration tests remain fast and deterministic

---

## Task 3: Eval planning

**Objective:** Generate eval skeletons for future with-skill vs without-skill comparisons.

**Files:**
- Modify: `tools/skill_intake_tool.py`
- Extend tests as needed

**Targets:**
- optional eval plan output from `audit` or `re-audit`
- minimal structure for scenarios / expected outcomes

**Verification:**
- output is structured and deterministic

---

## Task 4: Sandbox / CI hardening plan

**Objective:** Prepare the pipeline for safer operational rollout.

**Files:**
- Docs only in V2.1 if implementation is too large
- Potential plan doc for CI / sandbox follow-up

**Targets:**
- define non-goals for execution of third-party code
- define CI gates for `skill_intake`
- define how to keep tests stable and cheap

**Verification:**
- docs/checklist exists and is actionable

---

## Exit criteria for V2.1

- provenance improved in a test-covered way
- integration fixtures more realistic
- eval planning baseline exists
- sandbox / CI hardening path documented

## Not in scope for V2.1

- auto-install of converted skills
- auto-publish
- executing third-party scripts by default
- full remote SCM metadata resolution if it adds fragility
