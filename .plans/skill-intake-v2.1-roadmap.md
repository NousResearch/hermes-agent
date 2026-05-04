# skill_intake V2.1 — Executive Backlog / Roadmap

## Positioning

V2 is complete.

V2.1 should not redesign the pipeline. It should harden and simplify operations around the existing 4-stage flow:
- audit
- convert
- re-audit
- admit

## Objective

Increase confidence, realism, and maintainability with the smallest possible changes.

## Priority order

### P1 — Modular refactor (lightweight)
Goal:
- reduce maintenance cost in `tools/skill_intake_tool.py`
- keep behavior stable

Scope:
- extract helpers by responsibility
- keep the public tool interface unchanged

Deliverable:
- thin `skill_intake_tool.py`
- helper modules for source resolution, discovery, semantics, conversion, governance

### P2 — Richer provenance
Goal:
- improve traceability for remote and converted sources

Scope:
- stronger remote metadata when reliably available
- normalized source metadata in conversion results
- clearer provenance for generated artifacts

Deliverable:
- richer provenance fields
- tests that assert them

### P3 — Larger realistic fixtures
Goal:
- increase confidence with more representative inputs

Scope:
- richer generic workflow fixture
- richer CLAUDE-style instructions fixture
- richer OpenClaw-like package fixture
- one intentionally risky fixture

Deliverable:
- more realistic integration coverage

### P4 — Eval planning
Goal:
- prepare for with-skill vs without-skill quality comparison

Scope:
- output a lightweight eval plan structure
- no full eval harness yet

Deliverable:
- deterministic eval-plan artifact from audit or re-audit

### P5 — Sandbox / CI hardening
Goal:
- prepare safer operational rollout

Scope:
- document CI gate expectations
- document non-goals for third-party code execution
- optionally prepare a minimal hardening checklist

Deliverable:
- hardening note/checklist

## Suggested execution sequence

1. modular refactor
2. richer provenance
3. realistic fixtures
4. eval planning
5. sandbox / CI hardening

## Exit criteria for V2.1

- codebase easier to maintain
- provenance materially better
- integration tests more realistic
- eval planning exists at baseline
- hardening path is documented

## What not to do in V2.1

- no auto-install of converted skills
- no auto-publish
- no implicit trust by source
- no execution of third-party scripts by default
- no rewrite of the admission model
