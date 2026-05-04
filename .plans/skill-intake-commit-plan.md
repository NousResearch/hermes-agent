# skill_intake — Suggested Commit Sequence

## Goal

Break the work into reviewable commits that preserve the story of the feature:
- architecture and docs first
- audit pipeline next
- conversion and traceability next
- governance and integration last

## Suggested commit sequence

### Commit 1
`docs: add skill_intake architecture and planning artifacts`

Files:
- `.plans/skill-intake-pipeline.md`
- `.plans/skill-intake/output-schema.json`
- `.plans/skill-intake/policy-matrix.md`
- sample result JSONs
- `.plans/skill-intake-v2-summary.md`
- `.plans/skill-intake-readme.md`
- `.plans/skill-intake-pr-ready-summary.md`
- `.plans/skill-intake-v2.1-plan.md`
- `.plans/skill-intake-delivery-package.md`

### Commit 2
`feat: add skill_intake audit tool and toolset integration`

Files:
- `tools/skill_intake_tool.py` (initial audit mode)
- `toolsets.py`
- `tests/tools/test_skill_intake_tool.py` (audit-focused baseline)

### Commit 3
`feat: support remote GitHub/raw audit provenance in skill_intake`

Files:
- `tools/skill_intake_tool.py`
- `tests/tools/test_skill_intake_tool.py`

Focus:
- remote classification
- remote fetch attempts
- provenance baseline

### Commit 4
`feat: add generic skill conversion pipeline to skill_intake`

Files:
- `tools/skill_intake_tool.py`
- `tests/tools/test_skill_intake_tool.py`

Focus:
- convert mode
- generated draft skill
- mapping/discarded basics
- out_dir persistence

### Commit 5
`feat: add re-audit and admit stages to skill_intake`

Files:
- `tools/skill_intake_tool.py`
- `tests/tools/test_skill_intake_tool.py`

Focus:
- re-audit
- readiness
- policy gate
- admit mode

### Commit 6
`feat: refine generic conversion semantics for skill_intake`

Files:
- `tools/skill_intake_tool.py`
- `tests/tools/test_skill_intake_tool.py`

Focus:
- section inference
- prerequisites
- commands
- examples
- pitfalls / verification
- temporary-context discard improvements

### Commit 7
`feat: support OpenClaw-like packages in skill_intake convert`

Files:
- `tools/skill_intake_tool.py`
- `tests/tools/test_skill_intake_tool.py`

Focus:
- detect OpenClaw-like package
- preserve `scripts/` and `references/`
- mapping for preserved assets

### Commit 8
`test: add integration coverage for skill_intake pipeline`

Files:
- `tests/tools/test_skill_intake_tool_integration.py`

Focus:
- generic end-to-end flow
- OpenClaw-like end-to-end flow

## If you want a shorter sequence

Condensed version:
1. docs/spec
2. audit + remote provenance
3. convert + out_dir + generic refinement
4. re-audit + admit
5. OpenClaw-like + integration tests

## Review recommendation

If opening as one PR, still preserve the sequence in the PR description as milestones so reviewers can follow the evolution.
