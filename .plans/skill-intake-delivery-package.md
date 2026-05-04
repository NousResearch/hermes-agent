# skill_intake — Delivery Package Index

## Executive conclusion

The `skill_intake` V2 baseline is functionally complete and documented.

This package indexes the main artifacts produced during the build-out so review, handoff, and next-phase planning are straightforward.

---

## Core implementation

### Tool implementation
- `tools/skill_intake_tool.py`

What it contains:
- `audit`
- `convert`
- `re-audit`
- `admit`
- generic conversion heuristics
- OpenClaw-like package support
- provenance
- policy gate

---

## Test coverage

### Unit / focused tests
- `tests/tools/test_skill_intake_tool.py`

Covers:
- audit local / remote basics
- generic conversion
- prerequisites / commands / examples extraction
- discarded temporary context
- out_dir persistence
- OpenClaw-like package conversion
- re-audit
- admit

### Integration-style tests
- `tests/tools/test_skill_intake_tool_integration.py`

Covers:
- end-to-end generic flow
- end-to-end OpenClaw-like flow

---

## Design / planning artifacts

### Master implementation plan
- `.plans/skill-intake-pipeline.md`

### V2 executive summary
- `.plans/skill-intake-v2-summary.md`

### Operational README
- `.plans/skill-intake-readme.md`

### PR-ready review summary
- `.plans/skill-intake-pr-ready-summary.md`

### V2.1 follow-up plan
- `.plans/skill-intake-v2.1-plan.md`

---

## Spec artifacts

Directory:
- `.plans/skill-intake/`

Key files:
- `output-schema.json`
- `policy-matrix.md`
- `sample-audit-result.json`
- `sample-convert-result.json`
- `sample-reaudit-result.json`
- `sample-admit-result.json`

---

## Validation commands

### Focused tool tests
```bash
uv run --extra dev python -m pytest tests/tools/test_skill_intake_tool.py -q -o 'addopts='
```

### Integration tests
```bash
uv run --extra dev python -m pytest tests/tools/test_skill_intake_tool_integration.py -q -o 'addopts='
```

### Regression subset
```bash
uv run --extra dev python -m pytest tests/test_toolsets.py tests/tools/test_skill_intake_tool.py tests/tools/test_skill_intake_tool_integration.py -q -o 'addopts='
```

Latest validated subset result during delivery:
- all targeted tests passing

---

## Current V2 scope delivered

- audit-first intake workflow
- candidate classification
- compatibility scoring/classification
- security heuristic scanning
- semantic portability scoring
- conversion to Hermes draft
- discarded tracking
- source-to-output mapping
- optional out_dir persistence
- re-audit readiness
- policy-based admission
- OpenClaw-like package preservation

---

## Recommended next step after handoff

Treat V2 as complete.

Move to V2.1 with this order:
1. richer provenance
2. larger / more realistic fixtures
3. eval planning
4. sandbox / CI hardening
