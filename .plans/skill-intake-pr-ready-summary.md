# skill_intake V2 — PR-ready summary and checklist

## Summary

This change introduces and hardens `skill_intake`, a governed pipeline for evaluating and transforming candidate skills before admission into Hermes.

Delivered stages:
- `audit`
- `convert`
- `re-audit`
- `admit`

Key capabilities:
- local and remote source classification
- compatibility, security, and semantic portability scoring
- conversion to Hermes draft skill format
- `discarded` and `mapping` outputs for traceability
- optional artifact persistence via `out_dir`
- OpenClaw-like package preservation of `scripts/` and `references/`
- policy-based admission decisions
- focused integration tests

## Key files

### Added / implemented
- `tools/skill_intake_tool.py`
- `tests/tools/test_skill_intake_tool.py`
- `tests/tools/test_skill_intake_tool_integration.py`
- `.plans/skill-intake-v2-summary.md`
- `.plans/skill-intake-readme.md`
- `.plans/skill-intake-pipeline.md`
- `.plans/skill-intake/output-schema.json`
- `.plans/skill-intake/policy-matrix.md`

### Modified
- `toolsets.py`

## Validation performed

```bash
uv run --extra dev python -m pytest tests/tools/test_skill_intake_tool.py -q -o 'addopts='
uv run --extra dev python -m pytest tests/tools/test_skill_intake_tool_integration.py -q -o 'addopts='
uv run --extra dev python -m pytest tests/test_toolsets.py tests/tools/test_skill_intake_tool.py tests/tools/test_skill_intake_tool_integration.py -q -o 'addopts='
```

Latest result:
- all targeted tests passing

## Reviewer checklist

### Product / workflow
- [ ] `audit -> convert -> re-audit -> admit` is preserved
- [ ] conversion does not imply approval
- [ ] admission is policy-gated

### Technical
- [ ] tool schema matches supported modes and args
- [ ] local provenance is emitted
- [ ] remote provenance is emitted at current baseline
- [ ] generated skill output contains traceability (`discarded`, `mapping`)
- [ ] OpenClaw-like package assets are preserved

### Tests
- [ ] unit tests cover generic conversion
- [ ] unit tests cover temporary-context discard
- [ ] unit tests cover OpenClaw-like package handling
- [ ] integration tests cover end-to-end flow

### Follow-ups (not blocking V2)
- [ ] richer remote provenance
- [ ] larger fixtures
- [ ] eval planning
- [ ] sandbox/CI hardening
