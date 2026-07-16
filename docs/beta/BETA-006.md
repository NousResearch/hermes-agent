# BETA-006 — Specialist result consolidation

`consolidate_results()` produces Beta's single structured response while keeping facts, evidence, and hypotheses distinct. It deduplicates repeated findings, retains partial successes, detects opposing claims about the same subject, and applies confidence penalties for conflicts and failures.

A probable cause is emitted only when some fact or evidence exists. Without evidence, confidence is capped and the response explicitly says a categorical conclusion is unavailable.

Contradictions or high-risk recommendations set `qa_required`. When the orchestrator supplies a `qa_validator`, consolidation invokes it and includes the validated QA result. High-risk recommendations always set `authorization_required`; consolidation never executes them.

## Validation

```bash
python -m pytest -q tests/agent/beta/test_consolidation.py
```
