# CLOE Roadmap Completion & Certification Pattern

## When to use
When executing the final mission of a multi-mission CLOE roadmap chain — the stabilization/certification phase that validates the complete pipeline.

## Preconditions
- All prior roadmap missions completed and merged
- Operator approval for the final mission received
- Memory lock chain exists (each prior mission has a continuity file)

## Sequence

### 1. Full pipeline execution
Run every stage of the QA pipeline sequentially:

```bash
node tools/ci/lah-runtime-evidence-collector.mjs --all
node tools/ci/lah-reasoning-review-engine.mjs
node tools/ci/lah-root-cause-engine.mjs
node tools/ci/lah-targeted-improvement-engine.mjs
node tools/ci/lah-continuous-learning-engine.mjs
node tools/ci/lah-cognitive-release-gate.mjs
node tools/ci/cloe-reasoning-context-regression-guard.mjs
node tools/ci/lah-behavior-comparison-reporter.mjs
```

### 2. Extract certification metrics
Collect from all reports:

| Metric | Source |
|--------|--------|
| Global Score | `lah-reasoning-review.json` |
| Release Gate | `lah-cognitive-release-gate.json` |
| Regression Guard | `cloe-reasoning-context-regression-guard.mjs` |
| Reviewer Scores | `reviewer_summaries` in review report |
| Root Causes | `diagnoses` in root cause report |
| Learning Artifacts | `artifacts` in continuous learning report |
| V1→V2 Comparison | `lah-behavior-comparison.json` |
| Scenario Routing | `lah-runtime-evidence.json` |

### 3. Create certification report
Document in `docs/operator/CLOE_COGNITIVE_ASSISTANT_V1_CERTIFICATION_REPORT.md`:
- Executive summary with architecture version
- Global metrics table
- Reviewer scores with interpretation
- Pipeline diagram
- Component inventory (all 20 items)
- Root causes (automated)
- Improvement plans generated
- Known limitations
- Certification verdict
- Recommended next roadmap

### 4. Create stabilization architecture doc
Document in `docs/architecture/CLOE_COGNITIVE_ASSISTANT_STABILIZATION_V1.md`:
- Architecture overview
- QA pipeline overview
- Resource governance rules
- V1 baseline metrics

### 5. Final deliverables
- Certification report
- Stabilization doc
- Operator packet
- LOCAL_CI_VERIFIED receipt
- Continuity JSON with full mission chain

### 6. Known pitfalls
- Comparison baseline format mismatch: V1 domain names ≠ V2 capability IDs. Use `lah-behavior-comparison-reporter.mjs` with `lah-domain-capability-mapping-v1.json`.
- Evidence reviewer may report 0 due to format mismatch — calibrate before final certification.
- Continuous learning may show 0 artifacts if artifacts use nested `.items` arrays — check `.count` field.
- Resource gate must run first — confirm in release gate output.
