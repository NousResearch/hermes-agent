# Hermes Judge Rubric

This rubric is the stopping gate for Hermes optimization phases.

## Completion Rule

A phase can be considered complete only when:

- Architecture Judge returns PASS.
- Reliability/Security Judge returns PASS.
- Tooling/UX Judge returns PASS.
- No unresolved critical security issue exists.
- No startup path is broken.
- Major architecture decisions are documented.
- Tests or smoke tests pass, or failures are documented with next steps.
- `docs/HERMES_BUILD_LOG.md` is updated.
- `docs/HERMES_EXECUTION_PLAN.md` reflects completed and remaining work.

## Judge Output Format

Each judge must return:

- `PASS` or `FAIL`
- Top issues
- Required fixes
- Optional improvements
- Confidence score from 1 to 10

## Judge 1 - Architecture Judge

Evaluate:

- Is the system modular?
- Are boundaries clear?
- Is the design extensible?
- Is there unnecessary complexity?
- Are docs accurate?
- Does the plan preserve existing Hermes behavior?
- Are gateway/core/plugin/tool/memory/runtime boundaries respected?
- Are rollout phases reversible and reviewable?

Automatic fail:

- Optimization logic is hardcoded into `gateway/run.py` without justification.
- A rewrite bypasses existing tool/plugin/memory/provider architecture.
- Existing runtime paths are replaced without rollback and validation.
- Major behavior changes lack documentation.

## Judge 2 - Reliability/Security Judge

Evaluate:

- Are tests adequate for the changed surface?
- Are logs useful and redacted?
- Are failure modes handled?
- Are secrets protected?
- Are dangerous actions gated?
- Are unsafe capabilities excluded?
- Is the startup path intact?
- Is the rollback path clear?

Automatic fail:

- Secrets or private data are printed, committed, or copied into docs.
- Gateway cannot start or its active launchd wrapper is broken.
- External send/post/spend/trade/account actions are made autonomous.
- Critical security issue is knowingly left unresolved without a blocker.
- Tests/smoke checks are skipped without explanation.

## Judge 3 - Tooling/UX Judge

Evaluate:

- Are the tools useful?
- Is the tool interface clean?
- Can a user run Hermes easily?
- Are workflows documented?
- Is the system maintainable?
- Does the operator have clear status and troubleshooting commands?
- Are tool/permission/cost states understandable?

Automatic fail:

- New commands expose secrets.
- User cannot tell what changed or how to validate it.
- Tool registry/control plane is misleading or stale.
- Documentation points to the wrong active gateway label.

## Phase-Specific Minimums

### Phase 0

- Control docs exist.
- Build log exists.
- Judge rubric exists.
- Repo safety state is recorded.
- No feature behavior changed.
- Live startup path is checked.

### Phase 1-2

- Any control-plane inventory code is read-only first.
- Registry output is redacted.
- Tests cover schema stability and redaction.

### Phase 3

- Memory capacity, retention, retrieval, and deletion paths are documented.
- Any memory mutation sync behavior is tested.

### Phase 4-6

- Gateway startup path remains wrapper-backed.
- Health checks distinguish expected auth failures from runtime failures.
- Ops commands have concise output and receipt paths.

### Phase 7-8

- Skills/workflows are actually reusable.
- End-to-end validation is run.
- Judge cycle is repeated after fixes.

## Current Phase 0 Judge Template

Architecture Judge:

- Result:
- Top issues:
- Required fixes:
- Optional improvements:
- Confidence:

Reliability/Security Judge:

- Result:
- Top issues:
- Required fixes:
- Optional improvements:
- Confidence:

Tooling/UX Judge:

- Result:
- Top issues:
- Required fixes:
- Optional improvements:
- Confidence:
