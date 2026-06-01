# HAD-271 Reliability Runtime Surface Blocker

Date: 2026-06-01

## Evidence

- `origin/main` includes PR #140 at merge commit `615afe8d6a2ebeba6fd5bf865cbff869ed63fc59`.
- The active cron launcher at `/home/david/.hermes/scripts/hadto_self_improvement_pipeline.py` imports `hadto_hermes_plugin.tools.self_improvement` from `/home/david/.hermes/plugins/hadto`, not the repo-local `tools/self_improvement_tool.py`.
- The latest persisted benchmark history entry at `2026-06-01T07:12:25.362629+00:00` is in the legacy `evaluations` format and reports `score=78.72`, `critical_failures=["reliability_gate","leading_indicator_drift"]`, `reliability_gate=0.45`, `execution_loop=0.6`, and `leading_indicator_drift=0.5`.
- A repo-local `evaluate_self_improvement_pipeline(persist=False)` run from `/home/david/stacks/hermes-agent` on current `origin/main` reports `reliability_gate=1.0 pass`; the remaining local failures are from claimed-work and operator-value evidence, then `leading_indicator_drift`.

## Blocker

HAD-271 cannot be closed from PR #140 alone while the operational cron path continues to execute the installed Hadto plugin surface and write legacy benchmark entries. A repo fix to `tools/self_improvement_tool.py` must either be deployed into the active plugin path or the cron launcher must be switched to the repo-local self-improvement tool before cron benchmark evidence can prove the repaired reliability gate.

## Follow-up

This PR also fixes one repo-local scoring defect found during the reproduction: failed Codex attempts with `completed_at` are no longer counted as completed deliveries or claimed work. That correction keeps failed/stale attempts from inflating execution-loop throughput or dragging operator-value evidence as if they were shipped work.
