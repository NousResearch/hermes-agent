# Brief: preserve monitor follow-through without routine agent polling

## Decision and evidence

Hermes monitor jobs currently hash a monitor source and suppress an unchanged result before agent work (`cron/scheduler.py::_apply_monitor_gate`; `tests/cron/test_monitor_kind.py`). That correctly makes stable polling silent, but it is insufficient for three already-approved, **currently restored ordinary** agent jobs:

| Job | Cadence | Responsibility that must remain owned |
| --- | --- | --- |
| `d402b304bbdf` | 20m | PR309 completion/reconciliation and later approved activation; do not touch backfill owner `ae3f40ba2312` |
| `7dc76e654c44` | 15m | Byrdseed_OS PR167/168/174 follow-through |
| `e42cfb93506c` | 20m | bs-invoices PR169 follow-through |

The rejected local collectors treated output as a hash-only phase. Therefore an unchanged `stalled` or `ready_for_activation` phase could suppress a required continuation; an interrupted/partial Cursor review could not continue without a GitHub mutation; and a collector that printed an error to stdout with exit 0 could become an unchanged silent result. PR309's stall ordering could also hide exhaustion/reconciliation readiness.

The scheduler already treats a nonzero monitor source as an error and preserves the stored hash. Existing upstream draft [#97978](https://github.com/NousResearch/hermes-agent/pull/97978) covers one adjacent case—retrying a detected change after an agent failure—but is much broader (fallback-provider work) and does not establish continuation after a successful but unfinished agent turn. Do not copy or merge it wholesale.

## Required behavior

Implement the **smallest opt-in monitor continuation contract** in the existing cron monitor path. It must be usable by the three named jobs without adding a scheduler job, a second owner, an agent invocation on ordinary settled polling, or a local activation step.

1. Stable settled source output remains a no-agent, no-delivery silent tick.
2. A durable explicit unfinished/pending condition continues to wake the existing job even if its ordinary source snapshot is unchanged. It must clear only after the source reports settled/terminal; do not infer completion from a previous model response.
3. A monitor-source failure must remain retryable and visible: nonzero exit/error must never be baseline-committed or transformed into a stable stdout payload. The next scheduled tick must reattempt collection. Do not replay an agent action merely because delivery failed.
4. A failed or interrupted agent turn after a changed/pending condition must retain enough durable state for the next scheduled tick to retry/continue safely. The normal legacy monitor behavior remains unchanged unless the opt-in is set.
5. The transition selection for the PR309-shaped collector is ordered so exhaustion/reconciliation/activation readiness cannot be masked by a generic stalled state. This is a collector acceptance contract, not permission to create or install a collector now.
6. Preserve monitor script/url exclusivity, `no_agent` incompatibility, exact-output hashing for legacy jobs, snapshot redaction, current schedules, prompts, deliveries, enabled state, continuity, workdirs, and all existing production approvals.

## Proof required

Use `scripts/run_tests.sh`, not bare pytest. Add focused behavior tests (not source-shape/change-detector tests) proving:

- unchanged settled snapshot suppresses agent and delivery;
- unchanged durable pending snapshot invokes the existing agent on later ticks and clears back to suppression when settled;
- an agent failure/interruption retains pending/change retry state; a success alone does not clear source-owned pending;
- source nonzero/error is retried and is not committed as a stable baseline;
- legacy monitor jobs retain existing hash-only behavior;
- a read-only real feed sample for each target shape: PR309/backfill-owner state, Byrdseed_OS PR state, and bs-invoices PR state. Fixtures may redact IDs/content, but must derive from `gh pr view`/existing local durable evidence; no provider, production, Help Scout, Slack, or writer call.

Before production code, add RED tests and record `git diff --stat`; stop and report if tests consume the implementation budget. Prove RED against the reverted production change, then GREEN. Run `git diff --check`.

## File boundary and review

Expected owner files: `cron/monitor.py`, `cron/scheduler.py`, narrowly necessary persisted-field/CLI plumbing in `cron/jobs.py` and `hermes_cli/`, and `tests/cron/test_monitor_kind.py` or one focused sibling. Do not add a new generic watcher framework, provider fallback work, a new core tool, an environment flag, new schedule, or unrelated refactor.

Budget: approximately 80 production lines and 150 test lines; report before exceeding either. Tier 2: focused tests plus one independent read-only review of the final commit. The coordinator owns live job configuration/readback; Cursor must not alter `~/.hermes`, schedules, jobs, scripts, config, or production state.

## Out of scope and gates

- Do not modify, run, pause, or replace `ae3f40ba2312`; do not create another cron watcher.
- Do not merge, deploy, install, activate, call production, invoke provider mutation APIs, send Slack/Help Scout messages, or change customer/account data.
- No live enablement follows this PR. A separate exact-target configuration/activation decision remains required after implementation, review, and native scheduler readback.
