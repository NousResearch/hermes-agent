# Change: gateway-deferred-restart-g3-audit

## Why

The companion change `gateway-deferred-restart` (PR #140993) adds the deferred
restart channel and documents a known v1 coverage gap: a cron payload can drive
the channel, so a scheduled job can trigger repeated gateway restarts
(fire/restart loop). That change also carries no accountability trail for who
requested a restart. This change closes both gaps in one decoupled package:
source restriction (G3) + audit trail.

## What Changes

1. **G3, layer 1 — definition paths**: cron job create/edit/import/re-enable
   reject payloads referencing the deferred restart entrypoint (mention-level
   detection: no inert-body stripping; per-carrier scanning of prompt, monitor
   source and script paths, with merged-query splitting and active-home path
   binding).
2. **G3, layer 2 — dispatch pre-execution**: `check_deferred_restart_job`
   re-checks every payload at the real dispatch entry, fail-closed, including
   pre-existing jobs; a refused job books a failed execution and reaches zero
   user-code/model side effects.
3. **G3, runtime control point — CLI entrypoint provenance refusal**:
   `hermes_cli/defer_restart.py` refuses (exit 7) when the invocation carries
   trusted cron provenance (`HERMES_CRON_SESSION` / external-worker marker),
   before any request file is written or any signal is sent. A bash chain that
   assembles the command dynamically and executes it from a cron context must be
   refused with zero restart side effects.
4. **Audit trail**: every applicable decision or replay event is attempted into
   `restart_requests/audit.jsonl` (request id, bounded actor ref, timestamp,
   target generation, service identity, outcome). Placement rule: a decision
   audit fires only after the authoritative result record is persisted.

## Non-goals / explicit guarantee limits

- The audit chain is **evidence, not a gate**: append failures are reported and
  never alter the authoritative decision (the result record stays authoritative).
- **No no-gaps completeness is promised**: under storage failure the audit chain
  MAY have gaps (e.g. a duplicate replay whose append failed leaves no persistent
  trace of that replay). Every applicable event is attempted and failures are
  reported; that is the whole guarantee.
- Cooldown semantics remain as specified in the companion change: it limits new
  acceptance rate (anchored at accepted_at), not actual restart intervals; a
  persistent cron payload could still drive repeated restarts if G3 layers were
  bypassed — G3 is defense in depth, not a formal safety proof.

## Impact

- Affected specs: `gateway-deferred-restart-g3-audit` (new)
- Affected code: `cron/lifecycle_guard.py`, `cron/jobs.py`,
  `cron/job_definition.py`, `cron/scheduler.py`, `gateway/defer_restart.py`,
  `hermes_cli/defer_restart.py`
- Untouched: existing guard semantics (G1/G2 probe verdicts byte-identical);
  `tools/terminal_tool_guards.py`, `tools/code_execution_tool.py` unchanged.
- Depends on: `gateway-deferred-restart` (PR #140993). Stacked PR; will rebase
  onto main once the companion merges.
