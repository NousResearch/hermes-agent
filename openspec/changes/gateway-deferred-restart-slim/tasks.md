type: spec-tasks
project: hermes-agent (upstream PR)
task: gateway-deferred-restart-slim

# Tasks: gateway-deferred-restart-slim

## T1 CLI surface and target resolution
- [ ] Register `hermes gateway defer-restart` with no subcommand-specific arguments.
- [ ] Add a topical CLI sibling for cron refusal, target resolution, capability preflight, request, ACK validation, exit status, and output.
- [ ] Prefer a confirmed own gateway; use `host_multiplexer_serving(profile)` only for a confirmed same-tenant satellite with no own instance.
- [ ] Do not treat nullable PID/socket failure, starting/inaccessible ownership, stale/malformed records, or legacy default-home fallback as proof that an own instance is absent.
- [ ] Canonically verify the chosen control home and ensure identify and mutation target the same home.
- [ ] Reject standalone exclusions, unreachable/ambiguous owners, cross-tenant hosts, and failed resolution without guessing the default home.

## T2 Constrained gateway control verb
- [ ] Register a distinct `defer-restart` control verb; leave `pause-for-update` unchanged.
- [ ] On the receiving gateway, require `_detect_supervisor()` in `systemd|launchd|external` and `is_gateway_supervisor_process()` true.
- [ ] Reject non-POSIX hosts before any event-loop handoff; keep native Windows external-supervisor contexts unsupported.
- [ ] Read the runner's loaded drain budget; reject a non-finite budget before mutation rather than rereading or clamping config.
- [ ] Reject unsupported supervision without calling `request_restart()`.
- [ ] At callback execution time, reject an ordinary stop already in progress without changing its exit intent.
- [ ] Marshal `request_restart(detached=False, via_service=True)` to the event loop with distinct true, false, exception, and timeout outcomes.
- [ ] Return explicit `accepted` ACK data; never infer `already_stopping` from a missing result.

## T3 CLI result contract
- [ ] Use a bounded control request timeout that allows the handler result window but does not wait for restart completion.
- [ ] Exit 0 only for a strict valid positive ACK.
- [ ] Exit 1 for valid rejection/non-new acceptance.
- [ ] Define and validate the full negative ACK schema and recognized reasons.
- [ ] Treat missing, malformed, contradictory, or lost results as UNKNOWN; state that processing may have started.
- [ ] Do not retry or fall back to updater pause, signals, or ordinary restart.
- [ ] Explain multiplexer impact and that drain timeout is not a fixed restart delay.

## T4 Focused behavior tests
- [ ] Parser rejects delay/force/all/wait-ready/unknown arguments.
- [ ] Cron marker refuses before any mutation.
- [ ] Marked cron child coverage uses a real propagated child environment; document no-agent/pre-agent paths that currently lack the marker.
- [ ] Existing lifecycle guard still blocks restart/stop/uninstall and permits the new constrained command spelling.
- [ ] Target routing covers own owner, confirmed satellite host, standalone exclusion, unreachable/ambiguous owner, cross-tenant host, and failed host resolution.
- [ ] CLI preflight accepts only `systemd`, `launchd`, and `external`.
- [ ] Server-side gate rejects an unsupported replacement instance after supported CLI preflight.
- [ ] Native Windows external-supervisor test rejects without faking the host OS or calling `request_restart()`.
- [ ] Ordinary stop and late-callback orderings reject without creating a restart task or changing stop intent.
- [ ] Loaded finite drain metadata comes from the runner even when disk config differs; non-finite loaded budget rejects before mutation.
- [ ] Callback true, false, exception, and timeout produce the specified result classes.
- [ ] Timeout test proves UNKNOWN/no automatic retry while allowing a late callback to execute at most once.
- [ ] Strict ACK validation rejects boolean PIDs/timeouts, non-finite values, reasons on success, malformed false ACKs, and supervisor mismatch.
- [ ] Parser-to-dispatch or subprocess tests assert real process exit 0/1; handler return values alone are insufficient.
- [ ] Active-work test proves ACK precedes stop and stop waits for controlled eligible work under a positive wait budget.
- [ ] Same-runner duplicate request creates only one restart flow.
- [ ] Older unknown verb does not fall back; updater `pause-for-update` behavior remains unchanged.
- [ ] Include at least one temporary `HERMES_HOME` + real local socket + production bootstrap handler + real event-loop integration test.

## T5 Verification and documentation
- [ ] Update the minimal gateway CLI documentation/help with support, output, UNKNOWN, delivery, and multiplexer-impact boundaries.
- [ ] Run focused new tests through `scripts/run_tests.sh`.
- [ ] Run related existing restart/drain and delivery-ledger regression files through `scripts/run_tests.sh`.
- [ ] Run `python scripts/check` for blocking lint/health checks relevant to the diff.
- [ ] Record test totals, failures, skips, and any environment-only baseline failures.
- [ ] Perform the authorized four-step end-to-end verification: agent command accepted, final reply observed, gateway PID replaced, and the same conversation resumed.
- [ ] Run that E2E only in a disposable checkout-backed canary with isolated home/runtime/lock/HOME/XDG/transport/ports and a small external supervisor; do not modify the production gateway service.

## T6 Review and PR gate
- [ ] Submit the implementation diff and test evidence to Web ChatGPT at highest reasoning for the required review gate.
- [ ] Close every blocking finding with code and test evidence; do not silently absorb review findings.
- [ ] Prepare the proposed PR title, body, and diff summary for user approval before any push, PR edit, close, or public comment.
