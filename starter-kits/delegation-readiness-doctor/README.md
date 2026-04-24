# Delegation Readiness Doctor

Turn Hermes delegation from an assumed capability into a provable readiness surface.

## Outcome
narrow week-one claim: confirm the live delegation-readiness gap, then replace it with one canonical audit-to-fix path that ends in a successful delegated run.

## Who this is for
- Hermes operators
- AI builders using subagents for parallel work
- anyone who loses leverage because delegation looks available until it silently fails

## Current live wedge
The original wedge was a stubbed delegation readiness check. That gap is now closed in live code:
- `tools/delegate_tool.py` now implements `check_delegate_requirements()` via a real config-aware readiness check
- `hermes_cli/doctor.py` now exposes a canonical `◆ Delegation Readiness` section

This starter kit now packages the proof line, not just the kickoff gap, so the ship claim stays honest after implementation lands.

## Canonical proof path
1. treat `scripts/verify-current-gap.sh` as the historical kickoff verifier for the original stubbed-check gap
2. run `python -m hermes_cli.main doctor` and confirm `◆ Delegation Readiness` reports the live state
3. run `scripts/prove-broken-state-roundtrip.sh` to prove blocked → ready from an isolated temporary `HERMES_HOME`
4. run one real `delegate_task` proof from the live ready environment
5. freeze the ship/package decision in a durable artifact

## Folder layout
- `scripts/verify-current-gap.sh` — current-gap verifier that writes a durable report
- `scripts/prove-broken-state-roundtrip.sh` — isolated blocked→ready doctor roundtrip proof that leaves the real `~/.hermes/config.yaml` untouched
- `scripts/emit-pr-review-monitor.sh` — live GitHub PR monitor for the clean upstream handoff surface
- `scripts/emit-ci-result-interpreter.sh` — fail-closed first-CI interpreter that maps the first real check-run result back to the clean local proof line
- `scripts/emit-workflow-approval-brief.sh` — blocker-specific brief for the current fork-workflow approval gate
- `scripts/emit-workflow-approval-state-change.sh` — state-change detector; surfaces blocker-clear transitions vs. persist states so automation knows when maintainer approval happened without manual snapshot comparison
- `scripts/emit-workflow-approval-trigger.sh` — posting-state-aware nudge/approval packet for the repeated fork-workflow approval stall; prints `WORKFLOW_APPROVAL_TRIGGER_ALREADY_POSTED` when the maintainer request is already live so automation does not mistake a reference-only packet for a fresh action
- `scripts/sync-reviewer-handoff-baseline.sh` — keeps `latest-reviewer-handoff.md` aligned to the live PR head/base before state-change detection; polls GitHub mergeability before writing so the handoff does not regress to first-response `mergeability unknown` noise
- `scripts/refresh-upstream-blocker-packet.sh` — one-command refresh that syncs the reviewer handoff, reruns the state-change detector, PR monitor, CI interpreter, and approval trigger together, then emits a consolidated blocker packet from the same live PR state; prints `UPSTREAM_BLOCKER_PACKET_UNCHANGED` when the blocker signature is materially identical to the previous latest packet so cron can distinguish revalidation from a real transition; unchanged runs restore prior `latest-*` files and delete just-created timestamped component artifacts so approval-wait cron passes do not dirty the workspace with no-movement files
- `scripts/verify-unchanged-refresh-hygiene.sh` — proof harness for the external-wait loop breaker; snapshots canonical `latest-*` hashes and timestamped artifact names, runs the one-command refresh, and proves an unchanged blocker refresh leaves no local artifact churn behind
- `scripts/validate-artifact-consistency.sh` — fail-closed consistency check that requires every canonical blocker artifact to record the same live head/base pair before the packet is trusted
- `artifacts/latest-current-gap-report.md` — most recent proof packet emitted by the gap verifier
- `artifacts/latest-broken-state-roundtrip.md` — canonical blocked-state proof packet with before/after doctor output
- `artifacts/latest-pr-review-monitor.md` — canonical live review/merge monitor for PR `#14297`
- `artifacts/latest-ci-result-interpreter.md` — first-CI decision surface that fail-closes until real check runs exist, then routes pass/fail signals back to the proof artifacts
- `artifacts/latest-workflow-approval-brief.md` — exact maintainer action surface when GitHub Actions suites stay `action_required` with `0` check runs
- `artifacts/latest-workflow-approval-state-change.md` — state-change detector; records previous vs current action_required suite count and check-run count so automation can detect when the blocker cleared vs still persists
- `artifacts/latest-workflow-approval-trigger.md` — posting-state-aware maintainer nudge/reference packet + exact approval/recheck surfaces for the current head SHA
- `artifacts/latest-reviewer-handoff.md` — blocker-specific maintainer brief that maps the PR diff to proof, verification commands, and merge criteria
- `artifacts/latest-upstream-blocker-refresh.md` — consolidated one-command blocker packet used by recurring momentum blocks to decide whether the blocker changed, stayed external, or needs current-base refresh

## Fast start
From the Hermes repo root:

```bash
python -m hermes_cli.main doctor
bash starter-kits/delegation-readiness-doctor/scripts/prove-broken-state-roundtrip.sh
bash starter-kits/delegation-readiness-doctor/scripts/emit-pr-review-monitor.sh
bash starter-kits/delegation-readiness-doctor/scripts/emit-ci-result-interpreter.sh
bash starter-kits/delegation-readiness-doctor/scripts/emit-workflow-approval-brief.sh
bash starter-kits/delegation-readiness-doctor/scripts/emit-workflow-approval-state-change.sh
bash starter-kits/delegation-readiness-doctor/scripts/emit-workflow-approval-trigger.sh
bash starter-kits/delegation-readiness-doctor/scripts/refresh-upstream-blocker-packet.sh
bash starter-kits/delegation-readiness-doctor/scripts/verify-unchanged-refresh-hygiene.sh
```

Historical kickoff verifier:

```bash
bash starter-kits/delegation-readiness-doctor/scripts/verify-current-gap.sh
```

Expected results right now:
- `python -m hermes_cli.main doctor` includes `◆ Delegation Readiness`
- the roundtrip verifier exits successfully, writes a timestamped markdown report under `starter-kits/delegation-readiness-doctor/artifacts/`, and prints `BROKEN_STATE_ROUNDTRIP_PROVED`
- the historical gap verifier now exits non-zero because the original unconditional-stub gap is no longer present

That failure is honest evidence that the MVP moved past the kickoff gap and should now be judged on the readiness + roundtrip + delegated-run proof line.

## Honest Monday artifact freeze
What is now frozen on disk:
- the product thesis and scope
- the exact live blocker to attack first
- one executable verifier entrypoint that proves the blocker is real

## Week-one non-goals
- full delegation UX redesign
- multi-provider credential orchestration cleanup
- dashboard/control-plane expansion
- claiming delegation is fixed before one real delegated run passes
