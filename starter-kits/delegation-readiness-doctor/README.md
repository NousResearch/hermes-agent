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
- `artifacts/latest-current-gap-report.md` — most recent proof packet emitted by the gap verifier
- `artifacts/latest-broken-state-roundtrip.md` — canonical blocked-state proof packet with before/after doctor output
- `scripts/refresh-upstream-blocker-packet.sh` — one-command PR blocker refresh that distinguishes stale-base drift from pure workflow approval
- `artifacts/latest-upstream-blocker-refresh.md` — canonical live PR blocker packet with state signature + change-vs-previous marker

## Fast start
From the Hermes repo root:

```bash
python -m hermes_cli.main doctor
bash starter-kits/delegation-readiness-doctor/scripts/prove-broken-state-roundtrip.sh
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

## PR handoff monitoring
When the MVP is already open as a PR, use the blocker packet instead of freehand status checks:

```bash
bash starter-kits/delegation-readiness-doctor/scripts/refresh-upstream-blocker-packet.sh
```

Expected behavior:
- prints `UPSTREAM_BLOCKER_PACKET_REFRESHED` only when the live blocker signature actually changed
- prints `UPSTREAM_BLOCKER_PACKET_UNCHANGED` on timestamp-only reruns
- marks stale-base drift explicitly when the PR falls behind `origin/main`
- keeps workflow approval as the blocker only when the branch is current and check suites are still `action_required` with `0` real check runs
