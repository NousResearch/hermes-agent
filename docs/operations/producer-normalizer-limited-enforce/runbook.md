# Runbook — Producer Normalizer Limited Enforce

This runbook covers the operational procedure for monitoring and rolling back the
Producer Normalizer Limited Enforce in production.

## When to use this runbook

Use this runbook when:
- An automated CRITICAL alert fires (see SOP_Limited_Enforce.md § "Alerting").
- The monitor's `rollback_recommended` is `true`.
- An operator observes an unexpected behavior in production decisions.

## Pre-flight checks

Before executing any rollback, confirm:

1. The alert is genuine (not a test or false positive).
2. The condition is reproducible (re-run `evaluate_log()` manually).
3. The issue affects production (not a sandbox or test).

## Step 1: Confirm the monitor

```bash
python3 -c "
import sys
sys.path.insert(0, '/home/jr-ubuntu/.hermes/hermes-agent')
from pathlib import Path
from agent._limited_enforce_monitor import evaluate_log
r = evaluate_log(Path('/home/jr-ubuntu/.hermes/traces/production_limited_enforce.jsonl'))
import json
print(json.dumps(r.to_dict(), indent=2, sort_keys=True))
"
```

Inspect:
- `critical_violations` (non-empty = STOP violation, secret, enum violation, etc.).
- `rollback_recommended` (True if critical or major violation).
- `secrets_detected`, `serialization_errors` (counts).

## Step 2: Execute rollback

```bash
python3 -c "
import sys
sys.path.insert(0, '/home/jr-ubuntu/.hermes/hermes-agent')
from pathlib import Path
from datetime import datetime, timezone
from agent._limited_enforce_monitor import evaluate_log, execute_rollback

log = Path('/home/jr-ubuntu/.hermes/traces/production_limited_enforce.jsonl')
report = evaluate_log(log)
if not report.rollback_recommended:
    print('no rollback recommended')
    sys.exit(0)

env_path = Path('/home/jr-ubuntu/.hermes/.env')
result = execute_rollback(env_path, reason=report.rollback_reason)
import json
print(json.dumps(result, indent=2, sort_keys=True))
"
```

Result:
- `LIMITED_ENFORCE=0`
- `SHADOW_MODE=1`

The env file is mutated ONLY for these two keys. Other vars and comments are preserved.

## Step 3: Verify production

After rollback, verify:

```bash
# Confirm env is correctly flipped.
grep -E "HERMES_PRODUCER_NORMALIZER_(LIMITED_ENFORCE|SHADOW_MODE)" ~/.hermes/.env

# Run canary verifier to ensure gate behaves correctly in shadow.
python3 scripts/limited_enforce_canary.py

# Check that no new production_limited_enforce.jsonl entries have
# limited_enforce=True after rollback (gate is now in shadow mode).
tail -5 ~/.hermes/traces/production_limited_enforce.jsonl
```

Expected:
- env shows `LIMITED_ENFORCE=0`, `SHADOW_MODE=1`.
- canary verifier produces `gate_status=SHADOW` entries (not `ENFORCE_LIMITED_BLOCK`).
- new entries have `limited_enforce=false`, `reviewer_should_run=true`.

## Step 4: Open incident

Create an incident ticket with:
- Timestamp (UTC).
- Operator (the `whoami` of who executed the rollback).
- Trigger (which CRITICAL alert fired).
- Affected decisions count (from `report.total_decisions`).
- Rollback reason (from `report.rollback_reason`).
- Pre-rollback verdict distribution (from `report.verdict_distribution`).
- Pre-rollback gate_status distribution.

## Step 5: Preserve evidence

Save the following files to the incident ticket:

- `~/.hermes/traces/production_limited_enforce.jsonl` (last 100 lines).
- `~/.hermes/traces/decision.jsonl` (last 100 lines).
- The monitor report (JSON dump).
- The rollback executor output (JSON dump).

Do NOT modify these files before preservation.

## Post-rollback procedure

After rollback:

1. Identify the root cause.
2. Classify the incident (data corruption, code bug, environment misconfig, etc.).
3. Correct the issue in a feature branch.
4. Run canonical tests (must reach 157/157 PASS).
5. Run canary verifier (must produce 0 STOP violations).
6. Run monitor verifier (must produce `rollback_recommended=false`).
7. Request re-rollout via `HERMES_PRODUCER_NORMALIZER_LIMITED_ENFORCE_RE_ROLLOUT_APPROVAL`.

Re-rollout requires:
- Canonical tests PASS.
- Monitor PASS.
- Canary PASS.
- Zero STOP violations.
- Explicit operator approval.

DO NOT promote directly back to production after rollback.

## Escalation

If rollback fails or has unintended consequences:

1. Stop the pipeline (HERMES_PRODUCER_NORMALIZER_ENABLED=0).
2. Contact the on-call engineer.
3. Open a SEV1 incident.
4. Preserve all evidence before any further changes.

## Recovery time

Target recovery time: < 5 minutes from CRITICAL alert to LIMITED_ENFORCE=0.

Measure:
- `alert_time` = first CRITICAL alert timestamp.
- `rollback_complete_time` = execute_rollback return timestamp.
- `recovery_time_seconds` = `rollback_complete_time - alert_time`.

If recovery time > 5 minutes, open a postmortem.

## Communication

Notify the team via:
- Internal incident channel (Telegram/Discord/Slack).
- Pin to the relevant topic/thread.
- Include: timestamp, rollback reason, expected recovery time.

After recovery, send a follow-up with:
- Confirmation of rollback.
- Affected decision count.
- Next steps.