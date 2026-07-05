# Operator Checklist — Producer Normalizer Limited Enforce

Use this checklist daily and weekly to ensure the Limited Enforce is operating
within SLOs and that no anomalies have developed.

## Daily Checklist

Run once per business day (UTC).

### 1. Log integrity

- [ ] `~/.hermes/traces/production_limited_enforce.jsonl` exists and is append-only.
- [ ] Last line is a valid JSON object (not partial).
- [ ] No gap in line offsets (each line is a complete JSON).

### 2. Decision metrics

Run:
```bash
python3 -c "
import sys, json
sys.path.insert(0, '/home/jr-ubuntu/.hermes/hermes-agent')
from pathlib import Path
from agent._limited_enforce_monitor import evaluate_log
r = evaluate_log(Path('/home/jr-ubuntu/.hermes/traces/production_limited_enforce.jsonl'))
print(json.dumps({
    'total_decisions': r.total_decisions,
    'valid_decisions': r.valid_decisions,
    'invalid_decisions': r.invalid_decisions,
    'reviewer_call_saved': r.reviewer_call_saved,
    'reviewer_call_saved_rate': round(r.reviewer_call_saved_rate, 3),
    'verdict_distribution': r.verdict_distribution,
    'gate_status_distribution': r.gate_status_distribution,
    'engine_stop_count': r.engine_stop_count,
    'serialization_errors': r.serialization_errors,
    'secrets_detected': r.secrets_detected,
    'critical_violations': len(r.critical_violations),
    'major_violations': len(r.major_violations),
}, indent=2))
"
```

Confirm:
- [ ] `reviewer_call_saved_rate` is between 0.0 and 1.0.
- [ ] `engine_stop_count` < 3 (else WARNING).
- [ ] `serialization_errors` == 0.
- [ ] `secrets_detected` == 0.
- [ ] `critical_violations` == 0 (else CRITICAL → follow runbook).
- [ ] `major_violations` == 0.

### 3. SLO compliance

- [ ] Monitor latency < 1 s (re-run above; measure wall time).
- [ ] Rollback detection < 5 s (time from CRITICAL alert to recommendation).
- [ ] No rollback events in the last 24 h (else CRITICAL incident postmortem required).

### 4. Decision outcomes

- [ ] Verify distribution is consistent with baseline (no sudden shift).
- [ ] Verify no unexpected verdict (e.g., NOT_RUN outside ENGINE_STOP).

### 5. Sign-off

- [ ] Operator initials.
- [ ] Date (UTC).

If any CRITICAL item fails → follow `runbook.md`.

## Weekly Checklist

Run once per week (Monday 09:00 UTC recommended).

### 1. Trend analysis

Compare current week vs previous week:

- [ ] `reviewer_call_saved_rate` trend (should be stable or increasing).
- [ ] Verdict distribution trend (PARTIAL/PASS/BLOCKED ratios).
- [ ] Engine_STOP trend (should be 0 or decreasing).
- [ ] Log growth rate (linear, no sudden spikes).

### 2. False positive / false negative analysis

False positive = gate skipped reviewer but reviewer (in shadow) would have
rejected. False negative = gate allowed reviewer but reviewer (in shadow)
would have accepted.

Requires shadow-side data collection. If unavailable, skip.

### 3. Latency analysis

- [ ] Gate median latency < 100 ms.
- [ ] Gate p95 latency < 500 ms.
- [ ] Monitor latency < 1 s.

### 4. Capacity

- [ ] Log file size growth rate is reasonable (< 1 GB / week at current load).
- [ ] No accidental duplication (entry count == unique bundle_id count).

### 5. Contract / protocol drift

Verify NO changes in:
- [ ] Reviewer Protocol v1.0.0 (rubric + review_prompt hashes).
- [ ] Producer Normalizer contracts v1.0.0/v1.1.0.
- [ ] Ruleset (normalizer_ruleset.v1.0.0.yaml, normalizer_ruleset.v1.1.0.yaml).

```bash
# Compute current hashes; compare with documented values.
sha256sum tests/contract_tests/normalizer_ruleset.v*.yaml
sha256sum tests/contract_tests/normalizer_config.v*.yaml
# Compare with `~/.hermes/traces/canary_summary.json` field `by_gate_status`.
```

### 6. Re-rollout readiness

If rollback has occurred in the past 7 days:

- [ ] Canonical tests still 157/157 PASS.
- [ ] Canary verifier produces 0 STOP violations.
- [ ] Monitor verifier produces `rollback_recommended=false`.
- [ ] Re-rollout approval is in place.

### 7. Sign-off

- [ ] Operator initials.
- [ ] Date (UTC).
- [ ] Approval to continue production for next week.

## Incident Severity

| Severity | Trigger | Response Time | Action |
| -------- | ------- | ------------- | ------ |
| INFO | Any INFO alert fires | Next business day | Log and monitor |
| WARNING | Any WARNING alert fires | < 4 hours | Investigate |
| CRITICAL | Any CRITICAL alert fires | < 5 minutes | Runbook rollback |

## Reference

- `SOP_Limited_Enforce.md` — full SOP.
- `runbook.md` — rollback procedure.
- `agent/_limited_enforce_monitor.py` — monitor code.
- `agent/_normalizer_gate.py` — gate code.