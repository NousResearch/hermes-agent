# SOP — Producer Normalizer Limited Enforce (v1.0)

**Status**: PRODUCTION
**Owner**: Engineering
**Last updated**: 2026-06-28

## Scope

Standard Operating Procedure for daily operation of Producer Normalizer Limited Enforce on
the Producer → Normalizer v1.1 → Reviewer pipeline.

This SOP covers:
- Daily + weekly checks (operator_checklist.md).
- Runbook for rollback (runbook.md).
- SLOs and alerting thresholds.
- Audit requirements.

It does NOT cover:
- Code-level design (see agent/_limited_enforce_monitor.py, agent/_normalizer_gate.py).
- Promotion / canary flow (see previous approvals).
- Reviewer Protocol v1.0.0 or Producer Normalizer contracts.

## Architecture

```
Producer (MiniMax-M3)
    ↓
Producer Normalizer v1.1 (Limited Enforce gate)
    ↓ (verdict ∈ {BLOCKED, NO_EVIDENCE})
Skip reviewer (reviewer_call_saved=true)
    ↓ (verdict ∈ {PARTIAL, PASS})
Reviewer (openai-codex + gpt-5.5)

The gate is at `agent/_normalizer_gate.py::evaluate()` with the
following policy:

| normalizer_verdict | gate_status | reviewer_should_run |
| ------------------ | ----------- | ------------------- |
| NO_EVIDENCE | ENFORCE_LIMITED_BLOCK | False |
| BLOCKED | ENFORCE_LIMITED_BLOCK | False |
| PARTIAL | ENFORCE_REVIEW | True |
| PASS | ENFORCE_REVIEW | True |
| ENGINE_STOP | ENGINE_STOP | True (fail-open) |
```

The monitor (`agent/_limited_enforce_monitor.py::evaluate_log`) reads
`production_limited_enforce.jsonl` and reports violations.

The rollback executor (`execute_rollback()`) flips:
- `HERMES_PRODUCER_NORMALIZER_LIMITED_ENFORCE=1` → `0`
- `HERMES_PRODUCER_NORMALIZER_SHADOW_MODE=0` → `1`

## Configuration

Production env (set in `~/.hermes/.env`):

```
HERMES_PRODUCER_NORMALIZER_ENABLED=1
HERMES_PRODUCER_NORMALIZER_VERSION=1.1.0
HERMES_PRODUCER_NORMALIZER_LIMITED_ENFORCE=1
HERMES_PRODUCER_NORMALIZER_SHADOW_MODE=0
```

Logs:
- `~/.hermes/traces/production_limited_enforce.jsonl` — production decisions (append-only).
- `~/.hermes/traces/decision.jsonl` — main shadow log (append-only, untouched by limited enforce).

## SLOs

### Monitor SLOs

| Metric | Target | Critical threshold |
| ------ | ------ | ------------------ |
| monitor_latency | < 1 s | > 5 s |
| rollback_detection | < 5 s | > 30 s |
| rollback_execution | < 30 s | > 120 s |

### Quality SLOs

| Metric | Target |
| ------ | ------ |
| STOP violations | 0 |
| secrets_detected | 0 |
| serialization_errors | 0 |
| false_positive_rate | < 1% |
| false_negative_rate | < 5% |

### Availability SLOs

| Metric | Target |
| ------ | ------ |
| gate uptime | > 99% |
| monitor uptime | > 99% |

## Alerting

### INFO

- `reviewer_call_saved_rate` changes > 20% vs baseline.
- Verdict distribution shifts by > 10 percentage points.

### WARNING

- ENGINE_STOP count >= 2 in 24 h.
- Sustained PARTIAL increase.
- Reviewer latency increasing (p95 growing > 20% over 7 days).

### CRITICAL (auto-rollback)

- STOP violation (PASS/PARTIAL/ENGINE_STOP with reviewer_should_run=False).
- Secret detected in logs (Bearer/sk-/eyJ/xox[abprs]-).
- Log corruption (JSON parse error).
- Invalid gate_status (not in closed enum).
- Enum violation (limited_enforce_reason not in closed enum).
- Hash inconsistency.
- Missing log file.

## Audit trail

Each rollback event is recorded with:
- timestamp
- operator (from env: USER / USERNAME)
- version (HERMES_PRODUCER_NORMALIZER_VERSION)
- rollback_reason
- rollback_duration_ms
- affected_decisions (count of decisions during the incident window)
- recovery_time_seconds
- corrective_action (operator-provided)

The audit trail is appended to `~/.hermes/traces/rollback_audit.jsonl`.

## References

- `runbook.md` — Detailed rollback procedure.
- `operator_checklist.md` — Daily + weekly checklists.
- `agent/_limited_enforce_monitor.py` — Monitor implementation.
- `agent/_normalizer_gate.py` — Gate implementation.
- `agent/_normalizer_v1_1_impl.py` — Normalizer v1.1 implementation.