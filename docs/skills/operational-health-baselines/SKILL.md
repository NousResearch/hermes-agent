# Operational Health Baselines & SLAs

**Status:** DRAFT / NOT ACTIVE
**Created:** 2026-05-18
**Priority:** 1 of 7

## Purpose

Define per-system expected operating parameters, SLA thresholds, and escalation rules so that any Hermes agent (or the Town bridge) can distinguish normal variance from degraded state without operator interpretation.

## Preconditions

- This skill is READ-ONLY reference. It does not create alerts or modify infrastructure.
- All thresholds are initial estimates based on observed behavior through May 2026. They require operator calibration before activation.
- Escalation targets: operator brief (low), operator alert email (medium), Telegram/immediate (high).

---

## System SLA Definitions

### Herald Digest

| Metric | Expected | WARN | ALERT | CRITICAL |
| --- | --- | --- | --- | --- |
| Frequency | Daily (weekdays) | 1 missed day | 2 consecutive missed days | 3+ consecutive missed days |
| Max dark days | 0 | 1 | 3 | 5 (current state: 5+ weeks DARK) |
| Output | deduped + classified JSONL | deduped only (classification failed) | Neither file present | Pipeline unreachable |
| Escalation | None | Log + carry | Operator alert | Operator alert + block dependent jobs |

**Done predicate:** BOTH `data/press_releases/deduped/deduped_{date}.jsonl` AND `data/press_releases/classified/classified_{date}.jsonl` must exist. If classification fails but dedupe exists, next supervisor run retries classification.

### Bellringer

| Metric | Expected | WARN | ALERT |
| --- | --- | --- | --- |
| Sends per week | 5-10 distinct emails | < 5 | < 2 |
| Results emails per week | >= 3 | 1-2 | 0 |
| Reporter slot coverage | ~250 slots per send | < 200 | < 100 |
| Escalation | None | Log | Operator alert |

### Intraday Mover Alerts

| Metric | Expected | WARN | ALERT |
| --- | --- | --- | --- |
| HIGH-tier alerts per week | 40-80 | < 30 | < 15 or > 120 |
| Distribution (weekday) | 8-20 per day | < 5 any day | 0 for any weekday |
| Escalation | None | Log | Operator alert |

### PDUFA / Catalyst Alerts

| Metric | Expected | WARN | ALERT |
| --- | --- | --- | --- |
| Weekly output | >= 1 alert when catalysts are in window | 0 when catalysts exist in 14-day window | 0 for 2+ consecutive weeks with near-term catalysts |
| Escalation | None | Log + verify calendar | Operator alert |

**Note:** Zero alerts is expected when no catalysts are in the near-term window. The SLA applies only when the catalyst calendar shows pending events.

### Morning Briefing

| Metric | Expected | WARN | ALERT |
| --- | --- | --- | --- |
| Frequency | Daily Mon-Fri | 1 missed day | 2 consecutive missed days |
| Delivery time | Before 8:30 AM ET | After 9:00 AM | After 10:00 AM or missing |

### Daily Production Pipeline

| Metric | Expected | WARN | ALERT | CRITICAL |
| --- | --- | --- | --- | --- |
| Cron trigger | 5:30 PM ET weekdays | Late by > 30 min | Missed entirely | 2+ consecutive misses |
| Pipeline duration | < 60 min typical | > 80 min | > 100 min (approaching 6000s timeout) | Timeout kill |
| Step completion | 13/13 steps | 12/13 (non-critical skip) | < 12/13 | < 10/13 |
| Monday AACT | Longest run expected | > 90 min | Approaching timeout | Timeout kill |

### CI Pipeline

| Metric | Expected | WARN | ALERT | CRITICAL |
| --- | --- | --- | --- | --- |
| Status | GREEN | YELLOW (flaky test) | RED for 1-3 days | RED for > 5 days |
| Max red days | 0 | 2 | 5 | 10 (current state: ~10 days as of May 18) |
| Escalation | None | Log | Block non-critical merges | Operator alert + block all merges |

### Agent Fleet (OpenClaw)

| Metric | Expected | WARN | ALERT |
| --- | --- | --- | --- |
| Heartbeat pass rate | 100% of active agents | < 95% (1-2 agents stale) | < 85% (4+ agents stale) |
| Together AI success rate | > 95% | 80-95% | < 80% |
| Together AI avg latency | < 3s | 3-5s | > 5s |
| Stale agents (no heartbeat > 48h) | 0 | 1-2 | 3+ |

### 13F Refresh Cycle

| Metric | Expected | WARN | ALERT |
| --- | --- | --- | --- |
| Filing detection lag | < 24h after SEC posting | 24-48h | > 48h |
| Pre-refresh readiness | 5/5 guards PASS | 4/5 | < 4/5 |
| Quarantine verdict turnaround | < 48h after filing cluster | 48-72h | > 72h |

---

## Composite Health Score (Conceptual)

Not implemented. If activated, a simple weighted-average health score across all systems could be computed. The value is in having a single number that trends over time, not in the specific formula.

---

## Open Questions for Operator Calibration

1. Are the WARN/ALERT/CRITICAL thresholds appropriate, or do they need adjustment based on historical norms?
2. Should CI RED > 5 days actually block all merges, or is the current state (PR #285 open, CI red ~10 days) acceptable during freeze?
3. Is Herald Digest worth restoring during freeze, or is it deferred to post-freeze along with everything else?
4. Should composite health score be a real artifact, or is per-system monitoring sufficient?