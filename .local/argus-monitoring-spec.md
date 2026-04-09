# ARGUS Monitoring Services Spec

## Scope
High and medium value daemon monitoring services not currently covered by hermes.
Feature-level planning — no implementation code. For prioritization and recall.

---

## HIGH VALUE

### 1. Resource Exhaustion Detection

**What:** Monitor disk, DB size, session store growth. Alert before silent failures.

**Features:**
- Disk free space on work directories (`~/Projects`, `~/.hermes`)
- DB file size tracking (state.db, argus.db, holographic_memory.db)
- Session store bloat detection (count, oldest age, total file size)
- Rolling trend analysis (growing fast vs stable)
- Tiered alerts: warning at 80%, critical at 95%

**Why first:** Prevents the most common silent failure mode. Disk fills → DB corrupts → agent crashes with no signal.

---

### 2. API Provider Health

**What:** Per-provider, per-model error rate tracking. Detect outages before users notice.

**Features:**
- HTTP error rate per provider (nous, openrouter, anthropic, etc.)
- Rate limit (429) detection — immediate critical alert
- Timeout rate tracking (> 30% in last 20 calls)
- Latency degradation (p95 vs baseline)
- Model-not-found / invalid-key detection
- Provider outage correlation (same errors across multiple sessions = provider issue, not session issue)

**Why:** Most common user-facing problem. Currently invisible until sessions start failing.

---

### 3. Dead Session Cleanup

**What:** Find and mark orphaned sessions that should have ended.

**Features:**
- Sessions started > 2h ago with no messages in 30min
- Delegate sessions with no parent
- Cron sessions referencing deleted jobs
- Stale PID files for processes that died
- Periodic sweep (every 10 poll cycles, not every cycle)
- Mark as 'orphaned' with reason, don't delete (audit trail)

**Why:** DB hygiene. Orphaned sessions accumulate, pollute entropy detection, waste resources.

---

### 4. Config Drift Detection

**What:** Detect when config/credentials/skills change at runtime.

**Features:**
- Track mtime + hash of: config.yaml, .env, skills/, directives.yaml, watcher_schema.sql
- Compare between poll cycles
- Alert on change with old→new hash
- Differentiate: config change (new sessions affected), credential rotation (may need restart), skill update (new skills available)
- Does NOT restart — running sessions use their loaded config

**Why:** Catches misconfiguration. User edits config.yaml, doesn't realize running sessions are stale.

---

## MEDIUM VALUE

### 5. Cron Chain Detection

**What:** Detect when dependent cron jobs break their data pipeline.

**Features:**
- Parse cron job prompts for file path references
- Build dependency graph: job A writes path X, job B reads path X
- Detect when downstream job runs with stale upstream data
- Alert on broken chains, not just individual job failures

**Why:** Complex workflows (swarms, batch runners) break silently when one link fails.

---

### 6. Cascade Monitoring

**What:** Correlate failures across components. Alert on root cause, not symptoms.

**Features:**
- Gateway down → all platform adapters disconnected
- Provider outage → all sessions hitting that provider fail simultaneously
- DB locked → entropy detection stalls → decisions delayed
- WAL monitor thread dead → real-time entropy detection lost
- Correlate by timestamp window: if 3+ sessions fail within 60s, likely systemic

**Why:** Individual session failures are noisy. Cascade detection surfaces the real problem.

---

### 7. Predictive Budget

**What:** Project token spend. Alert before exhaustion, not after.

**Features:**
- Token burn rate per session (input + output tokens / time)
- Projected daily cost based on current rate
- Cron cycle cost trend (increasing = inefficient)
- Context compression frequency (high = expensive model)
- Cost per quality point (increasing = diminishing returns)
- Rolling averages: 1h, 6h, 24h windows

**Why:** Cost control. Users discover budget exhaustion after the fact.

---

### 8. Audit Trail

**What:** Immutable append-only log of all ARGUS decisions.

**Features:**
- Single audit table, INSERT-only (no UPDATE, no DELETE)
- Every restart/kill/inject decision recorded with full context
- Includes: entropy detections, directive checks, metrics at decision time
- Queryable by session, action type, time range
- Exportable (JSONL) for external analysis
- Foundation for all other monitoring — everything writes to audit

**Why:** Debugging. When something goes wrong, need to reconstruct what ARGUS decided and why.

---

## Implementation Phases

```
Phase 1 (foundational):
  1. Resource exhaustion — prevents silent failures
  4. Config drift — catches misconfiguration
  8. Audit trail — foundation for everything else

Phase 2 (after phase 1 data accumulates):
  3. Dead session cleanup — DB hygiene
  2. API provider health — user-facing reliability

Phase 3 (needs data from phases 1-2):
  7. Predictive budget — needs token history
  6. Cascade monitoring — needs failure correlation data
  5. Cron chain detection — needs prompt parsing
```

## Target Module Layout

```
argus/
  argus.py           — orchestrator, main loop
  actions.py         — restart/kill/inject (done)
  entropy.py         — entropy detection (done)
  directives.py      — directive checks (done)
  notifications.py   — multi-platform alerts (done)
  resources.py       — resource exhaustion
  drift.py           — config drift detection
  audit.py           — append-only audit trail
  provider_health.py — API provider monitoring
  cleanup.py         — dead session cleanup
  budget.py          — predictive token spend
  cascade.py         — cascade detection
  cron_chains.py     — cron dependency graph
  hermes_fallback.py — subprocess stubs (done)
  wal_monitor.py     — real-time tool call monitoring (existing)
```
