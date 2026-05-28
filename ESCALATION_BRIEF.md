# ESCALATION BRIEF — Full Context Review
# Date: 2026-06-02 | Status: Active Dev Sprint
# Escalation target: DeepSeek v4 (draft) → Claude (review) → Ring (gate)

## 1. Where We Are

The Hermes Agent context management subsystem has been under active development. The sprint started after foundational board review approval (BOARD_REVIEW_FOUNDATIONAL_2026-06-02 marked APPROVED).

### Completed Work
- **PR #32934** open: gateway integration wiring + 4 recovered source files
- **Trimmer tuning**: Threshold lowered 100K→80K, target 60K→40K, budget raised 12K→14K, post-trim lowered 6K→5K
- **Memory resilience**: Aggressive 48h compaction of low-importance episodes, WAL checkpoint hardening, auto-prune tuning
- **Auto-sculpt system**: Automatic context reset at 20-turn mark or under 90% token pressure, with performance review
- **Session watchdog**: Cron job fires every 5 min to detect stale heartbeat (dead man's switch)
- **Cascade protection**: All write paths (store_episode, store_fact, set_working) now wrapped in try/except with graceful degradation when DB is full

## 2. What We're Seeing (Problems)

### Problem 1: Memory Palace DB Overflow Cascade
- **Root cause**: DB hits 500KB hard cap → writes fail → error handler tries to log/write → cascading failure kills all tool operations
- **Observed**: errors.log shows memory at 1,516/2,200 chars attempting 1,021 char write; then background review tools denied; then skill_manage failures; finally terminal operation failures
- **Status**: FIXED — _check_capacity() pre-write guard added, all write paths wrapped in try/except

### Problem 2: Context Window Degradation
- **Root cause**: After ~20 turns, model coherence degrades even with trimmed context
- **Mechanism**: Turn-by-turn context drift accumulates; identity blocks consume 4-8K leaving only 4-8K for dialogue
- **Status**: FIXED — Auto-sculpt system triggers at 20 turns, persists T0-T3 to memory palace, rebuilds fresh context

### Problem 3: Trimmer Threshold Mismatch
- **Root cause**: Original thresholds (100K trigger, 60K target) too conservative for 12-14K budget system
- **Status**: FIXED — New values (80K trigger, 40K target, 14K budget, 5K post-trim)

### Problem 4: Gateway Integration Wiring
- **Root cause**: 4 source files (context_orchestrator, gateway_integration, memory_palace, consult_merge) deployed but never committed
- **Status**: FIXED — All recovered and committed

## 3. What Has Been Done

### Code Changes (all committed to feat/gateway-integration-wiring)
1. `scripts/auto_trim.py` — TRIM_THRESHOLD_TOKENS: 100000→80000, TARGET_TOKENS: 60000→40000
2. `scripts/context_orchestrator.py` — BUDGET_TOKENS: 12000→14000, TARGET_POST_TRIM: 6000→5000, WARNING_TOKENS: 9000→10000
   - Added: turn counter, auto-sculpt trigger, performance review, tuning observation log
3. `scripts/memory_palace.py` — Added _check_capacity() pre-write guard, try/except on all write paths, auto_prune aggressive 48h compaction
4. `DECISION-FRAMEWORK.md` — Cleaned duplicate Decision 5b block
5. `BOARD_REVIEW_FOUNDATIONAL_2026-06-02.md` — Flipped to APPROVED

### Infrastructure
- Session watchdog cron job (every 5 min)
- Heartbeat script and update mechanism
- PR #32934 updated with full change summary

## 4. Open Questions

1. Should the auto-sculpt turn limit (20) be configurable per-platform? Mobile may need lower, desktop higher.
2. The gateway/platforms/base.py pre-existing diff defers trim check — needs review for re-entrant recursion safety.
3. Claude API key not yet configured — board review chain incomplete (DeepSeek ✅ Grok ✅ Ring ✅ Kimi ⚠️ Claude ❌)

## 5. Recommendations

1. **Immediate**: Run integration test with the new tuning values under simulated load (20+ turns)
2. **Short-term**: Add telemetry endpoint for tuning metrics (turns-to-sculpt, trim frequency, DB size over time)
3. **Medium-term**: Replace the 2K injected context summary with checkpoint-based recovery (load from memory palace on restart instead of full state reconstruction)
4. **Ongoing**: Monitor the tuning_log entries across real sessions to calibrate thresholds empirically

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DB overflow cascade | LOW (fixed) | HIGH | _check_capacity guard |
| Memory corruption on crash | MEDIUM | MEDIUM | WAL mode + auto_prune |
| Sculpt loses important context | LOW | MEDIUM | T0-T3 always persisted before reset |
| Tuning too aggressive | MEDIUM | LOW | Parameters are env-overridable, auto-review catches it |
| Gateway base.py conflict | LOW | HIGH | Needs explicit review before merge |