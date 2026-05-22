# HOT-tier Memory — DEM Fleet (Wake Robin Biotech Screener)
# Cap: 100 lines. Loaded every session. Do not exceed.
# Last updated: 2026-05-22
# Source repo: C:\Projects\biotech_screener\biotech-screener\

---

## System Identity

- **Project**: Wake Robin Capital — institutional biotech investment screener
- **Decision Engine**: Two-stage selector/ranker → EW Top-30 output
- **Production pipeline**: 13-step daily cron, 5:30 PM ET
- **Governance**: CCFT (Canonical, Complete, Frozen, Timestamped). All outputs deterministic.

---

## Active Ruleset

- **Ruleset**: v1.14.0 (`8887576e`) — coinvest-only selector, pairwise_minimal ranker
- **Sort anchor**: selector_score (coinvest_score_z 100%)
- **inst_delta_z**: ZEROED in selector since 2026-05-04 (mean_ic=-0.097, two-frame confirmed)
- **Model fleet**: `deepseek/deepseek-v4-flash:free` (migrated 2026-05-20, all 27 active agents)
- **Pinned in**: `run_screen.py` AND `run_phase2_snapshot_delta.py` (must stay in sync)

---

## Architecture Freeze

- **Status**: ACTIVE through ~2026-05-26 (h20d checkpoint)
- **h20d decision**: May 26. Evidence collected. All 5 Path A conditions met as of 2026-05-22.
- **Expected outcome**: Path A (freeze lift), June 1 KG deployment + ranker shadow start
- **No changes allowed**: no enforcement logic, scoring, or ranking changes until freeze lifts

---

## 13F Cohort Status

- **Q1 2026**: CLEARED as of 2026-05-19 (Jaccard 0.875, 42/48 managers, all gates PASS)
- **Priority firms**: Fairmount, Deep Track, Logos — all filed 2026-05-15
- **Key positions**: VRDN (FM 14.04% + DT 5.30%), ORKA coinvest (FM + DT)
- **Next cycle**: Q2 2026, deadline ~2026-08-14

---

## Active Blockers / Held Specs

- **Spec 100** (ranker IC tooling): blocked by architecture freeze, implement post-May-26
- **Spec 095** (IC scope bug): CURRENT_TOOLS_CONFLATED — all composite_score IC claims invalid
- **score_rank_pct**: SPEC_REQUIRED — WARN streak (mean_ic=-0.0119, hit_rate=28.95%)
- **Spec 087 B2**: UNBLOCKED (B1b formally closed 2026-05-14), dashboard envelope ready to draft
- **Spec 088 Phase B**: HELD — pending Spec 087 full closure
- **KG Phase 2 Step 5**: blocked until freeze lifts (June 1 target)

---

## Governance Rules (Always Active)

- North star: backtests produce evidence only — never change production weights without governance
- No agent may modify production weights without full multi-gate promotion path
- Tier 4 changes (architecture, signal promotion) require memo + human approval
- Town-Hermes Feedback Protocol: FROZEN until after h20d (2026-05-26)
- Only `crt_resolution_watcher` holds `mutate_data` authority
- Lane A agents must not depend on LLM gateway tokens
- PIT rule: never call set "true PIT" unless archived raw inputs + archived code + archived artifacts all exist

---

## Forward Monitor

- Accumulating since 2026-04-03. ~35+ trading days as of 2026-05-22.
- coinvest_score_z pooled mean IC = -0.031 (14 dates, 28.6% hit rate) — OBSERVE verdict
- Ranker IC: UNMEASURED (Spec 095 scope bug; blocked until Spec 100)
- Post-13F refresh IC decomposition: pending (gate: quarantine cleared ✓)
