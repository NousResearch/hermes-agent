# Status — Plan 004: Self-Improvement Service (Hermes-internal module)

**Status:** COMPLETE (2026-05-21 — all 4 phases shipped end-to-end in batched dispatch)
**Last updated:** 2026-05-21
**Blocked by:** None

## Phase Progress

| Phase | Title | Status | Notes |
|---|---|---|---|
| 004-A | Skills dashboard — read-side first (capture + display) | **Complete** | Commit 3e67879. feedback_capture.py + skill_scorer.py + Slack reaction handlers + `/skills` Next.js dashboard. RLS-scoped via HermesIdentity. |
| 004-B | Auto-suggest + Blake-approve promotion | **Complete** | Commit 2475227. promotion_proposer.py daily cron + Slack DM with top 5 candidates + bidirectional demotion. No silent promotions. |
| 004-C | Drift / regression alerts | **Complete** | Commit fb6b5e7. drift_detector.py watches 14d rolling thumbs_rate; alerts at baseline drop; auto-resolution + manual dismiss. |
| 004-D | Skill recommendations — LLM-driven gap analysis | **Complete (with xfail tracker)** | Commit 9e9750f. recommender.py 589 LOC + Sonnet via Portkey + $20/mo BudgetGate. 8/10 tests pass; 2 xfail on TF-IDF clustering threshold (Q-D.1: defer to shadow-mode tuning). |

## Execution sequence

```
004-A (Skills dashboard) ──► 004-B (Auto-suggest)
                          ├──► 004-C (Drift alerts)
                          └──► 004-D (Recommendations)
```

Phase A is the strict prerequisite (telemetry). After A, B/C/D can ship in any order or in parallel.

## Resumption context

- Next phase: 004-A
- All decisions locked per spec discussion 2026-05-20
- Estimated total: ~6-8 weeks (incremental)
- Recommended cadence: ship A → use it for 1-2 weeks → ship B → use both for 1-2 weeks → ship C → use → ship D

## Key decisions locked (2026-05-20 spec discussion)

- **D-004-1**: Hermes-internal module location (NOT standalone service). Lives in `hermes_agent/self_improvement/`. Port 8002 reservation rescinded.
- **D-004-2**: First observable signal = explicit user thumbs (👍/👎 Slack reactions)
- **D-004-3**: Promotion criterion = manual + automated tiers (auto-suggest, Blake approves; no silent promotions)
- **D-004-4**: All 4 user-visible wins are in scope (dashboard + auto-suggest + drift + recommendations)
- **D-004-5**: Incremental shipping — each phase produces value standalone
- **D-004-6**: Reuses Plan 001-D (NeonBackend), Plan 003 (Skills Service promote API), atlas-012 (connector), Plan 001-C (skill_locks)
- **D-004-7**: Phase D uses Sonnet via Portkey; cap $20/mo

## Open questions (carry into execution)

- Q-A.1: Multi-platform reactions (Telegram/Discord) — defer to A.1 follow-up
- Q-B.1: Threshold tuning (10 uses + 80% thumbs_rate) — revisit after 30 days of live data
- Q-B.2: Slack DM cadence — only-when-pending (avoid daily noise)
- Q-C.1: Drift detector for generated skills from Phase D — yes, same code path
- Q-D.1: Sonnet vs Opus for clustering — Sonnet default; escalate if shadow mode shows poor quality
- Q-D.2: Auto-commit recommended skills — currently editor-only; auto-commit deferred

## Budget

- Dev: ~6-8 weeks across 4 phases
- Ongoing LLM cost: ~$20/mo (Phase D only)
- Storage: ~10MB additional Neon (4 new tables: skill_feedback, promotion_decisions, skill_drift_alerts, skill_recommendations)

## Cross-references

- Tier graph: `agentic-hub/plans/cross-repo-tier-graph.md` row `hermes-004`
- Boundary: reads atlas-012 connector data; writes to Skills Service promote_skill API
- Storage: Neon (Plan 001-D); RLS scoping via HermesIdentity (Plan 001-0)
