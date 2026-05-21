# Status — Plan 004: Self-Improvement Service (Hermes-internal module)

**Status:** APPROVED, NOT STARTED
**Last updated:** 2026-05-20
**Blocked by:** None — all prerequisites complete (Plan 001-A/B/C/D + Plan 002 + Plan 003 + atlas-012)

## Phase Progress

| Phase | Title | Status | Notes |
|---|---|---|---|
| 004-A | Skills dashboard — read-side first (capture + display) | Not started | 1.5w. Slack reaction → Neon writer + per-skill score aggregation + `/skills` dashboard. Foundation for B/C/D |
| 004-B | Auto-suggest + Blake-approve promotion | Not started | 1.5w. Daily job surfaces promotion candidates; Blake clicks approve → Skills Service promote_skill call. Bidirectional (demote on regression) |
| 004-C | Drift / regression alerts | Not started | 1w. Watch thumbs_rate over 14d rolling; alert on baseline drop |
| 004-D | Skill recommendations — LLM-driven gap analysis | Not started | 2-3w. Weekly LLM cluster analysis; generates SKILL.md drafts |

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
