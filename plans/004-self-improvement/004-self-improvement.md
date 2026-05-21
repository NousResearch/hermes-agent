# Plan 004 — Self-Improvement Service (Hermes-internal module)

**Author:** Blake + Claude Opus 4.7 · **Date:** 2026-05-20 · **Status:** APPROVED, not yet executed

## Context

Hermes runs agents that produce outputs (drafts, replies, summaries, decisions). Today nothing measures whether those outputs are useful — skills get used or ignored without telemetry, good skills don't surface as templates, bad skills don't get retired. Plan 004 closes that loop.

Three upstream plans make 004 viable now:
- **Plan 003 — Skills Service**: scoped registry with promote_skill API (CSS resolution + DDB locks)
- **Plan 002 — Self-organization**: directory layout + session isolation
- **atlas-012 — Hermes↔Atlas connector**: per-message metadata in Neon (`messages.metadata` JSONB)

Plan 004 builds a self-improvement module **inside hermes-agent** (not a standalone service) that:
1. Captures explicit user feedback (👍/👎 Slack reactions) on Hermes outputs
2. Tracks per-skill quality + usage telemetry
3. Auto-suggests skill promotions (Blake approves)
4. Detects regression / drift in previously-good skills
5. Recommends new skills based on observed task patterns

### Why Hermes-internal (not a standalone service)?

Per spec discussion 2026-05-20:
- Tightest integration with the runtime that actually generates outputs
- Reads atlas-012 connector data directly (no extra network hop)
- Writes promotion proposals to Skills Service via the existing promote_skill API
- Avoids the "another service to deploy + monitor" overhead

### Boundary with other components

- **Atlas memory** stores the raw feedback events. Self-Improvement reads them.
- **Skills Service** owns skill content + promotion mechanics. Self-Improvement triggers promotions; doesn't store skills.
- **Hermes runtime** generates outputs + hooks into reaction capture. No new gateway needed.

### Explicit non-additions

- ❌ No standalone service (port 8002 NOT reserved; that decision is rescinded)
- ❌ No multi-tenant skill ranking (single-user; per-tenant when SaaS expands)
- ❌ No automated promotion without Blake review (manual approval gate stays)
- ❌ No new database — uses Neon (Plan 001-D) + existing atlas-012 connector tables

## Critical files

- `hermes_agent/self_improvement/__init__.py` — NEW: module package
- `hermes_agent/self_improvement/feedback_capture.py` — NEW: Slack reaction listener → Neon writer
- `hermes_agent/self_improvement/skill_scorer.py` — NEW: per-skill usage + thumbs aggregation
- `hermes_agent/self_improvement/promotion_proposer.py` — NEW: daily job suggesting promotions
- `hermes_agent/self_improvement/drift_detector.py` — NEW: regression alert engine
- `hermes_agent/self_improvement/recommender.py` — NEW: LLM-driven skill-gap analysis (Phase D)
- `gateway/platforms/slack.py` — extend: wire reaction events to feedback_capture
- `web/app/skills/page.tsx` — NEW Next.js route: per-skill score dashboard
- `migrations/002_skill_feedback.sql` — NEW: skill_feedback + skill_scores tables in Neon

## Existing utilities to reuse (do not reinvent)

- `hermes_storage/neon_backend.py` — NeonBackend with RLS context manager (Plan 001-D)
- `agent/skill_utils.py` — RegistryEntry + scope resolution (Plan 003)
- `tools/skill_manager_tool.py::_promote_skill` — promotion API (Plan 003)
- `tools/skill_locks.py` — distributed locks for promotion writes (Plan 001-C)
- `atlas-012` connector — query `/v1/memory/hermes` for historical feedback events

## Phase Index

| Phase | Title | Effort | Risk | Priority | Status |
|---|---|---|---|---|---|
| 004-A | Skills dashboard — read-side first (capture + display) | 1.5w | Low | P0 | Not started |
| 004-B | Auto-suggest + Blake-approve promotion | 1.5w | Medium | P0 | Not started |
| 004-C | Drift / regression alerts | 1w | Low | P1 | Not started |
| 004-D | Skill recommendations — LLM-driven gap analysis | 2-3w | High | P2 | Not started |

**Total: ~6-8 weeks. Ships incrementally — each phase produces user-visible value.**

## Phases at a glance

```
004-A (Skills dashboard) ──► 004-B (Auto-suggest) ──► 004-C (Drift alerts) ──► 004-D (Recommendations)
       (read-side)               (write-side)              (regression)              (gap analysis)
```

Linear dependency: each phase builds on the prior's telemetry foundation.

---

## Phase 004-A — Skills dashboard (read-side first)

### Motivation

Before automating promotion decisions, we need observability: what skills exist, how often they're used, how users react. Blake's first observable win: "I can see which skills actually work." Pure read-side — no automation, no behavioral changes to the agent runtime.

### What

A new module `hermes_agent/self_improvement/` that:
1. Hooks into Slack platform adapter to capture reaction events (👍 / 👎) on Hermes message outputs
2. Stores feedback events in Neon (`skill_feedback` table) with (skill_name, agent_message_id, reaction, user_id, tenant_id, timestamp)
3. Aggregates per-skill scores: usage count + thumbs rate over rolling windows (7d, 30d, all-time)
4. Surfaces in a Next.js dashboard at `/skills` showing every skill across personal/team/global scopes with score columns

### Files to modify

- **CREATE**: `hermes_agent/self_improvement/__init__.py`
- **CREATE**: `hermes_agent/self_improvement/feedback_capture.py` (~200 LOC)
- **CREATE**: `hermes_agent/self_improvement/skill_scorer.py` (~250 LOC)
- **CREATE**: `migrations/002_skill_feedback.sql` — Neon migration
- **MODIFY**: `gateway/platforms/slack.py` — register reaction event handler
- **CREATE**: `web/app/skills/page.tsx` — dashboard route
- **CREATE**: `web/app/skills/SkillsTable.tsx` — client component with sortable columns
- **CREATE**: `tests/self_improvement/test_feedback_capture.py`
- **CREATE**: `tests/self_improvement/test_skill_scorer.py`

### Files to verify (no edits)

- `hermes_storage/neon_backend.py` — feedback_capture uses NeonBackend for writes
- `agent/skill_utils.py::get_all_skills_dirs()` — skill_scorer enumerates skills from this

### Explicitly out of scope

- No auto-promotion (Phase B)
- No drift alerts (Phase C)
- No skill recommendations (Phase D)
- No emoji-set customization beyond default 👍/👎

### Implementation notes

- **Reaction → output correlation**: Hermes outputs in Slack carry a `client_msg_id`. We store the skill_name → client_msg_id mapping at output time. Reactions reference client_msg_id → resolve back to skill.
- **Tenant scoping via HermesIdentity**: every event carries `tenant_id` from the originating identity. RLS via `app.tenant_id` GUC.
- **Score = thumbs_up / (thumbs_up + thumbs_down)`** when total ≥ 3. Below 3, score is null (insufficient signal).
- **Recency weighting**: default uniform within 30-day window; weighting strategy is a Phase B refinement.
- **Reactive feedback only**: we capture reactions, we don't poll for them. If user removes a reaction, we delete the row (atomic).

### Acceptance criteria

- [ ] **AC-A.1** — Adding 👍 reaction on a Hermes Slack output writes a row in `skill_feedback` within 5s
- [ ] **AC-A.2** — Removing the reaction deletes the corresponding row
- [ ] **AC-A.3** — `/skills` dashboard renders with a sortable table: name + scope + usage_30d + thumbs_rate_30d + last_used
- [ ] **AC-A.4** — Cross-tenant isolation: tenant A cannot see tenant B's feedback (verified via RLS test)
- [ ] **AC-A.5** — Skills with <3 reactions show score = "—" (insufficient signal) rather than misleading 0% or 100%
- [ ] **AC-A.6** — 26 pytest cases pass (10 feedback_capture + 16 skill_scorer)
- [ ] **AC-A.7** — Zero regressions on existing Hermes test suite
- [ ] **AC-A.8** — Live verification: react to 5 real Hermes outputs in Slack; dashboard shows correct counts within 1 minute

### Test-driven design (pytest)

- `test_slack_reaction_event_creates_feedback_row`
- `test_reaction_removal_deletes_row`
- `test_feedback_capture_resolves_skill_from_client_msg_id`
- `test_skill_scorer_aggregates_30d_window`
- `test_skill_scorer_returns_null_for_insufficient_signal`
- `test_skills_api_returns_tenant_scoped_data_only`
- `test_skills_dashboard_renders_score_correctly` (Playwright)

---

## Phase 004-B — Auto-suggest + Blake-approve promotion

### Motivation

With Phase A's telemetry in place, identify skills ready for promotion. Per spec discussion: **manual + automated tiers** — system surfaces "these skills look ready" + Blake clicks approve. No silent promotions.

### What

A new daily job + UI surface that:
1. Scans `skill_scores` for skills meeting promotion thresholds (default: usage_30d ≥ 10, thumbs_rate_30d ≥ 0.80)
2. For each candidate, emits a "Suggest promote {skill_name} from {personal} → {team}" notification (Slack DM to Blake + dashboard banner)
3. Blake clicks approve → calls Skills Service `promote_skill` API → skill moved to team scope (via Plan 001-C's DDB lock-protected write)
4. Persists decision history in `promotion_decisions` table for retrospective review

### Files to modify

- **CREATE**: `hermes_agent/self_improvement/promotion_proposer.py` (~300 LOC)
- **CREATE**: `migrations/003_promotion_decisions.sql`
- **MODIFY**: `web/app/skills/page.tsx` — add "Promotion suggestions" panel
- **CREATE**: `web/app/skills/PromotionPanel.tsx` — accept/reject UI
- **MODIFY**: `gateway/platforms/slack.py` — daily DM with pending suggestions
- **MODIFY**: `tools/skill_manager_tool.py` — wire promotion-approve callback
- **CREATE**: `tests/self_improvement/test_promotion_proposer.py`

### Implementation notes

- Thresholds configurable: `HERMES_PROMOTION_MIN_USAGE` (default 10), `HERMES_PROMOTION_MIN_THUMBS_RATE` (default 0.80), `HERMES_PROMOTION_WINDOW_DAYS` (default 30)
- Daily cron via existing scheduler (Plan 001-D's hermes runtime scheduler)
- DM to Blake = single message with ≤5 candidates ranked by combined-score (usage × thumbs_rate)
- Each candidate has "Approve" button → POST to internal promotion endpoint → calls Skills Service
- Decision history feeds Plan 014.1-style retrospective scoring (was the auto-suggestion right?)
- **Bidirectional close-out**: also suggest DEMOTIONS for team-scope skills falling below thumbs_rate 0.50 over 30 days

### Acceptance criteria

- [ ] **AC-B.1** — Daily promotion-proposer job runs at configured cron time
- [ ] **AC-B.2** — Skills meeting thresholds appear in `/skills` "Promotion suggestions" panel
- [ ] **AC-B.3** — Slack DM sent to Blake daily with ≤5 ranked candidates (if any)
- [ ] **AC-B.4** — "Approve" click triggers `promote_skill` call; skill appears in team registry within 5s
- [ ] **AC-B.5** — Decision (approve / dismiss) persisted in `promotion_decisions`
- [ ] **AC-B.6** — Demotion path: team-scope skills with thumbs_rate < 0.50 over 30d generate demotion suggestions
- [ ] **AC-B.7** — No auto-promotion without Blake approval (test asserts no silent promote_skill calls)
- [ ] **AC-B.8** — 18 pytest cases pass
- [ ] **AC-B.9** — Live verification: trigger threshold by reacting to a Hermes output 10× with 👍; verify suggestion appears

---

## Phase 004-C — Drift / regression alerts

### Motivation

Skills that were good can stop being good (LLM behavior drifts, prompt rot, external dependency changes). Plan 014's drift detection works at ontology level; this works at skill level. Catch regressions before they accumulate.

### What

A regression detector that:
1. Watches per-skill `thumbs_rate_30d` over time
2. Alerts when a skill that was previously ≥80% thumbs drops below 50% over a 14-day rolling window
3. Surfaces alerts in `/skills` "Drift alerts" panel + Slack DM
4. Provides context: recent feedback samples, output diffs vs prior good versions (if Skills Service has Git history from Plan 003-D)

### Files to modify

- **CREATE**: `hermes_agent/self_improvement/drift_detector.py` (~250 LOC)
- **CREATE**: `migrations/004_skill_drift_alerts.sql`
- **MODIFY**: `web/app/skills/page.tsx` — drift alerts panel
- **CREATE**: `web/app/skills/DriftAlertPanel.tsx`
- **MODIFY**: `gateway/platforms/slack.py` — DM on first detection of new alert
- **CREATE**: `tests/self_improvement/test_drift_detector.py`

### Implementation notes

- Alert criteria configurable: previous baseline thumbs_rate (default 0.80), regression threshold (default 0.50), detection window (default 14d), minimum signal volume (default 5 reactions in window)
- Deduplicate alerts: a skill already in "alerted" state doesn't re-alert until resolved
- Resolution: thumbs_rate climbs back above 0.65 → alert auto-clears
- Manual override: Blake can dismiss an alert or mark "iterate" (skill author should improve it)

### Acceptance criteria

- [ ] **AC-C.1** — Drift detector job runs daily
- [ ] **AC-C.2** — Skill matching alert criteria generates a row in `skill_drift_alerts`
- [ ] **AC-C.3** — `/skills` shows alert badge + drift panel with skill name + recent feedback samples
- [ ] **AC-C.4** — Slack DM sent on FIRST detection (deduplicates on subsequent runs)
- [ ] **AC-C.5** — Auto-resolution: thumbs_rate ≥ 0.65 clears the alert + sends "resolved" DM
- [ ] **AC-C.6** — Manual dismiss / iterate actions work in UI
- [ ] **AC-C.7** — 12 pytest cases pass

---

## Phase 004-D — Skill recommendations (LLM-driven gap analysis)

### Motivation

Most ambitious of the four. Detect patterns of Hermes work that have NO existing skill — repeated tasks where the agent does expensive reasoning from scratch. Suggest creating a skill. Closes the meta-loop: "you should write skill Y because you keep doing this pattern."

### What

A weekly LLM-driven analysis that:
1. Queries atlas-012 connector for recent Conversation/Turn/ToolCall records
2. Identifies clusters of similar task patterns (LLM-judged similarity)
3. For each cluster with ≥3 instances + no matching existing skill, generates a skill proposal
4. Surfaces proposals in `/skills` "Recommended skills" panel
5. Blake can click "Generate skill" → LLM drafts a full SKILL.md + script → opens PR in blake-cowork-plugins (or local-only commit)

### Files to modify

- **CREATE**: `hermes_agent/self_improvement/recommender.py` (~400 LOC)
- **CREATE**: `migrations/005_skill_recommendations.sql`
- **MODIFY**: `web/app/skills/page.tsx` — recommendations panel
- **CREATE**: `web/app/skills/RecommendedSkillsPanel.tsx`
- **MODIFY**: `tools/skill_manager_tool.py` — wire "Generate from recommendation" action
- **CREATE**: `tests/self_improvement/test_recommender.py`
- **CREATE**: `hermes_agent/prompts/skill_recommendation.md.j2` — LLM prompt template

### Implementation notes

- Highest LLM cost of any phase — cap at $20/mo budget via `BudgetGate("recommender", $20)` monthly
- Sonnet (not Haiku) for clustering quality — proposal generation is high-stakes
- Cluster threshold: cosine similarity ≥ 0.85 on Turn embeddings (reuse atlas-012's embedding pipeline)
- Minimum cluster size: 3 instances over rolling 30 days
- Generated SKILL.md follows Plan 003 frontmatter conventions
- Blake can "Generate skill" → drafts file, opens in editor (not auto-commit) → Blake reviews + commits when ready

### Acceptance criteria

- [ ] **AC-D.1** — Weekly recommender job runs; identifies ≥0 clusters
- [ ] **AC-D.2** — Each cluster surfaces as a recommendation card with: task summary, example turns, suggested skill name
- [ ] **AC-D.3** — "Generate skill" action produces a valid SKILL.md following Plan 003 conventions
- [ ] **AC-D.4** — LLM cost stays under $20/mo cap
- [ ] **AC-D.5** — Generated skill is opened in editor (or local commit) — NOT auto-pushed to a remote
- [ ] **AC-D.6** — 15 pytest cases pass
- [ ] **AC-D.7** — Live verification: 1-month observation produces ≥1 useful recommendation

## Budget

- One-time dev: ~6-8 weeks across all 4 phases (parallelizable: A first, then B/C/D in any order)
- Ongoing LLM cost:
  - Phase A: $0 (no LLM)
  - Phase B: $0 (rule-based)
  - Phase C: $0 (rule-based)
  - Phase D: ~$20/mo (Sonnet for clustering + generation)
- Storage: ~10MB additional Neon for 4 tables

## Timeline estimate

~6-8 weeks total. Recommended order: A → (B, C in parallel) → D. Phase A is the foundation (telemetry); without it, B/C/D have no data.

## Open questions

- **Q-A.1** — Should we capture reactions on non-Slack platforms in Phase A? Defer to Phase A.1 (Telegram + Discord + etc).
- **Q-B.1** — Threshold tuning: defaults (10 uses + 80%) are intuitive but untested. Plan to revisit after 30 days of live data.
- **Q-B.2** — Slack DM cadence: daily is the default. Should it be only-when-pending? Probably yes (no daily noise).
- **Q-C.1** — Should drift detector also watch Phase D's recommended-then-generated skills? Yes — same code path applies.
- **Q-D.1** — Sonnet vs Opus for clustering — Sonnet is the default. If recommendation quality is poor in Phase D shadow mode, escalate to Opus.
- **Q-D.2** — Auto-generate vs Blake-writes — Phase D drafts the file but Blake commits. Whether to add an "auto-commit after review" mode is deferred.

## Decisions locked (per spec discussion 2026-05-20)

- ✅ Hermes-internal module (not a standalone service; port 8002 reservation rescinded)
- ✅ First observable signal: explicit user thumbs (👍/👎 Slack reactions)
- ✅ Promotion criterion: manual + automated tiers (auto-suggest, Blake approves)
- ✅ All 4 user-visible wins are in scope (dashboard + auto-suggest + drift + recommendations)
- ✅ Ships incrementally — each phase produces value standalone
- ✅ Reuses Plan 001-D (NeonBackend), Plan 003 (Skills Service promote API), atlas-012 (connector), Plan 001-C (skill_locks for promotion writes)
