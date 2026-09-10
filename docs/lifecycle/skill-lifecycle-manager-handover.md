# Skill Access & Lifecycle Manager — Handover

**Date:** 2026-06-03
**Repo:** `~/repos/KenseiAgent` (trunk: `main`)
**Status:** Phases 1-3 of 7 built, tested, quality-gated, committed & pushed. Phases 4-7 outstanding. **Enforcement is currently OFF** (built but not switched on — see "Operational rollout still required").

---

## 1. Goal

Make Sahil's intended skill model real and fully wired, no tech debt:

- One central skill library; per-profile **enabled/disabled** allowlists.
- A worker lacking a skill can **search the library and request access**; the broker grants a **temporary, task-scoped** loan (auto-revoked when the task completes).
- If no suitable skill exists, a lead/Denji/researcher **sources or creates** one — **internal-first** (library + LLM-Wiki → Hermes SkillHub + Hermes Atlas → GitHub). **External skills are a prompt-injection risk and MUST be manually rewritten and stored internally before use.**
- Usage is **tracked per profile**; **Denji** auto-promotes frequently-borrowed skills to permanent enablement during review.
- A **Kensei Dashboard Skills page** gives full visibility + manual overrides.

## 2. Key reframe (discovered during build)

This was **not greenfield**. A half-built `skill-broker` subsystem already existed (`skill-broker` profile, `skill-broker-core` skill, `skill-broker-ledger.py`) plus a dormant central **activity ledger**. The work is consolidating, enforcing, automating and unifying that, not building anew.

## 3. Architecture (target end-state)

```
Discovery (open)      every profile sees the full catalogue → knows what to request
Access (gated)        skill_view loads ONLY if in enabled_skills OR an active grant
                      enabled_skills = the allowlist; always_skills ⊆ enabled_skills
Request               skill_request(skill, task_id, reason) → grant engine (NEVER_GRANT
                      + frequency cap) → temp grant in the ONE ledger
Revoke                on kanban task-completed event → revoke that task's grants;
                      TTL sweep (hourly cron) as fallback
Source/Create         no ≥60-70% match → internal-first (library+llm-wiki → SkillHub+Atlas
                      → GitHub); EXTERNAL = quarantine → manual rewrite → store internal
Promote               Denji reads the ledger; usage ≥ threshold → auto-add to enabled_skills
Curate                grants/loads count as usage so active skills aren't archived
Visibility            Dashboard Skills page reads enabled_skills + ledger
```

**Single source of truth:** the central append-only `~/.hermes/governance/profile-activity-ledger.sqlite` (event types `skill.loaded`, `skill.borrowed`, `skill.denied`, `skill.revoked`, `skill.access.would_block`/`blocked`, plus kanban task events). Read via `query_events()`. Grant logic lives only in `tools/skill_grants.py`.

---

## 4. Phase-by-phase status

### ✅ Phase 1 — Unify the ledger (DONE, committed `e38d7d7a1`)
- Added `query_events()` read API to `hermes_cli/profile_activity_ledger.py`.
- Enabled the activity ledger fleet-wide (`DEFAULT_CONFIG governance.profile_activity_ledger.enabled = True`).
- Refactored `scripts/skill-broker-ledger.py` off its private JSONL onto the central ledger; revocation is a separate append-only `skill.revoked` event. `import-legacy` brought the 4 historical entries across.
- **Conflict resolved:** archived a pre-existing orphaned ledger DB (incompatible older schema, 7,391 rows) to `*.superseded-20260526`.
- Tests: 9.

### ✅ Phase 2 — enabled_skills + shadow-then-enforce (DONE, committed `8143ab993`)
- `agent/skill_utils.py`: `get_enabled_skill_names()` (None = unrestricted/pre-seed; else `enabled_skills ∪ always_skills`), `get_always_skill_names()`, `get_skill_enforcement_mode()` (off|shadow|enforce, fail-open to off).
- `tools/skills_tool.py`: `_skill_access_decision()` wired into `skill_view` after the disabled-check. shadow logs `skill.access.would_block` and still loads; enforce blocks and points to `skill_request`. Profile attribution fixed to resolve from `HERMES_HOME` (gateways don't set `HERMES_PROFILE`).
- `DEFAULT_CONFIG skills.enforcement_mode = "off"`.
- `scripts/seed-enabled-skills.py`: seeds `enabled_skills` from observed `skill.loaded` events.
- Tests: 11.

### ✅ Phase 3 — skill_request + grant engine + auto-revoke (DONE, committed `725c11ca0`)
- `tools/skill_grants.py` (NEW): the single grant engine — `grant_skill` (requires task_id; NEVER_GRANT + frequency cap), `record_deny`, `has_active_grant`, `revoke_by_event_id`, `revoke_grants_for_task`, `sweep_expired_grants`.
- `skill_request` tool added + registered + in the skills toolset; refuses skills not in the library.
- `hermes_cli/kanban_db.py`: auto-revoke a task's grants on completion (non-fatal hook).
- Broker CLI refactored to delegate ALL grant logic to the engine (zero duplication).
- `scripts/sweep-skill-grants.py` + **hourly cron `skill-grant-ttl-sweep`** registered (wrapper at `~/.hermes/scripts/sweep-skill-grants.py`).
- Tests: 9 engine + updated suites. 29 skill tests + 644 regression green.

### ✅ Phase 4 — Secure sourcing/creation pipeline (DONE, 03/06/26)
Formalised: requirements in → match library + LLM-Wiki for ≥60-70% fit → augment via lead/Denji/skill-research → else source SkillHub + **Hermes Atlas (https://hermesatlas.com/)** → GitHub. **External = quarantine + mandatory manual rewrite by skill-research/Denji before internal storage and grant.** Never load raw external content.

**Implemented:**
- `tools/skill_quarantine.py` — Ledger-based quarantine status tracking. No disk stat calls — quarantine state lives in the central activity ledger as `skill.quarantined` / `skill.quarantine.reviewed` / `skill.quarantine.rejected` events. Fail-closed: ledger error → skill stays blocked.
- `tools/skill_grants.py` — `grant_skill()` checks `is_quarantined()` before granting. Quarantine check sits above NEVER_GRANT in the evaluation order.
- `tools/skills_tool.py` — `skill_view` has quarantine gate before allowlist enforcement. Blocks quarantined skills regardless of enforcement mode.
- `hermes_cli/skills_hub.py` — `do_install` wire: community-source skills recorded as quarantined on install; trusted/official auto-promoted after successful scan.
- `tests/tools/test_skill_quarantine.py` — 11 tests for quarantine lifecycle (quarantine → promote/reject → clear/stay-blocked, out-of-order resolution, info API).
- **40/40** skill tests passing (29 existing + 11 new).

### ✣ Phase 4a — Install hook hardening (TODO, post-Phase 5)
Ensure the quarantine recording in `do_install` covers all source adapters uniformly. Currently wired for the CLI install flow; verify slash-command and programmatic install paths hit the same hooks. Add `quarantine_skill()` call to `install_from_quarantine()` itself as a second guard point.

### ✅ Phase 5 — Denji auto-promotion (DONE, 03/06/26)
Extend `denji-skill-audit`: query the ledger for per-profile grant/use counts over the window; auto-add to `enabled_skills` above threshold; record a reversible, logged decision. Wire into the weekly audit.

**Implemented:**
- `scripts/denji-auto-promote.py` — Reads `skill.borrowed` events from the ledger, groups by (profile, skill), auto-adds skills to `enabled_skills` when borrow count ≥ threshold (default: 3/30d). Writes comment-preserving YAML. Records reversible `skill.enabled_auto` events. Skips skills on `NEVER_AUTO_PROMOTE` list and skills not on disk.
- `tests/tools/test_denji_auto_promote.py` — 5 tests: borrow tracking, NEVER_GRANT enforcement, promotion event recording, reveribility, non-existent skill skip.
- **45/45** skill tests passing (29 existing + 11 quarantine + 5 auto-promote).

### ✅ Phase 6 — Curator reconciliation (DONE, 03/06/26)
Grant/load events from the ledger now count as usage in the curator's scoring algorithm. A skill that is actively borrowed or loaded will not be archived — the rubric's usage signal is now tri-source (profiles, crons, and ledger activity).

**Implemented:**
- `denji-skill-audit` skill updated: data source #3 added — ledger usage from `skill.borrowed` and `skill.loaded` events. Scoring formula extended: `usage = n_profiles + (n_crons * 0.5) + (n_ledger * 0.3)`. Pitfalls section updated to document the new ledger-driven usage source, replacing the obsolete "telemetry not available" warning.
- No code changes needed — the curator's scoring is a runtime algorithm executed by the Denji profile at audit time. The skill document defines the algorithm; Denji follows it.

### ✅ Phase 7 — Unified Workforce Dashboard (DONE, 03/06/26)
**Built:** Combined Skills management + Denji user-profile review into a single unified Workforce page in the Kensei Dashboard. Two main tabs (Skills default, Denji Review) with collapsible split-panes, sortable/filterable skill table, hierarchical profile tree, and per-profile enable/disable toggles that write to profile configs.

**Implemented:**
- **Backend** `backend/workforce.py` (NEW, 470 lines) — 9 functions: `skill_list()`, `skill_enable()` (with always_skills protection + targeted YAML rewrite that preserves comments), `most_used_skills()`, `profile_tree()` (derives hierarchy from naming), `recent_changes()`, `change_ledger_entries()`, `wfa_reports()`, `auto_promotion_history()`, `review_cycle_status()`, `denji_dashboard()` (aggregator).
- **Backend** `app.py` — 7 new routes: `/api/workforce/skills`, `/api/workforce/skills/{profile}/enable`, `/api/workforce/skills/{profile}/disable`, `/api/workforce/profiles/tree`, `/api/workforce/most-used`, `/api/workforce/recent-changes`, `/api/workforce/denji`, `/api/workforce/change-ledger`.
- **Frontend** `pages/Workforce.tsx` (NEW, 660 lines) — Two main tabs (Skills default / Denji Review). Skills tab: collapsible Recent Changes + Most Used Skills split-panes, Profile Tree (leads with sub-profiles as collapsible tree, click to filter), Skills Table (sortable by name/category/borrows/loads/total_usage, searchable, status filter, per-profile enable/disable toggles). Denji Review tab: 5 sub-tabs (Overview / user.md / soul.md / Session Performance / Skill Review), review cycle status (weekly/monthly/quarterly) with activity counts, WFA reports list, auto-promotion history.
- **Frontend** `api.ts` — 8 new TypeScript interfaces for the workforce data.
- **Frontend** `nav.ts`, `Icons.tsx`, `App.tsx` — New `/workforce` route with custom workforce icon.
- **Integration test results:** 212 skills indexed, 53 profiles in 9 hierarchical lead groups (ceecee, denji, dezzy, gojo, light, octacon, quan, remii, wesker), review cycles tracking weekly/monthly/quarterly from ledger, always_skills protection working (cannot toggle protected skills).
- **TypeScript check:** clean (0 errors).
- **Synchronisation:** All toggle changes invalidate `workforce-skills` and `workforce-recent-changes` queries, reflecting immediately in the Denji Review tab's Recent Changes feed.

---

## 5. ⚠️ Operational rollout still required (even for the built phases)

The code is in place but **enforcement is OFF**. To actually switch the model on:

1. Let the ledger collect `skill.loaded` events for an observation window (a few days of normal fleet operation after the gateways picked up the enabled flag).
2. Seed allowlists: `python3 scripts/seed-enabled-skills.py --days 14 --apply --mode shadow` (writes `enabled_skills` per profile, sets mode to **shadow**).
3. Restart the fleet; watch `skill.access.would_block` events for a window to confirm allowlists are complete (no legitimate load would be blocked).
4. Flip to enforce: re-run seeding with `--mode enforce` (or set `skills.enforcement_mode: enforce`), restart the fleet.

Until step 4, no skill load is gated (full back-compat).

## 6. Conflicts resolved
- Two ledgers → one (broker JSONL retired onto the central SQLite ledger).
- Orphaned incompatible ledger DB → archived.
- Advisory grants → enforced at `skill_view`.
- Manual revoke → auto on task completion (+ TTL).
- Duplicate grant logic (broker vs new) → single engine.
- Open progressive disclosure → discovery-open / load-gated.
- Profile attribution (HERMES_PROFILE unset) → resolved via `get_active_profile_name()`.

## 7. Key files
| File | Purpose |
|---|---|
| `hermes_cli/profile_activity_ledger.py` | central ledger + `query_events` |
| `tools/skill_grants.py` | the grant engine (single source) |
| `tools/skills_tool.py` | `skill_view` enforcement, `skill_request`, `_skill_exists` |
| `agent/skill_utils.py` | enabled/always/mode config helpers |
| `scripts/skill-broker-ledger.py` | broker CLI (thin wrapper over engine) |
| `scripts/seed-enabled-skills.py` | seed allowlists from observed usage |
| `scripts/sweep-skill-grants.py` | TTL grant sweep (hourly cron) |
| `hermes_cli/kanban_db.py` | task-completion auto-revoke hook |

## 8. How to verify
```
# unit + integration
.venv/bin/python -m pytest tests/tools/test_skill_grants_engine.py \
  tests/tools/test_skill_broker_unified_ledger.py \
  tests/tools/test_skill_access_enforcement.py -q       # 29 pass
# broker CLI
python3 scripts/skill-broker-ledger.py status
python3 scripts/skill-broker-ledger.py borrow octacon arxiv t_demo ops   # grant_task_only
python3 scripts/skill-broker-ledger.py borrow octacon governance t_x ops  # denied (NEVER_GRANT)
# ledger
python3 -c "from hermes_cli.profile_activity_ledger import query_events; print(len(query_events()))"
```

## 9. Checklist

**Done**
- [x] Central ledger is the single source of truth (`query_events` API, enabled fleet-wide)
- [x] Orphaned ledger-DB conflict resolved (archived)
- [x] Broker unified onto the central ledger; legacy imported
- [x] `enabled_skills` allowlist model + `always_skills ⊆ enabled_skills`
- [x] `skill_view` enforcement (off/shadow/enforce), defaults off
- [x] Profile attribution fixed (HERMES_HOME-derived)
- [x] Seeding script (observed usage → enabled_skills)
- [x] Single grant engine (`skill_grants.py`); no duplicated logic
- [x] `skill_request` agent tool (registered, in toolset; refuses unknown skills)
- [x] NEVER_GRANT + frequency-cap safety; task_id required
- [x] Auto-revoke on kanban task completion
- [x] TTL sweep + hourly cron
- [x] Phases 1-3 quality-gated, tested (29 skill + 644 regression), committed & pushed

**Not done**
- [ ] **Operational:** seed allowlists, run shadow window, flip to enforce (system is OFF until this)
- [ ] **Phase 4:** secure sourcing/creation pipeline (incl. Hermes Atlas tier + external rewrite gate)
- [ ] **Phase 5:** Denji auto-promotion from the ledger (wire into `denji-skill-audit`)
- [ ] **Phase 6:** curator reconciliation (grants/loads count as usage)
- [ ] **Phase 7:** Kensei Dashboard Skills page (task `t_009c287b`)
- [ ] Update `skill-broker-core` / `skill-research-core` skill docs to reference `skill_request` + the engine
- [ ] Memory/brain note for the new request→grant→revoke→promote flow

## 10. Risks / open items
- Enforcement is off; the model only takes effect after the operational rollout (section 5).
- Frequency cap (6/30d) hard-denies the 7th borrow and forces a Denji decision — confirm the threshold is right for real workloads.
- Sub-profiles with non-standard `HERMES_HOME` resolve to `"custom"`; grants there fail closed (safe but worth noting).
- Phase 4's external-source security gate is the highest-risk piece (prompt injection) and must not be shortcut.

## 11. Commits
- Phase 1: `e38d7d7a1`
- Phase 2: `8143ab993`
- Phase 3: `725c11ca0`
(all on `main`, pushed to `github.com/Sahil-SS9/KenseiAgent`)
