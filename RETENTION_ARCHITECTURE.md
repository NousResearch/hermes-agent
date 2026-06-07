# Session Retention & Data-Lifecycle — Architecture & Patterns Plan

> Status: **IMPLEMENTED** — Tier 1 + Tier 2 done on branch
> `feat/chat-retention-inactivity`; Tier 3 deliberately deferred. 2026-06-07.
> Scope: the chat/session retention + soft-delete + DB-hygiene subsystem added to this
> vendored Hermes checkout (branch `feat/chat-retention-inactivity`, commit `793cd3c5b`).
> Method: web research (SOTA patterns) + 2 independent adversarial reviews (Haiku agents),
> claims **verified against the code**, then filtered through engineering judgment for *this*
> context (a vendored upstream fork, ~465 sessions / 369 MB, no custom Trash UI).

---

## 1. Context

We replaced an age-based auto-prune (`retention_days` from creation) with:
- **Inactivity retention** (delete by `ended_at` = last activity; active/archived exempt; continuing resets the clock),
- **Tiered** windows (automated/`cron` shorter than interactive),
- **Soft-delete Trash** (`deleted_at` → hidden, restorable, hard-purged after grace),
- **Orphan-message sweep**, reusing the existing SOTA `vacuum()` (optimize_fts → VACUUM → wal_checkpoint).

It works and is tested (19 unit + 256 regression + e2e on a real-DB copy). This plan is about
making it **cleaner, leak-proof, and easier to maintain against upstream** — not adding features.

---

## 2. As-is pain points (verified)

| # | Pain | Verified? | Severity |
|---|------|-----------|----------|
| P1 | `maybe_auto_prune_and_vacuum` has **7 params / 3 modes** | ✅ real | Med-High |
| P2 | Config→behavior wiring **duplicated** in `gateway/run.py` and `cli.py` (identical blocks) | ✅ real | Med (drift risk) |
| P3 | `deleted_at IS NULL` filter **scattered** across `list_sessions_rich`, `session_count`, `search_messages` (×2 branches), `search_sessions` → a future query can forget it | ✅ real | Med |
| P4 | By-id / export paths (`get_session`, `_get_session_rich_row`, `export_all`) **don't** filter trashed | ✅ real | **Low** — hidden from list/search so unreachable via normal UI; `get_session` must stay unfiltered for restore |
| P5 | Vendored-fork **merge-conflict** exposure on `git pull` | ✅ real | Med |
| — | "messages FK got ON DELETE CASCADE (v2)" (claimed by a reviewer) | ❌ **false** — that's the *telegram_dm_topic* migration; messages FK has no CASCADE, so the orphan-sweep is correctly required | n/a |

> Note: one Haiku reviewer rated P4 "SEVERE" and invented a messages-CASCADE migration.
> Both were corrected by reading the source. Trust, but verify.

---

## 3. Research — SOTA patterns (cited)

- **Soft-delete via a VIEW, not filter-everywhere.** A base table + `CREATE VIEW active_* AS … WHERE deleted_at IS NULL`; app reads the view. The "filter in every query" approach is the dominant *leak-bug* failure mode. ([bun.uptrace.dev soft-deletes](https://bun.uptrace.dev/guide/soft-deletes.html), [oneuptime](https://oneuptime.com/blog/post/2026-01-21-postgresql-soft-deletes-view))
- **Partial indexes** `… WHERE deleted_at IS NULL` reclaim index space / keep uniqueness for active rows. ([thisdot/Prisma](https://www.thisdot.co/blog/how-to-implement-soft-delete-with-prisma-using-partial-indexes))
- **Repository / global query filter** centralizes the active-row predicate to one enforcement point. ([EF Core global filters vs repo](https://medium.com/@farkhondepeyali/global-query-filter-vs-custom-repository-pattern-in-entity-framework-core-6b094924e83f))
- **Policy/Strategy object over long parameter lists**; configuration-as-data; tiered TTL; grace + restore UX. ([Firestore TTL](https://firebase.google.com/docs/firestore/ttl), [LangSmith TTL](https://docs.smith.langchain.com/self_hosting/configuration/ttl))
- **Idempotent, off-hot-path maintenance**: last-run marker, batch deletes, VACUUM after bulk delete. (Matches our existing `last_auto_prune` marker + gated `vacuum()`.)
- **Vendored-fork isolation**: keep local logic in a separate module / hook; **avoid monkeypatching**; expect to rebase a thin patch set on upstream. ([Shopify: case against monkey-patching](https://shopify.engineering/the-case-against-monkey-patching))

---

## 4. Target architecture (recommended subset)

Engineering judgment for *this* context: adopt the patterns that remove real pain at low risk;
**decline** abstractions whose cost exceeds their value on a vendored fork at this scale.

### ✅ ADOPT — Tier 1 (high value, low risk)

**A. `RetentionPolicy` value object + `from_config()` factory.** Collapse the 7 params into one
immutable dataclass built in **one** place. Fixes **P1** and **P2** at once.
```python
@dataclass(frozen=True)
class RetentionPolicy:
    retention_days: int = 90
    min_interval_hours: int = 24
    vacuum_after_prune: bool = True
    inactive_days: int | None = None
    automated_inactive_days: int | None = None
    automated_source: str = "cron"
    trash_grace_days: int | None = None
    @classmethod
    def from_config(cls, cfg: dict) -> "RetentionPolicy": ...   # single source of truth for keys
```

**B. One shared entry point** `run_session_retention_maintenance(db, sessions_dir)` that does
*load config → build policy → call db.maybe_auto_prune_and_vacuum(policy)*. Both vendored callers
(`gateway/run.py`, `cli.py`) shrink to a **single line** → also **shrinks the vendored edit
surface** (helps **P5**). `maybe_auto_prune_and_vacuum(policy, sessions_dir)` takes the policy
object (keep a back-compat shim if any other caller passes kwargs).

### 🟡 CONSIDER — Tier 2 (good, but mind the cost)

**C. `active_sessions` SQL VIEW** to centralize the soft-delete predicate (fixes **P3**, hardens
against future leaks). Caveat: `list_sessions_rich` uses a recursive CTE + correlated subqueries;
only the **primary** `FROM sessions s` should become `FROM active_sessions s` — lineage subqueries
must keep seeing the base table. Worth doing, but with care + the existing tests as a safety net.
*Lighter alternative:* keep the explicit filters (already tested) and add a single regression test
asserting "no query surfaces a trashed chat".

### ⛔ DECLINE — over-engineering for this context

- **Full policy engine** (`RetentionPolicy.should_delete_session()` strategy objects replacing the
  SQL prune methods): the set-based SQL prunes are simpler and faster than row-by-row policy
  evaluation; not worth it at this scale.
- **`local/retention.py` disk-space auto-override hook**: speculative; adds indirection and still
  requires edits in the vendored call sites, so it doesn't actually achieve isolation.
- **FK `ON DELETE CASCADE` on messages** (table rebuild): FK enforcement is already ON; manual
  cascade + orphan-sweep cover it; rebuild risk ≫ value.
- **External-content FTS migration**: destructive on a 369 MB live DB; bloat here is mostly
  reasoning-trace text, not FTS duplication → modest savings, high risk. DB also shrinks naturally
  as retention purges. Leave as a deliberate, backed-up opt-in only.
- **Golden-fixture test framework**: the 19 unit + e2e tests are adequate at this scale; add
  targeted tests instead of a framework.

---

## 5. Patterns catalog (what we are using / should use)

| Pattern | Where | Status |
|---|---|---|
| Soft-delete with `deleted_at` (timestamp, not bool) | sessions | ✅ done |
| Two-phase delete (Trash → grace → purge) + restore | retention | ✅ done |
| Tiered TTL (source-differentiated) | cron vs interactive | ✅ done |
| Idempotent maintenance + last-run marker | `maybe_auto_prune` | ✅ pre-existing |
| Optimize-FTS-before-VACUUM + WAL checkpoint | `vacuum()` | ✅ pre-existing |
| Value object + factory (config-as-data) | `RetentionPolicy` | ✅ done |
| Single maintenance entry point (DRY) | `run_session_retention_maintenance` | ✅ done |
| Active-rows VIEW (centralized filter) | `active_sessions` + partial index | ✅ done |
| Vendored-fork: thin patch on a feature branch | git | ✅ (branch) — keep edits minimal |

---

## 6. Outcome / sequencing

1. ✅ **Tier 1 (A+B) DONE** — `RetentionPolicy` + `from_config` + shared
   `run_session_retention_maintenance`; both callers are now one line; the
   method takes a policy (keyword args kept for back-compat).
2. ✅ **Tier 2 (C) DONE** — `active_sessions` VIEW + partial index; list/count/
   search read the view (also closed a latent CJK-LIKE search leak). Added the
   "no trashed leak" regression test. **24 retention + 256 regression green**;
   e2e on a copy of the real 369 MB DB confirms view + index auto-migrate.
3. ⏸️ **Tier 3 held** (recorded in §4) — risk/value poor here; revisit on need.

Open question for the owner: is a **Trash UI** ever in scope? Without a frontend rebuild, restore is
CLI/`restore_session()` only — which caps the realizable value of soft-delete. If a UI is wanted,
that's a separate (frontend) workstream.

---
*Plan derived from: 1 web-research pass + 2 adversarial code reviews (Haiku), all claims verified
against source. Synthesis & prioritization: Claude Opus 4.8.*
