# LAH Affiliate Outcome Semantics Contract

Established: CLOE_AFFILIATE_OUTCOME_SYNTHETIC_POSTBACK_E2E_CERTIFICATION_V1 (2026-08-11).
Use for ANY work touching affiliate outcome data: postbacks, `/events` ingestion,
payout semantics, conversion projection, zone contract paid_conversions /
downstream_events, or synthetic test data. This is the canonical outcome contract
across lah-core (WP) → lah-brain (LAHB).

## The 5 contract rules (encode, never regress)

1. **missing payout ≠ payout 0** — a payout-absent postback must stay null/unknown.
2. **payout 0 ≠ proof of sale** — an explicit 0 is an observed value, not a sale.
3. **sale requires explicit positive provider/goal/event semantics** — never infer
   sale from payout alone. `status=conversion` is a LAH-side constant, not proof.
4. **null stays null until proof** — no layer may coerce null to 0.
5. **synthetic events stay identifiable and never become business truth.**

Vocabulary: `event_semantics` ∈ {`PAID`, `UNVERIFIED`} — PAID only when payout > 0;
payout=0 or absent → UNVERIFIED. Never invent an unproven provider taxonomy;
when the provider nature is unknown, semantics = UNVERIFIED/UNKNOWN, not sale.

## The coercion chain (all layers that broke null→0 / sale)

| Hop | File | Old defect |
|-----|------|-----------|
| H4 LAH S2S | lah-core `includes/class-lah-s2s-push.php` `build_sale_event` | `is_numeric($n['payout']) ? (float) : 0.0` + hardcoded `event_type='sale'` |
| H5 schema | lah-brain `src/event-schema.js` | `normalizeEventType("conversion")→"sale"` + `toNumber(payout \|\| revenue)` → 0 |
| H5 adapter | lah-brain `src/liveaccesshub-adapter.js` | SECOND `conversion→sale` mapping in `normalizeCanonicalEventType` + `toNumber(firstNonEmpty(payout,revenue,0))` + whitelist rebuild DROPS new fields |
| H5 importer | lah-brain `src/money/conversion-importer.js` | `Number(obj.payout ?? obj.revenue ?? obj.amount ?? 0)` accepts absence → 0 |
| H5 DB | lah-brain `src/db.js` events table | `payout REAL NOT NULL DEFAULT 0` cannot persist NULL |

**Search both repos for every occurrence of a mapping/coercion** — the
conversion→sale mapping existed in TWO files (schema + adapter); fixing one
leaves the defect live.

## Repair recipe (smallest delta, proven green)

- **transport (lah-core)**: `$payout = is_numeric(...) ? (float)$n['payout'] : null;`
  `event_type = payout>0 ? 'sale' : 'conversion'`; add `event_semantics` (PAID|UNVERIFIED),
  `is_test`, `source` to the event. Normalize captures `source` verbatim and sets
  `is_test=1` when the source has the reserved `synthetic_e2e` PREFIX
  (`str_starts_with(strtolower(trim($source)), 'synthetic_e2e')`) or request `is_test=1`.
  STRICT EQUALITY on the marker is a defect — see production-trial pitfall below.
- **schema/adapter (lah-brain)**: remove BOTH `conversion→sale` mappings; payout
  nullable — **explicit null must NOT fall back to revenue** (revenue is 0 for
  null-payout events after the adapter; only `undefined`/`""` payout may fall back
  to revenue). Add `event_semantics` + `source` to: `normalizeEvent` output,
  `ALLOWED_KEYS`, adapter whitelist rebuild, `EVENT_COLUMNS`, events DDL,
  `ensureColumns`, `rowToEvent`. Missing any one = "Unknown named parameter" or
  silently-empty field.
- **importer**: missing payout → fail-closed error, never 0. `projectCanonicalSaleEvent`
  requires `event_type==='sale'` AND `event_semantics==='PAID'` (defense in depth).
- **KPI safety (mechanism A — marked + excluded, no cleanup needed)**: projected
  canonical rows for `is_test=1` events are stored `status='rejected'`; reconciliation
  and the zone contract filter `status != 'rejected'`; `isTestTrafficEvent` excludes
  them from default event listings. Synthetic markers: `is_test=1` + `source` +
  `transaction_id` prefix (`SYNTH_CR_*`) must survive every hop.

## Sale wire payload contract (LAH → LAHB `/events`, `import_source='lah_php_postback'`)

`event_type, event_semantics, click_id, affiliate_click_id, external_click_token,
transaction_id, offer_id, goal_id, campaign_id, zone_id, site_id, payout, revenue,
currency, network, traffic_source, reconciliation_state, import_source, source,
is_test, timestamp, event_id` (stable event_id generated pre-dispatch; retry queue
persisted BEFORE dispatch). Dedupe at LAHB: `INSERT OR IGNORE` on
`event_id` fallback composite (import_source|timestamp|type|click_id|...|transaction_id|goal_id).

## SQLite nullable-money migration (SQLite cannot ALTER to drop NOT NULL)

```js
// in createSchema, order: createBaseTables → ensureColumns → createDependentTables
//   → migrateDedupIndexToV2 → auditDuplicateEvents → migrateEventsPayoutNullable → createIndexes
function migrateEventsPayoutNullable(db) {
  const payout = db.prepare("PRAGMA table_info(events)").all().find(c => c.name === "payout");
  if (!payout || payout.notnull === 0) return;
  db.exec("BEGIN");
  try {
    db.exec("ALTER TABLE events RENAME TO events_old");
    db.exec(`CREATE TABLE events ( /* full DDL with payout REAL DEFAULT NULL + new columns */ )`);
    const shared = oldCols.concat(["new_col"]).filter((n,i,a) => a.indexOf(n) === i);
    db.exec(`INSERT INTO events (${shared.join(", ")}) SELECT ${shared.join(", ")} FROM events_old`);
    db.exec("DROP TABLE events_old");
    db.exec("COMMIT");
  } catch (e) { db.exec("ROLLBACK"); throw e; }
}
```
- `ensureColumns` (ALTER ADD COLUMN) must run BEFORE the rebuild so the renamed old
  table already carries the new columns (the INSERT SELECT shares them).
- Run the rebuild BEFORE `createIndexes` so indexes are recreated on the new table;
  `auditDuplicateEvents` (index-existence check) must run BEFORE the rebuild.
- Rebuild on a fresh DB: payout already nullable → early return (no-op).

## Pitfalls

- **JS comments inside template-literal SQL DDL** → SQLite `near "/": syntax error`.
  Keep `//` comments OUT of `db.exec(` ... `)` blocks.
- **Duplicate DDL blocks with different indentation** (migration vs createBaseTables):
  the patch tool matches one occurrence only — verify BOTH locations (search the
  column name afterwards).
- **Adapter whitelist rebuild** silently drops any field not explicitly listed —
  new contract fields must be added in: adapter event object, ALLOWED_KEYS,
  EVENT_COLUMNS, DDL, ensureColumns, rowToEvent. A missing EVENT_COLUMNS entry
  throws `Unknown named parameter 'source'`.
- **Old tests encode the OLD contract** (e.g. payout=0 → 'sale', 'conversion'→'sale'):
  updating those assertions is the repair, not masking a failure — annotate the
  change with the mission id.
- Non-paid/UNVERIFIED postbacks are NOT projected into affiliate_conversions
  (visible in the events store with payout NULL). The zone contract reads only
  affiliate_conversions — downstream_events visibility for UNVERIFIED events is a
  documented follow-up, not widened scope.
- **Strict equality on the synthetic marker is a KPI-safety defect (proven by the
  real production trial CLOE_AFFILIATE_OUTCOME_REAL_POSTBACK_PRODUCTION_TRIAL_V1).**
  The mission source `synthetic_e2e_production_trial` failed `source === 'synthetic_e2e'`
  → `is_test=0` → the S2S event was delivered with is_test=0 → LAHB projected an
  APPROVED paid conversion (payout 44) → business KPI contamination. Marker derivation
  MUST be prefix-based (`synthetic_e2e*`), and sims/tests must use the EXACT production
  string variants (e.g. `_production_trial` suffix), never only the shortest form.
- **Fire-and-forget S2S is STILL a delivery.** `wp_remote_post(..., blocking=false)`
  performs connect+write; the response is just never read. A retry-queue row stuck on
  SHADOW_PENDING does NOT mean "not delivered" — the initial send already went out.
  Containment via `LAH_Retry_Queue::mark_dead_letter()` stops RETRIES, not the original
  send. Check delivery likelihood by absence of PHP connect errors in error_log
  (unchanged mtime = no failure logged). See `references/real-production-trial-runbook.md`.

## Verification recipe

- Deterministic JS suite (temp SQLite): A paid survives E2E (events row payout 44 +
  projection rejected), A2 reconciliation path on approved is_test=0 event
  (direct_postback_dimensions, revenue 44), B null stays null, C importer
  fail-closed, D 0≠sale + no conversion→sale, E dedupe, F campaign/zone survive,
  G KPI exclusion, M migration rebuild.
- lah-core PHP sims (WP stubbed, capture `wp_remote_post`): normalize + build_sale_event
  semantics, 12/12 + 9/9 pattern.
- Behavioral harness (real `POST /events` handler, temp DB): 8 scenarios — paid insert,
  projection rejected, dedupe, null not projected, listings/reconciliation exclusion,
  reconciliation path, raw_json preserves payout:null. Harness lives in
  `lah-brain/tools/operator-validation/`; requires `../../src/` paths from there
  (see behavioral-operator-simulation skill).
