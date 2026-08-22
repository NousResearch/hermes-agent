# Affiliate Outcome Pipeline Diagnosis (CrakRevenue → lah-core → LAHB → readiness)

Class of task: determine where the production outcome chain is broken, prove provider
postback traffic, and certify the normalization/persistence/S2S/ingestion/reconciliation/
readiness hops read-only. Worked evidence: CLOE_AFFILIATE_OUTCOME_INGESTION_LIVE_READINESS_V1
(2026-08-11, verdict BLOCKED_POSTBACK_TRAFFIC).

## Chain topology

```
CrakRevenue
→ /lah-postback/affiliate            (liveaccesshub.com, lah-core WP plugin lah-core-router)
→ normalization                      class-lah-utils.php  normalize_affiliate_conversion_request
→ wp_lah_conversions (+ lah-logs/lah_conversions.ndjson)
→ LAH_S2S_Push                       class-lah-s2s-push.php  build_sale_event
→ POST https://leanframeworklab.com/events   (header X-Ingest-Secret)
→ events table + projectCanonicalSaleEvent   (lah-brain src/db.js runInsert → conversion-importer.js)
→ affiliate_conversions
→ reconciliation P1–P4               src/money/reconciliation.js
→ readiness gate                     routes/business-runtime.js computeAutocutDataReadiness
```

## Evidence surfaces (all read-only)

1. **Namecheap SSH/SFTP channel** (the only way to see WP-side truth):
   `livewmgu@server313.web-hosting.com:21098`, key `~/.ssh/lah_core_deploy_livewmgu` (dedicated, deploy-scoped).
   - Access logs: `~/logs/liveaccesshub.com-<Month>-YYYY.gz` → `gzip -dc <f> | grep -c lah-postback`
   - Application log: `~/lah-logs/lah_conversions.ndjson` — ONE line per stored conversion; contains raw_payload → REDACT on output (parse locally, print selected fields only).
   - `~/public_html/wp-config.php` — grep constant NAMES only (`LAH_S2S_ENABLED`, `LAH_S2S_LAHB_URL`, `LAH_S2S_INGEST_SECRET`); NEVER print secret values.
   - Log-fidelity check: if the access log is small (~700 lines/mo), verify it sees real traffic before trusting a zero-count conclusion.
   - User rule: display the exact inspection command BEFORE executing (security preference).
2. **LAHB live** (leanframeworklab.com): `GET /version` (deployed SHA), `GET /health`,
   `GET /events` (read-only, NO auth) → events table contents. `POST /events` with empty body
   and no secret → 401 proves endpoint live + auth enforced with ZERO side effects. Since
   2026-08 /business/runtime-context is auth-gated — use certified checkpoints for readiness.
3. **lah-mcp-bridge** (`127.0.0.1:25910/mcp`, 7 tools: site_status/plugins_list/snippets_list/
   runtime_status/security_status/snapshot_create/snapshot_compare) + `verify-lah-core.sh`
   (certified ALL_GREEN check: homepage 200, plugin ACTIVE, site_status OK,
   `/lah-postback/affiliate` no-params → 400 = route registered, no conversion persisted).
4. **Deployed code**: `git archive origin/main | tar -x -C /tmp/<tag>` — NEVER trust the dirty
   working tree (lah-core routinely sits on unmerged local branches with pre-existing changes);
   origin/main is the deployed authority, verify live file hashes via SFTP sha256sum.

## Provider traffic classification (Phase 1)

A. NO_PROVIDER_POSTBACK_TRAFFIC ← access logs 0 hits + conversions ndjson empty/stale + LAHB events 0.
Classify this BEFORE touching ingestion code. When classified A, stop code changes and route to
provider configuration. CrakRevenue has NO API credentials in the stack (certified) →
`CRAKREVENUE_POSTBACK_CONFIGURATION_UNVERIFIED`; the authenticated evidence path is the operator
dashboard (Postback settings + Statistics → Custom Parameters Report, dimension "Sub ID 2" —
proven zone-level aff_sub2 attribution 2026-08-11). Do NOT modify ingestion code before the
provider postback contract is known (aff_sub2/aff_sub3 echo + All-Event triggers are
dashboard-only unknowns). Correct fail-closed verdict: BLOCKED_POSTBACK_TRAFFIC.

## Semantic defects to look for (the "3 null→0 coercion" pattern)

Non-paid outcomes (All Event / lead, payout 0) must never become paid conversions or lose the
payout=null distinction. Known defect sites at SHA 7168b468/2f9bcc38:
1. lah-core `class-lah-s2s-push.php build_sale_event`: `payout = is_numeric(...) ? : 0.0` — null→0.
2. lah-brain `src/event-schema.js normalizeEvent`: `payout: toNumber(rawEvent.payout || rawEvent.revenue)` — null→0.
3. lah-brain `src/money/conversion-importer.js normalizeConversionRow`: `Number(obj.payout ?? ... ?? 0)` — null→0.
Plus: lah-core utils hardcodes `event_type='conversion'`; S2S builder is sale-only;
lah-brain `projectCanonicalSaleEvent` projects ONLY `event_type='sale'` +
`import_source='lah_php_postback'` — non-paid outcomes never reach affiliate_conversions.
Repair = outcome_type/paid_conversion semantics before real traffic flows.

## Schema gaps

`wp_lah_conversions` (schema v1.3.0) has NO campaign_id/zone_id/site_id columns (wp_lah_clicks
has them) — Mode B dims persist JSON-only at row level. Dedupe: UNIQUE(dedupe_key),
`tx:network:transaction_id` (idempotent replay safe).

## Readiness gate semantics (never relax)

`computeAutocutDataReadiness` (business-runtime.js):
BLOCKED_FRESHNESS (money not OBSERVED / provenance UNAVAILABLE) → BLOCKED_INCOMPLETE_ZONE_DATA
(truncated or 0 zones) → BLOCKED_ATTRIBUTION (unmatchedConversions>0) → BLOCKED_AFFILIATE_OUTCOME_DATA
(zero zone outcome visibility: downstream_events>0 || paid_conversions>0) → READY.
Zone outcome breakdown: downstream_events=COUNT(*), paid_conversions=SUM(payout>0) from
affiliate_conversions keyed campaign_id|zone_id — rows MUST carry both dims or outcomes never
attach. The correct fix is provider evidence, never gate relaxation.

## Pitfalls

- lah-brain AGENTS.md codegraph bootstrap (`node tools/codegraph/freshness-check.js --repo lahb`)
  is STALE — tools/codegraph/ does not exist in lah-brain (MODULE_NOT_FOUND). Emit
  `CODEGRAPH_UNAVAILABLE_FOR_<MISSION>` and proceed with bounded manual inspection.
- lah-start launcher: use `claude` (or chatgpt/codex) placeholder from lah-stack-tools with the
  ABSOLUTE repo path — see `lah-start-agent-name-workaround.md`.
- GET /events returns event rows incl. beacon data — summarize counts/import_source/event_type,
  never dump raw payloads.
- Certify deployed files by SHA/hash, not by branch name: origin/main ≠ local HEAD on lah-core.
