# Production Postback Trial Pattern (CLOE real trials on the LAH chain)

Pattern for bounded real HTTPS trials against `https://liveaccesshub.com/lah-postback/affiliate`
with the REAL postback secret, plus LAHB authority verification, containment, and
exact-ID remediation. Proven in CLOE_AFFILIATE_OUTCOME_REAL_POSTBACK_PRODUCTION_TRIAL_V1
(+ repair & retry).

## 1. Secret handling — server-side in-memory runner (mandatory)
- The postback secret lives ONLY in the WP option `lah_affiliate_postback_secrets`
  (array keyed by network, e.g. `crakrevenue`) in the wp2h_options table on the
  Namecheap server. It is NOT on the VPS, NOT in ~/.lah-secrets, NOT in any repo.
- NEVER copy the secret to the VPS. Run the trial FROM the production server:
  a PHP CLI script at `~/tmp/` (outside public_html, chmod 600) that
  `require`s wp-load.php, reads `LAH_Utils::affiliate_postback_secrets()['crakrevenue']`
  in memory, and performs the HTTPS GET itself.
- Send the secret ONLY as the `X-LAH-SECRET` header (the endpoint prefers header
  auth; the legacy `?secret=` query path puts it in the URL — never use it).
- Redact ALL output: `str_replace($secret, '[REDACTED]', $text)` in memory before
  printing. Evidence records only `secret_source / secret_loaded / secret_redacted`.
- Pass non-secret trial IDs via argv; never embed the secret in the script file.
- Access: `ssh -i ~/.ssh/lah_core_deploy_livewmgu -p 21098 livewmgu@server313.web-hosting.com`
  (php CLI 8.2 + wp-load available; plugin classes load via wp-load).

## 2. Trial runner structure (exit-code discipline)
- Precheck: snapshot table counts + confirm transaction IDs do NOT already exist
  (exact `WHERE transaction_id = %s`), else exit before sending anything.
- Trial A (paid): assertions = HTTP 200, `ok:true`, conversion_id present, payout 44,
  currency USD, campaign/zone/source preserved, `is_test=1`, S2S payload
  sale/PAID/payout 44, retry-queue row persisted BEFORE dispatch.
  If ANY assertion fails → exit 3 and do NOT send Trial B.
- Trial B (null): payout parameter ABSENT (not `payout=`, not `payout=0`). Assert
  payout null at normalize, persisted SQL NULL, S2S event_type `conversion` +
  `event_semantics=UNVERIFIED`, never `sale`. Fail → exit 4.
- S2S proof: the initial dispatch is fire-and-forget (`wp_remote_post blocking=false`)
  — the event is already sent at postback time; the retry row stays SHADOW_PENDING.
  After the grace window (`LAH_S2S_RETRY_GRACE_SECS`, default 300 s), invoke
  `LAH_S2S_Retry_Worker::run_batch()` ONCE (the canonical cron mechanism) →
  rows become DELIVERED with `last_http_status=202` = LAHB /events acceptance proof.
  Do not force retries before the grace elapses (worker skips young rows).

## 3. LAHB runtime DB authority — CRITICAL pitfall
- Production LAH Brain runs on HOSTINGER: `leanframeworklab.com` → A 92.113.23.159,
  AAAA 2a02:4780::/32 (Hostinger ranges). The VPS has NO nginx server block for the
  root domain and NO systemd LAH Brain service (only hermes-dashboard/gateway,
  lah-mcp-bridge, codex, studio/workspace/openclaw).
- `/home/deploy/lah-stack-repos/lah-brain/storage/lah-brain.sqlite` (local checkout)
  is NOT the production store. A 0-row result on that file proves NOTHING about
  production. The operator's manual sqlite3 check can silently target the wrong DB.
- Authority verification (read-only): `getent ahostsv4/v6 <domain>` (Hostinger IP
  ranges), `/etc/nginx/sites-enabled` (no root-domain block), `systemctl --user
  list-units`, `/version` response, `/proc/<pid>/cwd|environ` (often
  permission-denied for root processes — that itself is a signal).
- LAHB read endpoints (`/events/recent`, `/reconciliation`, `/business/runtime-context`)
  are auth-gated (401); no admin creds on the VPS → store-level H5/H6 proof is
  BLOCKED. Report `LAHB_RUNTIME_DB_AUTHORITY_UNVERIFIED`; never invent KPI proof.
  Verdict rule: when H2-H4 + LAHB 202 acceptance are proven but the store is
  unreadable → `..._CERTIFIED_WITH_LAHB_RUNTIME_AUTHORITY_LIMITATION`.

## 4. Containment (when a trial exposes a defect)
- Mark the retry-queue row DEAD_LETTER via the canonical
  `LAH_Retry_Queue::mark_dead_letter(event_id, 0, reason)` (small PHP script) —
  the worker never re-dispatches DEAD_LETTER rows. No SQL DELETE.
- Containment stops retries only: the initial fire-and-forget may have ALREADY
  delivered the event. Check server logs: an unchanged error_log around the trial
  window ⇒ connect+write likely succeeded ⇒ delivery probable.

## 5. Exact-ID LAHB remediation (operator-run when no DB access)
- No canonical conversion status-flip API exists. Remediation = exact-ID DB mutation:
  `UPDATE affiliate_conversions SET status='rejected' WHERE transaction_id='<exact>' AND payout=44 AND status='approved';`
  `UPDATE events SET is_test=1 WHERE event_id='<exact>' AND import_source='lah_php_postback';`
  Guards are exact-value anchors (defense-in-depth), not fuzzy predicates.
  No DELETE, no LIKE, no wildcards. Sequence: READ → interpret → operator UPDATE →
  VERIFY (status rejected + `COUNT/SUM ... WHERE status != 'rejected'` = 0/0).

## 6. Synthetic marker contract (reserved namespace)
- source→is_test derivation must be: exact `'synthetic_e2e'` OR
  `str_starts_with(source, 'synthetic_e2e_')`. NOT a broad prefix
  (`'synthetic_e2evil'` must be 0). The real trial caught the strict-equality bug
  (`synthetic_e2e_production_trial` → is_test=0 → KPI contamination risk).
- Boundary tests are mandatory (R1-R7 pattern: exact marker, trial variant,
  reserved suffix, evil boundary, normal source, paid transport, null transport).
- Trial ID conventions: `REAL_SYNTH_CR_PAID_<ts>`, `REAL_SYNTH_CLICK_*`,
  `REAL_SYNTH_ZONE_*`, `REAL_SYNTH_GOAL_*`; `source=synthetic_e2e_production_trial`.

## 7. lah-brain codebase pitfalls (from the repair)
- SQLite cannot ALTER NOT NULL→nullable: table-rebuild migration
  (rename → CREATE new DDL → INSERT shared columns → DROP old → COMMIT), placed
  between `auditDuplicateEvents` and `createIndexes` so indexes rebuild. `ensureColumns`
  (ALTER ADD COLUMN) runs BEFORE the rebuild so the old table already carries new
  columns and the copy list picks them up.
- NEVER put JS comments inside template-literal SQL DDL — `//` is a SQL syntax error.
- A new normalized field (event_semantics, source) must be added EVERYWHERE:
  normalizeEvent, ALLOWED_KEYS/validateEventPayload, the adapter whitelist rebuild,
  EVENT_COLUMNS, table DDL, ensureColumns, migration DDL, rowToEvent. Missing any
  one = 'Unknown named parameter' on insert or silently dropped field.
- `liveaccesshub-adapter.js` rebuilds the event from a whitelist AND coerces
  `payout: toNumber(firstNonEmpty(payout, revenue, 0))` — a second null→0 coercion
  site; fix the adapter's nullable money handling too, not just event-schema.
- `tools/operator-validation/*.mjs` harnesses need `../../src/` (two levels up),
  not `../src/` — path-depth bug bites on first run.

## 8. Verification order
READ (exact-ID) → interpretation (expected rows/status) → operator-run UPDATE →
VERIFY. Present commands in that order; never execute mutations before the operator
has run and returned the reads. Include expected outputs for every step.
