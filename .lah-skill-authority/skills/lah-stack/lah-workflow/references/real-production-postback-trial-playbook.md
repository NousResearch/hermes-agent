# Real Production Postback Trial Playbook (LAH/LAHB E2E)

Class of task: bounded real-HTTPS production trials against the liveaccesshub.com
postback endpoint (`/lah-postback/affiliate`) to certify the deployed chain
(endpoint → normalize → persistence → retry/S2S → LAHB ingest → projection →
reconciliation), plus the repair-and-retry loop. Derived from
CLOE_AFFILIATE_OUTCOME_REAL_POSTBACK_PRODUCTION_TRIAL_V1 (+_REPAIR_AND_RETRY).

## 1. Secret-safe server-side runner (never expose the postback secret)

- The postback secret lives ONLY in the WP option `lah_affiliate_postback_secrets['crakrevenue']`
  (Namecheap server, wp2h_options). It is NOT on the VPS and must NEVER be printed,
  persisted, or put in argv/URL/logs/evidence.
- Access: `ssh -i ~/.ssh/lah_core_deploy_livewmgu -p 21098 livewmgu@server313.web-hosting.com`
  (php CLI 8.2 + public_html/wp-load.php available; SFTP-only paths fail — shell works).
- Pattern: upload a php CLI runner to `~/tmp/` (chmod 600, OUTSIDE public_html),
  `require wp-load.php`, read the secret via `LAH_Utils::affiliate_postback_secrets()`
  IN MEMORY, send via `X-LAH-SECRET` header (preferred auth channel; never `?secret=`),
  redact any output through an in-memory `str_replace($secret, '[REDACTED]', ...)`.
- Auth is per-network: endpoint `validate_auth` accepts header first, legacy query fallback
  exists but header is preferred; a wrong header is rejected with 403 (no fallthrough).
- Evidence output: sanitized JSON — HTTP status, response body (redacted), tx ids,
  DB row excerpts (parse `normalized_payload_json` for campaign/zone/source/is_test since
  wp_lah_conversions has NO row-level dims columns), retry-queue payload excerpt.

## 2. Stop-gate discipline (mission-embedded, hard-coded in the runner)

- Distinct exit codes: 0=certified, 3=trial A failed (trial B MUST NOT be sent),
  4=trial B failed, 7=precheck tx-collision, 8/9=config/usage error.
- Trial B is executed ONLY inside `if (trial A all_pass)`.
- Script-level assertions per trial: HTTP 200 + ok=true + conversion_id present;
  payout preserved (44) / omitted stays NULL; currency; campaign/zone preserved;
  source marker preserved; is_test=1; S2S semantics (sale/PAID vs conversion/UNVERIFIED);
  queue row persisted before dispatch.
- Any divergence → exit non-zero + report exact hop, exact transaction_id, expected vs
  observed, sanitized evidence. Do NOT patch code during the trial (request a repair mission).

## 3. Synthetic marker contract — RESERVED NAMESPACE, not broad prefix

- Real defect found: strict equality `source === 'synthetic_e2e'` missed
  `synthetic_e2e_production_trial` → is_test=0 → paid S2S event left LAH without the
  KPI-safe marker → LAHB projected it as an APPROVED paid conversion (contamination).
- Correct derivation: `is_test = 1` when `source == exact_marker` OR
  `str_starts_with(source, marker . '_')` (e.g. `synthetic_e2e_`). Explicit `is_test=1`
  param preserved.
- Boundary tests are mandatory both ways: `synthetic_e2e_production_trial` → 1 AND
  `synthetic_e2evil` → 0 (a broad `str_starts_with(marker)` would wrongly match).
- Test at normalize level AND transport level (S2S payload carries is_test + source).

## 4. Fire-and-forget S2S containment caveat (CRITICAL)

- LAH S2S dispatch = `wp_remote_post(blocking=false, timeout=1)` → the HTTP request IS
  sent (connect+write at kernel level); the response is never read; the retry-queue row
  stays SHADOW_PENDING (last_http_status NULL).
- Containing the queue row (`LAH_Retry_Queue::mark_dead_letter` → DEAD_LETTER) stops the
  E-2.2 worker from re-dispatching, but CANNOT undo the initial fire-and-forget delivery.
- Infer whether the initial delivery happened: check `public_html/error_log` — unchanged
  around the trial window = no connect failure logged = delivery LIKELY. Direct LAHB reads
  are auth-gated (401) so certainty requires operator/DB access.
- Grace window: worker picks SHADOW_PENDING only after INITIAL_GRACE_SECS (default 300s) —
  act fast (containment within the window) but remember the initial send already happened.

## 5. LAHB store access reality + exact-ID remediation protocol

- LAHB (leanframeworklab.com): ALL read endpoints auth-gated (401) — /events/recent,
  /reconciliation, /business/runtime-context. /version and /health are public but expose
  NO counts. No admin creds on the VPS → LAHB verification/remediation requires the
  operator or DB access.
- No canonical status-flip mechanism exists for affiliate_conversions (the approvals /
  missions reject endpoints operate on other records). Remediation = exact-ID SQLite
  UPDATE (operator-run or with DB access).
- Protocol (give the operator this exact sequence, never execute blind):
  READ (exact event_id / transaction_id SELECTs) → interpret (row exists? status approved?
  contribution COUNT/SUM?) → UPDATE `status='rejected'` on affiliate_conversions +
  `is_test=1` on events, WHERE exact-ID + exact-value guards (payout=44, status='approved',
  import_source='lah_php_postback') → VERIFY (contribution = 0/0 after).
- Hard constraints: no wildcards, no LIKE, no campaign/offer/source-only predicates,
  no DELETE, no replay without new authorization.
- `status='rejected'` is the canonical KPI-exclusion mechanism (reconciliation uses
  `WHERE status != 'rejected'`; business-runtime zone contract filters it too).

## 6. Pre-retrial deployed-code probe (no event created)

- Before sending new production events: run a runtime normalization probe ON the server
  (php CLI + wp-load + `LAH_Utils::normalize_affiliate_conversion_request` with the trial
  source) — proves the DEPLOYED code maps source → is_test=1 with NO HTTP request and NO
  persistence. Local-file hash == deployed hash (verified via SFTP download) makes the
  local sim result apply to production code.
- Also verify: old trial's retry row is DEAD_LETTER (no pending synthetic retry) + counts.

## 7. Deploy + verification constants (lah-core)

- Canonical SFTP deploy: extract files from merge SHA (`git show <sha>:<path>`), hash,
  backup live file as `.bak-YYYYMMDD{c}` (increment suffix — preserve older backups),
  upload `.new`, download+hash VERIFY BEFORE mv, atomic rename, chmod 644, download+hash
  VERIFY AFTER. s2s-push.php unchanged → skip redeploying it.
- Endpoint health WITHOUT triggering a trial: `GET /lah-postback/affiliate` with no params
  → 400 (route registered, no conversion created). Homepage 200.
- CI: no GitHub Actions — local gates only (php sims / node --test), report honestly.

## 8. Evidence artifacts

- Keep runner scripts + evidence under `/home/deploy/cloe-diagnostics/<mission>/`;
  server copies in `~/tmp/` (chmod 600). Never leave the secret in any artifact; the
  plugin's own ndjson log redacts it as `***` (raw_payload write path).
- Retained synthetic records are the established audit-safe pattern on this system
  (pre-existing operator test rows are kept, never DELETEd for cosmetics).
