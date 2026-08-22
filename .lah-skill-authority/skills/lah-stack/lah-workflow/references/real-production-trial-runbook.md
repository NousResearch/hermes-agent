# Real Production Trial Runbook (authorized endpoint/postback trials)

Established: CLOE_AFFILIATE_OUTCOME_REAL_POSTBACK_PRODUCTION_TRIAL_V1 (2026-08-11).
Use for ANY operator-authorized bounded trial against a REAL production endpoint
(postback, webhook, /events, API) where the outcome must be certified on live
infrastructure — especially when a secret is involved and when a defect could
contaminate business truth. This is the phase AFTER the behavioral-simulation
gate (behavioral-operator-simulation) and BEFORE any repair.

## Authorization envelope (what the operator must explicitly grant)

- EXACT number of requests (e.g. "exactly TWO real HTTPS calls: A paid, B null").
- The secret source to use (canonical config) + "NEVER print/echo/log/persist".
- Explicit boundaries: no provider mutation, no campaign/bid/budget/status
  mutation, no third request, no duplicate replay without new authorization.
- Expected verdict strings + what BLOCKER means (stop, report, request repair
  authorization — DO NOT patch production code during the trial).

## Secret-in-memory server-side pattern (the only safe way)

When the secret lives in a remote production config (e.g. WP option
`lah_affiliate_postback_secrets` on a shared host) and the trial must run with it:

1. Upload a PHP runner via SFTP to a NON-WEB directory (`~/tmp/`, chmod 600) —
   never `public_html/`.
2. Run it over SSH: `ssh -i <key> user@host 'php /home/user/tmp/runner.php <args>'`.
   (The dedicated lah-core deploy key gives a SHELL — ssh works, not just sftp —
   and php CLI 8.2 is available on cPanel shared hosting.)
3. The runner does `define('ABSPATH', '/home/<user>/public_html/'); require
   wp-load.php;` then reads the secret IN MEMORY from the canonical option via the
   plugin's own accessor (e.g. `LAH_Utils::affiliate_postback_secrets()['crakrevenue']`).
   The secret NEVER enters argv, URLs, logs, or output.
4. Send the secret as a HEADER (e.g. `X-LAH-SECRET`), never in the query string.
5. Redact every printed string through `str_replace($secret, '[REDACTED]', $text)`
   before it can reach stdout; assert the persisted raw payload contains no secret.
6. Record only: `secret_source`, `secret_loaded=true`, `secret_redacted=true`,
   `secret_printed=false`. Never a value or hash.

Alternative (no shell): ask the operator to place the secret in a 0600 file the
runner reads in-process; never ask them to paste it in chat.

## Phase gating (abort semantics are part of the runner)

- PRE: verify deployed SHAs/hashes match the certified versions; snapshot counts
  (conversions, retry queue); generate unique IDs (prefix `REAL_SYNTH_*` +
  timestamp) and confirm they do NOT already exist.
- TRIAL A (positive): send; assert EVERY hop (response ok, persisted values,
  marker is_test=1, S2S event shape). If ANY assertion fails → exit non-zero,
  print BLOCKER + exact hop/transaction_id/expected-vs-observed, and DO NOT send
  trial B. The runner itself must enforce "abort before B".
- TRIAL B (negative): the discriminating case — e.g. the payout parameter must be
  ABSENT (not `payout=`, not `payout=0`). Assert null stays null, no sale
  inference, no projection. Any null→0 or sale inference → BLOCKER, no ad-hoc patch.
- RETRY/S2S proof: invoke the canonical worker once (`LAH_S2S_Retry_Worker::run_batch()`)
  — blocking=true sets `last_http_status` — then re-read the queue rows.
- Evidence: one sanitized JSON report (trial-report.json) with verdict, SHAs,
  per-trial traces, assertions, retained records, H1 status, code-changes: NONE.

## Fire-and-forget S2S: SHADOW_PENDING ≠ not delivered

`wp_remote_post(..., 'blocking' => false)` still performs connect+write; only the
response is unread. So after a postback, the LAHB may ALREADY have the event even
though the queue row says SHADOW_PENDING. When a trial fails (e.g. is_test=0):

1. Contain FIRST: `LAH_Retry_Queue::mark_dead_letter($event_id, 0, '<reason>')`
   (canonical method; DEAD_LETTER rows are excluded from `get_pending_batch`).
   This stops RETRIES — it does NOT undo the initial fire-and-forget send.
2. Assess delivery likelihood: the initial dispatch happens inside the grace
   window (SHADOW_PENDING eligible after ~300 s); check the server error_log
   mtime — an unchanged error_log around the dispatch timestamp means no
   connect failure was logged (high delivery probability).
3. LAHB-side read endpoints are often auth-gated (401). If so, state the
   inferred state honestly + require operator-authorized LAHB admin verification
   by exact event_id/transaction_id before any remediation.

## BLOCKER reporting protocol (on real defect)

- STOP coding. Zero production code changes during the trial (containment via
  canonical methods only).
- Report: exact hop, exact transaction_id, expected vs observed, sanitized
  evidence (no secret, no full URL).
- Request authorization for: (a) the bounded repair, (b) any LAHB-side
  remediation (exact-ID deletion only, operator access or operator-run query),
  (c) a re-trial with NEW transaction IDs.
- H1 authority: a synthetic trial NEVER proves the provider sent callbacks.
  Keep `CRAKREVENUE_POSTBACK_CONFIGURATION=UNVERIFIED` /
  `REAL_CRAKREVENUE_CALLBACK_TRAFFIC=NOT OBSERVED` unless an independently
  observed provider request exists.

## Case study (why prefix matching matters)

The deployed marker derivation used `source === 'synthetic_e2e'` (strict). The
mission-specified source `synthetic_e2e_production_trial` → `is_test=0` → S2S
delivered with is_test=0 → LAHB projected an APPROVED paid conversion (payout 44)
→ KPI contamination. The runner aborted before trial B (correct), containment
dead-lettered the retry row, but the initial fire-and-forget send was already out.
Fix proposal: prefix match `str_starts_with(strtolower(trim($source)), 'synthetic_e2e')`
and re-test with the exact production string variants.
