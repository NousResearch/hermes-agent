# Cron Reliability — Phase 0 Evidence Ledger (2026-09-11)

Baseline: KenseiAgent `98a4aa` (ancestor of HEAD `1ac908d9`, clean tree).
Sources: live `~/.hermes/cron/jobs.json`, `executions.db`, `deliveries.db`, per-job
output dirs, gateway logs, `/tmp/cronanalysis-verification-20260911/probe.py` results
(fixture harness, no live jobs, no network). All secrets redacted.

## Claim-by-claim verification (from Cronanalysisreport.md)

| # | Report claim | Verdict | Live evidence (2026-09-11) |
|---|---|---|---|
| 1 | Research job leaks verification narration | CONFIRMED | Final response (1,150 chars) is pure verification narration ("provenance lint... exit 0", "scripts/run_tests.sh", "ls -la"). Output file: `cron/output/463058f2566d/2026-09-11_06-39-37.md` |
| 2 | Mashup review fails with empty response | CONFIRMED (symptom) | Job last_status=error, "Agent completed but produced empty response", failure_streak=3. Final response = `(No response generated)` (23 chars). Provider-level cause still LIKELY (oversized input: 195 KB paper + 617 KB radar), not proven |
| 3 | Blog pregen fails on zero credit | CONFIRMED | Job last_status=error, failure_streak=15. Script rc=1. Log shows 4x `credit insufficient balance: balance=0` HTTP 400 per run. Current `llm_generate.py` rotates only on 401/402/403/429 — zero-credit 400 not routed |
| 4 | X morning article data-model defects | CONFIRMED | Fixture probe: same source staged twice (dedupe by draft ID not source ID), stale 2020 timestamp accepted, undeclared signal type accepted. Job interrupted by shutdown, streak=3 |
| 5 | System audit syntax error in skills_tool.py | HISTORICAL (not current) | Failed run 14:00:35 with `expected an indented block after 'if' statement (skills_tool.py:891)`. Current source passes `ast.parse()`. Report must mark this historical |
| 6 | Shutdown interruptions lose runs | CONFIRMED | research / mashup-adjacent / x-manager / mailbox jobs all last_status=error "Interrupted by shutdown before terminal completion", streak=3 each. `recover_interrupted_executions()` marks `unknown`; no retry scheduled |
| 7 | Artifact helpers never wired into run_job() | CONFIRMED | Instrumented fixture run: `_prepare_delivery_artifact` calls=0, `_recover_run_scoped_artifact_delivery` calls=0 through real entry point. Helper also accepts non-HTML files (type not validated) |
| 8 | Mailbox job violates output contract | REFUTED (currently) | Latest run: clean compact digest + single `MEDIA:` path (175 chars). Compliant. Keep under watch, no fix needed now |
| 9 | Routing: 264 unapproved profile changes | PARTIALLY CONFIRMED | Total 269 records correct. Exact split: 5 approved root, 4 unapproved root, 260 unapproved profile. Correct wording: "264 unapproved overall (260 profile + 4 root)" |
| 10 | Sanitiser lets leaks through | CONFIRMED | Probe: 1,150-char leak input passed through unchanged by current sanitiser |

## Global counters (last 200 deliveries)

- delivered: 102, failed: 36 (executions: 199 completed / 1 failed in window)
- Only `blog-backlog-pregen` and `kensei-system-audit-daily` have failure deliveries
  joined via execution ID in this window; the four shutdown-interrupted jobs have
  job-level last_status=error but no execution rows (executions.db does not record
  those runs — itself evidence for the Phase 5 replay gap).

## Routing-review boundary (Phase 8 input)

- Root-approved changes: 5. Root unapproved: 4. Profile unapproved: 260.
- No provider/model names inferred from signatures; snapshots required per event.
- No reverts performed. Disposition ledger still to be built (Phase 8).

## Approval boundaries for Phase 1+

1. CODE REPAIR (no approval needed beyond plan sign-off): provider failure
   classifier, scheduler artifact wiring, response sanitisation, shutdown replay,
   mashup preprocessor, X data model.
2. JOB PROMPT EDITS: per-job, minimal, listed in each phase before applying.
3. MODEL/PROVIDER POLICY CHANGES: none in this plan. Exact-model routing preserved.
   Zero-credit 400 handling is failure *classification*, not a route change.
4. PROFILE ROUTING RECONCILIATION: evidence-only until Sahil disposes each batch.

## Open decisions for Sahil

1. X account/source registry: authoritative list + freshness window
   (recommended 12h default, 6h optional for morning job if volume allows).
2. Confirm routing-reconciliation scope: all 269 records ledgered before any
   revert decision (recommended) vs. top-N profiles only.
3. Confirm Phase 1 may proceed on the KenseiAgent baseline (98a4aa lineage).
