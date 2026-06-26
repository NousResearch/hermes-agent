# Torben Backend Capability Canary Report

Generated: 2026-06-26T12:55:32Z

## Verdict

Torben is the only live Signal-facing operator for this slice. EA,
GTM/Magnus, and Finance/Ratatosk backend capabilities are wired behind Torben
with stage-only or approval-gated behavior.

The stage-only production canaries passed after one live GTM fix: Torben's
hidden Magnus refresh now uses Torben's profile credential context for Grok/xAI
instead of depending on the retired Magnus profile auth. The live Torben profile
script was synced to the repo snapshot and live-profile verification passes.

Important caveat: before the fix, the 2026-06-26 08:21 GTM radar failure and
the 2026-06-26 08:24 live-profile drift warning were delivered to Signal. The
current state is repaired and verified; those already-delivered messages cannot
be erased by this canary run.

Gate decisions are now resolved for testing scope:

- Public/personal mutations: limited testing autonomy. Torben may run narrow,
  auditable mutation canaries and apply explicitly approved handles. Unattended
  public X/LinkedIn writes remain blocked.
- Live finance: one tiny canary is approved for testing only after mandate,
  consent, kill switch, pre-trade guard, and reconciliation gates pass. The
  current Ratatosk circuit-breaker halt remains a hard block.

## Evidence

ADLC contract:

- `bin/adlc validate-artifact --schema build-brief --input /Users/ericfreeman/.hermes/hermes-agent/docs/build-briefs/torben-backend-capability-completion.json --json`
- Result: `valid=true`.
- `bin/adlc emit-work-items --target linear --build-brief /Users/ericfreeman/.hermes/hermes-agent/docs/build-briefs/torben-backend-capability-completion.json --dry-run --require-ready --json`
- Result: `ready`, 10 work items, 15 dependency links.

Hermes tests:

- `venv/bin/python -m pytest tests/test_signal_coo.py tests/test_signal_coo_gmail_realtime.py tests/test_signal_coo_live_profile_verify.py tests/test_torben_gtm_engagement_radar.py tests/gateway/test_torben_gtm_reply_router.py tests/test_torben_finance_radar.py tests/test_torben_gtm_radar_script.py -q`
- Result: 101 passed.
- `git diff --check`
- Result: clean.
- Focused post-gate slice:
  `venv/bin/python -m pytest tests/test_signal_coo_live_profile_verify.py tests/test_torben_gtm_radar_script.py tests/test_torben_finance_radar.py -q`
- Result: 27 passed.

Live Torben cron state:

- `hermes -p torben cron list`
- Result: 12 active Torben jobs, including morning brief, calendar alignment,
  Boardy brief, email hygiene review, meeting prep, GTM radar, Gmail Pub/Sub
  pull, Gmail watch renew, live-profile verify, GTM engagement radar,
  live-profile investigate, and finance radar.
- Current listed jobs have no `last_error` or `last_delivery_error`.
- `torben-gtm-radar` last ran at `2026-06-26T08:21:09.882421-04:00` with
  scheduler status `ok`; the post-run artifact was then repaired by rerunning
  the fixed stage-only path under Torben's profile context.
- `torben-gtm-engagement-radar` last ran at
  `2026-06-26T08:35:36.842043-04:00` with scheduler status `ok`.

Live profile verifier:

- `HERMES_HOME=/Users/ericfreeman/.hermes/profiles/torben venv/bin/python profiles/torben/scripts/torben_live_profile_verify.py`
- Result: `{"wakeAgent": false, "status": "pass", "task": "torben_live_profile_verify"}`.
- Latest verifier artifact:
  `/Users/ericfreeman/.hermes/profiles/torben/state/torben-live-profile-verify-latest.json`.
- Enabled jobs checked: 12.
- Gmail realtime health: pass, four accounts checked, Pub/Sub latest at
  `2026-06-26T12:23:38.047417Z`, soonest watch expiration
  `2026-07-03T07:15:35.887000Z`.
- Submanager contract health: pass, three contracts: EA, GTM, Finance.
- Live profile investigation:
  `HERMES_HOME=/Users/ericfreeman/.hermes/profiles/torben venv/bin/python profiles/torben/scripts/torben_live_profile_investigate.py`
  returned `status=idle`, `wakeAgent=false`.
- Scheduler result: the 08:42 live-profile investigation cron tick consumed the
  same no-pending-failure state and returned `[SILENT]`; the next investigation
  run is scheduled for `2026-06-26T08:57:24.191037-04:00`.
- Backend artifact health is now part of the live-profile verifier. Current
  GTM, GTM engagement, and finance artifacts all pass with
  `public_actions_taken=0`, `external_mutations=0`, and
  `broker_orders_submitted=0`.
- Scheduler result: the hardened verifier's 08:39 cron tick ran the production
  no-agent path and produced `status=pass`, no errors, no warnings, and
  `backend_artifact_health.status=pass`; the next verifier run is scheduled for
  `2026-06-26T08:54:20.494895-04:00`.

GTM credential-context fix:

- Failure found: the scheduled Torben GTM radar invoked Magnus with
  `HERMES_HOME=/Users/ericfreeman/.hermes/profiles/magnus`, where
  `xai-oauth` was unavailable.
- Metadata proof: `hermes -p magnus auth list` listed no `xai-oauth`;
  `hermes -p torben auth list` listed `xai-oauth`.
- Fix: `profiles/torben/scripts/torben_gtm_radar.py` now defaults hidden Magnus
  refresh credentials to Torben's profile home while preserving the explicit
  `TORBEN_GTM_MAGNUS_HERMES_HOME` override.
- Live sync proof: the live profile script and repo snapshot have matching
  SHA-256 `c4a676110dba0a5adc2d7212f911613e708c3aa3ce629927db95714d289e7dee`.
- Scheduler delivery proof: `agent.log` records the 08:21 GTM error and 08:24
  drift warning as delivered to Signal, so the recovery work includes verifier
  hardening to catch this class as an active health failure instead of relying
  on cron `last_status=ok`.

Gmail realtime no-op canary:

- Latest artifact:
  `/Users/ericfreeman/.hermes/profiles/torben/state/torben-gmail-pubsub-pull-latest.json`.
- Result: `wakeAgent=false`, reason
  `Gmail history fallback processed with no realtime candidates`.
- Diagnostics: `pubsub_messages_received=0`, `pubsub_messages_acked=0`,
  `new_message_count=0`, `gmail_reads=6`, `gmail_writes=0`,
  `external_mutations=0`.

Gmail realtime mutation canary:

- Command:
  `HERMES_HOME=/Users/ericfreeman/.hermes/profiles/torben TORBEN_GMAIL_CANARY_TIMEOUT_SECONDS=90 venv/bin/python profiles/torben/scripts/torben_gmail_realtime_canary.py`
- Result: `status=pass`, `wakeAgent=false`.
- Account: `personal_freeman`.
- Canary message: `19f03fecd8bed563`, subject
  `Torben realtime Gmail canary 1782478458`.
- Pub/Sub/history result: one notification received and acked,
  `new_message_count=1`, processed with no realtime candidate and no Signal
  wake.
- Scoped mailbox mutations: import synthetic message, add `INBOX`/`UNREAD`,
  trash the synthetic message. Diagnostics recorded
  `gmail_mailbox_mutations=3`, `external_mutations=3`, and
  `public_actions_taken=0`.

EA recommendation canary:

- Latest artifact:
  `/Users/ericfreeman/.hermes/profiles/torben/state/torben-email-hygiene-review-actions-latest.json`.
- Result: four approval-only `EA-*` recommendations:
  `EA-20260625-017`, `EA-20260625-018`, `EA-20260625-019`,
  `EA-20260625-020`.
- Diagnostics: `messages_scanned=1623`, `gmail_reads=2335`,
  `gmail_writes=0`, `external_mutations=0`.
- Mutation boundary: weekly review stages recommendations only; it does not
  trash, archive, label, delete, unsubscribe, or send until Eric approves a
  handle.

Calendar alignment canary:

- Latest artifact:
  `/Users/ericfreeman/.hermes/profiles/torben/state/torben-calendar-alignment-audit-latest.json`.
- Latest text output:
  `Torben / Calendar Alignment Audit / 2026-06-26T12:22:18.059998Z`.
- Result: checked 4 accounts, 53 events, found 0 drift candidates.
- Audit: `google_read_api_calls=4`, `google_write_api_calls=0`,
  `external_mutations=0`, warnings empty.
- Mutation boundary: only synthetic private Busy blocks are allowed where policy
  permits; source events are never edited or deleted.

GTM intelligence canary:

- Command:
  `HERMES_HOME=/Users/ericfreeman/.hermes/profiles/torben venv/bin/python profiles/torben/scripts/torben_gtm_radar.py`
- Result: staged `GTM-20260626-001`, `GTM-20260626-002`, and
  `GTM-20260626-003` from Magnus-backed GTM intelligence.
- Provider: `xai-oauth`; model: `grok-4.3`; X Search used: true.
- Magnus refresh result: `success=true`, `scanned_count=341`,
  `finding_count=7`, `public_actions_taken=0`, `external_mutations=0`.
- Selected source links:
  `https://arxiv.org/abs/2606.27027`,
  `https://arxiv.org/abs/2606.26479`,
  `https://arxiv.org/abs/2606.26627`.
- Required invariants held: `public_actions_taken=0`,
  `external_mutations=0`; staged action mutation status is `draft_only` and
  `publishing_blocked_until=explicit_signal_approval`.

GTM engagement canary:

- Command:
  `HERMES_HOME=/Users/ericfreeman/.hermes/profiles/torben venv/bin/python profiles/torben/scripts/torben_gtm_engagement_radar.py`
- Manual result: staged `GTM-20260626-004`, `GTM-20260626-005`, and
  `GTM-20260626-006` reply candidates.
- Scheduler result: the 08:35 cron tick ran the same no-agent production path,
  produced three accepted reply opportunities, and left the next run scheduled
  for `2026-06-26T10:35:00-04:00`.
- Provider: `xai-oauth`; model: `grok-4.3`; X Search used: true.
- Required invariants held: `public_actions_taken=0`,
  `external_mutations=0`; staged action mutation status is `draft_only` and
  `publishing_blocked_until=separate_explicit_public_reply_approval`.

Finance stage-only canary:

- Command:
  `HERMES_HOME=/Users/ericfreeman/.hermes/profiles/torben TORBEN_FINANCE_RADAR_PREVIEW=1 TORBEN_FINANCE_RADAR_FORCE_WAKE=1 TORBEN_FINANCE_PHASE=postmarket UV_PROJECT_ENVIRONMENT=venv uv run --extra dev python profiles/torben/scripts/torben_finance_radar.py`
- Result: emitted a preview `FIN-20260626-001` review card.
- Latest artifact:
  `/Users/ericfreeman/.hermes/profiles/torben/state/torben-finance-radar-latest.json`.
- Required invariants held: `public_actions_taken=0`,
  `external_mutations=0`, `broker_orders_submitted=0`.
- Preview mode did not append a durable FIN action to
  `/Users/ericfreeman/.hermes/profiles/torben/state/torben-action-ledger.json`.

Ratatosk validation:

- `UV_PROJECT_ENVIRONMENT=venv uv run python scripts/robinhood_v01_validate.py --json`
- Result: `ready=true`, `external_mutations=0`.
- Stage decision: `allowed=true`, `action=stage_for_review`.
- Live decision: `allowed=false`, blocked by missing matching human consent,
  active circuit breaker/trading halt, and disabled live env gates.
- Live env preflight:
  `UV_PROJECT_ENVIRONMENT=venv RATATOSK_LIVE_TRADING=true ROBINHOOD_LIVE=true uv run python scripts/robinhood_v01_validate.py --json`
- Result: `external_mutations=0`; stage remains allowed, live decision remains
  blocked by missing matching human consent and active circuit breaker/trading
  halt.
- Guard-only consent check: with explicit in-process consent and live env gates
  set, a `0.01` notional SPY equity candidate still returned `allowed=false`
  because `Circuit breaker tripped at 43.1% drawdown`.
- Broker mutation result: no order placed, cancelled, modified, reviewed, or
  submitted; `external_mutations=0`.
- `UV_PROJECT_ENVIRONMENT=venv uv run python -m pytest tests/test_robinhood_v01 -q`
- Result: 21 passed.
- `UV_PROJECT_ENVIRONMENT=venv uv run python -m pytest tests/test_engine/test_executor_robinhood.py -q`
- Result: 7 passed.

Hidden backend profile boundary:

- `hermes -p magnus cron list`
- Result: no scheduled jobs.
- `hermes -p ratatosk cron list`
- Result: profile `ratatosk` does not exist.

## Intentional Blocks

`TBC-DECIDE-PUBLIC-MUTATION` is resolved for limited testing autonomy.

This run executed only the synthetic Gmail realtime mutation canary. It did not
send email, apply a real hygiene handle, mutate a real calendar attendee event,
post to X, post to LinkedIn, or create any public write.

`TBC-DECIDE-LIVE-FINANCE` is resolved for a tiny testing canary, but the live
finance provider canary is still blocked by Ratatosk safety controls.

This run did not clear Ratatosk halts, place orders, cancel orders, modify
orders, mutate Monarch, or submit broker actions. Ratatosk remains stage-ready
and live-blocked because the active circuit breaker/trading halt is doing its
job.

## Regressions Added

- The Torben finance adapter now fails loud if a Ratatosk stage-only run reports
  `orders_submitted > 0`, `broker_orders_submitted > 0`, or
  `external_mutations > 0`.
- The Torben GTM radar wrapper now has regression coverage proving preview
  refresh behavior, newsletter file inclusion, JSON extraction, and Torben
  profile credential context for hidden Magnus refreshes.
- Live-profile verification now fails if GTM, GTM engagement, or finance latest
  artifacts contain explicit error payloads, failed source refreshes or LLM
  judges, public actions, external mutations, or broker orders hidden behind
  cron `last_status=ok`.

Regression tests:

- `tests/test_torben_finance_radar.py::test_finance_radar_fails_loud_if_ratatosk_reports_stage_only_mutation`
- `tests/test_torben_gtm_radar_script.py::test_torben_gtm_magnus_refresh_uses_torben_profile_credentials`
- `tests/test_signal_coo_live_profile_verify.py::test_live_profile_verify_fails_backend_error_artifact`
- `tests/test_signal_coo_live_profile_verify.py::test_live_profile_verify_fails_backend_forbidden_mutation_count`
