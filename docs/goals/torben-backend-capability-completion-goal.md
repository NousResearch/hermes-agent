# Torben Backend Capability Completion Goal

Last refreshed: 2026-06-26 08:55 EDT / 2026-06-26T12:55:32Z.

This goal executes the ADLC build brief:

`docs/build-briefs/torben-backend-capability-completion.json`

Local ADLC checkout:

`/Users/ericfreeman/Downloads/adlc`

## Objective

Finish the backend capabilities needed for Torben to operate as the single Signal-facing COO over three hidden scopes:

- EA / operations: email, calendar, meeting prep, relationship memory, daily brief, action nudges, cleanup proposals.
- GTM / Magnus: intelligence, article ideas, drafts, X/LinkedIn response candidates, X algorithm scoring, performance feedback.
- Finance / Ratatosk: Robinhood research, staged trade candidates, portfolio/risk context, performance learning, later live execution only after explicit approval.

The implementation must preserve one user-facing channel. Magnus and Ratatosk may remain separate repos or hidden submanagers, but they must not become separate Signal operators. Torben owns visible synthesis, approvals, action handles, dedupe, silence policy, and health escalation.

## Current Cutline

ADLC:

- Build brief validates against `/Users/ericfreeman/Downloads/adlc/docs/schemas/build-brief.schema.json`.
- `emit-work-items --dry-run --require-ready` returns ready.
- The brief contains 10 work items and 15 dependency links.
- Two Type-1 decision gates are resolved for testing scope:
  - `TBC-DECIDE-PUBLIC-MUTATION`: limited testing autonomy. Torben may run
    narrow, auditable mutation canaries and apply approved handles; unattended
    public writes remain blocked.
  - `TBC-DECIDE-LIVE-FINANCE`: tiny live finance canary approved for testing
    only after mandate, consent, kill switch, pre-trade guard, and
    reconciliation gates pass. The active Ratatosk circuit-breaker halt remains
    a hard blocker and must not be cleared by this PR.

Live Hermes:

- `hermes -p torben cron list` shows Torben active with morning brief, calendar alignment, Boardy brief, email hygiene weekly review, meeting prep watch, GTM radar, Gmail Pub/Sub pull, Gmail watch renew, live-profile verify, GTM engagement radar, and live-profile investigate.
- `hermes -p magnus cron list` returns no scheduled jobs.
- `hermes -p ratatosk cron list` fails because profile `ratatosk` does not exist.

Live Ratatosk:

- `/Users/ericfreeman/ratatosk/scripts/robinhood_v01_validate.py --json` returns `ready=true`.
- Stage candidate review is allowed.
- Research signals cannot trade directly.
- Live Robinhood order path is still blocked by the active circuit
  breaker/trading halt. A guard-only test with explicit consent and live env
  gates set still returned `allowed=false` with reason `Circuit breaker tripped
  at 43.1% drawdown`.

## Non-Negotiable Boundaries

- No OpenClaw resurrection for this build.
- No second user-facing Signal thread for Magnus or Ratatosk.
- No public posting to X or LinkedIn without an approved handle and adapter proof.
- No Gmail send, trash, delete, archive, or calendar attendee mutation without
  either a synthetic test canary boundary or an approved handle.
- No X or LinkedIn public write without an approved handle and provider-side
  adapter proof.
- No broker order placement, cancellation, margin, options execution, or
  Monarch mutation until the live-finance mandate, consent, kill switch,
  pre-trade guard, and reconciliation canary all pass.
- Crons may schedule evidence collection and invoke bounded LLM analysis. Crons must not encode static final decision logic for nuanced routing, posting, deletion, or trading.
- Successful background maintenance stays silent. Torben only notifies for user-actionable findings, failures, approvals, or degraded capability risk.
- Every visible action must have a stable handle: `EA-*`, `GTM-*`, or `FIN-*`.
- Every external mutation must have an action-ledger record, policy decision, provider result, and rollback or irreversibility note.

## Execution Plan

1. Validate The ADLC Contract

   Run:

   ```bash
   cd /Users/ericfreeman/Downloads/adlc
   bin/adlc validate-artifact --schema build-brief \
     --input /Users/ericfreeman/.hermes/hermes-agent/docs/build-briefs/torben-backend-capability-completion.json \
     --json
   bin/adlc emit-work-items --target linear \
     --build-brief /Users/ericfreeman/.hermes/hermes-agent/docs/build-briefs/torben-backend-capability-completion.json \
     --dry-run --require-ready --json
   ```

   Do not emit tickets unless Eric explicitly asks. Use the brief as the local engineering contract.

2. Implement Torben Submanager Contracts (`TBC-CONTRACTS`)

   Add or tighten the common contract for hidden scopes:

   - capability registry: owner, purpose, tools, auth mode, mutation classes, health state.
   - action envelope: handle, evidence ids, source scope, proposed action, approval state, mutation class, expiry.
   - silence policy: success stays local; failures and approvals surface to Signal.
   - LLM run contract: cron tick id, token budget, sources read, model used, policy version, output schema.
   - health/debug loop: failed live-profile verification can trigger investigation, but patches require explicit approval unless low-risk config repair is pre-authorized.

   Expected files:

   - `hermes_cli/signal_coo/`
   - `profiles/torben/scripts/torben_live_profile_verify.py`
   - `profiles/torben/scripts/torben_live_profile_investigate.py`
   - `docs/operations/torben-submanager-contracts.md`

3. Finish Magnus Hidden GTM Backend (`TBC-GTM-MAGNUS`)

   Keep Magnus as the GTM/content engine, but route visible output through Torben.

   Requirements:

   - GTM intelligence loop: arXiv, security/AI/security+AI sources, X/YCombinator/HN/THN/newsletters, source health, dedupe, trend clustering.
   - GTM content loop: article opportunities, article drafts, X/LinkedIn post drafts, visuals/image prompts, founder/Magnus voice.
   - GTM engagement loop: timely X/LinkedIn reply candidates, response windows, X algorithm scoring lens, public-action gate.
   - Grok/xAI should be the primary GTM judgment path where available; Codex/OpenAI can be used for code, long-form assembly, QA, and local engineering tasks.
   - Public posting remains disabled until `TBC-DECIDE-PUBLIC-MUTATION` and `TBC-PUBLIC-ADAPTERS`.

   Tests should prove:

   - crons invoke bounded LLM analysis, not static routing dressed up as automation.
   - no duplicate GTM findings are sent.
   - findings include enough context to act: why this matters, draft angle, links, and recommended next step.
   - `public_actions_taken=0` unless approved.

4. Finish EA Mutation Backends (`TBC-EA-MUTATION`)

   Keep EA as Torben-owned operations.

   Requirements:

   - Gmail realtime triage watches all authenticated accounts and turns net-new mail into LLM-scored action candidates.
   - Relationship learning asks one short question for important unknown senders and updates durable context.
   - Morning brief dedupes against realtime alerts and focuses on prior-day security/AI/tool/newsletter signal plus important unresolved replies.
   - Calendar alignment can maintain private busy blocks where policy already allows it, with quiet success.
   - Gmail cleanup/deletion is weekly recommendation-first until mutation policy is approved.
   - Staged response cards include one to two lines of context and the actual proposed response summary.

   Public/personal mutation examples that require an approved handle outside
   synthetic canary boundaries:

   - send email
   - trash/delete/archive Gmail
   - create calendar events with attendees
   - edit/delete real calendar events
   - public posting

5. Finish Ratatosk Hidden Finance Backend In Stage-Only Mode (`TBC-RATATOSK-FINANCE`)

   Do not create a separate Signal-facing finance agent. Either call the Ratatosk repo from Torben scripts or create a hidden/local-only Ratatosk submanager profile. Torben remains the visible operator.

   Requirements:

   - Robinhood-only v0.1 for now.
   - Cron ticks mint one bounded LLM analysis run.
   - Research signals create candidates, never orders.
   - Candidates include thesis, entry/exit idea, risk, horizon, source refs, and guard result.
   - Performance ledger records signal quality, execution quality, PnL, risk behavior, and strategy score.
   - Stage-only finance sends `FIN-*` review cards through Torben.
   - Live order path stays blocked until `TBC-DECIDE-LIVE-FINANCE`.

   Ratatosk validation:

   ```bash
   cd /Users/ericfreeman/ratatosk
   UV_PROJECT_ENVIRONMENT=venv uv run python scripts/robinhood_v01_validate.py --json
   UV_PROJECT_ENVIRONMENT=venv uv run python -m pytest tests/test_robinhood_v01 -q
   UV_PROJECT_ENVIRONMENT=venv uv run python -m pytest tests/test_engine/test_executor_robinhood.py -q
   ```

   Known risk to resolve before live finance:

   - focused executor tests have previously failed around submitted/unfilled Robinhood orders being treated as positions. Fix the behavior or the tests before live execution.

6. Resolve Public Mutation Gate And Enable Adapters (`TBC-PUBLIC-ADAPTERS`)

   Only after `TBC-DECIDE-PUBLIC-MUTATION` is resolved:

   - X create-post adapter: approval-gated, evidence logged, no blind public write.
   - LinkedIn Posts API adapter: approval-gated, member/org target explicit.
   - Gmail send/trash/archive/delete adapter: approval-gated and reversible status explicit.
   - Calendar mutation adapter: approval-gated, attendee-impact explicit.

   Canary requirements:

   - dry-run card
   - explicit approval handle
   - provider-side result
   - action ledger
   - rollback or irreversibility note
   - failure stays actionable but not noisy

7. Resolve Live Finance Gate And Enable Trading (`TBC-LIVE-FINANCE`)

   The live-finance policy is resolved as a testing-only tiny canary, but
   execution remains blocked while the Ratatosk circuit breaker/trading halt is
   active.

   - write mandate and consent artifacts.
   - preserve the active circuit breaker/trading halt unless Eric separately
     approves clearing it with current risk evidence.
   - enable live env gates only for the canary.
   - prove review before order.
   - prove reconciliation after order.
   - prove halt on divergence.
   - keep margin/options within explicit mandate only.

   First live canary should be the smallest possible permitted path. If options are enabled, force review-option-order before place-option-order and keep long premium-only constraints.

8. End-To-End Canary And Debug Loop (`TBC-VAL-CANARY`)

   Required proof:

   - Torben active cron list is clean.
   - live-profile verifier passes.
   - investigation job is quiet unless risk exists.
   - Gmail realtime canary detects a net-new email.
   - calendar alignment stays quiet on success.
   - morning brief dedupes realtime items.
   - GTM radar emits useful, link-heavy, non-duplicate intelligence.
   - GTM engagement radar emits timely reply candidates.
   - Ratatosk stage-only finance emits a `FIN-*` candidate and records no broker mutation.
   - public mutation canary passes only if approved.
   - live finance canary either passes only if approved and all guardrails are
     green, or records a fail-closed blocked-live result with zero broker
     mutations.

   Hermes validation:

   ```bash
   cd /Users/ericfreeman/.hermes/hermes-agent
   git diff --check
   venv/bin/python -m pytest \
     tests/test_signal_coo.py \
     tests/test_signal_coo_gmail_realtime.py \
     tests/test_signal_coo_live_profile_verify.py \
     tests/test_torben_gtm_engagement_radar.py \
     tests/gateway/test_torben_gtm_reply_router.py \
     -q
   hermes -p torben cron list
   ```

## Source References

- X create post API: https://docs.x.com/x-api/posts/create-post
- LinkedIn Posts API: https://learn.microsoft.com/en-us/linkedin/marketing/community-management/shares/posts-api
- Gmail send: https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/send
- Gmail trash: https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/trash
- Google Calendar event creation: https://developers.google.com/workspace/calendar/api/guides/create-events
- Robinhood Agentic Trading: https://robinhood.com/us/en/agentic-trading
- Robinhood Agentic Trading overview: https://robinhood.com/us/en/support/articles/agentic-trading-overview/
- Monarch MCP connector: https://help.monarch.com/hc/en-us/articles/50207234679956-Monarch-MCP-Connector

## Definition Of Done

This goal is complete when Torben is the only visible operator and can:

- brief and nudge the EA lane with deduped realtime plus morning context.
- call Magnus for GTM intelligence/content/engagement and surface useful `GTM-*` actions.
- call Ratatosk for finance/trading intelligence and surface useful `FIN-*` actions.
- stage or execute approved external mutations according to the resolved gates.
- stay quiet for successful maintenance.
- escalate failed capability health with actionable debug evidence.
- pass repo tests, ADLC validation, live cron checks, canaries, and provider-side mutation proofs where mutation is enabled.

Do not mark this complete because the brief is valid. Mark it complete only when the backend behavior is operational and verified.
