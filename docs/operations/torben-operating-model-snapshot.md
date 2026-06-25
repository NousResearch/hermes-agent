# Torben Operating Model Snapshot

Status: commit-ready source snapshot
Date: 2026-06-25
Primary profile: `/Users/ericfreeman/.hermes/profiles/torben`
Repo snapshot: `profiles/torben/`

## Intent

Torben is the single Signal-facing COO operator. The live product direction is
Hermes-native, not OpenClaw-mediated. EA and GTM/Magnus are active slices;
Finance is authenticated but still needs scoped risk gates before live use. All
scopes share the same Signal/action-ledger operating model.

Crons are wake-up triggers. Static code collects bounded evidence, enforces
mutation policy, writes action handles, and verifies provider results. LLMs
decide whether the evidence is worth interrupting Eric, synthesize the message,
draft replies, and keep the conversation useful.

## Current Commit Boundary

Commit as the Torben source/operating snapshot:

- `hermes_cli/main.py`
- `hermes_cli/subcommands/torben.py`
- `hermes_cli/signal_coo/*.py`
- `tests/test_signal_coo.py`
- `docs/prds/hermes-native-signal-coo-operator-prd.md`
- `docs/build-briefs/hermes-native-signal-coo-operator.json`
- `docs/security/torben-runtime-secrets-and-host-cleanup.md`
- `docs/operations/torben-operating-model-snapshot.md`
- `profiles/torben/**`

Keep out of this commit:

- `AutoHedge/` and `tinker-atropos/`: local reference clones, not Hermes source.
- `state/`: generated wiki/runtime cache.
- `mcp_cron.py` and `tests/test_mcp_cron.py`: legacy OpenClaw/cron-MCP bridge
  artifact, not required by the Torben reset.
- `/Users/ericfreeman/.hermes/profiles/torben/.env`, OAuth tokens, Signal data,
  DBs, logs, cron outputs, action ledgers, evidence artifacts, locks, and PIDs.

## Live Torben Jobs

The safe cron source lives in `profiles/torben/cron/jobs.snapshot.json`.
The live Torben profile has `cron.wrap_response=false`; background jobs should
not produce generic `Cronjob Response` wrappers. Jobs should be silent on
non-actionable success and notify only for actionable items or failures.

Active jobs at snapshot time:

- `torben-morning-brief`: `0 8 * * *`, Signal, LLM-backed, one-screen daily brief.
- `torben-calendar-alignment-watchdog`: `every 5m`, `no_agent=true`, silent
  background calendar alignment with private reminder-free busy blocks.
- `torben-realtime-email-watch`: disabled legacy 10-minute Gmail polling
  fallback. Keep disabled while Pub/Sub realtime is healthy to avoid duplicate
  email alerts.
- `torben-gmail-pubsub-pull`: `every 1m`, Signal, LLM-backed Gmail Pub/Sub
  realtime triage. Gmail notifications trigger bounded history reads; no-op
  pulls return `wakeAgent=false` and stay silent.
- `torben-gmail-watch-renew`: `15 3 * * *`, `no_agent=true`, daily Gmail
  watch renewal. Success is silent; registration failure fails the cron.
- `torben-live-profile-verify`: `every 15m`, `no_agent=true`, live profile
  drift guard. It verifies enabled cron scripts exist in the live profile,
  compile, have no stale `last_error` / `last_delivery_error`, and match the
  repo snapshot. Success is silent; drift or cron health errors fail loudly.
- `torben-boardy-brief`: `30 11,16 * * *`, Signal, twice-daily Boardy digest.
- `torben-email-hygiene-weekly-review`: `0 10 * * 1`, Signal, approval-gated
  cleanup recommendations only.
- `torben-meeting-prep-watch`: `every 5m`, Signal, rolling pre-call context
  alert; silent when no real meeting is approaching.
- `torben-gtm-radar`: `20 8,13,18 * * *`, Signal, no-agent adapter that
  refreshes the Magnus GTM intelligence radar under the Magnus profile before
  adapting findings; silent when there are no new findings.
- `torben-gtm-engagement-radar`: `35 8,10,12,14,16,18 * * *`, Signal,
  no-agent Grok/X Search response-opportunity scan. It selects the latest GTM
  topics, asks Grok with X Search for concrete reply opportunities, stages
  draft-only local reply ideas, and stays silent when there are no fresh
  opportunities.

## Auth Cutover

Safe repo snapshot: `profiles/torben/config/auth_cutover.snapshot.yaml`.

Torben uses OAuth/provider-native auth first. The live `runtime.env.op` is
limited to local Signal bootstrap values:

- `SIGNAL_ACCOUNT`
- `SIGNAL_HTTP_URL`

OpenAI uses Hermes auth provider `openai-codex`; Magnus/GTM uses
`xai-oauth`; Google uses per-account OAuth token files; finance uses hosted
OAuth MCP sessions. Do not add provider API keys, Google token JSON, X API
tokens, broker credentials, or Monarch passwords to the Torben runtime env.

Gmail realtime uses Google Cloud Pub/Sub in project `sigma-zodiac-485821-f0`:

- Topic: `projects/sigma-zodiac-485821-f0/topics/torben-gmail-watch`
- Subscription:
  `projects/sigma-zodiac-485821-f0/subscriptions/torben-gmail-watch-pull`
- Gmail publisher IAM member:
  `serviceAccount:gmail-api-push@system.gserviceaccount.com`

The watch state lives only in the live Torben profile at
`/Users/ericfreeman/.hermes/profiles/torben/state/torben-gmail-watch-state.json`.
It stores account aliases, emails, Gmail history cursors, and watch expiration
times; it must not be committed.

Manual Gmail realtime canary:
`profiles/torben/scripts/torben_gmail_realtime_canary.py` creates one
controlled canary message, adds the watched `INBOX` label, verifies Pub/Sub and
Gmail history processing, and trashes the canary message afterward. This is not
a scheduled job because it intentionally performs bounded Gmail mailbox
mutations for proof.

Finance MCP servers configured in the live Torben profile:

- `robinhood-agentic-mcp`: `https://agent.robinhood.com/mcp/trading`,
  `auth: oauth`
- `monarch-money-mcp`: `https://api.monarch.com/mcp`, `auth: oauth`

`torben auth-check --json` must fail closed if either required finance MCP
connector is missing or disabled. Hosted MCP OAuth login is complete for both
finance connectors; any finance mutation still needs the scoped risk gates and
approval policy before use.

## Magnus GTM Adapter

The GTM adapter refreshes the Magnus GTM intelligence radar first, then reads
`/Users/ericfreeman/magnus/state/gtm-intelligence-radar/latest.json`, stages
Torben action handles for new findings, and sends one concise Signal note with
reply aliases such as `draft 1`, `source 1`, and `hold 1`. A successful
no-new-finding run stays silent; a refresh failure is actionable and should
surface. The adapter does not post publicly, reply publicly, schedule content,
or mutate X/LinkedIn.

Preview mode must not stage ledger actions or mark findings delivered. Real cron
runs stage draft-only actions and mark only delivered finding fingerprints so the
same radar item does not create repeat chat noise.

Runtime toggles:

- `TORBEN_GTM_REFRESH_MAGNUS=0`: skip the upstream Magnus refresh and adapt the
  latest saved artifact only.
- `TORBEN_GTM_REFRESH_MAGNUS_PREVIEW=1`: allow preview runs to perform a
  dry-run Magnus refresh without persisting seen-state changes.
- `TORBEN_GTM_MAGNUS_HOURS_BACK`, `TORBEN_GTM_MAGNUS_MAX_ITEMS`, and
  `TORBEN_GTM_MAGNUS_MIN_SCORE`: bound the upstream radar scan.
- `TORBEN_GTM_NEWSLETTER_FILE` or `TORBEN_GTM_NEWSLETTER_FILES`: pass EA
  newsletter extraction artifacts into the Magnus radar.
- By default the Torben wrapper also passes
  `state/torben-morning-brief-inbox-context-latest.json` when it exists, so
  previous-day newsletter security, AI, and tool candidates feed the GTM radar.
  Set `TORBEN_GTM_INCLUDE_DEFAULT_NEWSLETTER=0` to disable that bridge.
- `TORBEN_GTM_MAGNUS_TIMEOUT_SECONDS`: cap the upstream refresh.
- `TORBEN_GTM_ENGAGEMENT_MAX_TOPICS` and
  `TORBEN_GTM_ENGAGEMENT_MAX_OPPORTUNITIES`: bound the Grok/X response scan.
- `TORBEN_GTM_ENGAGEMENT_MODEL`: override the default `grok-4.3` response
  opportunity model.
- `TORBEN_GTM_ENGAGEMENT_PREVIEW=1`: run the response-opportunity script
  without staging ledger actions or marking opportunities delivered.

X distribution lens:

- GTM radar actions and draft packages include the public
  `https://github.com/xai-org/x-algorithm` repository as a distribution
  pressure-test. The lens is used to improve hooks, reply value, repost/quote
  value, profile-click intent, dwell, and follow intent while avoiding
  not-interested, block, mute, and report signals.
- Treat this as directional public evidence from the open repository, not a
  claim about exact private ranking weights. It should shape drafts and response
  ideas; it should not create a standalone recurring Signal report.

## GTM Reply Router

Signal replies to GTM radar actions are intercepted before the generic agent
loop. `gateway/run.py` calls `route_gtm_radar_reply` against the Torben action
ledger when an inbound Signal message references an open Magnus GTM radar action
by handle or alias, such as `draft 2`, `source 2`, `hold 2`, or a multi-handle
message asking to thread an article together.

The router is approval-gated. It can stage a local GTM content package artifact
under `/Users/ericfreeman/.hermes/profiles/torben/state/gtm-content-packages/`
with article, LinkedIn, X thread, X single-post, and visual-plan draft objects;
append resolution history to the source radar actions; and send a concise Signal
acknowledgement with reviewable previews and approval options. It must not post,
schedule, reply publicly, or mutate X/LinkedIn.

Authoring is Grok-first. `hermes_cli.signal_coo.gtm_grok_writer` uses Torben's
`xai-oauth` credentials to call xAI's Responses API with the `x_search` tool
enabled by default, then merges Grok-authored article/LinkedIn/X/visual drafts
back into the local package. If Grok auth/API/search fails, the router keeps the
Signal flow useful by falling back to deterministic Magnus-radar drafting and
recording `grok_authoring.status=failed`. Runtime toggles:

- `TORBEN_GTM_GROK_DRAFTING=0`: disable live Grok authoring.
- `TORBEN_GTM_GROK_X_SEARCH=0`: disable X Search context inside the Grok call.
- `TORBEN_GTM_GROK_MODEL`: override the default `grok-4.3` authoring model.

Current live backfill from the 2026-06-25 GTM radar reply staged
`GTM-20260625-005` from `GTM-20260625-002` and `GTM-20260625-003` with
`mutation_status=approval_ready_draft_package`, `content_package_status=approval_required`,
Grok-authored article/LinkedIn/X/visual draft assets, `draft_source=grok`,
`grok_authoring.status=success`, `grok_authoring.provider=xai-oauth`,
`grok_authoring.x_search_enabled=true`, `public_actions_taken=0`, and
`external_mutations=0`; `GTM-20260625-001` remains open.

## Backup Rule

The repo should back up source, prompts, and safe relationship/source context.
The live profile should keep secrets and mutable runtime state. If a future
change matters operationally, update both the live profile and the safe snapshot
in this repo before treating the work as durable.

## Verification Set

Run from `/Users/ericfreeman/.hermes/hermes-agent`:

```bash
UV_PROJECT_ENVIRONMENT=venv uv run --extra dev python -m py_compile \
  hermes_cli/signal_coo/*.py \
  /Users/ericfreeman/.hermes/profiles/torben/scripts/torben_*.py

UV_PROJECT_ENVIRONMENT=venv uv run --extra dev python -m pytest tests/test_signal_coo.py -q
UV_PROJECT_ENVIRONMENT=venv uv run --extra dev python -m pytest \
  tests/test_signal_coo_gmail_realtime.py -q
UV_PROJECT_ENVIRONMENT=venv uv run --extra dev python -m pytest \
  tests/test_signal_coo_live_profile_verify.py -q
UV_PROJECT_ENVIRONMENT=venv uv run --extra dev python -m pytest \
  tests/gateway/test_torben_gtm_reply_router.py -q

git diff --check
```

Live sanity checks:

```bash
jq '.jobs[] | select(.name|startswith("torben-")) |
  {name,schedule_display,enabled,no_agent,deliver,script,last_status,last_delivery_error}' \
  /Users/ericfreeman/.hermes/profiles/torben/cron/jobs.json

HERMES_HOME=/Users/ericfreeman/.hermes/profiles/torben \
TORBEN_MEETING_PREP_WATCH_PREVIEW=1 \
TORBEN_MEETING_PREP_WINDOW_MINUTES=30 \
UV_PROJECT_ENVIRONMENT=venv uv run --extra dev python \
  /Users/ericfreeman/.hermes/profiles/torben/scripts/torben_meeting_prep_watch.py

UV_PROJECT_ENVIRONMENT=venv uv run python \
  profiles/torben/scripts/torben_gmail_watch_register.py

UV_PROJECT_ENVIRONMENT=venv uv run python \
  profiles/torben/scripts/torben_gmail_pubsub_pull.py

HERMES_HOME=/Users/ericfreeman/.hermes/profiles/torben \
TORBEN_GMAIL_CANARY_ACCOUNT=personal_michael \
UV_PROJECT_ENVIRONMENT=venv uv run python \
  /Users/ericfreeman/.hermes/profiles/torben/scripts/torben_gmail_realtime_canary.py

HERMES_HOME=/Users/ericfreeman/.hermes/profiles/torben \
UV_PROJECT_ENVIRONMENT=venv uv run python \
  /Users/ericfreeman/.hermes/profiles/torben/scripts/torben_live_profile_verify.py
```
