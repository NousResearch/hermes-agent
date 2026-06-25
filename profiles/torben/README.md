# Torben Profile Snapshot

This directory is the source-controlled, safe backup of the live Torben profile.
It is not a clone of `/Users/ericfreeman/.hermes/profiles/torben`.

Included:

- `SOUL.md`: Torben's operator persona and standing behavior contract.
- `scripts/`: profile-local cron wrappers that call repo-owned `hermes_cli.signal_coo` code.
- `skills/agent-ops/torben-operator-maintenance/SKILL.md`: maintenance rules for future Torben edits.
- `config/relationship_context.yaml`: current relationship/source routing context.
- `cron/jobs.snapshot.json`: sanitized schedule, delivery, script, and prompt definitions for Torben jobs.

GTM radar replies are handled by repo code before the generic agent loop:
`hermes_cli.signal_coo.gtm_reply_router.route_gtm_radar_reply` resolves handles
and aliases from Signal replies, stages approval-ready draft packages with
article, LinkedIn, X thread, X single-post, and visual-plan draft objects, and
records ledger resolution history without public posting.

The `torben-gtm-radar` cron is Torben-owned. It refreshes the Magnus GTM
intelligence radar under the Magnus profile, adapts only new findings into
Signal actions, and stays silent when there is no new signal. Preview mode is
non-mutating; refresh failures are actionable because the source loop is then
not trustworthy.

The `torben-gtm-engagement-radar` cron is the response-opportunity loop. It
uses the latest GTM radar findings as topics, invokes Grok with X Search, stages
draft-only reply ideas with the public X algorithm lens, and stays silent when
there is no fresh opportunity. It does not post or reply publicly.

Realtime Gmail is Pub/Sub-backed. `torben_gmail_watch_register.py` registers or
renews Gmail watches for the enabled OAuth accounts, and
`torben_gmail_pubsub_pull.py` pulls only Gmail change notifications, advances
history cursors, stages draft-only EA actions, and stays silent on no-op runs.
The old `torben-realtime-email-watch` polling cron is kept disabled as a manual
fallback to avoid duplicate alerts.

`torben_live_profile_verify.py` is the recurring drift guard for this snapshot:
enabled live cron scripts must exist, compile, have clean cron error fields, and
match the repo copy. `torben_gmail_realtime_canary.py` is manual only; it creates
and trashes one controlled Gmail canary message to prove Pub/Sub/history
processing end to end.

GTM package authoring is Grok-first through `xai-oauth`. The writer calls
xAI's Responses API with the `x_search` tool enabled by default, then stores the
Grok-authored drafts in the local approval package. If Grok or X Search is down,
the package falls back to deterministic Magnus-radar drafting and records the
failure in `grok_authoring`.

Excluded on purpose:

- `.env`, `runtime.env.op`, OAuth tokens, provider credentials, and 1Password material.
- Signal account state, attachments, avatars, and daemon data.
- SQLite state databases, action ledgers, generated evidence artifacts, cron outputs, logs, locks, and PID files.
- Decommission archives and old OpenClaw/Hermes runtime state.

Restore shape:

1. Restore the Hermes repo code.
2. Recreate or repair `/Users/ericfreeman/.hermes/profiles/torben`.
3. Copy `SOUL.md`, `scripts/`, `skills/`, and `config/relationship_context.yaml` from this snapshot into the live profile.
4. Recreate cron jobs from `cron/jobs.snapshot.json` using `hermes -p torben cron create`.
5. Re-auth providers through OAuth/MCP registration. Do not restore tokens from git.
6. Run the Torben verifier set before enabling live jobs.

This snapshot contains personal relationship context. Keep it in a private repo.
