# Control Plane / Discord Bot-to-Bot Decommission Plan

## Problem

Discord bot-to-bot messages were being used as an authority/control path between Hermes profiles. That failed because Discord is an unreliable authority plane: delivery is asynchronous, channel/thread routing is lossy, bot admission policy is complex, and structured BOT_MSG envelopes can create loops or bypass human-facing routing assumptions.

## Target State

1. Discord gateway returns to baseline mirror/ingress behavior.
2. Legacy Discord BOT_MSG authority is decommissioned, not env-gated: Discord bot-authored messages are ignored and bot-message model tools are not registered.
3. Local SQLite control-plane DB under the root Hermes home is the deterministic authority surface for profiles, routes, durable messages, approvals, dispatch lifecycle, and outbox observer events.
4. DB operations are explicit, redacted, lease/epoch fenced, and test-covered.

## Slices and Review Gates

### Slice 1 — DB bootstrap/schema/helper
- Add `hermes_cli/control_db.py`.
- Use root Hermes home, not active profile home, for shared control-plane state.
- Enable SQLite WAL, busy timeout, foreign keys.
- Refuse unsupported/partial schema versions instead of silently blessing unknown DBs.
- Gate: schema/concurrency tests and `py_compile`.

### Slice 2 — profiles/liveness/doctor/routes
- Register profiles and live instances.
- Add doctor checks for wrong root and stale instances.
- Implement default-deny route policy with admin/bootstrap-only mutation and deny-wins tie behavior.
- Gate: route/admin/liveness tests.

### Slice 3 — durable message/outbox/redaction/HMAC
- Store redacted message bodies/metadata.
- Create observer outbox rows for state transitions.
- HMAC external refs with a local secret scoped to the DB root.
- Gate: redaction and HMAC tests.

### Slice 4 — approvals and Discord authority removal
- Mirror dangerous command approvals into `cp_approvals`.
- Persist approval decision and consume approved grants atomically.
- In `control_db` authority mode, fail closed if DB approval persistence/consume fails.
- Remove `send_bot_message` / `send_bot_approval_decision` registration and handlers.
- Ignore inbound Discord bot messages/approval decisions regardless of legacy env vars.
- Gate: approval persistence tests and legacy-disable tests.

### Slice 5 — dispatch lifecycle
- Add durable dispatch rows, claim leases, lease epochs, live-lease advance, watchdog/reap to retry or dead-letter.
- Gate: single-winner, stale epoch, expired lease, invalid-status tests.

### Slice 6 — Discord gateway baseline
- Remove default operational routing guard/bot authority behavior from normal Discord gateway path.
- Do not preserve a Discord legacy bot-to-bot rollback path; rollback should be a git revert, not a runtime env switch.
- Gate: gateway bot filter/hardening tests updated to assert bot authors are ignored even when legacy env vars are set.

## Verification

- Targeted: control DB, approvals, Discord bot admission/decommission, send-message registry, gateway hardening.
- Syntax: `py_compile` for modified modules.
- Broad: run `pytest -q -o addopts='' --maxfail=1 tests`; classify unrelated environmental/baseline failures separately.

## Residual Risk

This introduces the DB primitives and disables Discord as default authority. It does not restart live gateway processes, migrate live profile policy, or build a full worker daemon consuming inbox/outbox. Those are deliberate non-actions unless separately approved.
