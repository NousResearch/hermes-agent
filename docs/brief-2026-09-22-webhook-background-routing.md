# Fix webhook background-task completion routing

## Problem

A one-shot webhook agent run can launch a terminal process with completion notification enabled. The webhook adapter closes its per-delivery session when the main turn finishes. If the background task completes afterward, `GatewayNotificationsMixin._build_process_event_source()` cannot restore the ended webhook session origin and falls back to `_parse_session_key()`.

Webhook session keys contain colon-bearing chat and user identifiers:

`agent:main:webhook:webhook:webhook:<route>:<delivery-id>:webhook:<route>`

The generic parser reduces this to `platform=webhook`, `chat_type=webhook`, `chat_id=webhook`. Rebuilding a source from that data produces a different session key. The base adapter rejects the internal completion event and `_drain_watch_notifications()` requeues it every two seconds indefinitely.

Observed production evidence:

- Expected key: `agent:main:webhook:webhook:webhook:pump-pr-events:<delivery-id>:webhook:pump-pr-events`
- Rebuilt key: `agent:main:webhook:webhook:webhook:webhook:pump-pr-events`
- Repeating warning: `Dropping internally routed event: expected session=... derived=...`

## Required behavior

1. A background task launched from a webhook run may finish after that webhook session is closed.
2. Its completion event must not enter an unbounded retry loop.
3. Routing recovery must preserve the webhook delivery's exact identity or otherwise terminally handle an ended one-shot webhook session.
4. Existing routing for Slack, Telegram, API-server sessions, named profiles, and relay scope must remain unchanged.
5. Do not weaken the expected-session-key guard or bypass session isolation.

## Tests first

Add a focused regression test that reproduces the real sequence:

1. Build a webhook source with a colon-bearing route and delivery ID.
2. Generate its actual session key.
3. Remove/close the session origin so recovery must use durable watcher metadata or the supported fallback.
4. Inject a completion/watch event.
5. Prove the event is either admitted to the exact original webhook identity or terminally acknowledged without requeue.
6. Prove a mismatched ordinary internal event is still rejected.

Run the test before implementation and record the expected failure.

Likely owner files:

- `gateway/run_notifications.py`
- `gateway/platforms/webhook.py`
- `tests/gateway/test_background_process_notifications.py`
- optionally the narrow watcher metadata producer if exact webhook identity must be persisted

## Acceptance

- The regression test fails before the fix and passes afterward.
- The event does not reappear in the completion queue after terminal handling.
- Existing background-notification and webhook adapter tests pass.
- No broad session-key format migration.
- No changes to delivery targets, webhook signatures, or user-facing webhook behavior.

## Scope

Target: no more than about 100 production lines plus focused tests. Prefer preserving full routing metadata at dispatch over teaching the generic session-key parser every platform-specific key grammar.

Do not merge, deploy, restart gateways, or call production systems.
