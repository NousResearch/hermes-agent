# Webhooks and Events

## Webhooks vs events

| Mechanism | Use when |
| --- | --- |
| **Webhooks** | Push notifications for file/folder changes |
| **Events API** | Catch-up sync, polling, backfill after downtime |

## CLI setup (agent tasks)

```bash
box webhooks:list --json
box webhooks:create --help   # confirm flags for your CLI version
box events --help
```

Service account must have scope and access to the target folder.

## Application implementation

When building a shipped app (not one-off agent tasks), see `references/sdk-development.md`:

1. Confirm which Box actor owns the subscription
2. Store signing secrets in env / secret manager — not in repo
3. Verify webhook signatures **before** parsing body
4. Persist idempotency keys — Box may deliver duplicates
5. Fetch file/folder metadata after event; do not trust payload alone

## Verification checklist

- Happy path: receive → verify signature → fetch metadata → log Box ID
- Duplicate delivery: same payload twice → one downstream action
- Bad signature: reject → no side effects
- Catch-up: events cursor/checkpoint persisted if using Events API

## Docs

- https://developer.box.com/guides/webhooks/
- https://developer.box.com/reference/resources/event/
