# Webhooks and events

Use webhooks for push notifications about a file or folder. Use Events API polling for catch-up, backfill, or a durable cursor. Webhook management requires a custom OAuth Platform App with the **Manage webhooks** scope; the official Box CLI OAuth app is not sufficient. Use the normal OAuth identity that owns or can access the target, not an administrator identity unless the target operation itself requires it.

## Create and inspect a webhook

```bash
box webhooks:list --json
box webhooks:create folder <FOLDER_ID> \
  --triggers FILE.UPLOADED,FILE.VERSION_UPLOADED \
  --address https://example.com/box/webhook --json
```

The current actor needs access to the target and the app needs appropriate scopes. Confirm the destination URL and event triggers before creating a webhook.

## Poll user events with a durable cursor

For user catch-up and backfill, use the User Events API through the selected OAuth identity. Do not use the CLI's default `box events` command: it defaults to enterprise admin-log streams. Persist the returned `next_stream_position` after every successful response, then use it on the next poll:

```bash
box request /events --query "stream_type=changes&stream_position=now" --json
box request /events --query "stream_type=changes&stream_position=<SAVED_CURSOR>" --json
```

Use `stream_position=now` only to initialize a future-events cursor. For backfill, begin with an approved historical cursor or reconcile the target folder first, then persist each returned cursor atomically with the processed event IDs.

## Application handler contract

When implementing a shipped application:

1. Verify the Box signature before parsing or acting on the body.
2. Persist idempotency keys because deliveries can repeat.
3. Acknowledge quickly and process work asynchronously.
4. Fetch the current file or folder from Box; do not trust an event payload as the final state.
5. Persist the Events API cursor when polling.

Test a valid event, duplicate event, invalid signature, and restart/catch-up path.

## Sources

- [Box webhooks](https://developer.box.com/guides/webhooks/)
- [Events resource](https://developer.box.com/reference/resources/event/)
- [User Events](https://developer.box.com/guides/events/user-events/for-user/)
