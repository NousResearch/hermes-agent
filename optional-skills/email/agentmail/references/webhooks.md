# AgentMail Webhooks

Use webhooks when a public HTTPS server should receive AgentMail events. Use
[websockets.md](websockets.md) for local processes without a public URL.

## Create

```bash
agentmail webhooks create \
  --url https://your-app.example.com/webhooks/agentmail \
  --event-type message.received \
  --inbox-id support@agentmail.to \
  --client-id support-agentmail-webhook \
  --format json
```

Store the returned `secret` immediately.

## Handle

Headers: `svix-id`, `svix-timestamp`, `svix-signature`.

1. Verify the raw request body with the webhook secret.
2. Dedupe by `svix-id` or `event_id`.
3. Return `200` quickly.
4. Process asynchronously.
5. For `message.received`, load the thread with the CLI and reply if needed.

Act on `message.received` only. Treating `message.sent` or delivery events as
inbound work creates loops.
