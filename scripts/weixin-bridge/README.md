# Hermes Weixin Bridge

Local Node.js bridge used by Hermes Agent's `weixin` platform.

This folder is intentionally self-contained so it can be split into a standalone repo later with minimal churn.

## What It Implements

- QR login against Tencent iLink bot endpoints
- Long-poll `getupdates`
- Text reply delivery through `sendmessage`
- Local persistence for session credentials, reply context tokens, and polling cursors

## Run Standalone

```bash
cd scripts/weixin-bridge
npm install
node bridge.js --port 3010 --session ~/.hermes/weixin/session
```

Optional environment variables:

- `WEIXIN_BOT_TYPE` — QR login bot type override (default: `3`)
- `WEIXIN_APP_ID` — iLink app id header override (default: `bot`)

## HTTP Endpoints

- `GET /health`
  Returns bridge status, paired account/user IDs, QR fallback URL, queue depth, and last error.
- `GET /messages`
  Drains the in-memory inbound queue and returns normalized message events.
- `POST /send`
  Sends a text reply.
  JSON body: `{"chatId":"<weixin-id>","message":"hello"}`
- `POST /typing`
  Best-effort typing endpoint. Currently a no-op ack used by Hermes.
- `GET /chat/:id`
  Returns minimal chat metadata for a Weixin conversation.

## Persisted State

The bridge writes these files under the session directory:

- `credentials.json`
- `context-tokens.json`
- `get-updates-buf.txt`

`context-tokens.json` lets Hermes keep replying in the same Weixin context when Tencent requires `context_token`.

## Split-To-Repo Notes

If this becomes a separate repo/package, keep the current contract stable:

- `bridge.js` remains the entrypoint
- `/health`, `/messages`, `/send`, `/typing`, `/chat/:id` stay compatible
- session persistence format remains backward compatible when possible

That keeps Hermes core able to vendor or depend on the bridge without changing the Python adapter.
