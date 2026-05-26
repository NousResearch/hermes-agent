# Slack Bot Bridge

Bot-to-bot ingress for Hermes instances that live behind a firewall. An
external bot — a GitHub Actions workflow, a webhook relay, a CI runner —
posts a tagged JSON envelope into a single dedicated Slack channel, and
Hermes treats it as if a human had typed the prompt themselves.

The plugin **does not** patch the Slack adapter or the gateway runner. It
hooks `pre_gateway_dispatch`, validates the envelope, deduplicates by
`request_id`, and rewrites the Slack message into a canonical prompt before
the normal dispatch path takes over.

## Enable

```bash
hermes plugins enable slack-bot-bridge
```

(Or check the box in the interactive `hermes plugins` UI.)

## Configure

### 1. Lock the Slack adapter to the bridge channel

In `~/.hermes/config.yaml`:

```yaml
platforms:
  slack:
    extra:
      allow_bots: mentions      # let bot/webhook messages through
      strict_mention: true      # but only when they @ Hermes
      allowed_channels:
        - C0123456789           # the bridge channel ID — single channel
```

Without `allow_bots: mentions` the Slack adapter drops bot messages before
the plugin ever sees them.

### 2. Allow the relay bot's Slack user id

The plugin rewrites `event.text` but cannot change `event.source.user_id`.
Slack stamps bot events with the sending bot's user id (`U0…`), so Hermes'
own authorization check still runs against that id. Add the relay bot's
user id to `SLACK_ALLOWED_USERS`:

```bash
# in ~/.hermes/.env
SLACK_ALLOWED_USERS=U_OWNER,U_RELAY_BOT
```

If you skip this step, the plugin will rewrite the prompt and the gateway
will then silently reject it as unauthorized.

### 3. Tell the plugin which channel and which bots to accept

```bash
# ~/.hermes/.env
HERMES_SLACK_BRIDGE_CHANNEL=C0123456789
HERMES_SLACK_BRIDGE_ALLOWED_BOT_IDS=B0AAAAAAA
HERMES_SLACK_BRIDGE_ALLOWED_APP_IDS=A0BBBBBBB
HERMES_SLACK_BRIDGE_ALLOWED_TEAMS=T0CCCCCCC

# Optional: require an HMAC signature on every envelope.
HERMES_SLACK_BRIDGE_HMAC_SECRET=use-a-real-secret-here

# Optional: override the 24-hour dedup window (0 = never expire entries).
HERMES_SLACK_BRIDGE_DEDUP_TTL_SECONDS=86400
```

- `HERMES_SLACK_BRIDGE_CHANNEL` is **required** — the plugin is inert
  without it.
- Team allowlists are combined with sender identity allowlists: if
  `ALLOWED_TEAMS` is set, the event team must match; if either
  `ALLOWED_BOT_IDS` or `ALLOWED_APP_IDS` is set, one of those sender
  identities must match too.
- Leave all three lists empty for channel-only gating during quick bring-up
  only; production should configure at least an app or bot id allowlist.
- For production, pair a bot/app allowlist with `HERMES_SLACK_BRIDGE_HMAC_SECRET`
  so channel membership alone is never enough to enqueue agent work.

## Wire format

A relay posts a Slack message like:

````text
<@U_HERMES_BOT> hermes-bridge
```json
{
  "request_id": "ci-pr-4242-attempt-1",
  "actor": "github-actions",
  "prompt": "Triage failures in PR #4242 — focus on the auth tests.",
  "metadata": {"repo": "org/repo", "pr": 4242},
  "signature": "f5c1...optional-hex-hmac-sha256"
}
```
````

- `<@U_HERMES_BOT>` is the standard Slack mention of your Hermes bot user.
- `hermes-bridge` is the literal marker token (case-insensitive).
- The envelope MUST sit inside a fenced ` ```json … ``` ` block (the plugin
  also accepts a bare `{ … }` for clients that strip code fences).

Field rules:

| field        | required | shape                                                                 |
| ------------ | -------- | --------------------------------------------------------------------- |
| `request_id` | yes      | opaque string `[A-Za-z0-9_\-\.:]{1,128}`, used for dedup              |
| `prompt`     | yes      | text — up to 32 000 characters                                        |
| `actor`      | no       | opaque string, same character set as `request_id`                     |
| `metadata`   | no       | JSON object — passed through to the agent inside the rewritten prompt |
| `signature`  | depends  | hex HMAC-SHA256 over `request_id|actor|prompt|canonical_metadata`; required iff secret set |

## What Hermes actually sees

For the envelope above, the agent receives:

```text
[slack-bot-bridge]
request_id: ci-pr-4242-attempt-1
actor: github-actions
metadata: {"pr": 4242, "repo": "org/repo"}

Triage failures in PR #4242 — focus on the auth tests.
```

## GitHub Actions example

```yaml
# .github/workflows/hermes-bridge.yml
name: Notify Hermes
on:
  pull_request:
    types: [opened, synchronize]

jobs:
  notify:
    runs-on: ubuntu-latest
    steps:
      - name: Build envelope and post to Slack
        env:
          SLACK_BOT_TOKEN: ${{ secrets.SLACK_BOT_TOKEN }}
          BRIDGE_CHANNEL: ${{ vars.HERMES_BRIDGE_CHANNEL }}
          HERMES_USER_ID: ${{ vars.HERMES_USER_ID }}
          BRIDGE_HMAC_SECRET: ${{ secrets.HERMES_SLACK_BRIDGE_HMAC_SECRET }}
        run: |
          set -euo pipefail
          REQUEST_ID="pr-${{ github.event.pull_request.number }}-sha-${{ github.event.pull_request.head.sha }}"
          ACTOR="github-actions"
          PROMPT="Triage failures in PR #${{ github.event.pull_request.number }}."
          METADATA=$(jq -S -nc --arg repo "${{ github.repository }}" '{"pr": ${{ github.event.pull_request.number }}, "repo": $repo}')
          # signature input is the same canonical form the plugin computes
          SIG=$(printf '%s|%s|%s|%s' "$REQUEST_ID" "$ACTOR" "$PROMPT" "$METADATA" \
                | openssl dgst -sha256 -hmac "$BRIDGE_HMAC_SECRET" \
                | awk '{print $2}')
          BODY=$(jq -nc \
            --arg request_id "$REQUEST_ID" \
            --arg actor      "$ACTOR" \
            --arg prompt     "$PROMPT" \
            --arg signature  "$SIG" \
            --argjson metadata "$METADATA" \
            '{request_id:$request_id, actor:$actor, prompt:$prompt, metadata:$metadata, signature:$signature}')
          # Slack message text — fenced JSON, mention Hermes, include marker.
          TEXT=$(printf '<@%s> hermes-bridge\n```json\n%s\n```\n' "$HERMES_USER_ID" "$BODY")
          curl -fsS -X POST https://slack.com/api/chat.postMessage \
            -H "Authorization: Bearer ${SLACK_BOT_TOKEN}" \
            -H "Content-Type: application/json; charset=utf-8" \
            --data "$(jq -nc --arg ch "$BRIDGE_CHANNEL" --arg t "$TEXT" '{channel:$ch, text:$t}')"
```

## Security model

- **Channel allowlist** is mandatory — the plugin is inert without
  `HERMES_SLACK_BRIDGE_CHANNEL`.
- **Identifier allowlists** (bot/app/team) narrow the set of senders the
  plugin will accept inside the bridge channel. Configure at least one app
  or bot id for production; team allowlists are an additional scope gate.
- **HMAC** (optional but strongly recommended for production) makes the
  envelope tamper-evident; without it any member of the bridge channel could
  craft a payload.
- **No free-form text.** The plugin only accepts a structured envelope —
  arbitrary bot chatter in the bridge channel is ignored.
- **Dedup** by `request_id` for 24 hours by default — Slack retries are
  effectively idempotent.
- **Auth still runs.** The plugin returns a `rewrite` action; the
  gateway's normal `SLACK_ALLOWED_USERS` / pairing check still applies to
  the sending bot's Slack user id. Add the relay bot to that allowlist
  explicitly.
