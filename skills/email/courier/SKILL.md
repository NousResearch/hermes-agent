---
name: courier
description: "Courier Protocol (getcourier.dev) — self-sovereign email infrastructure for autonomous AI agents. Create inboxes, receive SMTP email, extract OTPs and magic links, send agent-to-agent messages. No human signup, no API keys, no OAuth."
version: 1.0.0
author: Antonio Lombardo / Courier Protocol (MIT)
license: MIT
metadata:
  hermes:
    tags: [Email, Courier, Inbox, OTP, Agent, Protocol]
    homepage: https://getcourier.dev
---

# Courier Protocol

Courier Protocol (getcourier.dev) gives autonomous AI agents real email inboxes with:
- **No human signup** — agents self-provision inboxes via a single API call
- **Real SMTP email** — any service can send to Courier inboxes
- **Automatic extraction** — OTP codes and magic links are classified and extracted from incoming messages
- **Self-hostable** — single VPS, MIT license, open source on GitHub

## When to Use

- The agent needs its own email identity (inbox) to receive verification codes, OTPs, or magic links
- The agent needs to receive email autonomously without Gmail OAuth or human dashboard signup
- Sending operational messages between agents (agent-to-agent)
- Disposable inboxes for testing and signup flows

**Do not use for:**
- Reading a human's existing personal inbox — use Gmail API, IMAP, or Himalaya
- High-volume outbound to real-world email addresses — Courier's `/incoming` routes to Courier inboxes; use Resend, Postmark, or SendGrid for outbound to actual people

## Prerequisites

- `curl` (for API calls)
- No API keys, no signup, no configuration needed for free tier

## API Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `POST /alias` | POST | Create an inbox (returns `inbox_id` + `read_token`) |
| `GET /inbox/{inbox_id}/messages` | GET | Retrieve messages with extracted codes and links |
| `GET /health` | GET | Service health check |
| `POST /incoming` | POST | Send an email to a Courier inbox |
| `GET /capabilities` | GET | Full protocol capabilities document |
| `GET /llms.txt` | GET | Protocol overview for LLM context injection |
| `GET /agent.json` | GET | Machine-readable agent metadata |

Base URL: `https://getcourier.dev`

## Quick Start

### 1. Check service health

```bash
curl -s https://getcourier.dev/health
```

### 2. Create an inbox

```bash
curl -s -X POST https://getcourier.dev/alias \
  -H "Content-Type: application/json" \
  -d '{"purpose":"my-task","agent":"agent-name","service":"my-service"}'
```

Response includes `inbox_id` and `read_token` — save both.

### 3. Read messages

```bash
curl -s "https://getcourier.dev/inbox/{INBOX_ID}/messages?limit=10"
```

Messages include extracted `codes` (OTP, verification) and `links` (magic links) under each message object.

### 4. Send to another Courier inbox

```bash
curl -s -X POST https://getcourier.dev/incoming \
  -H "X-Forwarded-To: target-alias@inbox.getcourier.dev" \
  -H "X-Forwarded-From: my-alias@inbox.getcourier.dev" \
  -H "Content-Type: text/plain" \
  -d "From: Sender Name <sender@domain.com>
To: Recipient
Subject: Subject Line

Message body here."
```

## Common Operations

### Create an alias for a specific purpose

```bash
curl -s -X POST https://getcourier.dev/alias \
  -H "Content-Type: application/json" \
  -d '{"purpose":"otp-verification","agent":"my-bot","service":"signup-service"}'
```

If you omit the `alias` field, one is auto-generated.

### Receive OTP codes

```bash
curl -s "https://getcourier.dev/inbox/{INBOX_ID}/messages?limit=5" \
  | jq '.messages[] | select(.codes != []) | {subject, codes}'
```

### Receive magic links

```bash
curl -s "https://getcourier.dev/inbox/{INBOX_ID}/messages?limit=5" \
  | jq '.messages[] | select(.classification.type=="magic_link") | {subject, links}'
```

### Send a structured JSON message between agents

```bash
curl -s -X POST https://getcourier.dev/incoming \
  -H "X-Forwarded-To: target@inbox.getcourier.dev" \
  -H "X-Forwarded-From: me@inbox.getcourier.dev" \
  -H "Content-Type: application/json" \
  -d '{"type":"status_update","status":"completed","workflow_id":"wf-123"}'
```

## Rate Limits & Free Tier

| Resource | Free Limit |
|----------|-----------|
| Aliases (inboxes) | 10 |
| Ingests per month | 500 |
| Alias creation | 10/hr per IP |
| Messages read | 50/day |
| Message retention | 7 days (ephemeral) |

## Error Codes

| Code | Meaning | Retryable? |
|------|---------|-----------|
| `ALIAS_NOT_FOUND` | No matching alias | No |
| `ALIAS_EXISTS` | Alias name taken | No |
| `INGEST_FAILED` | Could not parse/storing | Yes |
| `RATE_LIMITED` | Too many requests | Yes (60s backoff) |
| `PAYMENT_REQUIRED` | Free quota exceeded | Yes |
| `SERVICE_UNAVAILABLE` | Service temporarily down | Yes |

## Pricing Tiers (Lightning Network)

| Tier | Cost | Description |
|------|------|-------------|
| Free | $0 | Development and evaluation |
| Hobby | 5000 sats/mo | Production for single agent |
| Agent | 25000 sats/mo | Multi-agent operations |
| Autonomous | 100000 sats/mo | Unlimited agent operations |

Payments via Lightning Network (BTC). Experimental — `POST /x402/invoice`.

## Self-Hosting

```bash
git clone https://github.com/antonioac1/courier.git
cd courier && npm install
```

Requirements: Node >= 18, ~512MB RAM, ~5GB disk. Single VPS, ~$4/month.

## Common Pitfalls

1. **Wrong service.** Courier Protocol (getcourier.dev) is NOT courier.com / trycourier. The latter is a notification platform for human apps and does not give agents their own inbox identity.
2. **Not for outbound to real people.** `/incoming` routes to Courier inboxes only. For sending to real-world email addresses, use a separate outbound provider (Resend, Postmark, SendGrid, or SMTP via Himalaya).
3. **Messages are ephemeral.** Retention is 7 days. Do not use Courier as a long-term email archive.
4. **Rate limits are per IP.** If you exhaust alias creation rate limits (10/hr), wait or use a different source IP.
5. **`read_token` is sensitive.** Treat it like an API key — anyone with it can read your inbox.

## Integrations

| Platform | Link |
|----------|------|
| OpenAI Agents SDK | `examples/openai-agents-integration.md` |
| Claude Code | `examples/autonomous-onboarding.sh` |
| Cursor | `examples/cursor-integration.md` |
| Python | `pip install requests` |
| Node.js | `npm install courier-protocol` |
| MCP | `io.github.antonioac1/courier` |

Source: https://github.com/antonioac1/courier
npm: https://www.npmjs.com/package/courier-protocol

## Verification Checklist

- [ ] `curl -s https://getcourier.dev/health` returns `"status": "running"`
- [ ] `POST /alias` returns `success: true` with `inbox_id` and `read_token`
- [ ] Messages are retrievable at `GET /inbox/{inbox_id}/messages`
- [ ] Incoming SMTP reaches the inbox (test by sending to `your-alias@inbox.getcourier.dev`)
- [ ] OTP codes and magic links appear in the `codes` / `links` fields of message objects
