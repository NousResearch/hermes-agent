---
name: courier
description: "Courier Protocol (getcourier.dev) — full-duplex email infrastructure for autonomous AI agents: receive SMTP email, extract OTPs and magic links, send governed outbound to real inboxes, maintain threaded conversations. No human signup."
version: 1.1.0
author: Antonio Lombardo / Courier Protocol (MIT)
license: MIT
metadata:
  hermes:
    tags: [Email, Courier, Inbox, OTP, Outbound, Agent, Protocol]
    homepage: https://getcourier.dev
---

# Courier Protocol

Courier Protocol (getcourier.dev) gives autonomous AI agents **full-duplex email communications**:
- **No human signup** — agents self-provision inboxes via a single API call
- **Real SMTP email** — any service can send to Courier inboxes
- **Automatic extraction** — OTP codes and magic links classified and extracted from incoming messages
- **Governed outbound** — send to real internet inboxes with identity tracking, rate limiting, and audit
- **Thread continuity** — reply within threads, maintain conversations across restarts
- **Self-hostable** — single VPS, MIT license, open source on GitHub

## When to Use

- The agent needs its own email identity to receive verification codes, OTPs, or magic links
- The agent needs to **send** email replies, follow-ups, or autonomous outreach
- The agent needs to maintain threaded conversations with human counterparties
- Avoiding Gmail OAuth, human dashboard signup, or any human-in-the-loop

**Do not use for:**
- Reading a human's existing personal inbox — use Gmail API, IMAP, or Himalaya
- courier.com / trycourier — that is a separate notification platform for human apps

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
| `POST /outbound/send` | POST | Send governed outbound email to real internet |
| `POST /thread/reply` | POST | Reply within a thread, maintaining continuity |
| `GET /threads` | GET | List and search thread history |
| `POST /identity/create` | POST | Establish persistent agent identity |
| `GET /capabilities` | GET | Full protocol capabilities document |
| `POST /x402/invoice` | POST | Lightning payment invoice |

Base URL: `https://getcourier.dev`

**Note:** Outbound and thread endpoints require an `Authorization: Bearer <token>` header. Obtain a token via `POST /identity/create`.

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

Messages include extracted `codes` (OTP, verification) and `links` (magic links).

### 4. Send outbound to a real email address

```bash
curl -X POST https://getcourier.dev/outbound/send \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"to":"user@example.com","subject":"Re: your message","body":"This is an agent reply.","in_reply_to":"msg_123"}'
```

### 5. Reply to a thread

```bash
curl -X POST https://getcourier.dev/thread/reply \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"thread_abc123","body":"Continuing this conversation."}'
```

### 6. Send to another Courier inbox

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

### Create an identity

```bash
curl -s -X POST https://getcourier.dev/identity/create \
  -H "Content-Type: application/json" \
  -d '{"purpose":"my-agent","agent":"agent-name","service":"my-service"}'
```

Returns an `agent_id` and sets up reputation tracking and provider routing.

### Create an alias

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

### Agent follow-up after verification

```bash
curl -X POST https://getcourier.dev/outbound/send \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"to":"service@company.com","subject":"Follow-up on application","body":"Following up on my application submitted earlier."}'
```

### Agent confirms signup

```bash
curl -X POST https://getcourier.dev/outbound/send \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"to":"support@startup.io","subject":"Signup confirmation","body":"OTP verified. Account active. Agent ID: agent_abc."}'
```

### Agent communicates with a human

```bash
curl -X POST https://getcourier.dev/thread/reply \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"thread_def456","body":"Hi Alice, the report you requested is ready."}'
```

## Outbound Delivery Characteristics

Courier outbound is NOT an open SMTP relay. Every outbound send is:

- **Governed** — Allowed domains, send quotas, policies enforced per identity
- **Rate-limited** — Per-identity limits with Retry-After headers (10/min per identity)
- **Audited** — Every send recorded in append-only JSONL audit stream
- **Reputation-aware** — Sender reputation tracked, abusive patterns trigger suspension
- **Provider-routed** — Automated failover across delivery providers
- **CISO-controlled** — Operator-level policy: suspend, revoke, emergency shutdown

## Rate Limits & Free Tier

| Resource | Free Limit |
|----------|-----------|
| Aliases (inboxes) | 10 |
| Ingests per month | 500 |
| Outbound sends | 50 |
| Identities | 3 |
| Threads | 50 |
| Alias creation | 10/hr per IP |
| Messages read | 50/day |
| Outbound rate | 10/min per identity |
| Message retention | 7 days (ephemeral) |

## Error Codes

| Code | Meaning | Retryable? |
|------|---------|-----------|
| `ALIAS_NOT_FOUND` | No matching alias | No |
| `ALIAS_EXISTS` | Alias name taken | No |
| `INGEST_FAILED` | Could not parse/storing | Yes |
| `OUTBOUND_FAILED` | Send failed — check provider status | Yes |
| `OUTBOUND_QUOTA_EXCEEDED` | Monthly outbound quota hit | No |
| `OUTBOUND_DOMAIN_DENIED` | Target domain not in allowlist | No |
| `RATE_LIMITED` | Too many requests | Yes (60s backoff) |
| `PAYMENT_REQUIRED` | Free quota exceeded | Yes |
| `THREAD_NOT_FOUND` | Thread ID not found | No |
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
2. **Outbound is governed, not open relay.** You need a valid identity and token. Domain allowlists and monthly quotas apply.
3. **Messages are ephemeral.** Retention is 7 days. Do not use Courier as a long-term email archive.
4. **Rate limits are per IP.** If you exhaust alias creation rate limits (10/hr), wait or use a different source IP.
5. **`read_token` is sensitive.** Treat it like an API key — anyone with it can read your inbox.

## Integrations

| Platform | Link |
|----------|------|
| Hermes Agent | `hermes config set tools.courier.enabled true` |
| OpenAI Agents SDK | `examples/openai-agents-integration.md` |
| Claude Desktop / Claude Code | `examples/autonomous-onboarding.sh` |
| Cursor | `examples/cursor-integration.md` |
| OpenHands | `examples/openhands-integration.md` |
| Python | `pip install requests` |
| Node.js | `npm install courier-protocol` |
| MCP | `io.github.antonioac1/courier` |

Source: https://github.com/antonioac1/courier
npm: https://www.npmjs.com/package/courier-protocol

## Verification Checklist

- [ ] `curl -s https://getcourier.dev/health` returns `"status": "running"`
- [ ] `POST /identity/create` creates a persistent identity
- [ ] `POST /alias` returns `success: true` with `inbox_id` and `read_token`
- [ ] Messages are retrievable at `GET /inbox/{inbox_id}/messages`
- [ ] Incoming SMTP reaches the inbox (send to `your-alias@inbox.getcourier.dev`)
- [ ] OTP codes and magic links appear in the `codes` / `links` fields of message objects
- [ ] `POST /outbound/send` with auth token sends governed outbound
