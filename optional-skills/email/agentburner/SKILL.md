---
name: agent-burner
version: 0.2.0
description: Disposable email API. Use when you need a temporary email address -- receiving mail, reading contents, extracting URLs. Triggers include "create a temp email", "I need a burner email", "disposable inbox", or any task requiring a throwaway email address.
homepage: https://agentburner.com
tags:
  - email
  - disposable
  - burner
  - temp-mail
  - verification
  - otp
  - inbox
  - api
  - no-auth
  - zero-config
metadata:
  api_base: https://api.agentburner.com
  hermes:
    tags: [email, disposable, burner, temp-mail, inbox, no-auth]
    category: email
---

# Agent Burner

Disposable email API. No signup, no API key.

## Quick Start

```bash
# create an inbox
curl -X POST https://api.agentburner.com/inbox
# -> { "address": "a3f7b2c1@caledy.com", "key": "550e8400-...", "ttl": 3600 }

# read emails
curl https://api.agentburner.com/inbox/550e8400-...
# -> { "address": "...", "entries": [{ "id": "...", "from": "...", "subject": "...", "receivedAt": "..." }] }

# get full email
curl https://api.agentburner.com/inbox/550e8400-.../EMAIL_ID
# -> { "body": "...", "html": "...", "urls": ["..."], ... }

# delete inbox (optional -- auto-expires)
curl -X DELETE https://api.agentburner.com/inbox/550e8400-...
```

## API Reference

### POST /inbox

Create a disposable inbox.

```
Body (optional): { "ttl": 60 }   // TTL in minutes, default 60, max 360
Response: { "address": string, "key": string, "ttl": number }
```

- `address` -- the email address
- `key` -- inbox key (UUID), the only credential you need
- `ttl` -- inbox lifetime in seconds

Rate limited to 10 creations per minute per IP.

### GET /inbox/:key

List received emails.

```
Response: { "address": string, "entries": [{ "id": string, "from": string, "subject": string, "receivedAt": string }] }
```

Returns 404 if key is expired or invalid.

### GET /inbox/:key/:emailId

Get full email content.

```
Response: {
  "id": string,
  "from": string,
  "to": string[],
  "subject": string,
  "body": string,        // plain text
  "html": string | null, // raw HTML if present
  "urls": string[],      // all extracted URLs
  "receivedAt": string   // ISO 8601
}
```

### DELETE /inbox/:key

Delete inbox and all emails. Optional -- inboxes auto-expire.

```
Response: { "ok": true }
```

## Key Details

- No auth required -- inbox key is the only credential
- Multiple trusted .com domains
- Auto-expiry -- 1 hour default, 6 hours max
- 50 emails per inbox (oldest dropped first)
- 10MB max email size
- URLs extracted from both HTML and plain text

## Errors

| Status | Meaning |
|--------|---------|
| 404 | Key expired or invalid |
| 429 | Rate limit exceeded |
