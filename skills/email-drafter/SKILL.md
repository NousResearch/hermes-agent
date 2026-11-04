---
triggers: ["draft an email", "compose an email", "email this person", "reply to this email", "create a gmail draft", "threaded email reply"]
name: email-drafter
description: Creates Gmail drafts — new emails or threaded replies.
metadata:
  hermes:
    tags: [email, gmail, draft, communication, reply, thread]
category: communication
version: 3.0.0
author: ndr@draas.com
---

# Email Drafter Skill

Drafts new Gmail emails or threaded replies as drafts for the user. It never
sends email — it only creates drafts, which the human sends from their own
Drafts folder (see `hermes-data/SOUL.md`, "Email Sending — HARD RULE").

## When to Use

Use when the user asks to compose an email ("email [name]", "draft an email to
[name]", "reply to [name]'s email", "send [name] an email"). For replies,
locate the existing Gmail message and create a properly threaded reply draft.

## Prerequisites

- The session user's Google account must be authorized in the GWS vault
  (a `service_name` with a stored token must exist for the account).
- The Gmail scope must be in `gws_auth.HERMES_GWS_SCOPES` for that account.

## How to Run

1. Resolve the account with `gws_resolve_account` — never guess the
   `service_name`. Default account is `ndr@draas.com`; use
   `nishantranka@gmail.com` or `ndr@ahfl.in` only when explicitly asked.
2. Create the draft with `execute_code` + `tools.gws_skill_bridge`.

## Quick Reference

Resolve the account:

```
tool: gws_resolve_account
input:
  account: "ndr@draas.com"   # or omit to list all accounts
```

Returns the exact `service_name` (e.g. `google-draas`) for that account.

Create a new-email draft (via `execute_code`):

```python
from tools.gws_skill_bridge import call
print(call("draft_create",
           service_name="google-draas",
           to="raghu@example.com",
           subject="Ranka Oasis: Site Visit",
           body="Please confirm your availability for a site visit this week.",
           cc="",
           from_header="",
           html=False))
```

Create a threaded reply draft (via `execute_code`):

```python
from tools.gws_skill_bridge import call
print(call("draft_reply_create",
           service_name="google-draas",
           message_id="<id-of-the-message-being-replied-to>",
           body="Thanks — confirmed for Wednesday."))
```

Find a message to reply to (via `execute_code`):

```python
from tools.gws_skill_bridge import call
print(call("gmail_search", service_name="google-draas",
           query="from:raghu subject:land valuation", max=5))
```

Read a thread (via `execute_code`):

```python
from tools.gws_skill_bridge import call
print(call("gmail_thread_get", service_name="google-draas",
           thread_id="<thread_id>"))
```

## Procedure

### Stage 1 — Context gathering

- For a NEW email: resolve the recipient with `contact_resolver` — never guess
  email addresses. When `auto_selected` is true, confirm the chosen address
  with the user first. If no match, report it; do not fall back to guessing.
- For a REPLY: find the message with `gmail_search` using the sender/subject,
  then read the thread with `gmail_thread_get`. Capture the message id of the
  message being replied to, the subject, and the participants. Confirm the
  reply scope (reply vs reply-all) with the user.

### Stage 2 — Draft

- Work email tone: no greeting, straight to the point, numbered asks, no
  boilerplate ("Hope you're well", "Dear [name]") unless explicitly asked.
- Subject: `[Project/Entity Name]: [one-line description]`.
- Personal/casual tone: warmer, plain text, no subject prefix. Roshni Ranka
  ("RO") is always personal tone.
- Use `html=True` only when the user asks for formatted/HTML email; otherwise
  keep it plain text.

Present the draft for confirmation before creating it.

### Stage 3 — Create the draft

- New email: `call("draft_create", ...)`.
- Reply: `call("draft_reply_create", message_id=..., ...)` — the bridge pulls
  the original thread, subject, and sender automatically, so the reply lands
  in the right thread. Pass `to=` only when replying to someone other than
  the original sender.
- Never use `gmail_send`/`gmail_reply` — they are permanently blocked in the
  bridge. Sending is the human's action from their own Drafts folder.

## Pitfalls

- ALWAYS call `gws_resolve_account` first and use the returned `service_name`.
  A hand-typed slug can look exactly like "not authorized" even when the token
  exists under the correct key.
- `draft_reply_create` needs the original `message_id` (from `gmail_search`),
  not the thread id.
- Do not add boilerplate greetings unless asked.
- Do not send without showing the draft and getting confirmation.
- Only use `nishantranka@gmail.com` / `ndr@ahfl.in` when the user explicitly
  asks — default is `ndr@draas.com`.

## Verification

Confirm the JSON result contains `"status": "draft_created"` and a
`draft_id`. Tell the user the draft is in their Drafts folder; on failure,
report the exact error and suggest re-auth via `send_oauth_url` if the vault
reports no token.