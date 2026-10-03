---
name: jmap
description: "JMAP CLI: read, search, triage, and send email."
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Email, JMAP, AtomicMail, Fastmail, Communication]
---

# JMAP Email Integration

Manage emails directly from the terminal via standard JMAP (RFC 8620 / RFC 8621). This skill enables Hermes Agent to inspect inboxes, triage unread threads, read full messages, and compose/send emails with support for Atomic Mail, Fastmail, and Stalwart.

## When to Use

- When the user asks to check their email, search for messages, or read recent threads via JMAP or Atomic Mail.
- When the user asks to send, reply to, or draft an email.
- When performing automated inbox triage or priority summaries.

> **Note**: For legacy IMAP/SMTP providers (Gmail, Outlook, iCloud without JMAP), use the `himalaya` skill instead.

---

## Configuration & Credentials

The skill reads credentials from (in order of precedence):
1. Command-line flags (`--session-url`, `--token`, `--account-id`)
2. Environment variables: `JMAP_SESSION_URL`, `JMAP_API_TOKEN`, `JMAP_ACCOUNT_ID`
3. Credentials file: `~/.hermes/jmap-credentials.json` (see `templates/credentials.template.json`)

Supported session presets:
- **Atomic Mail**: `https://api.atomicmail.io/jmap/session`
- **Fastmail**: `https://api.fastmail.com/jmap/session`
- **Self-Hosted**: `https://mail.example.com/.well-known/jmap`

---

## CLI Usage & Workflows

Run the client via Python from the repository root:

```bash
# Set script alias or path
SCRIPT="python3 skills/email/jmap/scripts/jmap_client.py"
TRIAGE="python3 skills/email/jmap/scripts/email_triage.py"
```

### 1. List & Search Emails

```bash
# List recent 10 emails in INBOX
python3 skills/email/jmap/scripts/jmap_client.py list --limit 10

# List only unread messages
python3 skills/email/jmap/scripts/jmap_client.py list --unread --limit 5

# Search emails for a keyword
python3 skills/email/jmap/scripts/jmap_client.py list --query "project roadmap"
```

### 2. Read Full Email Content

```bash
# View complete message by ID
python3 skills/email/jmap/scripts/jmap_client.py get "M18f3a9b2c"
```

### 3. Send or Draft Email

```bash
# Send an email
python3 skills/email/jmap/scripts/jmap_client.py send \
  --to "alex@example.com" \
  --subject "Meeting Notes & Next Steps" \
  --body "Hi Alex,\n\nHere are the updated meeting notes..."

# Send with BCC and CC
python3 skills/email/jmap/scripts/jmap_client.py send \
  --to "dev@company.com" \
  --cc "lead@company.com" \
  --bcc "archive@atomicmail.io" \
  --subject "Release v2.4 Scheduled" \
  --body "Deployment starts at 18:00 UTC."

# Save as draft without sending
python3 skills/email/jmap/scripts/jmap_client.py send \
  --to "client@example.com" \
  --subject "Draft Proposal" \
  --body "Proposal text..." \
  --draft
```

### 4. Triage Inbox

```bash
# Generate categorized summary (Action Required vs Notifications)
python3 skills/email/jmap/scripts/email_triage.py --limit 20
```

---

## Safety & Best Practices

1. **Confirmation before Sending**: Always summarize recipient, subject, and body to the user before executing `send` commands unless explicitly requested to send autonomously.
2. **Plain Text Fallback**: The client automatically strips HTML markup to prevent context pollution in terminal output.
3. **Draft Safety**: When uncertain about sending immediately, use `--draft` to create the email in the user's Drafts mailbox for manual review.
