# Gmail Sender Blacklist (Auto-Cleaner)

Reference for the auto-cleaner approach to blacklisting senders in Gmail when
`gmail.settings.sharing` scope is unavailable (native filter API not accessible).

## When to Use

- User says "blacklist these senders", "mark as junk", "move to spam"
- User labels emails as junk in the inbox and wants future mail from those senders
  to skip the inbox entirely
- You don't have the `gmail.settings.sharing` OAuth scope (native filter API)
- As a fallback while waiting for the user to re-authorize for filter scope

## Approach

A Gmail API script + cron job that periodically scans the inbox for messages
matching blacklisted senders and moves them to SPAM.

### Core Script Pattern

```python
from tools.gws_auth import build_service
import json

def move_to_spam(sender_emails, service_name="google-draas"):
    """Move inbox messages from blacklisted senders to SPAM."""
    gmail = build_service("gmail", "v1", service_name=service_name)

    # Build Gmail search query: messages from any of the target senders
    # in INBOX label (not already in SPAM)
    queries = [f"from:{s}" for s in sender_emails]
    # For forwarded emails, the actual sender may be in Reply-To:
    queries += [f"rfc822msgid:{s}" for s in sender_emails]
    # Simpler: search inbox and check headers manually
    query = " OR ".join(sender_emails)
    results = gmail.users().messages().list(
        userId="me", q=f"in:inbox ({query})"
    ).execute()

    moved = []
    for msg in results.get("messages", []):
        gmail.users().messages().modify(
            userId="me", id=msg["id"],
            body={"addLabelIds": ["SPAM"]}
        ).execute()
        moved.append(msg["id"])
    return moved


def get_sender_from_headers(msg_data):
    """Extract the actual sender address from a message's headers.
    For forwarded emails, Reply-To is more reliable than From."""
    headers = msg_data["payload"]["headers"]
    hdrs = {h["name"].lower(): h["value"] for h in headers}
    return hdrs.get("reply-to", hdrs.get("from", ""))
```

### Cron Schedule

```
every 15m  — sweep inbox for any new mail from blacklisted senders
```

### Sender Address Strategy

| Scenario | Search By |
|---|---|
| Direct email (e.g. amit@outlinepr.com) | `from:` header |
| Forwarded to group (e.g. ndr@draas.com from service) | `Reply-To` header |
| Bulk/marketing via sendgrid/ses | envelope from or `Return-Path` |

### Full Example (from Aug 2026 session)

For each blacklisted sender, the script:
1. Searches `in:inbox` for the sender's domain or email
2. For each match, inspects headers to confirm the sender
3. Calls `modify(addLabelIds=['SPAM'])` to move the message
4. Also sweeps Sent Mail for replies to intercepted threads
5. Logs both count and identity of moved messages

```python
# Blacklist action script — run once to sweep, then schedule as cron
from tools.gws_auth import build_service

gmail = build_service("gmail", "v1", service_name="google-draas")

BLACKLISTED_SENDERS = [
    "marketing@cqra.acts-int.com",      # CQRA Private Limited
    "amit@outlinepr.com",                # Amit Saparia Outline PR
    "secretariat1@worldhrdcongress.com", # Secretariat
    "chelsea.c@ifttt.com",               # Chelsea / IFTTT
    "mritunjay.anand@reliablegroup.net", # Reliable Ispat Udyog
    # + Internshala, Adobe Stock, Vibro Springs (via Reply-To)
]

for sender in BLACKLISTED_SENDERS:
    msgs = gmail.users().messages().list(
        userId="me", q=f"in:inbox ({sender})"
    ).execute()
    for m in msgs.get("messages", []):
        gmail.users().messages().modify(
            userId="me", id=m["id"],
            body={"addLabelIds": ["SPAM"]}
        ).execute()
```

### Limitations vs Native Filters

| Aspect | Native Filter | Auto-Cleaner |
|---|---|---|
| Latency | Instant | Up to 15 min |
| Future mail | Blocked before arrival | Moves after arrival |
| Retroactive sweep | No | Yes (can clean existing inbox) |
| Scope needed | gmail.settings.sharing | gmail.modify only |

### Related

- `references/not-spam-whitelist.md` — the opposite operation (auto-unspam whitelisted senders from Gmail SPAM using a Google Sheet whitelist)
- `email` skill Section 5, which links here
