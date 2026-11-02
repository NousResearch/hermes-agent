# Threaded Reply-All via Raw Gmail API

When the `gws_skill_bridge` or `google_workspace_manager` tool can't produce a properly threaded reply-all draft (or isn't available), use the raw Gmail API via `tools.gws_auth.build_service()`.

## When to use this pattern

- Need a **reply-all** that preserves all To/CC recipients from the original thread
- The bridge's `draft_reply_create` doesn't support the level of control you need
- Need to set specific threading headers (In-Reply-To, References) for Gmail to nest the reply correctly
- The `google_workspace_manager` tool is not exposed in the current session

## Workflow

### 1. Find the right email to reply to

The "right" email is the **most recent message in the thread from someone other than the user**. Replying to the most recent message from another participant ensures the reply is properly threaded.

```python
from tools.gws_auth import build_service

service = build_service("gmail", "v1", service_name="google-draas")

# Search for the thread
results = service.users().messages().list(
    userId="me",
    q='subject:"Hermitage matters"',
    maxResults=10
).execute()

# Get the most recent message ID
latest_id = results["messages"][0]["id"]  # most recent first
```

### 2. Get the message to extract threading headers

Two approaches — both work:

**Option A — `format="metadata"` (simpler, no base64 decode needed):**
```python
msg_data = service.users().messages().get(
    userId="me", id=latest_id, format="metadata",
    metadataHeaders=["From", "Subject", "Message-ID", "To", "Cc", "References"],
).execute()
headers = {h["name"]: h["value"] for h in msg_data["payload"]["headers"]}
msg_id_header = headers.get("Message-ID", "")
references = headers.get("References", "")
sender = headers.get("From", "")
all_cc = headers.get("Cc", "")
```

**Option B — `format="raw"` (full MIME access):**
```python
import base64
import email as email_parser

msg_data = service.users().messages().get(userId="me", id=latest_id, format="raw").execute()
raw = base64.urlsafe_b64decode(msg_data["raw"]).decode("utf-8", errors="replace")

parsed = email_parser.message_from_string(raw)
msg_id_header = parsed.get("Message-ID", "").strip()
references = parsed.get("References", "")
```

**Note:** `Message-ID` MAY come back EMPTY via `format="metadata"` even when requested in `metadataHeaders` (observed Sep-2026: Google-draas message returned all headers except Message-ID). Do not trust it — if `headers.get("Message-ID")` is empty, fall back to `format="raw"` and parse with `email_parser.message_from_string(raw)` to get the real Message-ID. Threading depends on it, so always verify it's non-empty before building the reply.

### 3. Create the reply-all draft with proper threading

Two approaches — pick one:

**Option A — `MIMEText` (simpler, for plain-text replies):**

```python
from email.mime.text import MIMEText
import base64

message = MIMEText(body_text, "plain")
message["To"] = sender  # original sender from Step 2
if all_cc:
    message["Cc"] = all_cc  # preserve original CCs for reply-all
message["Subject"] = f"Re: {original_subject}"
if msg_id_header:
    message["In-Reply-To"] = msg_id_header
    message["References"] = msg_id_header

raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
draft_body = {
    "message": {
        "raw": raw,
        "threadId": original_thread_id  # from the original msg_data
    }
}
draft = service.users().drafts().create(userId="me", body=draft_body).execute()
```

**Option B — `EmailMessage` (more control, when you need strict threading):**

```python
import re
from email.message import EmailMessage

def clean_header(val):
    return re.sub(r'\s+', ' ', val).strip()

reply = EmailMessage()
reply["In-Reply-To"] = clean_header(msg_id_header)
reply["References"] = clean_header(references) if references else clean_header(msg_id_header)
reply["Subject"] = "Re: " + original_subject
reply["To"] = sender
reply["Cc"] = all_cc
reply.set_content(body_text)

raw = base64.urlsafe_b64encode(reply.as_bytes()).decode()
draft = service.users().drafts().create(
    userId="me",
    body={"message": {"raw": raw, "threadId": original_thread_id}}
).execute()
```

**Note on `threadId`:** Always include `threadId` in the draft body dict. Without it, Gmail creates a disconnected new thread despite the In-Reply-To/References headers.

### 4. Verify the draft

```python
draft_data = service.users().drafts().get(userId="me", id=draft["id"], format="full").execute()
```
print("To:", headers.get("To", ""))
print("Cc:", headers.get("Cc", ""))
print("In-Reply-To:", headers.get("In-Reply-To", "")[:60])
```

### Day-of verification warning (2026-08-10)

A draft created and verified here (label `['DRAFT']`, clean References, `drafts().get()` passes) can later turn out to be **SENT** — either the user sent it from the Gmail UI, or it auto-sent. You cannot rely on the create-time label. The definitive "is it still a draft" check is:

```python
res = service.users().drafts().list(userId="me").execute()   # NOT drafts().get()
print([d["id"] for d in res.get("drafts", [])])
```

If the user later says "I can't see the draft", run `drafts().list()`; if the draft is absent, subject-search the account (`q='subject:"<keyword>"'`) and check `labelIds`. A `['SENT']` hit means it went out — tell the user it was sent (give time, To, Cc) and do NOT recreate a fresh draft (that yields a duplicate they might send twice).

## Important: sender address must match

The email will be created in the **account** the `build_service` was constructed for. `service_name="google-draas"` means `ndr@draas.com`. If you need a different sender (e.g. `nishantranka@gmail.com`), resolve the account first via `gws_resolve_account()` and pass the correct `service_name`.

## Key pitfalls

- **Newlines in References/In-Reply-To**: Gmail may store these across multiple lines. Use `re.sub(r'\s+', ' ', val).strip()` to flatten them before passing to `EmailMessage.__setitem__`, which rejects headers containing linefeed characters.
- **Message-ID content**: The raw Gmail API stores `Message-ID` with angle brackets: `<CAMf8n+Hun2Yr...@mail.gmail.com>`. Keep them — they're part of the valid header value.
- **Threading only works with both In-Reply-To AND References**: Gmail nests replies based on these two headers. The References header should include all ancestor message IDs, with the immediate parent being the last entry.
- **The `email.message.EmailMessage` `__setitem__` method is strict** about header values: no raw newlines, no trailing whitespace. Always clean headers before passing them.
- **`format="metadata"` with `metadataHeaders` returns Message-ID**: The Message-ID header IS available via `format="metadata"` when you explicitly request it in `metadataHeaders`. You do NOT need `format="raw"` just for header extraction. Use `format="raw"` only when you need the full MIME body content.

## Non-ASCII subjects in hand-built MIME (en-dash, accents)

Option B (`EmailMessage`) encodes non-ASCII headers correctly by itself. The **hand-built raw
string** pattern (the one in `templates/html-reply-with-attachment.py`, where you join header
lines with `\r\n` yourself) does NOT — dropping a raw `–` (U+2013) or any accented character
into a `Subject:` line produces a mangled subject.

DRAAS subjects regularly carry an en-dash, e.g.
`OASIS Site – Revised Saucer Drain Design and Section Updates`. Wrap the header in an RFC 2047
encoded-word — base64 is the safest because the Q-encoding of an already-long subject easily
blows past the 76-char limit:

```python
SUBJECT_PLAIN = "Re: OASIS Site \u2013 Revised Saucer Drain Design and Section Updates"
SUBJECT_ENC   = "=?UTF-8?B?" + base64.b64encode(SUBJECT_PLAIN.encode("utf-8")).decode() + "?="
head = [..., "Subject: %s" % SUBJECT_ENC, ...]
```

Do NOT silently "fix" it by substituting a hyphen — that retitles the thread. Verify on the way
out: `drafts().get(..., format="full")` and confirm `Subject:` decodes back with the en-dash
intact (Gmail's API returns it decoded, so the check is just `hdrs["Subject"] == SUBJECT_PLAIN`).

The absence of `\r\n`-safe encoding is only a problem for headers. The `text/plain` and
`text/html` **bodies** in the same raw string are fine as literal UTF-8 because their
`Content-Type` carries `charset="UTF-8"`.
