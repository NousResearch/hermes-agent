#!/usr/bin/env python3
"""Reply-all (+ optional BCC) as a Gmail DRAFT, correctly threaded. NEVER sends.

WHY THIS FILE EXISTS
--------------------
Two verified gaps force a hand-built MIME for reply-all / BCC replies:

1. The NATIVE `gws_gmail_reply_draft` (and `gws_dwd_gmail_reply_draft`) tool IGNORES
   its `to=` parameter and always addresses the reply to the original sender only.
   Verified 2026-10-17: called it with
       to="mdr@draas.com, dharmesh.ranka1@gmail.com", bcc="rnr@draas.com"
   and the created draft came back `to: Manish Ranka <mdr@draas.com>` — Dharmesh was
   silently dropped, and there was no Cc either. It DOES honour `bcc`.
2. The bridge's `draft_reply_create` has no `bcc` parameter at all.

So: reply-all and/or BCC  ->  use this template. Plain single-recipient reply with no
BCC  ->  the bridge's `draft_reply_create` is fine and simpler.

RUN
---
    /opt/hermes/.venv/bin/python3 templates/reply-all-bcc-threaded.py

Copying the body into `execute_code` also works — `tools.gws_auth.build_service` is
importable there, unlike the bridge (which needs `gws_fetch_token` from the venv).

OUTPUT
------
Prints the draft read back from `drafts().get(format='full')`: To, Cc, Bcc, Subject,
In-Reply-To, threadId, labels. Gmail does NOT strip `bcc` on an UNSENT draft, so you
can assert on it directly (the older "Gmail strips Bcc" claim applies to sent mail).
"""

import base64
import json
import sys

sys.path.insert(0, "/opt/hermes")

from email.mime.text import MIMEText
from tools.gws_auth import build_service

# ---- configure -------------------------------------------------------------
SERVICE_NAME = "google-draas"   # resolve via gws_resolve_account(); never guess a slug
ORIGINAL_MESSAGE_ID = ""        # the message being replied to -- NOT the thread id
TO = ""                         # reply-all: sender + other original To/Cc, minus self
CC = ""                         # optional
BCC = ""                        # optional, e.g. "rnr@draas.com"
BODY = ""                       # minimal HTML only: <p>, <br>, <b>. NO css, NO divs,
                                # NO spans -- heavy HTML renders as raw text in the
                                # Gmail compose view (see
                                # messaging-drafts/references/gmail-draft-formatting-pitfalls.md)
IS_HTML = True
# ---------------------------------------------------------------------------

svc = build_service("gmail", "v1", service_name=SERVICE_NAME)

# Original headers: Message-ID is what threads the draft; threadId keeps it in place.
orig = svc.users().messages().get(
    userId="me",
    id=ORIGINAL_MESSAGE_ID,
    format="metadata",
    metadataHeaders=["From", "Subject", "Message-ID"],
).execute()
h = {x["name"].lower(): x["value"] for x in orig["payload"]["headers"]}
orig_msgid = h["message-id"]          # keep the angle brackets exactly as returned
subject = h["subject"] if h["subject"].lower().startswith("re:") else "Re: " + h["subject"]

msg = MIMEText(BODY, "html" if IS_HTML else "plain", "utf-8")
msg["To"] = TO
if CC:
    msg["Cc"] = CC
if BCC:
    msg["Bcc"] = BCC
msg["Subject"] = subject
msg["In-Reply-To"] = orig_msgid
msg["References"] = orig_msgid        # a single Message-ID is sufficient for a reply

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = svc.users().drafts().create(
    userId="me",
    body={"message": {"raw": raw, "threadId": orig["threadId"]}},
).execute()

# Verify before reporting: recipients are the thing that silently goes missing.
back = svc.users().drafts().get(userId="me", id=draft["id"], format="full").execute()["message"]
hb = {x["name"].lower(): x["value"] for x in back["payload"]["headers"]}
print(json.dumps({
    "draft_id": draft["id"],
    "message_id": back["id"],
    "threadId": back["threadId"],
    "to": hb.get("to"),
    "cc": hb.get("cc"),
    "bcc": hb.get("bcc"),
    "subject": hb.get("subject"),
    "in_reply_to": hb.get("in-reply-to"),
    "labels": back.get("labelIds"),
}, indent=2, ensure_ascii=False))

# Housekeeping: if an earlier bad draft was created on this thread, delete it first
# with svc.users().drafts().delete(userId="me", id=<draft_id>) so only one draft
# sits on the thread.
