# Editing a Live Draft In Place Without Breaking Its Thread

Recurring NDR/PS pattern: a long HTML reply draft is already sitting in a mailbox,
created earlier and verified, and the user then asks for **one more change** to it
("add this link", "remove that line", "change the wording"). You must edit the
draft WITHOUT disturbing its threading, recipients, or Cc.

Verified 2026-09-21 (V-Gro Finserv consolidated-pack reply, psingh@draas.com —
three successive edits: add a spreadsheet line, remove it, change a sentence).

## Second worked case (2026-09-28): a voice re-dictation REPLACES the draft

NDR dictated a follow-up to external counsel about three agreements. The first draft
was built from his initial wording; two turns later he re-dictated the same content
with the sequence corrected (*"It's not registration… we need to get one sale deed
registered in order to get the fixation process completed"*). That dictation was not an
addition — it replaced the framing.

- **Treat a re-dictation of the same ask as a REPLACEMENT, not a patch.** Delete the
  existing draft and recreate it threaded to the same thread. Do not bolt the new
  wording onto the old framing — the two readings contradict each other and the
  recipient sees whichever survives.
- **Diff the two dictations and say what changed** in the reply to the user (here:
  the wrong "done and the registration going" phrasing removed; the sequence restated
  — sale deed + construction agreement registered together *completes the fixation*;
  the clubhouse / common-area agreement reframed as a separate **third** agreement to
  be done later with the first purchaser).
- **Read the thread tail BEFORE drafting any follow-up.** That is where prerequisites
  and the real state of play live. Here the tail showed a colleague had already sent
  the counterparty the full 19-document RERA set two days earlier — so the draft must
  not re-ask for documents already delivered, and the reply should tell the user the
  prerequisite was already met.
- **A garbled word inside the clarification gets a flag, not a silent fix** — and a
  precedent the user names ("Sobha, Gardenia and other projects") stays as dictated
  unless he asks for it softened.

## Core rule — CORRECTED 2026-10-10: `drafts().update()` edits the body IN PLACE

**A draft body CAN be replaced in place. Prefer `users().drafts().update()`; treat
delete + recreate as the fallback.**

An earlier version of this note claimed the raw payload was immutable and that delete + recreate
was the only route. **That is wrong** — verified 2026-10-10 (Ranka Iris "Note of Gratitude"
reply, ndr@draas.com): a one-paragraph insertion into a ~17 KB HTML draft went through
`update()` and **kept the same draft id (`r2095987758268732886`) and the same `threadId`** — no
orphan draft, no new thread, none of the failure modes below.

```python
# Read the live draft, edit the HTML, write it back — same id, same thread
d   = gmail.users().drafts().get(userId="me", id=DRAFT_ID, format="raw").execute()
raw = base64.urlsafe_b64decode(d["message"]["raw"]).decode("utf-8", "replace")
msg = email.message_from_string(raw)                # see "Parsing the raw draft" below
html = <the decoded text/html part>                 # edit as text; `assert old in html` first

m = MIMEMultipart("alternative")                    # rebuild with the SAME headers
m["To"], m["Cc"], m["Subject"] = TO, CC, SUBJ
m["In-Reply-To"], m["References"] = IRT, REFS
m["From"] = "Nishant Ranka <ndr@draas.com>"
m.attach(MIMEText(html, "html", "utf-8"))

gmail.users().drafts().update(userId="me", id=DRAFT_ID, body={
    "message": {"raw": base64.urlsafe_b64encode(m.as_bytes()).decode(),
                "threadId": THREAD_ID}          # ← still required even on update
}).execute()
```

Use delete + recreate only when the draft must move to a **different** thread or when `update()`
errors. Harvest headers before any destructive step either way.

## Parsing the raw draft — never hand-split the MIME

Get headers **and** body from one `format="raw"` fetch:

- **`format="raw"` is the reliable route for everything.** On a bridge-created draft,
  `format="full"` returned `Subject`, `To` and `Cc` as **`None`** — only `From`, `In-Reply-To`,
  `References` and `threadId` survived. If you need the recipient set, use `raw` (or `metadata`).
- **Do NOT split on `Content-Transfer-Encoding: base64` and `b64decode` the remainder.** It fails
  with `binascii.Error: number of data characters (N) cannot be 1 more than a multiple of 4`,
  because the split leaves the blank-line separator inside the payload. Parse properly:

```python
msg  = email.message_from_string(raw)
html = next(p.get_payload(decode=True).decode("utf-8", "replace")
            for p in msg.walk() if p.get_content_type() == "text/html")
```

- **Verify the edit by content, not by the success return.** Assert the new string is present,
  that the formatting-block counts moved by exactly the expected amount (e.g. blue `#0000CC`
  blocks 22 → 23 while red `#CC0000` stayed at 9), and report the character-length delta.
  `drafts().update()` returning a draft id proves nothing about the body.

## The recipe — FALLBACK ONLY (delete + recreate)

Use this only when `drafts().update()` is unavailable for the case, or the draft must land on a
different thread. For an ordinary body edit, the `update()` recipe above is the correct one: it
keeps the draft id, the threadId, and avoids the orphan-draft risk entirely.

```python
import base64
from email.message import EmailMessage
from tools.gws_auth import build_service

THREAD_ID = "<original threadId>"          # from the existing draft
DRAFT_ID  = "<existing draft id>"
gmail = build_service("gmail", "v1", service_name="google-draas")

# 1) Harvest the ORIGINAL threading headers BEFORE deleting anything.
orig = gmail.users().drafts().get(userId="me", id=DRAFT_ID, format="metadata").execute()
oh = {h["name"].lower(): h["value"] for h in orig["message"]["payload"]["headers"]}

# 2) Rebuild the message with the edited HTML body.
html = open("/tmp/body.html").read()
msg = EmailMessage()
msg["To"] = TO
msg["Cc"] = CC
msg["Subject"] = SUBJ
msg["In-Reply-To"] = oh.get("in-reply-to", oh.get("message-id", ""))
msg["References"]  = oh.get("references",  oh.get("in-reply-to", ""))
msg.set_content("Please view this email in HTML.")     # plaintext fallback
msg.add_alternative(html, subtype="html")

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

# 3) Delete, then recreate ON THE ORIGINAL THREAD.
gmail.users().drafts().delete(userId="me", id=DRAFT_ID).execute()
new = gmail.users().drafts().create(userId="me", body={
    "message": {"raw": raw, "threadId": THREAD_ID}     # ← threadId is mandatory
}).execute()
```

## Pitfalls that actually bit

1. **`threadId` must be passed explicitly.** Raw-MIME insert (unlike a bridge
   `draft_reply_create`) does NOT re-thread from the `References` header — omit
   `threadId` and you get a brand-new thread. Always pass it, then assert
   `new["message"]["threadId"] == THREAD_ID`.

2. **Harvest `In-Reply-To`/`References` BEFORE the delete.** Once the draft is
   gone there is no easy source for them. Note `References` is often absent —
   fall back to `In-Reply-To`, then to `Message-ID`.

3. **Work on a body file, not string surgery in the call.** Read the current body
   out to a temp file, edit it as text with an `assert old in html` guard, write
   the result back out. The assert is what stops a silently-applied no-op edit.
   Keep the intermediate files (`body_final.html`, `body_final2.html`, ...) so
   "undo the last change" is a re-upload, not a reconstruction.

4. **Reverting is expected, not exceptional.** The user may ask you to undo a
   change one turn later ("stop, remove this"). Byte length is the fastest
   confirmation you restored the exact prior state — a 9,359 → 10,013 → 9,359
   round trip proves the revert is clean. Report the byte count.

5. **After every edit, re-verify all four things:** To, Cc, Subject, and
   `threadId`. Then list `drafts().list()` and assert the count is still 1 — a
   failed delete leaves an orphan and the user sees two drafts.

6. **Verify the RENDERED body, not just the draft's existence.** Re-fetch with
   `format="full"`, walk `payload.parts` for the `text/html` part, decode, and
   grep for the added/removed string. Reading back the header block alone proves
   nothing about the body content.

## When the edit also touched Drive artifacts — revert those too

If the change you're undoing included a Drive action (a file added to a delivery
folder, grants issued to recipients), the email revert is only half the job.
Undo the Drive side as well: trash the shortcut/upload, `permissions().delete`
the grants you created, then **list both back** (folder contents + target
permissions) and show the user the clean state. "Stop" means stop the whole
change, not just the sentence in the email.

## Where the draft lives — state it every time

Name the account in every report (`the draft is in psingh@draas.com`) and confirm
the thread matches. The user cannot see which mailbox holds the draft, and an
unstated account reads as a failure. See
`google-workspace/references/gmail-draft-account-and-provenance.md`.
