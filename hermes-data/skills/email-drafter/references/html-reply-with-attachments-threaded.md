# HTML reply-draft with attachments, on the SAME thread (hand-built raw MIME)

Verified 2026-09-29 (Oasis 50-cents critical reply to Sinchana): NDR dictated a strong
HTML+CSS follow-up with a user-sent image attached, as a REPLY-ALL on the existing
"Oasis 50 cents Area planning" thread. The bundled `create-draft-with-attachments.py`
only emits a text/plain part — when NDR asks for HTML presentation (see
ndr-email-presentation-2026.md) AND an attachment AND threading, build it by hand.

## When to use
- Threaded reply (same threadId as parent) with a styled HTML body + one or more
  attachments (usually a user-provided image from /data/hermes/image_cache/).
- reply-all semantics = To: primary recipient (Sinchana), Cc: the thread participants
  + newly-marked people (Bharat, Anbu, Kantesh, Gowri, ...). Resolve in-house emails
  from Gmail headers (`from:` search), never the contacts sheet — e.g. Anbu sends as
  **anbarasan@draas.com** (his Sep-2026 mails), Gowri = Gowri Singh **gsingh@draas.com**
  (NOT the sheet's gowrisingh@yahoo.com legacy rows).

## MIME shape (multipart/mixed → multipart/alternative → base64)
```
multipart/mixed; boundary=B1
└─ multipart/alternative; boundary=B2
   ├─ text/plain (paragraphs on single logical lines, NO manual wrapping)
   └─ text/html  (full-width table/banner/callout styling per ndr-email-presentation-2026.md)
└─ image/jpeg attachment (base64, filename preserved)
```

## Steps (from the worked session)
1. `gws_gmail_thread_get` the parent thread first — read the LATEST messages so the
   reply actually addresses what the other side just said (NDR often answers their
   latest mail, not the thread head).
2. Pull the parent Message-ID via `messages().get(format='metadata',
   metadataHeaders=['Message-ID','References'])`; set `In-Reply-To` = parent MID and
   `References` = parent References + parent MID.
3. Fetch the user's image attachment path (from the chat's image_cache) and base64-encode.
4. Build raw MIME by STRING CONCATENATION (head list + '\r\n'.join) — exactly like
   create-draft-with-attachments.py; do NOT use EmailMessage for HTML+attachment mixes
   when threading matters (Gmail folds/rewrites References and the draft can fail to
   open in the web UI). Hand-built raw MIME avoids that.
5. `drafts().create(body={'message': {'raw': ..., 'threadId': parent_thread}})`.
6. VERIFY with `drafts().get(format='full')`:
   - labelIds == ['DRAFT']
   - threadId == parent thread
   - To / Cc / Subject match intent
   - walk payload parts and assert every attachment filename is present.

## Re-attaching the ORIGINAL attachment (byte-identical) — verified 2026-10-09

When the reply must carry back a document the user originally sent (a form to be
signed, a statement to be corrected), do NOT regenerate or re-export it — pull
the exact bytes off the parent message and attach those:

```python
svc = build_service('gmail', 'v1', service_name='google-draas')
att = svc.users().messages().attachments().get(
    userId='me', messageId=PARENT_MSG, id=ATTACHMENT_ID).execute()
pdf = base64.urlsafe_b64decode(att['data'])
open(dest, 'wb').write(pdf)          # then base64 it into the MIME part
```

Take `ATTACHMENT_ID` from the parent's `messages().get(format='full')` parts.
Assert the byte count matches the original (Ranka Amber Form 5 chase: 58,212
bytes; verified equal after save). That byte match is the proof the recipient is
signing the same document the user sent — a re-export from the source HTML/GDoc
can reflow and silently differs. Print the byte count in the report.

## Reply-all to an EXTERNAL consultant on the principal's own thread (verified 2026-10-09)

- **Keep the To the principal himself used on the thread**, even when the
  consultant only ever *sends* from a different address of the same firm
  (Form 5 chase: To `BK@fnw.homes`, but Bhuvanesh only ever sends from
  `bk@findingform.design`). Add the sending address on **Cc** as a delivery
  safeguard and state the choice in the report — do not silently swap the To.
  A quick `gws_gmail_search` for the address shows whether it ever appears as a
  *sender*; if it never does, say so.
- **Mark the third party the user names** (e.g. the certifier on record) using
  the address from the contacts sheet, and say explicitly that it is the only
  address held + that the RERA document carries the *firm's* address instead, so
  the user can swap it before sending.
- **Cc any in-house colleague already participating in the thread** (Prakash
  here) — that is what "reply all" means in this shop.
- Threading: `In-Reply-To`/`References` = the user's **own** sent message, plus
  `threadId` on `drafts().create`. Replying to the principal's own message (not
  the thread tail) still threads correctly and keeps the subject stable.

## Pitfalls
- A `drafts().get(format='metadata')` reports 0 attachments — use format='full' and
  walk `payload.parts[].filename`.
- Keep the text/plain fallback flowing (no mid-sentence '\n'); NDR reads on mobile
  and rejects hard-wrapped bodies.
- When NDR says "reply all", do not just reuse the thread To: — he usually wants
  additional people marked (Cc) who were not on the thread (manager/planner FYI).
- HTML callout boxes that landed well: red left-border "⚠️ VERY CONCERNING & CRITICAL"
  banner, amber requirement box, navy FYI box for a cc'd party ("hold until X fixes").
- **Do NOT build this draft at all if the source thread cannot be located.** A
  reply-all needs the parent message's real `Message-ID`/`References`; with no parent
  there is nothing to thread onto, and guessing a vendor plus a stale thread to satisfy
  an instruction that literally said "reply all" produces a misdirected formal mail to
  a firm the user may no longer deal with. Report the blocker and name the single word
  that unlocks it — method in `email` →
  `references/vendor-email-forensic-sweep.md`. (Contrast: when the thread IS known and
  only a *field* is uncertain — e.g. the certifier's individual address vs the firm
  address on the registration — DO draft, and flag the swapped field.)
- Draft only — never send; report draft ID + that it is unsent.
- **The attachment filename on the wire is `os.path.basename(ATTACH)`.** User-uploaded
  images live in image_cache as `img_<hash>.jpg` — if you point ATTACH there, the recipient
  sees `img_ac6afe7ca910.jpg` on a formal/board mail. Copy the image to a descriptive name
  FIRST (`cp image_cache/img_...jpg /tmp/Recipient_Context_What.jpg`) and point ATTACH at
  that. Verified 2026-10-13 (Dharmesh approval screenshot → `Dharmesh_Ranka_WhatsApp_Approval.jpg`).
- **Re-running the template to fix a filename creates a SECOND draft** — the script calls
  `drafts().create()` unconditionally each run. If you re-ran, delete the stray draft
  (`gws_gmail_draft_delete`) and confirm only the verified one remains, or NDR will find a
  duplicate board email in Drafts.