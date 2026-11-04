# Email body formatting — NDR's hard rule (no "poem format")

> **SUPERSEDED AS AUTHORITY (2026-11-02):** the canonical standard now lives in the
> `email-writing` skill, which also ships the renderer
> (`skills/communication/email-writing/scripts/email_draft.py`, markdown → verified
> multipart draft). Use that. This file is kept for the diagnosis, the verification
> recipe and the reuse-in-place rules below.

## The rule (stated by NDR, 31 Oct 2026, verbatim intent)

> *"It is written in some kind of poem format where after every fourth, fifth or seventh word
> there is a newline character, which is not how I ever want my emails to be written.
> Paragraphs should be made without any unnecessary newline characters at all. You can use rich
> text format or HTML, either way, so that it is easy to read and presentable."*

What this means in practice — apply to **every** email draft and to WhatsApp long-form too:

- **Never hard-wrap.** Do not insert `\n` at ~70–80 chars to mimic a terminal. A paragraph is
  ONE continuous line in the source; the client wraps it.
- **Blank line = paragraph break only.** Newlines separate ideas, never lines of an idea.
- **Send multipart**: `text/plain` (unwrapped paragraphs) + `text/html` (the presentable one).
  Clients that strip HTML still get clean paragraphs.
- **HTML is light, not decorative.** Inline style once on `<body>`
  (`font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; font-size:14px;
  line-height:1.55; color:#1a1a1a`), `<p>` per paragraph, `<strong>` for section heads, a
  `margin: 22px 0 6px 0` on section-head paragraphs for air. No tables, no colours, no logo blocks
  unless he asks.
- **Long single-sentence scope lists go INSIDE one paragraph**, joined with semicolons
  (`…finalise the content calendar with you today; discuss the scope of procurement…`). Do not
  turn them into a bullet list unless the list is genuinely itemised data (figures, dates,
  deliverables).
- Non-ASCII: em dashes as `&ndash;` in the HTML part, plain `-` in the text part.

Why it happens: MIMEText/`scheme`-style helpers and copy-pasting a terminal-rendered body
reflow at fixed width. Build the strings as single-line paragraphs and the problem disappears.

## Skeleton (multipart, threaded-safe)

```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import base64

msg = MIMEMultipart('alternative')
msg['From'] = 'Nishant Ranka <ndr@draas.com>'
msg['To'] = 'A <a@draas.com>, B <b@draas.com>'
msg['Subject'] = 'Re: <original subject>'
msg['In-Reply-To'] = ORIG_MID
msg['References']  = ORIG_MID            # empty-References case: the root's own Message-ID
msg.attach(MIMEText(plain, 'plain', 'utf-8'))
msg.attach(MIMEText(html,  'html',  'utf-8'))
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
```

## Replacing an existing draft without leaving two

NDR often has already edited the draft in Gmail (he corrects spellings himself). Read it first —
`gws_gmail_get(message_id=<draft msg id>)` — then rebuild and **put the new raw into the SAME draft
id**, so Drafts holds one version and his edits inform the rewrite:

```python
g.users().drafts().update(userId='me', id=DRAFT_ID,
        body={'message': {'raw': raw, 'threadId': THREAD_ID}}).execute()
```

Only delete-and-recreate when the draft is in the wrong thread or addressed wrong. `draft_id` is
stable across `update`; `message_id` is NOT (it changes) — re-read it before verifying.

## Verify before reporting (non-optional)

Read the draft back `format='full'` and assert all of:

- `parts == ['text/plain', 'text/html']`
- every `text/plain` line length is a real paragraph, not ~72: `max(len(l) for l in body.splitlines())`
  should be a full sentence's length, and `len(splitlines())` should equal the paragraph count + blanks
- the corrected name/spelling count: `html.count('Gowri') == n` and `html.count('Gauri') == 0`
- `To`, `Cc`, `In-Reply-To`, `threadId`, `labelIds == ['DRAFT']`

## Consistency across channels (caught in review, 31 Oct 2026)

When a mail goes to a group and the same meeting is confirmed by WhatsApp to one person, the
**times must match in both**. NDR dictated "11:30, or just tell her 12 to 1 — the 1 to 2 slot is
with Anbu, so tell her 12 to 1"; he asked only for the WhatsApp to change, but leaving the thread
mail at 11:30 would have put two different times for the same meeting in the same conversation.
Adjust the mail too and **say in the report-back that you changed it and why**, so he can veto.

Same rule for names: when he spells a name out letter by letter (`G-O-W-R-I`), that supersedes the
contact sheet, the thread headers and your own earlier draft — sweep every occurrence in the body,
the To line, and the subject, and prove it with the `count()` check above.
