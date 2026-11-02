# Sending internal DRA material to an outside party (lender / bank / buyer)

**Triggers:** "forward Nishant sir's mail to <lender>", "send this to Sudheer / V-Gro / the
bank", "attach the partnership deed to that mail", "compile a mail for <external party>
using NDR's details".

Worked case: V-Gro Finserv (Sudheer Desu) information request, 29–30 Oct 2026.

## 🔴 The leak check — do this before any forward leaves the building

An internal mail forwarded to an outside party carries **NDR's instructions to staff**
with it. Real example that nearly went out:

> "Shareholding pattern … I will send these separately — **please keep this mail on hold
> with Sudheer until the full set is ready, so he gets one complete response rather than
> it going out in pieces.**"

Sending that tells the lender (a) the answer is incomplete and (b) we are managing him.
**Always read the full quoted body looking for internal-only content and cut it:**

- hold / delay / "don't send yet" instructions
- "I will send X separately" — contradicts anything you attach in the same mail
- names of staff the mail is delegating to, internal deadlines, "Sarthak, do this"
- anything addressed to us personally ("Dear Sarthak,")
- internal assessments of the counterparty

Mechanically: locate the substance section start and the internal section start, keep
`body[i_substance:i_internal]` + any trailing footnote, and re-check the result:

```python
for bad in ('keep this mail on hold', 'Dear Sarthak', 'Still to come',
            'I will send these separately'):
    assert bad not in final_body, bad
```

Note `"to follow"` is a FALSE POSITIVE — "JDA to follow the term sheet" is legitimate
land-bank text. Grep for the full phrase, not the fragment.

## Mask Aadhaar / PAN before attaching deeds

Reconstitution deeds, partnership deeds and KYC sets print the signatories' Aadhaar
numbers. Never hand those to a lender raw. Redact, then **prove** the numbers are gone
from the text layer:

```python
import re, pymupdf as fitz
doc = fitz.open(src)
for page in doc:
    txt = page.get_text("text")
    cands = set(re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b", txt)) | set(re.findall(r"\b\d{12}\b", txt))
    for c in cands:
        for r in page.search_for(c):
            page.add_redact_annot(r, fill=(0, 0, 0))
for page in doc:
    page.apply_redactions()
doc.save(out, garbage=4, deflate=True)

# verify — must print []
d2 = fitz.open(out)
left = [m for p in d2 for m in re.findall(r"\b\d{4}\s?\d{4}\s?\d{4}\b", p.get_text("text"))]
print("remaining:", left)
```

Say in the mail body that Aadhaar has been masked and unmasked copies follow on request —
the lender's KYC file then knows the difference is deliberate, not a bad scan.

`apply_redactions()` must be called on every page *after* all annots are added; skipping
it leaves the text extractable under the black box.

## Draft with attachments — the native tool cannot do it

`gws_gmail_draft_create` has no attachment parameter. Build the raw MIME yourself:

```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

msg = MIMEMultipart('mixed')
msg['To'] = 'lender@example.com'
msg['Cc'] = 'ndr@draas.com, rnr@draas.com'
msg['Subject'] = 'Re: <original subject>'
msg['In-Reply-To'] = orig_msgid          # FETCH IT, never invent it
msg['References'] = orig_msgid
msg.attach(MIMEText(body, 'plain', 'utf-8'))
for path, name in attachments:
    part = MIMEApplication(open(path,'rb').read(), _subtype='pdf')
    part.add_header('Content-Disposition', 'attachment', filename=name)
    msg.attach(part)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
g.users().drafts().create(userId='me',
    body={'message': {'raw': raw, 'threadId': orig_thread_id}}).execute()
```

- **Never invent the Message-ID.** Read it off the original:
  `messages().get(..., format='metadata', metadataHeaders=['Message-ID','References'])`.
  A fabricated `<...@mail.gmail.com>` threads nowhere and can be flagged as spoofed.
- Pass `threadId` so the draft lands in the lender's existing conversation even when the
  subject changes to `Fwd:`.
- Gmail's 25 MB limit: three PDFs ≈ 8.7 MB of `sizeEstimate` after base64 — fine; check
  `sizeEstimate` on the created draft if in doubt.

## Verify the draft before telling the user it is ready

Read the draft back (`drafts().get(..., format='full')`) and assert:

1. To / Cc / Subject / In-Reply-To as intended
2. every attachment present with the exact filename the receiver should see
3. body length is what you wrote — and the leak strings above are absent
4. the key facts are actually in the body (`assert '2,500 equity shares' in body`)

Then delete any superseded draft so Drafts holds exactly one sendable version.

## Workflow lesson from the V-Gro case

When NDR's own mail already contains the substance, the user will usually want it
**compiled into one clean response**, not re-authored. Sequence that worked:

1. read the counterparty's request and NDR's reply mail
2. gather the missing piece from primary documents (deeds, certificates) — do not
   paraphrase what the deeds say, quote the structure clause
3. draft **one** mail answering the request **in the counterparty's own order**, flagging
   any item that is genuinely pending
4. verify, then report what you changed and what still needs the user's decision

Do not promise a follow-up date you cannot keep ("by [date — to confirm]"); write "will
follow separately" and let the user decide whether to hold the mail.
