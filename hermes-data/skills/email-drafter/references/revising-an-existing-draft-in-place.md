# Revising an EXISTING draft in place (body + attachment)

Triggers: "update that draft", "change the figures in the draft", "don't put in figures there",
"can you please update that draft", or any correction to an email already sitting in Drafts.

A draft is a live object NDR has already read and shaped. Do NOT create a second draft — edit
the one he means, in place, and leave its id, To and Subject unchanged.

## 1. Find the draft, don't guess
`drafts().list(maxResults=50)` then `drafts().get(id, format='metadata')` and print
To / Cc / Subject / Date for each. Pick by subject + recipient (`rnr@draas.com` = Roshini).
If two candidates match, ask — never edit both.

## 2. Read the body AND inventory the attachments
Walk `payload['parts']` recursively, base64url-decode `text/plain` and `text/html`.
Then, separately:
```python
[c.get('filename') for c in payload.get('parts', []) if c.get('filename')]
```
A footer line saying "Attachment: <X> (PDF)" is NOT proof an attachment exists, and an
attachment exists even when the body never mentions it. Check the parts, always.

## 3. There is no "edit body" API — you replace the whole raw
```python
svc.users().drafts().update(userId='me', id=SAME_DRAFT_ID,
                            body={'message': {'raw': raw}}).execute()
```
Build `raw` as `MIMEMultipart('mixed')` → `MIMEMultipart('alternative')` (plain + html) →
`MIMEApplication(pdf)`, base64url-encode, and pass the SAME draft id. The draft keeps its
slot in Drafts and its thread; no duplicate appears. See
`templates/update-draft-in-place.py` for a working skeleton.

## 4. Body and attachment must not contradict
The most common trap: NDR corrects a figure in the body, but the attached working PDF was
generated from the old figure. A figure-free email with a figure-bearing "Recovery Working"
PDF attached is worse than the original.

- Download: `messages().attachments().get(userId='me', messageId=msg_id, id=attachmentId)`,
  base64url-decode, write to disk.
- Read: `pdftotext -layout file.pdf -` (poppler). `fitz`/`reportlab` are NOT in the system
  python; `pypdf` may be. If you must regenerate, build a venv:
  `uv venv /path/pv && uv pip install --python /path/pv/bin/python reportlab`.
- Regenerate as a no-figures instruction memo and REPLACE the attachment (new filename,
  updated body footer). Tell NDR in the summary that you did, and why.

## 5. "Don't put figures in" means no figures
- Delete the wrong figure **and** do not substitute the correct one if he didn't state it.
  ("Fixed CTC is not 50,000, we've been paying 75,000 — don't put in figures there" ⇒ neither
  number appears anywhere.)
- Keep only amounts NDR dictated himself and that are needed to *identify* an item
  (e.g. ₹5,000 paid today, ₹5,00,000 advance) — plus their payment refs (UPI ref, last-4 a/c).
- Strip derived totals too: pro-rata, "amount retained", "total recoverable", per-month
  recovery rates. Replace the arithmetic table with a table of HEADS and TREATMENT
  ("Withheld in full" / "Adjusted first" / "Recovered" / "Nil or a recovery"), not amounts.
- Keep one sentence saying explicitly that no figures are stated by design and who is to
  compute them — this is what makes the email actionable rather than vague.

## 6. A correction usually flips more than the point he named
Re-read every numbered point against the fresh dictation. In the Prakash Singh revision the
instruction "plus the ₹5,000 I gave him earlier today … all of that has to be adjusted"
silently reversed the old draft's "the ₹5,000 is NOT to be recovered". Sweep the whole body
for other statements the new instruction contradicts.

## 7. Never invent the account you were asked to produce
If NDR says "give him that account" (school fee, uniform, ledger utilisation) and you have no
ledger — Gmail will not have it — do not fabricate a statement. Put it in the email as an
explicit instruction to accounts to prepare it from the ledger holder's running statement, say
in the completion note that you could not produce it, and offer to have accounts route it to
NDR first before the counterparty sees it.

## 8. Verify before reporting
Re-read the draft and assert all of:
- `To` / `Subject` unchanged, attachment filename is the NEW one, exactly one draft in Drafts;
- a stale-figure grep returns False for every number you removed
  (`for bad in ['50,000','45,000','43,548','38,548']: assert bad not in all_text`);
- HTML has no `%%` left from `%`-formatting the CSS, and `width:100%;` survived;
- the PDF round-trips: re-download it and `pdftotext` the last page.

Drafts are never sent. Say "draft updated, still in your Drafts" — never "sent".
