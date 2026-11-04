# A threaded reply that must CARRY A FILE — the deltas

> **The full MIME recipe already exists**: `messaging-drafts/references/threaded-reply-pdf-attachments.md`
> (hand-built MIME, Drive/Gmail attachment download, in-place attachment swap, verification).
> Read that for the mechanics. **This file is only what that one does not cover**, plus the routing
> half that lives in `references/threaded-reply-all-routing.md` (anchor choice, threadId, References
> chain).
>
> Runnable builder for this exact shape: `scripts/draft_reply_with_attachment.py` in this skill —
> args `--anchor --thread --to --cc --body-file --attach --service`, prints the draft id and a
> verified flag, never sends.

**Durable rule, both skills:** none of the draft tools accept an attachment —
`gws_gmail_draft_create`, `gws_gmail_reply_draft`, the DWD variants, and the bridge's
`draft_create` / `draft_reply_create` all build `MIMEText` and stop there. A reply that must carry a
file is hand-built MIME + `drafts().create()`. That is a DRAFT, not a send, so it is allowed;
`.send()` never is.

## 1. Anchor on an OLD chain if that is where the subject lives

Verified 2026-10-27, "DRA Projects - Assets Sale to Directors - Transaction Analysis"
(`19cacba9e0cf52ec`). The substantive chain was **eight months old** and its tail was NDR's own
forward to the adviser. It was still the correct home: the advisory memorandum and a co-director's
written assent sat under it, so the recipient re-reads the basis without leaving the mail.

Don't create a fresh subject just because the thread is stale — and don't fall for the anchor's
own To-list: that forward went to the adviser alone, while the instruction was to mark two other
directors. **Recipients come from NDR's instruction; headers come from the anchor.**

## 2. The `Re: Fwd:` subject trap

Anchoring on a *forward* yields `Re: Fwd: <subject>`. Strip every leading `re:`/`fw:`/`fwd:` and add
one `Re:`:

```python
subj = re.sub(r'^\s*(re\s*:|fwd?\s*:|fw\s*:)\s*', '', subj, flags=re.I)
subj = 'Re: ' + subj
```

The first draft of 2026-10-27 went out as `Re: Fwd: DRA Projects - …`. Fix = delete that draft,
recreate, then list drafts and confirm exactly one live one. `drafts().list()` has been seen to
under-report immediately after creation — confirm by fetching the draft id directly.

## 3. The file NDR points at may be a WhatsApp image, not a document

"Attach his invoice or whatever he shared on WhatsApp" → `wa_get_media(messageId=…)` writes it to
`/data/hermes/tmp/wa-media/`. For a photograph of a screen (his own Excel fee-split workings), ship
a one-page PDF so the recipient gets something printable rather than a sideways JPEG:

```python
Image.open(path).convert('RGB').save(out_pdf, 'PDF', resolution=150)
```

Name it after the substance — `Vishwas Rao - fee split shared 26 Oct 2026.pdf`, never
`IMG-20261026.jpg`. The same discipline as Drive filenames: the attachment name is read before the
body is.

## 4. Reading a photographed-screen spreadsheet

Whole-frame OCR garbles on screen moiré. Crop the particulars column and the amount column
separately, upscale ×2–3 with LANCZOS, and run vision on each; line items, blanks and the total
column then resolve cleanly. Do this before writing a note that quotes the numbers — the split in
this session only became reliable after the two-column crop.

## 5. Verify the PART, not just the headers

Walk `payload.parts` recursively on `drafts().get(format='full')` and assert each filename plus
`body.size` equals the file on disk (2026-10-27: 212,441 bytes, matching the local PDF). A draft with
perfect To/Cc/In-Reply-To and no attachment part looks like success in the Gmail list and fails in
front of the recipient.

## 6. Large attachments out of Gmail — the same truncation trap

`gws_gmail_attachment_get` on a ~260 KB PDF returned a >350 KB tool result which the harness
persisted to a file, and whose base64 would not decode (`Incorrect padding`; ~8.7k invalid chars
after sanitising). Same failure the messaging-drafts reference documents for >1 MB — it is
truncation, not a padding bug, and it cannot be repaired. Use
`messages().attachments().get()` from `execute_code`, or read the document from Drive, or **say the
document could not be read** and write the note from sources you did read. Never paraphrase an
unread memorandum from memory — and flag the gap in the hand-off, as here.
