# Mass / bulk outreach drafts (lead list -> N individual Gmail drafts, inline image, per-recipient CC)

Use when the user gives a lead spreadsheet and wants one draft per recipient ("send individually"),
with CC to a supervisor (e.g. ndr@draas.com) and a vetted marketing creative embedded in the body.

## Recipe (verified 2026-10-07, 1,136 drafts, Ranka Udaya RG to Hosur SMEs)

1. **Extract + dedupe the lead list** via Sheets API (`build_service('sheets','v4', service_name='google-draas')`).
   - Tab by GID: read `spreadsheets().get()` sheet properties, match `sheetId` -> title, then values().get.
   - Dedupe by lowercased email; keep company names per email (an email shared by 2 firms => both names in salutation).
   - Validate with `^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$`; count + report rows with no/malformed email; write JSON for the batch script.
2. **Render the vetted ad/creative** to a compact low-res JPEG for embedding:
   - `pymupdf` (import pymupdf, not fitz) -> `page.get_pixmap(dpi=70)` -> `pix.save(out, jpg_quality=72)` (~120 KB is fine for email).
   - ALWAYS verify the creative content with vision_analyze before mass use (grabbed PDF -> OCR -> confirmed it was the RG offer with the right figures).
3. **Build MIME** with `email.message.EmailMessage`:
   - `To` = recipient, `Cc` = supervisor, `Subject` as given.
   - `msg.set_content(plaintext_fallback)` then `msg.add_alternative(html, subtype='html')`.
   - Inline image: after add_alternative, `msg.get_payload()[1].add_related(img, maintype='image', subtype='jpeg', cid='rg_ad')` and reference `<img src="cid:rg_ad">` in the HTML.
   - Personalise the salutation with the company name from the sheet.
   - `raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()`.
4. **Create the draft — PITFALL (400 "Missing draft message")**:
   - `gmail.users().drafts().create(userId='me', body={'message': {'raw': raw}})` — **the raw message MUST be nested under `message`**. Passing `body={'raw': raw}` fails with HttpError 400 "Missing draft message".
5. **Batch**: run as a background process (1,136 drafts took ~12 min). Log progress every 50 with timestamp + last email; log each error with the email; `mkdir -p` the log dir BEFORE redirecting stdout (shell redirect fails if the dir doesn't exist). Small sleep (0.25s) every 20 to stay under quota.
6. **Verify ground truth from Gmail**, never trust the script's own counter:
   - Page through `drafts().list(userId='me', maxResults=500, pageToken=...)` until no nextPageToken.
   - Count by Subject substring; sample a few drafts; check `To`/`Cc` headers.
   - `drafts().get(format='metadata')` — **do NOT pass `metadataHeaders=[...]`** (TypeError: unexpected keyword argument). Headers live at `g['message']['payload']['headers']`.

## Hard rules that apply

- DRAFTS ONLY. Never call `.send()`. The user (or their boss) reviews/sends from the Drafts folder.
- CC the supervisor on EVERY draft when the user asks for it ("add CC to ndr@draas.com in each email").
- Use the approved creative's own figures (the ad PDF), not figures from memory — flag to the user if the sheet/offer numbers differ so they can choose before mass creation.