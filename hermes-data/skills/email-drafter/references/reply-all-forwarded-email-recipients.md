# Reply-all to a FORWARDED email — reconstruct the real recipients

## When this applies
NDR's mailbox receives a **forward** (very common: Roshini <rnr@draas.com> forwards an
external email to ndr@draas.com — e.g. "Fwd: JD for Sales Manager - DRA Homes"), and NDR
says **"reply all to that email"**. Gmail-native reply-all is NOT enough: the forwarded
message's headers are only `From=forwarder / To=NDR`. The original sender and their Cc
chain live INSIDE the forwarded body as quoted text ("Begin forwarded message:" /
`<blockquote>`).

## Recipe (verified 2026-09-09, People Fractions JD feedback)
1. `messages().get(userId='me', id=<forward>, format='full')`; decode the body by walking
   `payload.parts` for `text/plain` or `text/html` and base64-decoding `body.data`.
2. Parse the embedded original block for **From / Date / To / Cc / Subject** — those are
   the REAL participants of the conversation.
3. Rebuild To/Cc by business intent, not header mechanics:
   - **To** = the original sender who asked for input (e.g. Ekta <ekta@peoplefractions.com>
     — her mail said "let me know if any changes are required").
   - **Cc** = original Cc chain + the forwarder (Roshini), so everyone stays in the loop.
   - Addresses taken from the actual embedded headers are ground truth and satisfy the
     contacts rule (same standard as `voice-transcribed-recipient-resolution.md`).
4. Threading: use the FORWARD's `threadId`; `In-Reply-To` = the forward's `Message-ID`
   (fetch via `format='metadata', metadataHeaders=['Message-ID','References']` — that
   kwarg IS accepted on `messages().get`, it is `drafts().get` that TypeError-s);
   `References` = forward's References + forward Message-ID. Subject stays Gmail-style:
   `Re: Fwd: <original subject>`.
5. Verify by refetching the created draft (full format, read headers off payload): To,
   Cc, Subject, In-Reply-To, and attachment count (`drafts().get` does NOT accept
   `metadataHeaders` — see `draft-create-verify-api.md`).
6. In the delivery summary, show NDR the reconstructed recipient list and offer to flip
   To/Cc if he intended otherwise.

## HTML reply body with PDF attachments
- `MIMEMultipart('mixed')` → attach `MIMEMultipart('alternative')` (plain + html), then
  one `MIMEBase('application','pdf')` per attachment (read payload, `encoders.encode_base64`,
  `Content-Disposition: attachment; filename=...`).
- Run via `terminal` with `/opt/hermes/.venv/bin/python3` (vault socket present there;
  not in `execute_code`).
- Convert markdown source docs (requirements / analysis) to branded PDFs with
  `scripts/md_to_pdf_dra.py` — `markdown` lib (tables/fenced_code/sane_lists) +
  `weasyprint`, DRA palette (navy #1F3864, gold #C99A2E, cream #F5EFDF zebra), A4 with
  page numbers. No pandoc/LaTeX needed.

## Real case
"JD for Sales Manager - DRA Homes": Ekta <ekta@peoplefractions.com> → rnr@draas.com
(Cc Sikha, Deepti, Richa @peoplefractions.com) → Roshini forwards to ndr@draas.com.
Build: To Ekta, Cc Sikha/Deepti/Richa + Roshini, subject "Re: Fwd: JD for Sales Manager -
DRA Homes", attachments = requirements PDF + JD-analysis PDF (both md→PDF), all verified.