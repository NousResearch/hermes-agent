# Bulk Personalized Gmail Draft Campaigns (mail-merge style, drafts only)

Canonical example: Hosur SMEs (HOSTIA) campaign — 1,196 rows in a Google Sheet tab,
1,136 unique valid emails (4 blank, 0 malformed), one Gmail draft per recipient,
each CC'd to ndr@draas.com, personalized "Dear <CompanyName>" salutation, one inline
image. NEVER sends anything — the deliverable is N drafts in the sender's Gmail.

## Build the recipient list first
- Read the sheet tab, extract (company name -> email). Some emails shared by 2 companies:
  show BOTH names in the salutation. Dedupe by email; if duplicates have different
  company names, merge the names.
- Report coverage honestly: total rows, rows with no email, malformed emails, unique
  valid count. This drives the batch loop.

## drafts.create API quirk (critical)
- The Gmail API createDraft expects the raw MIME call NESTED:
  `service.users().drafts().create(userId='me', body={'message': {'raw': base64.urlsafe_b64encode(mime_bytes).decode()}}).execute()`
  NOT `body={'raw': ...}` at top level — that 400s.

## Always test ONE draft first
- Build the full MIME (HTML body + inline image via `cid:` + CC header + To), create one
  draft, then GET it back and verify: subject, To, CC present, image part embedded with
  the right content-id, correct filename. Delete the test draft after verification.
- This catches MIME structure bugs before the mass loop costs you 1,100 drafts.

## Long loops: background + resume-capable (600s timeout kills foreground)
- 1,136 drafts x (get + create) at ~40 drafts/min exceeds any foreground timeout. The
  process gets killed at 600s mid-run.
- Run the loop with terminal(background=True, notify_on_complete=True). Write progress
  to a log file. Before starting, list existing drafts and SKIP any whose raw body
  already contains the target image md5 (idempotent resume) — DON'T rely on a separate
  "done" file that the killed process may not have flushed.
- If an earlier killed run already swapped ~700, resume must detect and skip, not redo.

## Verification = Gmail ground truth, not the loop's own log
- After the run, count actual drafts by subject via the API
  (`gws_gmail_search` / drafts.list with the exact subject) and sample random drafts
  (e.g. every 7th = statistical sample) checking body text, CC, image md5.
- The loop reporting "1136 updated, 0 errors" is a self-report; verify externally.

## Image swap across all drafts in place
- To replace the embedded image in every draft: for each draft, get raw, strip old
  attachment part, splice new image (same cid), create updated raw, drafts.update
  with the same draftId (the `message` id field). Preserve To/Cc/body; only swap media.
- Verify with md5 of the extracted attachment — exact byte match to the source image.

## Deliverability / non-spam prep for cold blasts (asked every time)
- Check DNS FIRST: SPF (must include google.com for Workspace), DKIM (google selector
  published?), DMARC (draas.com frequently has NONE — this is the #1 gap). dig may be
  missing; use DNS over HTTPS (e.g. curl to cloudflare-dns.com/dns-query?name=_dmarc.draas.com&type=TXT)
- Order of operations for Postmaster Tools: publish DMARC first (p=none + rua), then
  enable DKIM in Google Admin (Apps → Google Workspace → Gmail → Authenticate email,
  generate the two CNAMEs, publish), then postmaster.google.com → add domain →
  publish google-site-verification TXT → verify. Data lags 24–48h; day one looks empty.
- Cold-blast reality: Gmail spam decision is per-recipient engagement (opens/replies/
  complaints). Never send 1,136 at once — 50–100/day batches, morning IST, personalized
  To. Watch domain reputation (want High/Excellent), spam rate (<0.1%), user-reported
  spam (zero), auth pass rates. Postmaster only sees Gmail recipients; Yahoo/Rediffmail
  in the list are a separate signal.
- Do NOT promise "not spam" — say honestly it depends on engagement + setup, and give
  the concrete fix list instead.

## Content hygiene pitfalls
- Chat-room sentences must NEVER leak into drafted email bodies. When the user says
  "remove this from the email content", first VERIFY the draft body from Gmail — the
  offending text may have been in the chat reply only, and no rebuild is needed.
- Remove hard figures from body copy for cold lists (RERA/claim risk): replace with
  "we will share the exact numbers on a call". Images the user explicitly chose are
  untouched even if they carry figures.