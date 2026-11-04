# Bulk individual email drafts from a spreadsheet (one draft per recipient, CC fixed)

Verified 2026-10-07 on the **Hosur SMEs (HOSTIA)** campaign for Ranka Udaya RG: 1,136
individually-addressed Gmail drafts built from a Google Sheet tab, each with the same
HTML body + an inline image + CC ndr@draas.com + a per-company salutation.

## When to reach for this
User hands a sheet ("email all these companies") and wants REAL drafts (not a BCC blast):
each recipient sees only their own To, the salutation names their company, the same CC
goes on every draft, and an image is embedded in the body. Everything stays a DRAFT —
the user clicks send themselves (hard rule: never auto-send).

## Workflow
1. **Read the sheet tab, extract emails.** Sheets API → the named tab (e.g. "Hosur SMEs
   (HOSTIA)"). Deduplicate by email (lowercased, stripped), keep the company-name mapping.
   Some addresses belong to 2 companies — join both names into the salutation.
   Report the raw rows → unique-valid-email delta (1,196 rows → 1,136 unique here).
2. **Render the image to low-res email-JPEG.** If the body image comes from an approved
   PDF creative, render page 1 via pdftoppm → JPEG at ~600px long edge (123 KB for
   579×819). Keep the approved creative untouched — never rebuild the ad, never change
   its figures; embedding the low-res render of the existing asset is the point.
3. **Build the HTML body once** (personalized salutation placeholder + inline image via
   `cid:rg_ad` + CC). MIME: multipart/alternative (plain + html), HTML references the
   image as `cid:`, and the image is attached as an inline part with
   `Content-ID: <rg_ad>` + `Content-Disposition: inline`.
4. **drafts.create per recipient** with `body={'message': {'raw': raw}}` — the raw must be
   nested under `message` (missing that wrapper is the classic 400).
   `gmail.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()`.
5. **Verify by counting, not by trust.** After the run, call drafts.list and confirm the
   count matches recipients; pull a few drafts and assert subject / CC / salutation / the
   embedded image bytes (decode attachment; md5 against the source JPEG).

## Mid-campaign image swap (idempotent, resume-capable)
User says "swap the image to the other ad" AFTER 1,136 drafts exist. Do NOT recreate —
update in place:
- drafts.list → drafts.get(format='full') to read the current raw → decode the MIME,
  drop the old inline part, splice the new image part, re-encode → drafts.update(draftId,
  {'message': {'raw': new_raw}}).
- **Make the loop resume-safe**: write progress to disk (draft-id list) as you go; on any
  timeout/kill, restart and SKIP already-swapped drafts. Detection of "already swapped":
  fetch the embedded image md5 per draft and compare — but attachment fetch may fail on
  some drafts, so also accept the draft-id progress file as the source of truth.
- **Expect the foreground 600s timeout to kill a 1,100-draft loop** (~40 drafts/min →
  ~25 min). Run a long batch with terminal(background=True, notify_on_complete=True), or
  a resume script in the background; the FIRST foreground attempt will die mid-loop and
  that is normal — the resume logic is not optional.
- **Finding the drafts**: subject changes between generations (v1 "Guaranteed Rental
  Income Homes", v2 "Rental Income Homes | Phase 1 Units Now Open"). Match by the CURRENT
  subject, not memory of the old one. drafts.list estimate can exceed the message count —
  iterate nextPageToken properly.
- Verify after swap: sample every Nth draft (163/1136 here), assert the NEW image md5
  present and the OLD one absent, and confirm body text / subject / CC untouched.

## Cold-blast deliverability (SPF / DKIM / DMARC / Postmaster)
Answers to "will these be treated as spam?" must be HONEST (nobody can confirm a cold
blast won't be spammed; placement is driven by per-recipient engagement + complaints):
- Check the actual DNS before answering: SPF include google.com present, MX → Google
  Workspace = good base. `dig` is often NOT installed on the VPS — use DNS-over-HTTPS
  (e.g. https://dns.google/resolve?name=draas.com&type=TXT) via web_extract/curl.
- Missing DMARC + unpublished DKIM selector = the two biggest technical negatives. On
  Google Workspace: publish `v=DMARC1; p=none; rua=mailto:...` first (order matters —
  Postmaster Tools shows nothing until a DMARC record exists), then enable DKIM in Admin
  console (Apps → Google Workspace → Gmail → Authenticate email → Generate new record →
  publish the two CNAMEs → Start authentication), THEN add the domain in Postmaster
  (postmaster.google.com → TXT verification → verify).
- The real risk is the send pattern, not the wording: 1,000+ cold recipients, no history,
  a burst from one mailbox. Recommend 50–100 drafts/day max, morning IST, monitor the
  Postmaster dashboard: reputation (want High/Excellent), spam rate (<0.1%), user-reported
  spam (target 0 — even 2–3 complaints can train the filter), delivery errors, and
  SPF/DKIM/DMARC pass rates (DKIM pass should climb past 95% after CNAMEs propagate).
  Expect 1–3% replies on cold B2B; don't re-blast non-responders, a single re-send to
  non-openers is the ceiling. Postmaster only measures Gmail recipients — a multi-domain
  list (Yahoo/Rediffmail) needs send-test copies to those providers separately.

## Pitfalls
- **Salutation personalization**: substitute per-company from the sheet; shared emails get
  both company names — do this in the loop, never a single generic greeting.
- **CC is per-draft, not a list-level setting** — draft.create has no CC-at-batch option;
  put `Cc` in each MIME header (To is the recipient, Cc is the fixed manager address).
- **Never claim the emails were SENT** — they are drafts; report the count in Drafts and
  that each carries the CC. The user clicks send (batch by batch per the deliverability
  plan).