# Mass Personalized Gmail Draft Campaigns (spreadsheet → N drafts with inline image + CC)

Proven 2026-10: HOSTIA Hosur-SMEs campaign for Ranka Udaya RG — 1,196-row sheet → 1,136 unique
valid emails → 1,136 INDIVIDUAL Gmail drafts, per-company salutation, RG insert ad embedded inline
(cid:), CC ndr@draas.com on every draft. Drafts only — never send (email-sending hard rule).

## Recipient list prep
- Read the named sheet tab (e.g. "Hosur SMEs (HOSTIA)"). Rows can be 1,000+; extract to a working
  JSON locally first: email → company name(s). Rows with no email → drop and count. An email shared
  by 2 companies → show BOTH names in the salutation.
- Dedupe by normalized email. Unique valid count is the number of drafts to expect.

## Body build
- One HTML template; per-recipient personalization = `<p>Dear <Company Name>,</p>` at the top.
- Embed a single low-res image inline: render the approved creative page to JPEG (~120-150 KB,
  ~580-820 px) and reference in HTML as `<img src="cid:rg_ad">`; attach with
  `msg.get_payload()[1].add_header('Content-ID', '<rg_ad>')` and
  `Content-Disposition: inline; name=...`.
- CC: add `msg['Cc'] = 'ndr@draas.com'` (or the user's requested CC) on EVERY draft.
- API shape (Gmail API drafts().create): the raw message goes NESTED — `body={'message': {'raw': raw}}`
  — NOT `body={'raw': raw}`. The flat form fails. base64.urlsafe_b64encode the EmailMessage.

## Create loop (1,000+ drafts)
- `drafts().create(userId='me', body=...)` per recipient; ~50 drafts per 30 s → an 1,100-draft run
  takes ~10-11 min. Run with terminal(background=true, notify_on_complete=true); do NOT block a
  600 s foreground.
- Track progress by counting Gmail drafts with the exact subject (list then tally) — never trust the
  loop's own last line (a post-loop log-write crash can look like failure while all drafts landed).

## Editing ALL drafts afterwards (image swap / copy change)
- Each draft needs get + update → ~40 drafts/min; a 1,100-draft update runs ~25 min. NEVER run this
  in a 600 s foreground; background + resume-capable.
- **Scope discipline (user correction, 2026-10-09):** when the user says "remove THIS part from
  the email content", remove exactly the named text — do NOT also swap, re-render, or touch the
  embedded image unless they ask. The user had to explicitly add "Dont change any contents of the
  image" after a text-removal request that also contemplated the image. If the named sentence is
  chat-text only (never part of the drafts), say so plainly and prove it by reading one draft
  back from Gmail — do not silently rebuild the whole campaign on a text-only correction.
- Resume pattern: for each draft, fetch full message, compare md5 of the embedded attachment payload
  to the target image's md5; skip if already swapped. Re-run is idempotent and cheap to re-run.
- Caution: the md5 pre-check can miss (attachment fetch quirk) → a re-run may re-swap everything;
  that is fine (idempotent), just slower. Verify at the end by sampling — e.g. every 7th draft —
  and checking md5/cid of the embedded image.
- drafts.list pagination quirk: `maxResults=500` can return 0 messages on some calls while a
  `maxResults=100` loop pages correctly; use 100 and page through.

## Deliverability — honest assessment (before any send)
- Check the sender domain's real DNS first: SPF (TXT at root), MX, DKIM (CNAME google._domainkey),
  DMARC (_dmarc). `dig` often absent → use DNS-over-HTTPS (https://dns.google/resolve?name=...).
- A cold blast pattern (many recipients, single mailbox, one burst, promotional urgency copy) is
  decided by recipient engagement + complaint rate; NOBODY can promise "not spam". Say so.
- Order of fixes: publish DMARC p=none + rua first (Postmaster shows NOTHING until DMARC exists),
  then enable DKIM signing in Google Admin console (generate CNAMEs there — never hand-write),
  then verify the domain in Google Postmaster Tools with the google-site-verification TXT.
- Sending cadence: 50-100 drafts/day morning IST beats one 1,100 burst; send a test copy to Gmail +
  Outlook + Yahoo first; expect 1-3% replies on cold B2B; re-send only to non-openers.
- Postmaster data lags 24-48 h; monitor reputation (want High/Excellent), spam rate (<0.1% comfy),
  user-reported spam (aim 0), auth pass rates (adkim=s strict → DKIM must climb to ~95%+).
- Postmaster measures Gmail recipients only — fine given a list heavy in Yahoo/Rediffmail uses the
  reply rate as its own signal.