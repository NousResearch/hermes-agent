# Mass personalized Gmail draft campaigns + cold-blast deliverability (verified 2026-10, HOSTIA 1,136-lead RG campaign)

Class: "draft the same marketing email individually to N leads found in a spreadsheet tab, CC ndr@draas.com, embed an image in the body". The deliverable is N separate Gmail DRAFTS — never a send.

## Input prep
- Read the lead sheet tab (e.g. "Hosur SMEs (HOSTIA)"), extract company name + email.
- Dedupe on the email (lowercased). Some emails appear against 2 companies — show BOTH names in the salutation ("Dear T.K.M ENTERPRISES / Yiyasu Power tech Pvt Ltd").
- Valid unique emails only; report rows with missing emails honestly (1,196 rows → 1,136 unique emails here). 4 rows had no email; 0 malformed.
- Expect the loop to be SLOW: ~1,136 drafts ≈ 25 min at ~40-50/min (one create per second-ish). **Run in background with notify_on_complete.** A 600s foreground timeout kills the loop part-way — the v3 pattern is a RESUME-CAPABLE script: write processed draft-ids to a state file, on restart skip already-done recipients. Re-running is idempotent (fresh draft per recipient each pass if you don't dedupe — prefer the state file or verify by counting drafts with the exact subject AFTER the run, not during).

## Draft construction (draft-only; NEVER send)
- Account = the session user's own (psingh@draas.com → google-draas). `sys.path.insert(0,'/opt/hermes')` then `from tools.gws_auth import build_service`.
- MIME: `MIMEMultipart('alternative')` → attach `MIMEText(plain)`, then `MIMEText(html,'html')`. HTML body carries `<img src="cid:rg_ad">`.
- Inline image: `email.mime.image.MIMEImage(png_bytes)`, `img.add_header('Content-ID','<rg_ad>')`, `img.add_header('Content-Disposition','inline',filename='rg_ad.jpg')`, attach to the root msg (or a `related` part).
- CC: `msg['Cc'] = 'ndr@draas.com'` — every draft CC'd, verified after creation.
- Create: `gmail.users().drafts().create(userId='me', body={'message': {'raw': raw}})` — **the raw must be nested under `'message'`**, NOT at top level (`TypeError` otherwise).
- raw = `base64.urlsafe_b64encode(msg.as_bytes()).decode()`.

## APIs / quirks that bit us
- **`drafts().list` with a large `maxResults` returned estimate≈201 but 0 messages** — pagination quirk. Use `maxResults=100` and paginate on `nextPageToken`. The subject used to FIND drafts matters: the campaign subject CHANGED between generations ("… Rental Income Homes | Phase 1 Units Now Open" v2 vs the v1 subject in the first build script) — match the live subject, not the memory of it.
- **`drafts().get(id=..., format='full')` — `metadataHeaders` kwarg raises TypeError.** Parse `payload.headers` yourself.
- **Updating an existing draft in place** (swap the embedded image, keep the same subject/body): `gmail.users().drafts().update(userId='me', id=<draft_id>, body={'message': {'raw': raw}})` with the new full MIME. It is NOT message-id — draft ids differ from message ids; list drafts to get `draft['id']`.
- **Verification (never assume):** (a) count drafts: search `in:drafts subject:"<exact subject>"` — BUT a subject search returned 0 in one pass while the drafts clearly existed; instead enumerate `drafts().list(maxResults=100)` and count matching `message['subject']` headers. (b) Sample drafts across the range (every Nth) and check the embedded image: `drafts().get(format='full')` → find the attachment part → `attachments().get(userId='me', messageId=<msg_id>, id=<att_id>)` (camelCase `messageId=`) → decode → md5 must equal the source file's md5. Also assert the HTML still references `cid:<name>`. (c) Confirm Cc present on samples.
- The earlier v2 copy cleanup ("remove the ₹ figures from the body") had ALREADY been applied to the active 1,137 drafts; the chat-level sentence the user flagged ("I used your ad's figures…") was NEVER in the drafts. **Verify against the live drafts before claiming a string is/ isn't present — don't trust build-script memory.**

## Cold-blast deliverability — the honest assessment (DNS measured 2026-10)
- Check the domain's real DNS before promising anything: SPF; DKIM selector records (`google._domainkey`); DMARC (`_dmarc.<domain>`). `dig` may be absent — use DNS-over-HTTPS: `curl -s "https://dns.google/resolve?name=_dmarc.draas.com&type=TXT"`, same for MX/SPF.
- draas.com had SPF(+google) and MX→Google but **NO DMARC and NO published DKIM selector**. Postmaster Tools will show NO data until a DMARC record exists — so the ordering is: publish DMARC first, enable DKIM second, THEN Postmaster is meaningful.
- **Do not promise "won't be treated as spam".** Gmail spam placement for a cold blast is driven by per-recipient engagement (opens/replies/spam complaints); zero-history senders have no buffer. 1,136 recipients from ONE mailbox in ONE burst is the risk, not the wording.
- Fix order (10 minutes, biggest technical negatives first):
  1. DMARC TXT at `_dmarc.draas.com`: `v=DMARC1; p=none; rua=mailto:postmaster@draas.com; ruf=mailto:postmaster@draas.com; adkim=s; aspf=s; fo=1` — p=none is observe-only, changes no flow.
  2. Enable DKIM in Google Admin console (Apps→Google Workspace→Gmail→Authenticate email → Generate new record → publish the TWO CNAMEs the console shows → Start authentication). Takes a few hours to show "Authenticating".
  3. Postmaster Tools: postmaster.google.com → add domain → publish the `google-site-verification=...` TXT → Verify. Data lags 24-48h; the dashboard looks empty on day 1.
- **Send discipline:** not 1,136 at once — 50-100/day, morning IST, each To personalized (drafts already are); never a second blast to everyone (re-send to non-openers only, after ~a week); test one copy to Gmail + Outlook + Yahoo first.
- **What to watch week 1:** domain reputation (Low→stop), spam rate (0 comfortable, <0.1% ok, 0.1-0.3% shrink batches, above = stop), user-reported spam (target ZERO; even 2 complaints in batch 1 = replan), delivery errors (4xx transient is normal, 550 at volume = trouble), authentication pass rates (DKIM should climb to ~95%+ once CNAMEs propagate). Postmaster measures GMAIL only — the HOSTIA list is heavy on Yahoo/Rediffmail; those are a separate signal.
- Expect 1-3% replies on cold B2B at best; the win is the reply rate, not delivery.

## Layout / copy notes
- HTML body: greeting personalized per company, offer block, resource list with compressed-Drive links (see whatsapp outbound pattern for the shared-assets approach), signature with RERA no. Keep hard ₹ figures OUT of body copy when the user asked for that; the approved ad IMAGE may carry figures (it's the client's approved creative) — that's the user's call, not the agent's to delete.