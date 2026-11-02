# Mass individualized email drafts (N drafts from a sheet — HOSTIA order, verified 2026-10-07)

Distinct mode from the one-draft BCC blast in `html-body-email-drafting.md`.
The user shares a spreadsheet of companies/industries (e.g. "Hosur SMEs (HOSTIA)"
tab, ~1,196 rows of factories/outlets in the Bagalur-Hosur belt) and says
*"prepare a draft email for the email IDs mentioned… the email draft is to be
sent individually and add CC to ndr@draas.com in each email."* → build ONE draft
PER recipient, personalized, NOT a BCC blast.

## Recipe

1. **Read the sheet** with the session user's own token (`build_service("sheets",
   "v4", service_name="google-draas")`, `spreadsheets().values().get(..., range="A1:Z2000")`).
   Harvest emails + company name per row. Normalize keys to `.lower()`.
   Dedupe by lowercased email; keep company for the salutation. Rows with no email
   are skipped but REPORT the drop count (4 rows had none in the HOSTIA run).
   One email shared by two firms → both names in the salutation
   ("Dear T.K.M ENTERPRISES / Yiyasu Power tech Pvt Ltd").
2. **Build one EmailMessage per recipient** — `To:` the recipient, `Cc:`
   `ndr@draas.com` (hard rule from the user), Subject constant, HTML body with
   `{SALUTATION}` = company name, inline CID image attached via
   `msg.get_payload()[1].add_related(img, maintype="image", subtype="jpeg", cid="...")`
   (multipart/alternative: plain fallback + HTML).
   Loop `gmail.users().drafts().create(userId='me', body={'message': {'raw': raw}})`.
   1,136 drafts ≈ 8–10 minutes API time (about 50 drafts / 40 s). Run as a
   background process with `notify_on_complete=True`; track progress in a log file
   (write the DONE line INSIDE the `with open(...)` block — an outside write after
   the block closes raises `ValueError: I/O operation on closed file` which is
   cosmetic but looks like a crash).
3. **MASS UPDATE in place when the copy changes** — `drafts().update(draftId,
   body={'message': {'raw': raw}})` preserves draft IDs; do NOT delete+recreate.
   List all drafts paginated (`maxResults=500`, nextPageToken loop), filter by
   old subject, map `To` → company from the saved sheet, rebuild each raw, update.
   Same ~1,136 / 8–10 min.
4. **Verify ground truth against the API, not the script's log.** List all drafts
   and, per draft (`format='full'`), read the Subject header, the Cc header, the
   snippet, and walk payload parts for the image Content-ID. Assert: N drafts with
   the NEW subject, 0 with the old, N Cc'd to ndr@draas.com, 0 snippets matching a
   banned-figure regex, N with the CID image. A write whose log crashed may still
   have completed the loop — API readback is the only truthful count.

## Swapping the embedded image in ALL drafts (verified 2026-10-07, 1,136 drafts)

User said "add RG model image in the body, remove the existing one, previous image
was better, update all the drafts." Same loop as the mass update — rebuild each raw
with the new image + new `cid:` and `drafts().update(...)` in place. Hard-won API
facts:

- **Draft IDs ≠ message IDs.** `messages().list(q='in:drafts')` returns MESSAGE ids
  (`1a1163...`) but `messages().get(id=<that>)` 404s — drafts are only addressable
  through `drafts().get(userId='me', id=<draft_id>)` where the id comes from the
  `drafts().list(...)` resource (`r-8760145070967265287` style). Don't mix them up.
- **`drafts().get()` rejects `metadataHeaders=`** (TypeError — that kwarg is for
  `messages().get()`). Use `format='full'` or `format='raw'`.
- **`drafts().list()` with `maxResults=500` can return `[]` with a nextPageToken
  while the same call at `maxResults=100–200` returns the drafts.** Estimate may
  also disagree with yield (resultSizeEstimate 201 but 1,137 actual). Loop with a
  smaller page size and don't trust the estimate.
- **Resume, don't restart, long update loops.** A foreground run was killed at
  700/1,136 by the tool timeout (600 s). Re-run is idempotent: maintain a progress
  file of completed draft ids; at start, for every draft compare the CURRENT
  embedded image md5 (`messages().attachments().get()` on the image part) against
  the target image's local md5 and skip matches. Even if the md5 pre-check fails
  (attachment fetch quirks), re-running the whole update loop is safe — `drafts()
  .update()` replaces the raw wholesale.
- **Verify by sampling, not by count alone:** pick random drafts, decode raw via
  `email.message_from_bytes(..., policy=policy.default)`, walk parts, confirm the
  image part Content-ID and `hashlib.md5(data)` match the local file, and the
  HTML's `<img src="cid:...">` matches. Then a wider every-Nth sample (every 7th
  of 1,137 = 163 checked) proves 100% swapped, 0 left on the old image.

## Gmail API gotchas (both hit in the HOSTIA run)

- **`drafts().get()` does NOT accept `metadataHeaders=`** — that kwarg exists only
  on `messages().get()`. Passing it raises
  `TypeError: Got an unexpected keyword argument metadataHeaders`.
  Use `format='full'` on drafts.
- **`drafts().list(q='subject:"..."')` returns 0 even when drafts exist.**
  Draft subject search via `q` does not work the way message search does. List all
  drafts (paginated) and filter subjects client-side.

## No-PRICE-FIGURES rule for cold marketing emails (user correction, 2026-10-07)

Same family as the `channel-partner-empanelment` NO-PRICE-REFERENCES rule: do NOT
quote offer figures in the cold email — prices/terms change per management
decision. In the Ranka Udaya RG email the user ordered removal of
"2.12 Cr total, 8 x 1-BHK, Rs 60,000/month for 5 years, +5% escalation".

- Body says the model exists in plain terms ("guaranteed monthly rental income
  for a fixed term from completion") and pushes the numbers to a conversation:
  *"We will share the full rental model and the exact numbers with you on a call."*
- **The image is a separate decision from the body copy (user correction 2026-10-07).**
  The NO-PRICE rule governs BODY TEXT only. The approved creative image — the RG
  insert ad with "2.12 Cr | 8 x 1-BHK | ₹60,000/month | 5 YEARS" — is the user's
  PREFERRED email image. When a v2 run swapped it for a non-priced layout visual,
  he ordered it back ("add RG model image … previous image was better") and
  explicitly said "Dont change any contents of the image." So: never crop prices
  out of, or auto-replace, an approved creative unless the user asks. If asked for
  a layout/master-plan alternative, offer it as an option, don't impose it.
- Soften subject/trigger words: "Guaranteed / Limited / Act now" → factual
  subject ("Ranka Udaya, Hosur - Rental Income Homes | Phase 1 Units Now Open").
- Keep a polite footer: sender org, cities, "reply unsubscribe and we will not
  write again."

## Spam / deliverability assessment (answering "will these be treated as SPAM?")

Never guarantee inbox placement — give the honest factors and a verdict.

- **Check domain auth via DNS-over-HTTPS when `dig`/`host` are absent:**
  ```python
  import urllib.request, json
  def q(name, typ="TXT"):
      with urllib.request.urlopen(
          f"https://dns.google/resolve?name={name}&type={typ}", timeout=10) as r:
          d = json.load(r)
      return [a.get('data') for a in d.get('Answer', [])]
  q("draas.com")                    # SPF   — expect include:google.com
  q("_dmarc.draas.com")             # DMARC — empty = none published
  q("google._domainkey.draas.com")  # DKIM  — also probe s1/d1/k1/dkim selectors
  q("draas.com", "MX")              # MX — Google Workspace
  ```
- **Interpretation (HOSTIA facts):** SPF exists and includes Google (mail sent
  from Gmail passes SPF). DKIM selectors ALL empty and NO DMARC record = weaker
  trust at Outlook/Yahoo/corporate filters; Gmail→Gmail usually still lands.
  Recommend the admin add DMARC (`p=none`, then tighten) as the cheap fix.
- **Volume:** 1,136 cold emails in one day is the classic filtered pattern AND
  hits Gmail daily caps. Recommend batches of 100–150/day over 8–10 days and
  watching bounce rate. Never send 1,136 from one account in a day.
- **List quality:** cold emails to a scraped/purchased business list will sit
  unread or junk for many recipients no matter the content. Response rate, not
  delivery rate, is the metric. "Individual To + company-name salutation" is the
  single biggest anti-spam factor; low image ratio, no attachments, plain-text
  fallback, and opt-out footer help the rest.
- Drafts are never auto-sent — the user reviews and sends in batches. State that.

## Google Postmaster Tools setup + first-week watch (answered 2026-10-07 for draas.com)

Order matters: **publish DMARC FIRST, then set up Postmaster.** Postmaster shows
spam/complaint data only once a DMARC record exists.

1. **DMARC TXT** at `_dmarc.draas.com` (or host `_dmarc`):
   `v=DMARC1; p=none; rua=mailto:postmaster@draas.com; ruf=mailto:postmaster@draas.com; adkim=s; aspf=s; fo=1`
   — `p=none` = observe only, changes no mail flow.
2. **DKIM** in Google Admin console (admin.google.com → Apps → Google Workspace →
   Gmail → Authenticate email): generate record, publish the two CNAMEs it shows
   (the console generates values — never hand-write them), Start authentication.
   Takes effect in a few hours.
3. **Postmaster** (postmaster.google.com): sign in with an admin/shared account
   that owns the dashboard, Add domain, publish the `google-site-verification`
   TXT it gives, Verify. Re-click later if it doesn't resolve immediately.
4. **Watch in week 1** (data lags 24–48 h, so day 1 looks empty — normal):
   - Reputation circle: want Excellent/High; Low/Poor = stop batches immediately.
   - Spam rate: 0 is the only comfortable goal; <0.1% fine, 0.1–0.3% shrink batch,
     above = reset with warmer/smaller sends.
   - User-reported spam: aim for exactly zero; even 1–2 complaints in the first
     batch is a stop-and-replan signal.
   - Delivery errors: 4xx normal early, meaningful 550s = reputation problem.
   - Authentication tab: SPF/DKIM/DMARC pass rates should climb toward 95%+ after
     DKIM propagates (adkim=s demands strict alignment).
   - LIMITATION: Postmaster measures Gmail recipients only — a HOSTIA-style list
     has real Yahoo/Rediffmail share Postmaster never sees; treat their responses
     as a separate signal. Sent from Google Workspace, IP reputation partly
     reflects Google's shared pool — trust the domain panel over the IP panel.