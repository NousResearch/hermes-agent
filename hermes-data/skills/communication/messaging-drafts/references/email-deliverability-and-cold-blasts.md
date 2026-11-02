# Email deliverability for cold B2B campaigns — draas.com (2026-10)

Used when the user asks "will these emails be treated as SPAM", "set up Postmaster",
"how do I send this cold list safely". Applies to bulk personalized Gmail draft
campaigns (e.g. Hosur SMEs HOSTIA RG outreach — 1,136 individually-drafted emails).

## Check the user's DNS first — order matters
- Confirm SPF + MX exist (Gmail Workspace). Check via DNS-over-HTTPS
  (`dig` is not installed on the box — use curl against dns.google/resolve,
  or `python -c` with a simple A/TXT lookup helper).
- draas.com baseline seen: SPF includes google.com, MX → Google Workspace,
  NO DMARC published, NO google DKIM selector published. Fix those before volume.

## Correct setup order
1. Publish DMARC FIRST (Postmaster shows nothing until DMARC exists):
   `v=DMARC1; p=none; rua=mailto:postmaster@draas.com; ruf=mailto:postmaster@draas.com; adkim=s; aspf=s; fo=1` at `_dmarc.draas.com`. p=none = observe only, safe.
2. Enable DKIM in Google Admin console (admin.google.com → Apps → Google Workspace →
   Gmail → Authenticate email): generate the CNAME records Google shows (google._domainkey
   etc.), publish them verbatim, start authentication. Takes hours to roll out.
3. Postmaster Tools: postmaster.google.com, add domain, publish the
   google-site-verification TXT it gives, verify. Use a real admin/ops account.
4. Only then start the campaign in batches — small until auth pass rates climb.

## What to watch in the first week (Postmaster lags 24-48h)
- Reputation circle: Excellent/High good; Low/Poor → stop and diagnose.
- Spam rate: 0 is comfortable; <0.1% fine; 0.1–0.3% edge → shrink batches;
  above 0.3% → classified as spam at scale, stop and reset warmer.
- User-reported spam (complaints): aim exactly zero; even 1–2 complaints on a
  cold first batch is a stop-and-replan signal.
- Delivery errors: temporary 4xx normal at first; permanent 550 at meaningful
  share = IP/domain rep trouble.
- Authentication: SPF/DKIM/DMARC pass rates → DKIM should climb to 95%+ after
  CNAME propagation; with adkim=s, DMARC pass follows once DKIM aligned.
- Postmaster only measures Gmail recipients — Yahoo/Rediffmail (common on Indian
  SME lists) are invisible to it; treat their replies as a separate signal.

## Sending cadence that protects reputation
- Never send 1,100+ cold drafts in one burst. 50–100 per day, morning IST.
- Each draft must have its own personalized To (drafts API already builds this);
  never mass BCC from one message.
- Expect 1–3% reply rate on cold B2B; chase non-openers once after a week, never
  re-send a second blast to everyone (complaint magnet).
- Send one test copy to a Gmail, an Outlook, and a Yahoo address first; check the
  spam folder before the first real batch.
- Google Workspace sending shares the pool IP reputation — watch domain rep, not IP.

## Cold-list drafting rules that matter
- Cold recipients never opted in: complaints, not wording, decide spam placement.
- Urgency copy ("limited units", "act now") is promotional-weight; keep it minimal.
- Never put price/figure claims into the body that you can't verify against the
  approved creative — if the user says remove figures, strip body numbers but ask
  before altering an approved image (the image may legally carry the figures).

## Technical deliveries from the batch-drafting session
- Drafts.create needs the message nested: `body={'message': {'raw': base64url_mime}}`
  (a flat `raw=` at the top level 404s or silently no-ops).
- Inline image in HTML body: multipart/related with the image part and
  `Content-ID: <cid>` + `src="cid:cid_name"`; the file upload to the drafts API
  with the image inside the MIME works but mime/draft get loops can be slow
  (~40 drafts/min); run in background and resume by md5-checking the embedded
  attachment, not by assuming a count.
- Mass drafts land in the sender's own account (psingh@draas.com here); verify
  count by listing drafts with the campaign subject, not by trusting the script's
  completion. drafts.list pagination can return estimates (e.g. "~201") with 0
  messages on a mis-parameterized call — search by subject instead.