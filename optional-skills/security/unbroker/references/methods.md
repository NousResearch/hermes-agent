# Opt-out method playbooks

How the agent executes each broker `optout.method` using native Hermes tools. Always obey the
**verify-before-disclose** rule: confirm a real listing exists, then submit only the fields the broker
requires (`pdd.py plan` lists them per broker).

**Autonomy:** `pdd.py next <subject>` sequences all of this - it decides which method applies, orders
parents first, and routes human-only work to the digest. In `autonomy=full` (default), execute its
actions without pausing per submission; the consent recorded at intake is the authorization. These
playbooks are the HOW for each action type.

## Scan ladder (all methods)

Confirm exposure before acting, cheapest first. Run **every** `search_vectors` entry from `pdd.py
plan` (each name x location, phone, email, and address the broker's `search.by` supports) - different
vectors surface different listings for the same person; dedupe found URLs.

1. `web_extract` on the broker `search.url` (fast HTML -> markdown). Look for `search.match_signal`.
   Build per-vector URLs from `search.url_patterns` and heed `search.url_format_quirks` (see below).
1b. **`site:` search-engine probe (cheap, do it early and in parallel).** `web_search` with
   `site:<broker-domain> "First Last"` (add a city/ZIP or a unique phone/address to cut namesake
   noise) often returns the **exact profile-slug URL** in one shot - which both confirms the listing
   exists AND hands you the opaque `/find/person/<id>` or `/p/<slug>` URL you'd otherwise have to
   derive. Two big wins seen in the field: (a) it disambiguates namesakes fast - the SERP snippet
   shows age/city so you can tell the subject from a same-name relative before fetching anything; and
   (b) a broad `"First Last" <ZIP OR unique-address>` search (no `site:`) surfaces **brokers not yet in
   your DB** (e.g. information.com, peoplefinders.com) - record those as bonus exposures. Note: empty
   `site:` results are INCONCLUSIVE (many broker pages aren't indexed / are `noindex`), not `not_found`.
2. If the page is JS-rendered or returns nothing useful, `browser_navigate` + `browser_snapshot`
   (and `browser_type`/`browser_click` to run the site's search box).
3. If blocked by stealth/Cloudflare, use the `scrapling` skill via `terminal`. **If the broker record
   has `search.antibot` set (e.g. `datadome`), results are behind a device-check CAPTCHA**: a
   cloud/stealth browser (Browserbase) or `scrapling` may get through; if none is available, do **not**
   burn attempts - `pdd.py record <subject> <broker> blocked` and move on (a re-scan with a stealth
   backend can pick it up later).
3b. **Operator-browser path (the reliable unblock for anti-bot sites).** Cloudflare/DataDome key on
   datacenter IPs + headless fingerprints, so `web_extract`, the proxyless agent browser, and even a
   cloud browser often fail - but the **operator's own everyday browser (residential IP, real
   fingerprint) sails straight through**. For any `blocked` site, hand the operator a paste-ready
   search URL (built from `search.url_patterns`), give them the identity anchors to judge by (current
   + prior addresses, age, a distinguishing detail) and the namesake/relative watch-list, and ask for
   the verdict or a screenshot (the agent can read screenshots). This is a **first-class scan path, not
   a fallback** - treat the operator's live check as authoritative and record the real verdict
   (`found` / `not_found` / `indirect_exposure`), citing `scanned_via: operator_browser`. Same for
   opt-out forms the agent's browser can't reach: guide the operator field-by-field (least-disclosure),
   pausing before submit. (This is exactly why the same trick clears email-verification links the agent
   can't open - see the Verification loop.)
4. Capture evidence: save listing URLs and a `browser` screenshot into the subject's `evidence/` dir,
   then `pdd.py record <subject> <broker> found --found true --evidence '{"listing_urls":[...]}'`.

If a listing genuinely does not exist: `pdd.py record <subject> <broker> not_found` and move on.

### A 404 (or empty body) is INCONCLUSIVE, not "not_found"

A constructed search URL that 404s almost always means the **URL pattern is wrong**, not that the
person is absent. Never record `not_found` off a 404. Instead:
  1. Re-check the broker's `search.url_patterns` / `url_format_quirks` and rebuild the URL.
  2. Fall back to the **on-site search box**: `browser_navigate` to the search page, `browser_type`
     the raw query, `browser_click` Search, then read the **canonical result URL** the site lands on.
  3. Only after the site's own search returns an empty result set do you record `not_found`.
  4. If a pattern was wrong, fix it in `references/brokers/<id>.json` (`url_patterns` +
     `url_format_quirks`) so the next run is correct - see the rule below.

### Log URL/format quirks for every site you scrape

Whenever you discover how a broker's URLs are actually shaped (path layout, hyphen-vs-slash joins,
whether ZIP is required, abbreviation handling, query-param search, anti-bot gating), record it in
that broker's `references/brokers/<id>.json` under `search.url_patterns` (the templates) and
`search.url_format_quirks` (the gotchas, including which forms 404). Bump `last_verified`. This makes
the deterministic URL path reliable across runs and subjects instead of rediscovered each time. If the
opt-out form's real requirements differ from the record (extra required fields, a CAPTCHA, an account),
fix `optout.requires` / `optout.inputs` / `optout.tier` too - those drive tier selection and
least-disclosure. Log opt-out mechanics gotchas (a broker that needs a profile URL but doesn't expose
one for the subject, an email-only fallback, an authorized-agent toggle) in `optout.quirks` - the
planner surfaces these as `optout_quirks` per broker. Example: Radaris sometimes shows the subject only
as a static address-table row with no "View Profile" link, so `/control-privacy` (which needs a profile
URL) can't be used - fall back to `optout.email` rather than submitting a namesake's URL.

### Distinguish the subject from namesakes and relatives

People-search sites are dense with namesakes and family clusters. Before recording `found`, confirm the
record is the **subject themselves** (corroborate via DOB, a known current/prior address, or the
identifier you searched). Two non-removable patterns to record as evidence but NOT as the subject's own
listing:
  - **Namesake:** same name, different person (different DOB/location with no overlap). Not the subject.
  - **Relative record:** the listing is about a *different* person (a relative) and merely *names* the
    subject in a "Family" field, or carries the subject's email/phone as a secondary datum. This is a
    third party's record - the consent gate correctly blocks acting on it. See "Indirect exposure" in
    the web_form section for what the subject *can* still request.

## web_form

1. `browser_navigate` to `optout.url`; `browser_snapshot` to read the form.
2. Fill only the planned `disclosure_fields` with `browser_type`/`browser_click`; for `profile_url`,
   paste the confirmed listing URL from evidence.
3. Submit; `browser_snapshot` to confirm the success state; screenshot to `evidence/`.
4. `pdd.py record <subject> <broker> submitted --disclosed <field> --disclosed <field> --channel web_form`.
5. If the broker requires email verification, follow **Verification loop** below.

### Indirect exposure (named as a relative / your email on someone else's record)

You asked the right question: if a broker lists a *relative* and names you in their "Family" field, or
shows **your** email/phone on **their** record, that IS personal information about you - even though the
record's primary subject is a third party. Resolve it in two distinct lanes:

- **The self-service opt-out form does NOT cover this.** That form removes a record whose *primary
  subject* is you. It has no notion of "scrub my identifiers from this other person's record," and
  submitting it with the relative's address to force a match would be (a) disclosing data the listing
  doesn't tie to you and (b) acting on a third party's record. Don't. The consent gate exists to stop
  exactly that.
- **What you CAN do - a targeted "delete my personal information" request (CCPA 1798.105 / GDPR Art.17).**
  These rights attach to *your* personal information *wherever the business holds it*, including as a
  data point on another person's profile. So the subject may email the broker's privacy address and
  request suppression of **their own specific identifiers** (this email address, this phone number, my
  name in family/relative associations), citing the relative listings as the locations. This is a
  narrower request than a full opt-out and does not require the relative's consent - you are only asking
  them to delete data about *you*. Use `render-email` with the `ccpa`/`gdpr` template, list only the
  subject's own identifiers + the URLs where they appear, and record it as a normal `submitted` →
  `awaiting_processing` email case. Verify by re-scanning those identifier vectors (email/phone) after
  the statutory window - `confirmed_removed` only when the subject's identifier no longer appears.
- **Caveat:** the broker may decline to alter a third party's record beyond removing your specific
  identifiers, and "your name in a family graph" can be derived from public records they'll re-list.
  Note residual exposure in the report rather than marking a clean removal. (Operational guidance, not
  legal advice.)

## email

`pdd.py send-email <subject> <broker> --listing <url> [--kind ccpa|gdpr|ccpa_indirect]` always does
the deterministic parts (recipient locked to an address the broker record declares, refusing anything
else; `--listing` mandatory; records `submitted`, logs disclosure, stamps `next_recheck_at`). How it
actually sends depends on `email_mode`:

1. **browser mode (no password, autonomous):** the command returns a recipient-locked `compose`
   payload (`to`/`subject`/`body`). Compose a NEW message in the operator's **logged-in webmail** via
   `browser_*` (paste `compose.body` exactly, disclosing nothing beyond it) and send. No credentials
   stored. Requires the inbox signed in in the browser Hermes uses.
2. **programmatic mode (SMTP creds):** the command SMTP-sends it directly, no human.
3. **draft_only fallback:** `pdd.py render-email <subject> <broker> --listing <url>`; a digest entry
   tells the operator to send it, and the agent records `submitted --channel email` afterward.

Then follow the **Verification loop** if the broker emails a confirmation link.

## Verification loop (email_verification brokers)

- **browser mode (autonomous, no password):** open the broker's confirmation email in the operator's
  webmail (`browser_*`), then `pdd.py verify-link <subject> <broker> --text '<email body>'` returns
  the anti-phishing-scored link. `browser_navigate` it **in the same browser** (several brokers, e.g.
  PeopleConnect, bind the session to the browser that opens the link), finish the flow, record
  `awaiting_processing`.
- **programmatic mode (IMAP):** `pdd.py poll-verification <subject>` polls IMAP for every in-flight
  case, extracts the link (anti-phishing scored: only opt-out-looking links on the broker's own
  domains), and auto-advances `submitted → verification_pending`. Then `browser_navigate` the link in
  the agent's own browser, finish the flow, record `awaiting_processing`.
- **draft_only:** the digest tells the operator to click the link in the subject's inbox; the agent
  records `awaiting_processing` on their word.
- Either way, the due queue (`pdd.py due`) brings the case back after the broker's processing window
  for the verifying re-scan; only that re-scan justifies `confirmed_removed`.

## phone_callback (e.g. Whitepages)

Submit the web form, then the site places an automated call with a numeric code. If the operator is
available to read the code, capture it and complete the form (T2). Otherwise queue a human task.

## phone (voice menu) / fax / mail / gov_id  ->  human task (T3)

Do **not** attempt to automate. Create a `todo` task and `pdd.py record <subject> <broker>
human_task_queued` with exact instructions and an explicit **withhold** list (never SSN; never a
driver's-license number unless the subject chooses to and crosses out the ID number). Capture the
confirmation reference back into the ledger when the operator completes it.

## captcha

**Default: soft/managed CAPTCHAs clear automatically.** The recommended baseline backend is the
Browserbase cloud browser (`setup --auto` selects it when `BROWSERBASE_API_KEY` is set). Being a
real browser on a residential IP, it passes managed challenges - Cloudflare Turnstile, hCaptcha /
reCAPTCHA checkbox - as normal operation, so those brokers stay T1 and you just proceed. This is
**not** CAPTCHA solving: no solver service, no fingerprint spoofing.

Only a **hard** challenge the browser genuinely can't pass (interactive image grids, behavioral
scoring that flags the session) becomes a fallback: `record ... blocked` and requeue it for the
stealth/operator-browser pass (`methods.md` → scan ladder 3b - the operator's own residential
browser is the reliable unblock). Without a cloud browser configured, soft-CAPTCHA brokers drop to
T2 and become human tasks. **Never use a third-party CAPTCHA-defeating service.**

## Ownership clusters - DO PARENTS FIRST (playbooks live in the broker records)

Many brokers are resold shells of a few parents, so **one parent removal clears a whole cluster of
children** (see `owns` in each record). In Phase 2 you MUST work the cluster **parents first**, then
the standalone listings - doing a child before its parent wastes a submission the parent would have
covered. `pdd.py plan <subject> --batch` **orders the `found` group parents-first** and emits a
`parent_playbook` whose `steps` come verbatim from each record's **`optout.playbook`** - the single
source of truth, field-verified, updated as live runs discover mechanics. What follows is the
operating doctrine; the exact steps are in `references/brokers/<id>.json`.

**Deletion beats suppression, email lanes beat forms.** Each parent record carries a structured
`optout.deletion` lane (`via: in_flow | email | email_followup`, plus the privacy address). The
autopilot routes accordingly:

- **`in_flow`** (PeopleConnect): the deletion control lives INSIDE the web flow - complete the flow
  and use the **Right to Delete / DELETE MY USER DATA** control, never just the suppression step.
- **`via: email`** (Whitepages): the fully-autonomous lane - `send-email` the request (residency-picked
  kind: CCPA for US-CA, GDPR for EU/UK, generic otherwise), then `poll-verification` for their reply
  and answer identity questions with least-disclosure. This is also the **rescue lane**: any broker
  whose form demands a phone-callback/gov-ID/account but that declares a deletion email gets routed
  here instead of the human digest.
- **`email_followup`** (BeenVerified, Spokeo): the opt-out form is the fast primary (it clears the
  listing), and the playbook then sends a right-to-delete email for full erasure beyond suppression.

Verified parent facts (live-checked 2026-07-01; details + steps in the records):

- **Intelius/PeopleConnect** (~15+ sites in one flow): portal entry asks only email + consent →
  verify link is **session-bound to the browser that opens it** → guided-mode → **DELETE MY USER
  DATA** at the bottom (suppression alone re-lists). Fallback `privacy@peopleconnect.us` - their own
  published metrics: 33.5k deletion requests, median response < 1 day.
- **Whitepages**: `privacyrequest@whitepages.com` (or the Zendesk form) handles removal + CCPA
  deletion **without the phone-callback tool** - that phone call is only required by the automated
  tool. One removal also drops "all known connected listings". ≤15 days; check 411.com + Premium.
- **BeenVerified**: opt-out tool (footer "Do Not Sell" link → `/svc/optout/search/optouts`) + email
  verification; one opt-out per email address. Then `privacy@beenverified.com` deletion follow-up -
  controller is The Lifetime Value Co., so name their sister properties (NeighborWho, Ownerly,
  NumberGuru, Bumper) in the same request, and verify each separately.
- **Spokeo**: form takes ONE listing URL at a time and **each listing must be opted out
  individually** - collect every listing URL from all search vectors first, then submit one opt-out
  per URL. 24-48h processing. `privacy@spokeo.com` for full deletion beyond free-search suppression.

After each parent removal is confirmed, **re-scan its children** before submitting anything for them -
usually they drop out and need no separate opt-out.

### Any other parent
A parent without a hand-verified `optout.playbook` gets synthesised steps from its structured record
(URL/email, `requires` flags, deletion lane, notes/quirks). Follow those, and **write what you learn
back into `references/brokers/<id>.json`** (`optout.playbook`, `optout.deletion`, `quirks`,
`last_verified`) so the next run is exact - that file, not this one, is where per-broker knowledge
accrues.

