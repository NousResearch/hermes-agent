# Forensic Email-Thread Lookup — "who was marked on that email?"

Class of request: NDR (or a stakeholder) refers back to a SPECIFIC past email from a
partial/fuzzy/voice description and wants the recipient list (To/Cc) — or wants to know
whether a particular person/group was marked on it. Often phrased as:
"we sent an email to Sarthak about the ₹25L refund from [some collective] account — who all
were marked on it?", "did that same email also go to Roshni and Eshwari for ITR?"

## Why this is a distinct workflow

Finding a *specific* past email is different from triage (scan the last N hours) or drafting.
You are reconstructing an email you may not have sent from memory, from a voice-memo
description whose proper nouns and amounts are frequently mis-transcribed. The danger is
locking onto the verbatim spoken words and searching for something that never existed.

## Step 1 — Do NOT trust the verbatim transcription of proper nouns / amounts

Voice messages butcher entity names, group names, person names, and figures. Examples seen:
- User said "**Reformers Collective**" / "regsorformers collective" → the real entity is
  "**Red Soul Farmers Collective**" (also appears as "Redsol Farmers Collective", "Red Sol
  Farmers Collective" in the threads).
- The amount spoken as "25 lakh" was "₹25,00,000" — fine — but still try both digit and
  word forms ("25 lakh", "2500000", "₹25,00,000").

Treat every spoken name/amount as a *hypothesis*, not a search term. Resolve it against what
the emails actually contain, then correct the user's naming in your reply.

## Step 1b — The user's name for the sender is often a DIFFERENT PERSON (entity-collision trap)

Distinct from phonetic mangling (Step 1): sometimes the spoken name resolves cleanly to a real
contact who is *in the story but is not the sender you need*. Verified 2026-09-19 (Millers Road
structural-testing hunt): NDR said *"the report is in the email sent by **Amit**"*.

- The real conduit was **Aamir Khan** `<khan.hussain.aamir@gmail.com>` — an unrelated person whose
  name sounds adjacent, and who is a recurring DRAAS counterparty (Terra Greens, Riverstone Farm).
- `contact_resolver("Amit")` returned 33 matches (Amit Agarwal / Amit Bajaj / Amit (Dabba) / …) —
  **every one of them wrong.** Contact-sheet resolution cannot solve this; the correct hit was not
  in the contacts sheet at all at the time.

**Rule: when the named sender produces a *plausible* contact, still confirm by locating the THREAD,
not the person.** Search subject/domain terms from the *content* the user described (the project,
the counterparty firm, the document type), then read the actual `From:` header. Only then name the
sender back to the user. If you had drafted to any of the 33 "Amit" contacts you would have mailed
a stranger.

## Step 1c — Two vendors in one deal can have near-identical names (surface)

Same session: NDR said *"Nova Space is something… about this entire structural testing"*.

- **Nova Spaces** (`zaak@novaspaces.in`, Zakir Kalami) = the *façade / asset-upgradation proposal*
  vendor for Millers Road. Sent "DRA Asset Upgradation Proposal", 10 Aug.
- **Nordstar Novaspaces** (Kiran Kumar) = the *structural testing* vendor, to whom NDR's revised
  test-scope email went on 31 Aug.

These are **two different firms with two different scopes**, and the user's spoken description
blended them. Do not merge them into one thread; and do not conclude "the report is here" because a
similarly-named proposal exists. Report both, labelled by scope.

## Step 2c — "the email I sent to X" usually means X was CC'd, not To'd

Verified 2026-09-19 (Regalia/Iris inventory hunt). NDR asked: *"I sent some email to Prakash,
my colleague, about giving him the entire inventory detail about Regalia, Iris and others…
for sale and for rental."* He described it as an email **to** Prakash.

In fact, of the emails found, **neither was addressed to Prakash** — he was `Cc`'d on both:
- "Century Regalia (9th Floor) & Ranka Iris — Available Units for Sale" — `To:` an *external*
  party (Rajesh Talreja), `Cc:` Prakash Singh + Bharat Hawaldar.
- "Gandhinagar — Warehouse / Dark Store Space" — `To:` Shadowfax (and separately Blinkit/Zepto/
  BigBasket), `Cc:` Prakash Singh.

The colleague was the *coordinator named in the body* ("my colleague Prakash Singh (CC'd
above) will get in touch") — which is why NDR remembers it as "sent to Prakash."

**Consequences — do all three:**
1. **Search Cc, not just To.** `to:person` and `in:sent person` behave differently; a colleague
   who only ever appears in `Cc` will be missed by a `to:` query. Use a bare `in:sent <name>`
   pass as well as `in:sent to:<addr> (…)`, then read the actual `To`/`Cc` headers of each hit.
2. **Never answer the reply-all question from the user's description.** Before offering or
   drafting a reply-all, print the real `To` and `Cc`. The reply-all audience here included an
   **outside party** the user was not thinking about — re-opening a thread with an external
   contact is a materially different action from replying to a colleague. Surface the true
   audience and let him choose.
3. **Offer the audience explicitly when the target was only Cc'd.** Present the options rather
   than picking: reply-all (which includes the external To-recipient), a reply to the colleague
   only, or a fresh mail on-thread.

**Multiple hits on the same theme are normal — report all of them, grouped.** The same inventory
existed in several threads: the Regalia/Iris sale mail (8 Aug), a fuller Regalia unit list to an
investor-advisor (27 Aug, Prakash NOT on it), the Gandhinagar rental/warehouse mail (15 Jul), and
adjacent "available for offering" mails (Serenity Hillview, Ranka Amber UDS column). Group by
**sale / rental / adjacent** and state which colleague appears on each, rather than collapsing to
a single "the email you mean."

## Step 3b — Deliverable: a clickable link to the thread

When the user asks "give me a link to that email," the answer is the Gmail **thread** deep-link,
built from `threadId` (returned by any `messages().get` / `messages().list`):

```
https://mail.google.com/mail/u/0/#all/<threadId>
```

- `/u/0/` maps to the account the message actually lives in. For ndr@draas.com use `/u/0/`.
  If you resolved the thread on a non-default account, that account's index may differ — state
  which mailbox the link opens.
- `#all/<threadId>` (not `#inbox/` or `#sent/`) is the robust form: it finds the thread regardless
  of which label it currently carries, including archived mail.
- Print the threadId in full. Do **not** truncate it (see the 16-hex-char pitfall — a truncated id
  produces a dead link and a 404 on `messages().get`).
- Also report `thread msgs: N`. A single-message thread (N=1) means there is nothing to "reply
  into" beyond the original — worth saying, because the user may expect a back-and-forth that
  never happened.
- The user prefers **plain links, not code blocks.**

## Step 2b — "Analyse this report" means VERIFY THE REPORT EXISTS BEFORE ANALYSING

Highest-value pattern from the 2026-09-19 session. NDR asked: *"review the testing… give me your
analysis of what the testing report does… does it show the structure is good enough?"*

**There was no report.** No testing had ever been commissioned — the matter was still sitting at the
quotation stage, pending the vendor's revised quote. The only two possibilities are (a) analyse the
real artifact, or (b) say plainly it does not exist. Analysing an *adjacent* document (the Nova
Spaces façade proposal) and presenting it as the test findings would have been a serious error.

**Exhaustive absence-verification sweep — run all of these before saying "not found":**

```python
# 1. Broad inbox sweep with ANYWHERE (not just inbox) — catches archived/filed mail
gmail.users().messages().list(userId='me', q='after:2026/08/25 in:anywhere', maxResults=250)

# 2. Term matrix — instrument names AND standards AND the deliverable noun
#    structural, NDT, rebound hammer, UPV, ultrasonic pulse, core test, load test,
#    structural audit, structural report, test report, RCC test
# 3. Sender-domain sweeps: from:<vendor-domain>, from:<conduit-domain>
# 4. Drive: name contains '…' AND fullText contains '…' (rebound hammer, structural audit)
# 5. Drive recency: modifiedTime > '<start>' and mimeType='application/pdf'
# 6. ALL THREE accounts: google-draas, google-ahfl, google-gmail
```

Then state the negative with the **scope of what you checked** (accounts, date window, term list),
and — critically — **say where the matter actually stands instead**. In this case the tracker
(`kind='NDR_Master_Task_Tracker'`, tab `All Tasks`) already carried it: **P-211** = revised-scope
email sent to Kiran Kumar, status `Ongoing`, i.e. no quotation and no report yet. The task tracker
is the fastest way to establish "what stage is this at" once email comes up empty.

**Report shape that works:** (1) the direct answer — nothing has been tested, no report exists;
(2) what you found instead and who the real parties are; (3) where the matter actually stands, per
email + tracker; (4) what you checked (so he knows the negative is trustworthy); (5) offer the
concrete next action (chase the vendor / ask the conduit).

### Do not let a missing artifact block the part that does not depend on it

The user's *second* question — "we're only cutting away the chajjas and adding cladding, is the
structure good enough?" — is answerable from first principles **without** the test report, and
answering it preserves the value of the turn. Give the structural reasoning anyway (see
`real-estate-legal-compliance`, `references/structural-condition-assessment-and-facade-refit.md`),
while being explicit that the *reported numbers* cannot be assessed because no report exists.
Distinguish "answered from principle" from "answered from the document" so he never mistakes
reasoning for evidence.

Run several independent Gmail `q=` searches so you don't depend on any single (possibly
wrong) term. The dimensions that uniquely pin a thread:
1. **Entity-name variants** — try phonetic/substring variants ("reformers", "farmers",
   "red soul", "redsol", "red sol") and the domain of the related counterparty
   (e.g. "redsoul.co.in" surfaced the bank-account email).
2. **The person involved** (e.g. "nishant prakash" — gave this thread plus a separate
   partnership-deed email).
3. **The amount** ("25 lakh", "₹25,00,000").
4. **The recipient** named by the user ("sarthak").
5. **Subject-entity combos** ("red soul", "farmers collective").

Whichever dimension overlaps is the thread. In the ₹25L case, "sarthak" + "refund" +
"nishant prakash" + "red soul" all landed on the same subject line
"Red Soul Farmers Collective — Refund of ₹25,00,000".

## Step 3 — Pull exact To/Cc for EVERY variant of the email

This is the part that matters and the part that trips people up: the recipient list of the
**original**, any **forward**, and any **reply** inside the same thread are all DIFFERENT, and
the user's follow-up ("did that same email go to Roshni and Eshwari?") depends on that
distinction. Answer only after checking each relevant variant:

- **Original** (NDR → Sarthak, Cc Nishant Prakash): `To` = the addressee, `Cc` = who was copied.
- **Forward to a third party** (e.g. NDR → Manohar): frequently `Cc: None` even though the
  original body carries the original To/Cc inside it. Do not infer the forward's recipients
  from the embedded headers in the body — read the forward message's OWN top-level headers.
- **Reply** (Sarthak's confirmation): To/Cc usually mirror the original but not always.

Use `format='full'` and parse `payload.headers` for `From`/`To`/`Cc` of each message id.
Report them separately, e.g. "Original: To Sarthak, Cc Nishant Prakash. Forward to Manohar:
To Manohar only, Cc None — no Roshni/Eshwari."

## Sender-centric sweep — you know the SENDER/domain, not the recipient

Variant of the lookup where the user names a likely sender ("emails from drahomes.in,
Compliance / Ravi / Prabhakar") but the recipient/To is unknown. Recipient-oriented Gmail
queries (`"Lalit Kumar" "Vishwas"`) come up empty because the missing link is a party name,
not a party on the envelope. Instead:

1. **Inventory the sender's whole outbox with pageToken pagination** (up to ~300+ msgs):
   `q='from:ravi.v@drahomes.in'`, loop `nextPageToken` until exhausted.
2. **Keyword-match To + Cc + Subject per message** (case-insensitive): the hit list beats
   any single-query guess. In the Aug 2026 TIDCO/RLDA hunt, scanning
   `from:prabhakar.s@draas.com` (192 msgs), `from:ravi.v@drahomes.in` (35),
   `from:admin2.blr@drahomes.in` (353), `from:compliance@drahomes.in` (54) surfaced the
   whole CPPL/RLDA thread family in one pass.
3. **Cross-party queries** still help once you have a name: `"Lalit Kumar" "Vishwas"`,
   `"Lalit" "Chennai" "shares"` find threads where BOTH parties appear even if the spoken
   name is mangled ("Lalit" ↔ "Lalith").
4. **Financial-artifact tracing**: people who never email directly often appear in bank
   artifacts inside other threads. Search the RTGS/UTR remitter name (`"LALIT KUMAR"` in an
   RTGS row), share folio numbers, certificate numbers, "Mr. X @49%" shareholding workings.
   This is how "Mr. Lalith" resolved to the 49% investor behind Chennai Premium Projects.

## threads.get 404 fallback — use messages().get instead

Some message IDs returned by `users().messages().list()` fail
`threads().get(id=...)` with `HttpError 404 "Requested entity was not found"` — most often
for older/archived threads or list results crossing mailbox states. Do NOT declare the email
gone; fall back to `users().messages().get(userId='me', id=..., format='full')` which returns
the same headers/body for a single message. Thread-view aggregation is lost, but the To/Cc
evidence you need is intact. (Seen in the Aug 2026 Mundhara & Co folio lookup: 8 of 10
`threads().get` calls 404'd while `messages().get` worked.)

## Missing-account gate — re-auth before concluding "not in email"

After exhausting all term variants on the primary account, if the personal account
(`google-gmail`, nishantranka@gmail.com) shows an expired token (`invalid_grant`), THAT is
the unsearched surface — especially when compliance/secretarial senders write to personal
gmails directly (DRA compliance addresses dranjeeth@gmail.com / nishantranka@gmail.com).
Send ONE `send_oauth_url` (Google-only tool) and offer to resume the sweep after the user
taps the button. Also flag any mailbox you cannot reach at all (e.g. a partner's personal
Gmail) so the user can search it themselves.

## Working code pattern

Search (list) then fetch full headers per message:

```python
import os, sys
sys.path.insert(0, '/opt/hermes')   # REQUIRED — see note below, else ModuleNotFoundError
os.environ.setdefault("HERMES_SESSION_USER_ID", "ndr")
from tools.gws_auth import build_service
svc = build_service('gmail', 'v1', service_name='google-draas')  # google-draas, NOT 'google'
profile = svc.users().getProfile(userId='me').execute()
print("MAILBOX:", profile.get('emailAddress'))   # confirm = ndr@draas.com

res = svc.users().messages().list(userId='me', q=query, maxResults=10).execute()
for m in res.get('messages', []):
    full = svc.users().messages().get(userId='me', id=m['id'], format='full').execute()
    hdrs = {h['name']: h['value'] for h in full['payload']['headers']}
    print(hdrs.get('Date'), '|', hdrs.get('From'), '|', hdrs.get('To'), '|',
          'Cc:', hdrs.get('Cc'), '|', hdrs.get('Subject'))
```

Notes:
- **`sys.path.insert(0, '/opt/hermes')` is mandatory** when the script is run via `execute_code`
  or `terminal` from a cwd other than `/opt/hermes` (the default cwd is `/opt/data`). Without it,
  `from tools.gws_auth import build_service` raises
  `ModuleNotFoundError: No module named 'tools'`. The `tools` package lives at
  `/opt/hermes/tools/`. This is a path-resolution issue, not a broken tool — do not conclude
  Gmail access is unavailable. Locate it with
  `search_files(pattern='gws_auth.py', target='files', path='/')` if it ever moves.
- Pass `service_name='google-draas'` explicitly; the default may not be what you want.
- The vault may log `canonical_uid: vault has no identity mapping for '<uid>' -- using raw id as
  fallback key` to stderr. This is **noise** — the call still succeeds. Do not report a vault
  outage to the user because of it.
- Iterate: loop the query over the variant list in Step 2.

## Related matters that surface in the same lookup

When answering "who was marked," the same entity often appears in OTHER threads that are
related but distinct. Flag them rather than merging:
- A *bank-account-details* email ("Bank Account Details – Redsol Farmers Collective",
  Sarthak → Manohar, Cc NDR) is separate from the *refund instruction* email.
- A *partnership-deed* email (NDR → Nishant Prakash, Cc Bharat) is separate again.
- An *ITR* thread for land partners (Prakash → Eshwari, Cc Roshini, Bharat, NDR) belongs to a
  different matter — do not let "did they get CC'd on the ITR" merge with "who was on the
  refund email." Report the correct thread and clearly delimit the others.
