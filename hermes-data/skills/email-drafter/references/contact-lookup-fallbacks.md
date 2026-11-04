# Contact Lookup Fallbacks & Internal Address Rules (2026-09-15)

When the `contact_resolver` / `find_contact.py` path can't surface an address,
use this fallback chain. Validated on Ranka Oasis CA-team lookup, Sep 2026.

## Internal colleagues — canonical domain

**All DRA employees/colleagues use @draas.com, NEVER drahomes.com (user
correction 2026-09-15, Bharat Hawaldar case).** When the contacts sheet shows a
drahomes.in address for a team member, the draas.com version is canonical:
- Bharat Hawaldar = sales1.blr@draas.com (sheet's sales1.blr@drahomes.in is stale)
- Piyush Ranka = piyush@draas.com, Bhagya G = crm.blr@draas.com
- Prakash Singh = psingh@draas.com, Sarthak Sharma = admin3.blr@draas.com
- Gowri Singh = gsingh@draas.com (Content & Marketing Head — NDR's voice-dictated
  "Gauri"; team lead for content, Sai Neha Vaddadi nVaddadi@draas.com reports to her)

## contact_resolver tool can be dead in-session

Fails with either:
- `"User '<uid>' has no gws_service configured in their profile"` — session
  profile lacks a gws_service binding; the tool cannot resolve any account.
- `"Account email 'X' is not associated with your session user"` — passing an
  explicit account_email that isn't linked to the session user.

Do NOT loop on it (tool-loop warning fires after ~3 tries). Fall back:

1. **find_contact.py** (personal-messaging skill):
   `/opt/hermes/.venv/bin/python /data/hermes/skills/productivity/personal-messaging/scripts/find_contact.py "<name>"`
   Gives a "vault has no identity mapping ... using raw id as fallback key"
   warning — that is benign, results still work.
2. **Name-match blind spots:** compound sheet rows do NOT match plain queries
   ("Nitin Osthwal Saurabh CA" or "Saurabh CA Ref VK" won't match
   "Saurabh Jain"). Substring-scan the sheet instead (below).
3. **Direct sheets read** via `build_service('sheets','v4', service_name='google-draas')`
   with `HERMES_SESSION_USER_ID=<uid>` prefix to pin identity (whoami-guard it
   first: `gmail.users().getProfile` must equal ndr@draas.com).

   Contacts sheet 1: `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`
   - Tab is named **'NDR DRAAS Google contacts.csv'** (NOT 'Sheet1')
   - Scan `!A1:AH4300`; header columns: A=First Name, C=Last Name, N=Notes,
     Q=Labels; emails in cols S–W (E-mail 1–4 Value), phones in AB–AC
     (label 'Mobile' + value). Rows with NO email show only Q + AB + AC filled.
   - Other tabs: projects (7 rows), land_proposals (5), entities (9), topics (634),
     vocab_corrections (2), employees (10) — cheap to scan for names.

   Contacts sheet 2: `1fYa-t2RY1siy2qBgAH8uu_Jd2chjJ716BbcpxilpOK0` — tab is
   **'NDR CONTACTS'** (find_contact.py fails on it with "Unable to parse range:
   Sheet1"). Only ~16 rows; scan `A1:AH20`.

4. **Sweep ALL THREE Google accounts** before concluding "no email exists":
   `google-draas` (ndr@draas.com), `google-ahfl` (ndr@ahfl.in),
   `google-gmail` (nishantranka@gmail.com). For each: People API connections
   list (names+emails+phones) and Gmail `messages().list(q='from:<name>')` /
   `q='<surname>'` metadata-header scan. People API `people/me` can 403 (no
   profile scope) but `connections().list` with contacts scope still works via
   find_contact.py's path; direct `build_service('people',...)` may 403 —
   use find_contact.py for the People leg.

## Resolving a voice-dictated recipient that find_contact.py can't match

NDR dictates recipients by voice ("Prakash, Markin, Bharat and Gauri", "Nikhil who's
doing digital marketing"). When `find_contact.py "<name>"` returns no usable match,
run this rung order before concluding "not in contacts" (validated 2026-09-15 on the
Ranka Udaya strategy email):

1. **Employees tab of contacts sheet 1XbSRA** — `employees!A1:H60`. Headers:
   name | email | phone | role | telegram_id | notes. This resolves role-named
   dictations instantly: "Gauri" → Gowri Singh, Content & Marketing Head.
   Also has Bharat, Prakash, Anbarasan, Sai Neha Vaddadi and TG-pending rows.
2. **Substring scan of the main contacts tab** (`NDR DRAAS Google contacts.csv!A1:AH4300`)
   for the exact name AND role keywords ("content", "digital", "marketing"). One scan
   script, print row numbers + the full row so the agent can judge compounds.
3. **Gmail header sweep on the session account** — `gmail.users().messages().list(q="from:(@draas.com) <firstname>")`
   plus `q="<firstname>"` generally; also check group DLs `to:marketing@draas.com`,
   `to:leads@draas.com` to see who actually receives marketing/lead traffic. Listing
   threads on Gowri's content emails (`from:gsingh`) shows the real working group.
4. **People API searchContacts** (via find_contact.py's path) for first-name + "draas".
5. **Vault identity search** — `VaultClient().search_identities("<name>")`; 0 hits for
   people absent from the vault (e.g. Markin, digital-marketing Nikhil). A 0 here is
   authoritative-ish, not a vault fault.
6. **Past sessions** — session_search for the name; usually empty for brand-new names.

If ALL rungs miss: the person is a new hire, external vendor, or voice-transcription
garble. Do NOT guess an address or reuse a same-surname lookalike. Draft the email to
the verified recipients, reference the unresolved names in the body if needed, and
flag the exact missing addresses to NDR in the handoff note — he supplies them.

## If the user insists an address exists but every source is phone-only

Report the verified rows with row numbers + phones, and ask for the address
(usually reachable via WhatsApp). NEVER reuse a lookalike contact of the same
name — e.g. the sheet's "Saurabh Jain" with saurabhjain@gmail.com (row 3251) is
the 2010 IRCA-listing contact (introduced by DDR), NOT the CA.

Phone-only rows as of 2026-09-15 — BOTH RESOLVED same day (NDR supplied):
- Row 2333 "Nitin Osthwal Saurabh CA" — +91 80889 79777 → **Nitin Osthwal
  Kalpesh, Kalpeshnitin@gmail.com** (full name given verbally by NDR; official
  DRA CA; partner/cousin of Saurabh T Jain)
- Row 3248 "Saurabh CA Ref VK" — +91 98864 89873 → **Saurabh T Jain (TSJ &
  Associates LLP), stjain@gmail.com + firm info@tsjca.com** — email was NEVER
  in the sheet/People API/Gmail; came from the RERA fund certificate PDF NDR
  attached (see enrichment below)

## Enriching a CA/professional contact from a certificate PDF (2026-09-15)

When NDR attaches a professional's certificate (RERA fund certificate, audit
letter etc.) to resolve identity: OCR it (`pdftoppm` → pages as PNG → tesseract)
and pull the CA-specific fields — ICAI membership no, UDIN, firm, office
address, phone, email. For Saurabh T Jain (Svamitva Emerald Square /
Shankeshwar Landmark LLP, RERA PRM/KA/RERA/1251/308/PR/171028/000667):
ICAI Membership 221686, UDIN 21221686AAAAMA1354, firm TSJ & Associates
LLP, #40 4th Floor Lakshmi Complex KR Road BLR-560002, Ph 080-26705995.
Firm site (tsjca.com) publishes only info@tsjca.com + varun.info@tsjca.com —
no firstname@ pattern; partners listed on AboutUs: CA Tarachand, CA Saurabh
T. Jain (also CA Rahul Mehta). Do NOT invent a firstname@tsjca.com address.

## noun_learner tool — working vs broken actions (2026-09-15)

- **BROKEN: `learn_correction`** — raises `name 'VOCAB_CORRECTIONS_TAB' is
  not defined` (constant referenced at noun_learner_tool.py:180-230 but never
  defined; file is root-protected, cannot be patched). Fallback: write the
  `voice_misspellings` column (col 91 = CN) directly via Sheets API, comma-
  separated. Do NOT loop on the error.
- **WORKING (safe to use normally):** `add_alias` (col CE), `update_associations`
  (CA/CF/etc), `append_history` (CG), `increment_score` (CO). These never touch
  the vocab_corrections tab and succeed even when learn_correction fails — use
  them for the aliases/associations/history/score parts of a contact update.
- **contact_resolver** itself works when passed the explicit session account:
  `contact_resolver(query=..., account_email='ndr@draas.com')` — use this form
  whenever default resolution is flaky, and to VERIFY your sheet updates landed
  (it reflects the sheet live, so re-query after writing).