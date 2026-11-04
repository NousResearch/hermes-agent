# Contacts Lookup — Absorbed from data-science/contacts-lookup

## What This Reference Covers

Resolves a person's name to a full contact record: mobile numbers (mobile, work, DRA, personal), email addresses, organization, and notes. Searches Google Contacts Sheet (primary) and Google People API (fallback).

**Skill status:** Absorbed into `messaging-drafts` umbrella (2026-05-29). Original at `data-science/contacts-lookup/`.

## When to Use

Trigger: "Find [name]", "Look up [name]", "What is [name]'s number", "Search contacts for [name]", "Who is [name]", "Get me [name]'s details"

**Do NOT activate for:** personal relations ("my wife's number"), calendar lookups, email drafting.

## Resolution Order

The user's Google contacts live across multiple sources. **Neither is a superset** — always check ALL sources before declaring "person not found."

### Phase 1 — Search Drive for All Contact Sheets THEN Check Each Tab

When the user says "look up on my Google contacts AND my online contact sheet," the "online contact sheet" typically refers to the NDR DRAAS Google contacts spreadsheet — but other contact spreadsheets on Drive may have data the primary sheet doesn't.

**Workflow:**
1. Search Drive for contact-related spreadsheets:
   ```python
   drive.files().list(
       q="name contains 'Contact' and mimeType = 'application/vnd.google-apps.spreadsheet'",
       pageSize=20
   ).execute()
   ```
   Known sheets (June 2026):
   - **NDR DRAAS Google contacts** (`1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`) — main business contacts CSV export (53 columns, phone numbers at col AA-AB)
   - **NDR CONTACTS** (`1fYa-t2RY1siy2qBgAH8uu_Jd2chjJ716BbcpxilpOK0`) — additional contacts
   - **List of contact numbers & Email ID** (`1GRrndy0btgF-uB_v1XqWUNu-FfbieAfS1eKqTMmSaYU`)

2. For each sheet, get all tabs (`sheets.spreadsheets().get(fields='sheets.properties')`) and scan each tab for the target name
3. Search the **full row** (not just first/last name columns) — people may appear in Notes, Organization, or other columns
4. Phone columns are at AA-AB (col 27-28) for Mobile label+value, AC-AD (col 29-30) for Phone 2, etc.

### Phase 2 — Google People API

Search the user's live Google Contacts via People API:
```python
people.people().searchContacts(query="Manohar", readMask="names,phoneNumbers,emailAddresses,organizations")
```

**Pre-flight:** Check that the OAuth token includes `contacts.readonly` scope. Without it, People API returns HTTP 403.

**Limitation:** People API `connections().list()` only returns contacts the user has explicitly saved as "My Contacts" — it excludes "Other Contacts" and social-media-synced entries that the CSV export includes.

### Phase 3 — DRA Internal Staff Spreadsheets

Check these Drive files for DRAAS employees:
- `Employee Details.xlsx` (ID: `1q06-GS0Nd2PJwPH0iJSkYIjbzk8MAvgD`)
- `Updated DRA staff contact list 2016` (ID: `1GvQKVaCSVZLMUBmzBlSEkdFmezX9-hm0AdQunxBueCI`)

### Phase 4 — Drive PDFs (Personal Contacts)

For personal relations (doctors, dentists, family friends, school contacts) not in any business sheet:
- Search: `name contains '[first name]'` OR `name contains '[last name]'` OR `fullText contains 'dentist'`
- PDF phone extraction: `pymupdf.open(path).get_text()` — NOT `fitz.open().get_text()` (fitz has no `.get_text()` method)
- Phone number regex: `\\+91` followed by space-separated 5 digits, or `\\d{5}\\s\\d{5}`
- Session example (June 2026): Dr. Kenneth F.H. Tan → `20241021 Ruhaan P Dentist Tan.pdf` (ID: `1jPnNg6trQABJb6SxhzWlaJVwB91hviLn`) → found `98440 17643` (Cell)
- **Auth:** Use `tools.gws_auth.build_service('drive', 'v3')` with per-user token

## Critical Auth Pattern (ONLY working pattern here)

```python
import json, os, tempfile, urllib.request
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

telegram_id = "ndr"  # from HERMES_SESSION_USER_ID env var
token_path = f"/data/hermes/users/{telegram_id}/the gws-vault daemon (no token files exist on disk — see api-references/google-workspace-api/references/token-access-canonical.md)"
token_data = open(token_path).read()

with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
    tf.write(token_data)
    tmp_path = tf.name

try:
    creds = Credentials.from_authorized_user_file(tmp_path, [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
    ])
finally:
    os.unlink(tmp_path)

if creds.expired:
    creds.refresh(Request())
    open(token_path, "w").write(creds.to_json())
```

> ⚠️ `Credentials.from_authorized_user_json` does NOT exist in google-auth ≥2.x. Always use `from_authorized_user_file` with a real temp file.

## People API — Correct Endpoints (working as of 2026-05-28)

```python
# Create: POST https://people.googleapis.com/v1/people:createContact  (NOT connections:createContact)
# PATCH notes: requires etag in body
# People API query= is unreliable — always list all + filter locally
```

## Sheet Column Reference

The NDR DRAAS Google contacts.csv has **53 columns (A–BA)**. Phone columns start at **Col 27 (AA)** — not M–R as previously assumed:

| Col | Letter | Content |
|-----|--------|---------|
| 0 | A | First Name |
| 2 | C | Last Name |
| 10 | K | Organization Name |
| 14 | O | Notes |
| 16 | Q | Labels |
| 18 | S | E-mail 1 - Value (primary) |
| **27** | **AA** | **Phone 1 - Label** (e.g. "Mobile") |
| **28** | **AB** | **Phone 1 - Value** (e.g. "+91 99720 42131") |
| 29–30 | AC–AD | Phone 2 |
| 31–32 | AE–AF | Phone 3 |
| 33–34 | AG–AH | Phone 4 |
| 35–36 | AI–AJ | Phone 5 |
| 37–38 | AK–AL | Phone 6 |
| 39–52 | AM–BA | Address fields |

**⚠️ Phone columns are NOT at M–R.** The earlier assumption was wrong. Always read up to BA. To get the full header: `spreadsheet.values().get("'NDR DRAAS Google contacts.csv'!A1:BA1")`.

**⚠️ Neither source is a superset of the other — always check BOTH.**

Confirmed patterns (Jun 2026):
- **CSV-only:** Ashwin Pai (ashwin.pai@centuryrealestate.in, +91 9972042131) found in CSV Row 473 but NOT in People API `connections().list()` scanning 2,175 contacts. The CSV export captures contacts from "Other Contacts" and social-media-synced sources that People API `connections()` excludes.
- **People-API-only:** Roshini Ranka (rnr@draas.com, +91 9845026390) found in People API `connections().list()` but NOT in the CSV export at all. The CSV is a snapshot export of Google Contacts and can be stale/incomplete.

**Rule:** When resolving any contact, check BOTH sources in parallel. If one returns nothing, always check the other before declaring "person not found." The CSV has wider coverage for business contacts; People API has deeper coverage for personal/family contacts.

## Column Search Behavior — Broader Matching Required

**Contacts are often stored with "first name" containing the full name when "last name" is empty.** This is a Google Contacts export artifact — if a contact was created with only a "Name" field (no separate First/Last), the entire name lands in the First Name column and Last Name is blank.

**Example (June 2026):** "Vivek Chandra" → stored as `First Name: "Vivekchanda"`, `Last Name: ""`. A search for `"Vivek Chandra"` (with space) returns nothing. A partial prefix search for `"vivek"` catches it.

**Rule:** When searching for a full name `"First Last"`, do NOT search for the exact string. Instead:
1. First-pass: search for just the first-name prefix (`"vivek"`)
2. Filter results for matching org or role context
3. If that returns nothing, also try concatenated form (`"vivekchanda"`) and any obvious phonetic variants

This is distinct from phonetic misspelling traps (e.g., Gauri/Gowri) — it's a field-boundary issue where the name boundary itself is lost.

## Hard Stops

- **Name mismatch = hard stop** — do not return a number if name doesn't match
- **Relation columns** — do NOT use "Relation 1/2" to find people; those describe relationships TO others
- **Telegram limitation** — sheet phone numbers cannot be used for Telegram DMs (requires username/chat_id, not mobile number)

## Name Collision Hazard

Multiple contacts can share the same name. ALWAYS present ALL matches numbered with Name + Organization. Let user confirm. Do NOT auto-select.

## Cross-Source Discrepancy

When same person found in multiple sources with different details, flag ALL variants and ask user to confirm.

## Quick-Reference: 4-Phase Contact Resolution (validated 10 Jun 2026)

When user says "find X's number" or "look in my Google contacts", try these in order — STOP at the first one that returns a verified match:

### Phase 1 — tools.contact_resolver_tool
- Import path: `from tools.contact_resolver_tool import _handle_contact_resolver`
- Check availability: `_check_available()` — returns `False` in this environment (the sheets SA path relies on `tools.gws_sa`, which does not exist in this deployment — removed 2026-05-08)
- **If `_check_available()` returns False, skip directly to Phase 2. Do not waste a tool call trying to invoke the resolver.**
- Tool arguments: `{"query": "Manohar Singh", "context": "REDSOUL", "account_email": "ndr@draas.com"}` (account_email is the SA DWD subject — ndr owns the contacts sheet)

### Phase 2 — Search Drive for All Contact Sheets

When the user specifically says "check both my Google contacts AND my online contact sheet", this is the workflow:

1. Search Drive for contact spreadsheets (see Phase 1 in Resolution Order above)
2. Check all known sheets for the target name
3. This catches contacts in the CSV export that People API misses (e.g. Manohar Singh found in NDR DRAAS Google contacts.csv Row 1940 with 3 phone numbers, but NOT in People API search)

**Only proceed to Phase 3 if Phase 2 returns nothing.**

### Phase 3 — gws_auth People API (per-user OAuth, confirmed working)

**Pre-flight: Check token scope BEFORE calling People API.**

People API requires `https://www.googleapis.com/auth/contacts.readonly` (or `contacts` for write access). If this scope is missing, the API returns HTTP 403 "Insufficient authentication scopes".

**How to check token scope:**
```python
import json
token_path = f"/data/hermes/users/{telegram_id}/the gws-vault daemon (no token files exist on disk — see api-references/google-workspace-api/references/token-access-canonical.md)"
with open(token_path) as f:
    token = json.load(f)
    scopes = token.get('scopes', [])
# If 'contacts.readonly' not in scopes, People API will 403
```

**Also read scopes from the auth URL itself:**
```python
from tools.gws_auth import get_auth_url
url = get_auth_url(telegram_id)  # contains full scope list in the URL
```

**If `contacts.readonly` is missing:** Tell the user the token doesn't cover contacts and ask for the number directly. Do not attempt People API — it will 403 and waste a turn. Do not fall back to Gmail email body searches for phone numbers; email threads rarely contain phone numbers in a parseable format.

**When People API is available:** Use this when the contact_resolver is unavailable.
- See "Critical Auth Pattern" block above for the working code
- Sample confirmed search: `people.people().searchContacts(query="Manohar", readMask="names,phoneNumbers,emailAddresses,organizations")`
- The People API returns the user's full personal Google contacts — useful for business partners, family, doctors, etc. that may not be in the business contacts sheet
- **Always present all name matches with organization** (e.g. "Manjunath Manohar Singh @ O3 Infotech" vs "Manohar Prabhu" vs "Manoharji Bhandari") — never auto-pick the top hit
- **Mobile-number priority for wa.me:** Pick the entry whose `type == "mobile"`. If multiple mobile entries exist, ask the user which one to use. For Manjunath Manohar Singh, 6 phone numbers were returned — selected `+91 9845890316` (mobile) for the WhatsApp link.

### Phase 4 — Drive PDF search (for personal contacts not in People API)
- See "Drive PDFs" section above
- Triggered when both contact_resolver and People API return nothing (e.g. doctors, dentists, school contacts)

## CRITICAL: pm2.vlr vs anbarasan.murugaperumal — Hard Stop (2026-06-02)

**Anbu (Anbarasan Murugaperumal) — the SAME person appears under TWO different emails in contacts:**
- ✅ **Correct / active DRAAS email:** `pm2.blr@draas.com`
- ❌ **Old / legacy email (do NOT use for Drive sharing):** `anbarasan.murugaperumal@draas.com`

On 2026-06-02, user caught that `anbarasan.murugaperumal@draas.com` was incorrectly shared with Drive folders (OS 7-2025 subfolder and Legal Notice subfolder). The wrong email had to be manually removed by the user.

**When resolving Anbu by name:**
1. Always present both email variants found
2. Explicitly confirm which email is current before any Drive sharing action
3. Default to `pm2.blr@draas.com` for Drive permissions

---

## Known Phonetic Name Traps (2026-06-02)

### "Gauri Singh" / "Gowri Singh" — G-A-U-R-I vs G-O-W-R-I

**Problem:** User says "Gauri" (G-A-U-R-I) but contacts show "Gowri" (G-O-W-R-I). People API search for `gaurisingh` returns nothing; search for `gowrisingh` succeeds.

**Confirmed contacts (June 2 2026):**
- Name in contacts: **Gowri Singh** (G-O-W-R-I spelling)
- Emails: `gowrisingh72@yahoo.com` (primary), `gowrisingh1341@gmail.com`
- Role: DRAAS HR / Recruitment
- `gaurisingh@draas.com` does NOT exist — do not use

**Pattern:** When user names a DRAAS contact and People API returns nothing for the exact string, try phonetic spellings before assuming the email. Always present the found variant to the user and confirm before using in a calendar event.

**Same trap applies to Roshini:** User said "Roshini" — People API found "Roshini Ranka" immediately with multiple emails. Primary correct: `rnr@truss.com` (NOT `roshini.ranka@draas.com` which doesn't exist). When in doubt, verify all variants with user before creating events.

**Anbu's confirmed numbers (2026-06-02):**
- Primary (WhatsApp): `+91 81500 29900`
- Secondary: `+91 99942 213535`
- Tertiary: `+91 90365 13535`
