# Contacts Sheet — Naming Edge Cases That Block People API / contact_resolver

## Observed patterns (Aug 2026)

### 1. Parenthetical name suffixes

Some contacts in the NDR DRAAS Contact Sheet have a parentheses qualifier in
the First Name field that is NOT present in Google Contacts, e.g.:

- Sheet: `Vinod Kumar Das (Rahul)` — People API has no match for "Rahul"
- `Vinod (Srinivas kerala Investor)` — no Google Contacts entry at all

**Effect:** `contact_resolver(query="Vinod Kumar Das Rahul")` returns only
V-name false matches (V Gopal, V Jitendra, etc.) scoring ~90 because the
People API sees the first letter only. The real contact never shows up.

**Fix:** when `contact_resolver` returns only low-confidence matches (all
same-first-letter, no contextual relevance), go direct to sheet search:

```python
from tools import gws_auth
from googleapiclient.discovery import build

sheets = gws_auth.build_service('sheets', 'v4', service_name='google-draas')
sheet_id = '1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g'
all_data = sheets.spreadsheets().values().get(
    spreadsheetId=sheet_id,
    range='NDR DRAAS Google contacts.csv!A:Z'
).execute()

for i, row in enumerate(all_data.get('values', [])[1:], 2):
    first = (row[0] if len(row) > 0 else '').lower()
    # Search both the full first-name field AND strip parenthetical
    stripped = first.split('(')[0].strip()
    if query_word in stripped or query_word in first:
        # Found it — extract phone, email, labels
```

### 2. Role/designation suffix appended to name

- E.g. `Nagrajappa JDTP` stored in sheet, user says just "Nagrajappa"
- `Mohan Sir JDTP GBA East` stored, user says "Mohan ADTP"
- Already covered in P2.1 (designation-as-name-suffix)

### 3. Employee-style contacts with no phone column

DRAAS internal colleagues (like Vinod Kumar Das Rahul with
vkdas@draas.com) may have email-only rows in the contacts sheet.
The phone column (col AC / index 28) is empty. Don't assume every
contact in the sheet has a phone.

**Checklist when phone is missing:**
1. Check the email domain — if @draas.com or @drahomes.in, it's an
   internal colleague. No WhatsApp number means you need the user to
   provide one.
2. Check honcho memory — the user may have mentioned their number
   in a past session (e.g. "Rahul's number is +91...")
3. If still nothing, generate a phone-less WhatsApp link (opens to
   a blank recipient selector) and flag it.

### 4. employees sheet is empty (as of Aug 2026)

The `employees` tab in the contact spreadsheet has no data rows.
DRAAS colleagues live in the main `NDR DRAAS Google contacts.csv`
sheet alongside external contacts. Do NOT search `employees` first.

### 5. Common sheet-column layout (for manual access)

Range: `NDR DRAAS Google contacts.csv!A:DA`
- Col A (0): First Name (includes parentheticals and role suffixes)
- Col B (1): Middle Name
- Col C (2): Last Name
- Col Q (16): Labels (e.g. `* myContacts`, `RelA`)
- Col AC (28): Phone 1 - Value
- Email fields appear from Col R (17) onwards, multiple email columns
  with their own label columns interspersed.