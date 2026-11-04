# Google Contacts Sheet Access (NDR DRAAS)

## Sheet Details

- **Sheet ID:** `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`
- **Sheet URL:** `https://docs.google.com/spreadsheets/d/1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g/edit`
- **Sheet name:** `NDR DRAAS Google contacts.csv` (note: contains spaces and `.csv` — must be URL-encoded)
- **Credential:** `ndr@draas.com` → `/data/hermes/oauth-draas.json`

## Address Columns (0-based indices in Sheets API response)

The sheet has 4 address blocks (Address 1 through Address 4). Address 1 is the primary/home address. Column indices are 0-based (index 0 = column A):

| Index | Column | Header |
|-------|--------|--------|
| 39 | AN | Address 1 - Label |
| 40 | AO | Address 1 - Formatted |
| 41 | AP | Address 1 - Street |
| 42 | AQ | Address 1 - City |
| 43 | AR | Address 1 - PO Box |
| 44 | AS | Address 1 - Region |
| 45 | AT | Address 1 - Postal Code |
| 46 | AU | Address 1 - Country |
| 47 | AV | Address 1 - Extended Address |

**PITFALL — Column letter vs index:** A1 notation is 1-indexed (A=1), but the Sheets API returns arrays 0-indexed. To write to index 39 (Address 1 - Label), target column AN (the 40th column, A=1). Always verify by checking the header row first — use `headers[index]` to confirm the target column name before writing.

**Example — writing a Home address (verified June 2026):**
```python
# Range: AN{row}:AU{row}  (indices 39-46)
body = {'values': [[
    "Home",                          # AN - Label
    "Full formatted address\nLine 2\nCity PIN",  # AO - Formatted
    "Street address line",           # AP - Street
    "Bangalore",                     # AQ - City
    "",                              # AR - PO Box
    "Karnataka",                     # AS - Region
    "560037",                        # AT - Postal Code
    "India",                         # AU - Country
]]}
sheets.spreadsheets().values().update(
    spreadsheetId=sheet_id,
    range=f"AN{row_num}:AU{row_num}",
    valueInputOption='USER_ENTERED',
    body=body
).execute()
```

**PITFALL:** If you use the wrong column letter (e.g., AM instead of AN), the entire address shifts by one column — Label gets the formatted string, Formatted gets the street, etc. Always verify with a read-back after update.

**Total columns: 93 (A through CC).** The Relation columns (BX–CA) are toward the end of the sheet. Use header-name lookup, not index assumptions.

Key columns (use header name lookup, not hardcoded index — some columns may not exist in all sheets):

| Header | Notes |
|--------|-------|
| First Name | Col A |
| Middle Name | |
| Last Name | Col C |
| Nickname | Col I |
| E-mail 1 - Value | |
| E-mail 2 - Value | |
| Phone 1 - Value | |
| Phone 2 - Value | |
| Phone 3 - Value | |
| Organization Name | |
| Organization Title | |
| Notes | |
| Project Association | |
| Land Proposal Association | |
| People Association | |
| Relation 1 - Label | |
| Relation 1 - Value | |
| Relation 2 - Label | |
| Relation 2 - Value | |
| Spouse | |
| Alias | |

**Always look up columns by name, not by index.** Use this defensive pattern:

```python
def col(headers, name):
    return headers.index(name) if name in headers else None

headers = rows[0]
data_rows = rows[1:]

r1l_i = col(headers, 'Relation 1 - Label')
r1v_i = col(headers, 'Relation 1 - Value')
# ...
for i, row in enumerate(data_rows, start=2):
    r1l = row[r1l_i] if r1l_i is not None and r1l_i < len(row) else ''
```

Hardcoding column indices (e.g. `row[28]`) breaks when columns shift or are missing — `list.index()` on a missing header raises `ValueError`, and accessing `row[None]` raises `TypeError`.

## Python Recipe — Search for a Contact

```python
import json, urllib.request, urllib.parse

with open('/data/hermes/oauth-draas.json') as f:
    creds = json.load(f)

data = urllib.parse.urlencode({
    'client_id': creds['client_id'],
    'client_secret': creds['client_secret'],
    'refresh_token': creds['refresh_token'],
    'grant_type': 'refresh_token',
}).encode()
req = urllib.request.Request('https://oauth2.googleapis.com/token', data=data)
with urllib.request.urlopen(req, timeout=15) as resp:
    token = json.loads(resp.read())['access_token']

SHEET_ID = '1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g'
SHEET_NAME = urllib.parse.quote("NDR DRAAS Google contacts.csv")

# Fetch full sheet (header + data) in one call
url = f'https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/{SHEET_NAME}?majorDimension=ROWS&valueRenderOption=FORMATTED_VALUE'
req2 = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'})
with urllib.request.urlopen(req2, timeout=60) as resp:
    all_rows = json.loads(resp.read()).get('values', [])

headers = all_rows[0]
data_rows = all_rows[1:]

# Helper: get column value safely by header name (NOT by index)
def get(row, header):
    idx = headers.index(header) if header in headers else None
    return row[idx] if idx is not None and idx < len(row) else ''

# Search by name
for i, row in enumerate(data_rows, start=2):
    name = f"{get(row, 'First Name')} {get(row, 'Last Name')}".strip().lower()
    if 'target_first' in name and 'target_last' in name:
        phones = [get(row, 'Phone 1 - Value'), get(row, 'Phone 2 - Value'), get(row, 'Phone 3 - Value')]
        print(f"Row {i}: {get(row, 'First Name')} {get(row, 'Last Name')} | Phones: {[p for p in phones if p]}")
        print(f"  Org: {get(row, 'Organization Name')} | {get(row, 'Organization Title')}")
        print(f"  Email: {get(row, 'E-mail 1 - Value')}")
        print(f"  Notes: {get(row, 'Notes')}")
```

**Always look up column values by header name** (as shown above). Do not hardcode column indices — the sheet layout is stable but indices are fragile. Use `headers.index(header)` to find the right column dynamically.

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `HTTP Error 400: Bad Request` on `values/{sheet}` | Sheet name not URL-encoded (contains spaces, dots) | Wrap sheet name in `urllib.parse.quote()` |
| `HTTP Error 401` on CSV export URL | CSV export (`/export?format=csv`) bypasses OAuth | Use Sheets API v4 `values/{sheet}!` endpoint instead |
| `ModuleNotFoundError: contact_resolver` | Module does not exist in this environment | Do not attempt to import it; use OAuth + Sheets API directly |
| Cell reads back as `#ERROR!` after a write | `valueInputOption=USER_ENTERED` parsed the value as a **formula** — Indian phone numbers written as `+91 …` start with `+` | Re-write that cell with `valueInputOption=RAW` (see pitfall below) |
| Wrong value in the wrong column (label holds the phone, value holds the label) | Column letter miscounted — A1 notation is 1-indexed (A=1) but the API array is 0-indexed, so array index 27 is **AB**, index 28 is **AC** | Re-read the header row and map `headers[i]` → column letter before writing; verify by reading the row back |

### PITFALL — `+91 …` phone numbers break under USER_ENTERED (verified 2026-09-28)

Appending a contact row with `valueInputOption=USER_ENTERED` and a phone value of
`+91 99305 44141` stores **`#ERROR!`** in that cell: Sheets treats a leading `+` as the
start of a formula. The row writes "successfully" and the API returns `updatedCells`, so
the only way to catch it is a read-back.

Two-part fix:

```python
# 1. Append the row (USER_ENTERED is fine for names/emails/notes)
# 2. Immediately re-write the phone cells with RAW so no formula parsing happens
url = (f"https://sheets.googleapis.com/v4/spreadsheets/{SHEET_ID}/values/"
       f"{SHEET_NAME_Q}!AB{row}:AC{row}?valueInputOption=RAW")
body = {'values': [["Mobile", "+91 99305 44141"]]}   # AB = Phone 1 - Label, AC = Phone 1 - Value
```

Always read the row back and assert the phone cell equals the number you wrote (not
`#ERROR!`, not a label) before reporting the contact as done.

### A contact the user just added may exist ONLY in People API (verified 2026-09-28)

"I have added a contact called X" means it went into **Google Contacts / the phone**, not
into this sheet. The sheet (4,251 rows) had **zero** `Thesia` hits while
`people().searchContacts(query='Thesia')` on `google-draas` returned the person
immediately. A zero-hit sheet search is **not** "no such contact" — run
`find_contact.py` (which queries People API *and* both sheets) before concluding anything,
then enrich BOTH stores. Full recipe: `references/contact-enrichment-people-api.md`.

## Notes

- The sheet title includes `.csv` but it is a Google Sheets tab, not an actual CSV file — the Sheets API export/exportFormat approach fails; use `values` API instead.
- Row 473: **Ashwin Pai** — +91 9972042131 (single mobile, no org)
- Always present matched contacts with all available phone numbers for user to choose from.
