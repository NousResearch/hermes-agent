# Contact Search Transparency

**Trigger:** User asks "show me the raw search results" or "what API did you call and how many results" when you look up a contact.

## User Preference

When looking up a contact and reporting results, the user wants to see:
1. **What API was called** (e.g. Google People API v1, Sheets API v4)
2. **What endpoint/method** (e.g. `people().searchContacts()`, `spreadsheets().values().get()`)
3. **What search parameters** (query string, pageSize, readMask)
4. **Number of results returned**
5. **All matching entries** with full details (name variations, ALL phone numbers, ALL emails)

This applies to BOTH:
- Google Contacts (People API)
- Google Sheets (NDR CONTACTS, NDR DRAAS Google contacts)

## Why This Matters

The user often knows a contact exists but under a slightly different name or with multiple numbers. Showing raw results with full transparency lets them spot the right entry and correct you on the spot, rather than you drawing an incorrect conclusion and wasting a round-trip.

## Output Format

When the user asks or when search results are ambiguous, show:

```
=== GOOGLE CONTACTS SEARCH ===
API: Google People API v1
Method: people().searchContacts()
Query: "Jitu Virwani"
Parameters: pageSize=10, readMask=names,phoneNumbers,emailAddresses
Results count: 1

  Name: ["Jitu Virwani"]
  Phone: ["+91 9844065000"]
  Email: ["jitu@embassyindia.com"]

=== NDR CONTACTS SHEET ===
API: Google Sheets API v4
Method: spreadsheets().values().get()
Sheet: NDR CONTACTS
Query: searched all 13 rows for "jitu", "virwani"
Results count: 0
```

## Implementation Pattern

```python
import json
from tools.gws_auth import build_service

# Google Contacts
people = build_service('people', 'v1')
r = people.people().searchContacts(query='Name', pageSize=10, readMask='names,phoneNumbers,emailAddresses').execute()
results = r.get('results', [])
print(f'Results count: {len(results)}')
for res in results:
    p = res['person']
    print(f'  Name: {json.dumps([n.get(\"displayName\") for n in p.get(\"names\",[])])}')
    print(f'  Phone: {json.dumps([ph.get(\"value\") for ph in p.get(\"phoneNumbers\",[])])}')
    print(f'  Email: {json.dumps([e.get(\"value\") for e in p.get(\"emailAddresses\",[])])}')

# Sheets (fallback to gws_auth when SA key unavailable in terminal)
sheets = build_service('sheets', 'v4')
result = sheets.spreadsheets().values().get(spreadsheetId='<sheet_id>', range='<range>').execute()
rows = result.get('values', [])
print(f'Total rows: {len(rows)}')
```

**Note:** `tools.gws_sa` does not exist in this deployment (removed 2026-05-08). Use `gws_auth.build_service` for sheets — the user's OAuth token includes the spreadsheets scope.
