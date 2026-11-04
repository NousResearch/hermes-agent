# Not-Spam Whitelist Cron

Moves legit email from SPAM back to INBOX based on whitelist rules in a Google Sheet. Opposite of the sender-blacklist auto-cleaner.

## Sheet Schema

**Sheet:** `Whitelist` tab in the DRAAS whitelist spreadsheet  
**URL:** `https://docs.google.com/spreadsheets/d/1w8_R0JzfHP1PIdPoCFpqdhDh9TFU0qPqbt3V2vfDyw0/edit`

| Index | Column | Header | Content |
|-------|--------|--------|---------|
| 0 | A | `#` | Row number (not always sequential — gaps from reordering) |
| 1 | B | `Category` | Functional category (Banking, Legal, Medical, Internal, etc.) |
| 2 | C | `From Email / Domain` | Email address OR domain fragment |
| 3 | D | `To Email` | Which of NDR's mailboxes this applies to (not used in matching currently) |
| 4 | E | `Subject Keywords` | Comma-separated keywords for subject-based rules |
| 5 | F | `Content Description` | Human-readable note about what this rule covers |
| 6 | G | `Rule Type` | `exact_from` / `domain_from` / `subject_contains` / `combined` |
| 7 | H | `Date Added` | When the rule was added |
| 8 | I | `Notes` | Voice corrections, context, discovery notes |

**Current count (Oct 2026):** 48 rules across 8 rule numbers (sheet rows 2–49, with gaps from reordering). Categories covering: Banking, Legal, Medical, Internal, Insurance, Architecture, School, Payroll, Government, Venture, Broker, Personal.

## Rule Types & Matching Logic

| Rule Type | Column C Usage | Column E Usage | Match Condition |
|-----------|---------------|----------------|-----------------|
| `exact_from` | Full email (e.g. `apsaraa.sridhar@cms-induslaw.com`) | Ignored | `sender.lower() == from_domain.lower()` |
| `domain_from` | Domain suffix (e.g. `@kotak.bank.in`, `hdfcbank.bank.in`, or sometimes a full email that acts as exact) | Ignored | `sender.lower().endswith(from_domain.lower())` |
| `subject_contains` | Ignored in this mode | Comma-separated keywords | Any keyword (case-insensitive) is a substring of subject |
| `combined` | Domain suffix | Comma-separated keywords | Sender matches domain AND subject contains any keyword |

**Catch-all internal rules** (these are in the sheet as explicit `domain_from` rules):
- `@draas.com` with type `domain_from` (Rule #11)
- `@drahomes.in` with type `domain_from` (Rule #19)

**Note:** There is NO catch-all for `@o3infotech.com` — only a specific NVIDIA-domain rule (Rule #41: `nvidia.com`, domain_from) that happens to target ndr@o3infotech.com as the To address. If @o3infotech.com mail from other senders lands in SPAM, there is no rule covering it — add an explicit domain_from rule if needed.

## Working Script

The canonical script lives at `/tmp/not_spam_check.py` (regenerated each run). Core logic:

```python
from tools.gws_auth import build_service

# Build services (omit service_name for default google-draas, or be explicit)
gmail = build_service('gmail', 'v1')  # defaults to google-draas for ndr
sheets = build_service('sheets', 'v4')

# Read whitelist sheet
result = sheets.spreadsheets().values().get(
    spreadsheetId=SHEET_ID,
    range='Whitelist!A:I'
).execute()

# Fetch SPAM messages — TWO approaches (both verified Oct 2026 on 25-spam scan):

# Approach A: labelIds (direct API param)
spam = gmail.users().messages().list(
    userId='me',
    labelIds=['SPAM'],
    maxResults=200
).execute()

# Approach B: query string (equivalent, verified working)
spam = gmail.users().messages().list(
    userId='me',
    q='in:spam',
    maxResults=200
).execute()

# Get headers with metadataHeaders (VERIFIED working Oct 2026 — contrary to
# earlier uncertainty, Gmail API client accepts this kwarg and returns only
# the requested headers, reducing payload size):
detail = gmail.users().messages().get(
    userId='me', id=msg['id'],
    format='metadata',
    metadataHeaders=['From', 'Subject', 'Date']
).execute()
headers = detail.get('payload', {}).get('headers', [])
sender = next(h['value'] for h in headers if h['name'].lower() == 'from')
subject = next(h['value'] for h in headers if h['name'].lower() == 'subject')

# Extract email from "Name <email>" format
import re
m = re.search(r'<([^>]+)>', sender)
sender_email = m.group(1) if m else sender

# Move matching to INBOX — TWO APPROACHES:

# Approach A: per-message modify (MORE RELIABLE — verified on 63-spam, 4-move run)
# Gmail API batch ops can silently partial-fail (see email skill pitfall #9).
# Per-message is slower (~0.5s each) but each failure is caught immediately.
for msg in match_messages:
    try:
        gmail.users().messages().modify(
            userId='me', id=msg['id'],
            body={
                'addLabelIds': ['INBOX'],
                'removeLabelIds': ['SPAM']
            }
        ).execute()
        moved.append(msg)
    except Exception as e:
        errors.append({'id': msg['id'], 'error': str(e)})

# Approach B: batchModify (FASTER for bulk moves, but test on first few first)
gmail.users().messages().batchModify(
    userId='me',
    body={
        'ids': [m['id'] for m in match_messages],
        'addLabelIds': ['INBOX'],
        'removeLabelIds': ['SPAM']
    }
).execute()
```

## Running

Run as a cron job daily. Establish the session user identity first:

```python
os.environ['HERMES_SESSION_USER_ID'] = os.environ.get('NDR_UID', '7449813913')
```

## Service Name

The Gmail/Sheets service for ndr@draas.com uses `service_name='google-draas'`. For other accounts resolve first:

```python
# Explicit is safe — no guesswork
gmail = build_service('gmail', 'v1', service_name='google-draas')
sheets = build_service('sheets', 'v4', service_name='google-draas')
```

Alternatively, for the session user's default account, omit `service_name` entirely — it resolves automatically to the vault's primary key.

## Pitfalls

1. **`sys.path.insert(0, '/opt/hermes')` is required** before `from tools.gws_auth import build_service` — the `tools` package isn't on the default Python path when running from `/opt/data`.
2. **From header parsing**: the Gmail `From` header is often `"Display Name" <email@domain.com>`. Extract the bracketed email with `<([^>]+)>`, or fall back to the whole string.
3. **batchModify with `body=` syntax**: Use `body={'ids': batch, 'addLabelIds': ['INBOX'], 'removeLabelIds': ['SPAM']}` — NOT keyword-level params. Verified working on batches up to 5 (Oct 2026); larger batches should also work per Gmail API docs.
4. **SPAM API quirk**: Two equivalent fetch approaches: `labelIds=['SPAM']` (more direct) OR `q='in:spam'`. Do NOT use `is:spam` — the Gmail API query syntax uses `in:` prefixes.
5. **Cron context**: no interactive user is present — the script must be fully autonomous. No questions, no confirmations. If nothing to move, emit a clean report with zero counts.
6. **`metadataHeaders` IS supported** — the Gmail Python client accepts `metadataHeaders` with `format='metadata'` and returns only the requested headers (reduces payload for large SPAM sweeps). Verified working Oct 2026 on 25 messages.
7. **Default service**: For NDR's primary ndr@draas.com, calling `build_service('gmail', 'v1')` without `service_name` resolves to `google-draas` automatically. Explicit is safe too: `service_name='google-draas'`.
8. **Catch-all domains**: The whitelist sheet explicitly lists `@draas.com` (Rule #11) and `@drahomes.in` (Rule #19) as `domain_from` rules. There is NO hardcoded catch-all for `@o3infotech.com`.
9. **Most common match pattern is @draas.com sub-address marketing** (verified Oct 2026 run: 5/5 matches were `marketing@draas.com`, `info@draas.com`, `subscription@draas.com`). These are third-party emails (TCS webinars, Internshala posts, Adobe Photoshop updates) routed through DRAAS mail infrastructure — Gmail flags the content as marketing, but the `@draas.com` domain rule catches and restores them. This is the primary false-positive pattern.
10. **`domain_from` rules catch sub-addresses** — a rule for `@draas.com` also catches `marketing@draas.com`, `info@draas.com`, `subscription@draas.com`, etc. A rule targeting a specific sub-address like `subscription@draas.com` with `domain_from` type matches any sender ending in that complete string. This is correct Gmail address semantics — `@draas.com` suffix match covers all sub-addresses.