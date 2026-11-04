# Contact-from-Gmail-Signature Lookup

When Google Contacts `searchContacts()` returns zero results for a person the user wants to WhatsApp, the person's phone number may still be recoverable from their **email signature** in a past Gmail thread.

## Trigger

- `searchContacts(query="Mohammad Sadiq Ali")` → empty results
- `searchContacts(query="Sadeq")` → empty results
- The user says "check my emails, he's a friend of mine" or similar

## Procedure

### Step 0: Check past sessions first (fastest fallback)

Before searching Gmail (which can take multiple API calls), try `session_search` — past conversations often have the person's details from prior contact lookups or WhatsApp messages:

```python
# Search past sessions for the person's name + "phone" or "contact"
session_search(query='"Prakash Singh" phone', limit=3)
# Or by broader context
session_search(query='Prakash Singh contact number', limit=3)
```

Session search returns message snippets that may contain the phone number from an earlier contact lookup (Google Contacts, email signature extraction, or the user's own dictation). This is the **cheapest** fallback — no API calls needed.

### Step 1: Find the person in Gmail

If session search finds nothing, search their name (and phonetic/context variants) in Gmail using `gws_skill_bridge.call('gmail_search', ...)`:

```python
from tools.gws_skill_bridge import call
import json

results = call('gmail_search', query='"Sadiq Ali" OR "Mohamed Sadeq"', max=5)
data = json.loads(results) if isinstance(results, str) else results
```

**Key search strategies — try in this order:**

| Strategy | Example query | When to use |
|---|---|---|
| Exact name | `"Sadiq Ali"` | User gave a clear name |
| Phonetic variant | `"Sadeq"` OR `"Sadiq"` | STT-garbled name (Sadiq↔Sadeq) |
| Company name from voice context | `AccurKardia` OR `"Root NYC"` | User mentioned where they work |
| Email domain | `@root-nyc.com` OR `@accurkardia.com` | You found the domain from a prior search |
| Colleague + context | `"mali@accurkardia.com"` | You know one of their emails from a past thread |
| Project + partial name | `"Vishwas Intro" Sadeq` | User mentioned context of prior interaction |

### Step 2: Identify an email FROM the person

In the results, look for records where the `from` field matches the person's name:

```json
{
  "id": "19053b210cb5acdc",
  "from": "Mohamed Sadeq Ali <mali@root-nyc.com>",
  ...
}
```

Note: the Gmail API's `gmail_search` returns **headers only** (From, To, Subject, date, snippet). You need a separate `gmail_get` to extract the body with the signature.

### Step 3: Get the full body to extract the phone number

Use `gmail_get` with the message_id:

```python
res = call('gmail_get', message_id='19053b210cb5acdc', format='full')
data = json.loads(res) if isinstance(res, str) else res
body = data.get('body', '')
```

**Parameter gotcha:** `gmail_get` expects `message_id=`, not `id=`. Passing `id=` raises `AttributeError: 'types.SimpleNamespace' object has no attribute 'message_id'`.

### Return format of gmail_get

The `gws_skill_bridge.call('gmail_get', ...)` return value is a **flat dict**, not the raw Gmail API response. Known fields:

```python
{
    "headers": {
        "From": "...",
        "To": "...",
        "Subject": "...",
        "Date": "..."
    },
    "body": "Full decoded email body text\n\n...signature with phone numbers..."
}
```

**Caveats:**
- The `headers` dict may return **empty strings** for all fields in some cases — the headers are available from the earlier `gmail_search` result, so you should already have `From`/`To`/`Subject` before calling `gmail_get`.
- The `body` field contains the **decoded plain text** (both text/plain and extracted text from text/html parts). It is NOT base64-encoded — the skill bridge decodes it automatically.
- If the body is empty, the email may have been multipart and only the HTML part was preserved. In that case, scan the `snippet` from the gmail_search result instead.

### Step 4: Extract the phone number from the signature

Email signatures typically sit at the bottom of the body with a standard format:

```
Sincerely,
Sadeq
*Managing Director*
Root NYC
+44 7704 135902 (UK Mobile)
+1 201 805 6578 (US Mobile/WhatsApp)
+91 73587 89324 (India Mobile)
```

Scan the body text for:
- Lines starting with `+` (country code prefix)
- Look for "WhatsApp" in the label — that's the number to use in the wa.me link
- If no WhatsApp label, prefer the mobile/cell number over landline

### Step 5: Phone number selection rules

When multiple numbers are present:

| Label | Priority | Reason |
|---|---|---|
| "WhatsApp" | Highest | Explicitly listed for WhatsApp |
| "Mobile", "Cell" | High | Likely WhatsApp-enabled in India |
| "US Mobile", "UK Mobile" | Medium | May work on WhatsApp |
| Landline, Office | Low | Typically not on WhatsApp |
| India mobile (+91) | High | NDR's contacts are India-based |

### Step 6: Call the WhatsApp tool

```python
# Use the WhatsApp-specific number from the signature
whatsapp_link(phone="+12018056578", text=message)
```

## Pitfalls

### P1. The gmail_search result may only show people the user emailed, not the other way around

Gmail search covers both Inbox and Sent. However, the email from the target person may have been **archived**, **deleted**, or be **very old** (years ago). Extend the search with `newer_than:` or search by domain name to increase recall.

### P2. The email body may be base64-encoded

`gmail_get` with `format='full'` returns the body decoded as plain text in the `body` field. But if the email was sent as multipart/alternative with HTML/text parts, the body returned may be the HTML version. Fall back to scanning the textual parts or looking for numeric patterns (`\+?\d{7,15}`).

### P3. The person's email display name may differ from how the user refers to them

The user says "Mohammad Sadiq Ali" or "Mali" — but the contact's email display name is "Mohamed Sadeq Ali" (spelling variation). Search by **first-name fragment + company domain** rather than expecting an exact match.

### P4. The gws_skill_bridge call() parameter names are irregular

Discovered parameter mapping:
| Operation | kwarg name | Example |
|---|---|---|
| `gmail_search` | `query`, `max` | `call('gmail_search', query='Sadiq', max=5)` |
| `gmail_get` | `message_id`, `format` | `call('gmail_get', message_id='...', format='full')` |
| `contacts_list` | `query`, `max` | `call('contacts_list', query='Prakash', max=5)` |

Using `max_results` or `id` instead of `max` or `message_id` will raise `AttributeError`.

### P5. Not all email signatures include a phone number

Some people's signatures are just a name and company. If no phone number is found in any email body, tell the user and suggest they share the contact's number or ask the person directly.

## Firm-name resolution from a voice-garbled name (25-08-2026)

When the user asks to "check my past emails with [garbled person/firm]" — e.g. **"architect Balan and Nambisar"** — the STT word is often a mangling of a proper noun (here the firm **Balan + Nambisan Architects / BN Architects**):

1. **Grab the project context the user gives.** "Ranka Oasis proposals" pins the search. Search Gmail for the subject line / context phrase (e.g. `Balan Oasis`, `Nambiar Oasis`, `architect Balan`) rather than the exact garbled string.
2. **The user may have drafted the email to the firm but not sent it from that exact name.** A "narration" email (e.g. subject `Email for Balan and Amritsar... my total narration`) confirms the intended firm name verbatim — "the architecture firm Balan and Nambisar". This is the ground truth that de-garbles the STT.
3. **Find the firm's email domain** from the actual thread. Search `from:<domain-suffix>` / `bnarchitects` to list everyone on the firm's side (principals + associates CC'd).
4. **Read the sender's signature block** — it carries the full contact card: office address, landline (`Ph: +91-80-25217543 / 44`), website, Instagram, and the named principals. Signature blocks from these firms list the whole team, giving you every contact email in one shot (e.g. Janice Rodrigues / Arjun Nambisan / Arun Balan / Suresh / Jeevan, all `@bnarchitects.co.in`).
5. **Identify the main point of contact** as the person who authored the scope-of-work / proposal email (usually an Associate covering the deal), and the principals to CC.
6. **Confirm not-in-contacts**: cross-check the people/firm against the DRA contacts sheet + People API across all three vault accounts (they're often absent — Gmail signatures are the only source). Note that only the landline is on file, not partner mobiles, unless the signature shows them.
