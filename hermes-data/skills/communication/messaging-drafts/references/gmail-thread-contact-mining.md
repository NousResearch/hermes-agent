# Gmail Thread Contact Mining (For People Not In Sheet/People API)

## When to Use

Trigger: user says "I/we have messaged him before" or "he works with [firm X] — find his contact" or "search emails for his number". The person is an external counterparty (advocate, CA, broker, agent) and the standard resolution pipeline (Phase 1 contact_resolver / Phase 2 People API / Phase 3 Drive PDFs) returns nothing.

This is a fourth path: **mining Gmail threads the user was on (To/From/Cc) for the counterparty's other email addresses, phone numbers, and firm context.**

## Why This Path Exists

Both the contact_resolver and People API are scoped to the user's *contacts* — not to the universe of people who have been on email threads. An advocate who was Cc'd on three years of invoice reminders is someone we have *extensive* context on, but they may never have been added as a Google contact.

The Google Contacts sheet (DRAAS internal registry) is even narrower — only business contacts the user explicitly chose to register. External professionals they engage via email are often missing.

## Workflow

### Step 1 — Broad search by firm / person name

```python
from tools.gws_auth import build_service
gmail = build_service("gmail", "v1")

# Search by both person's name AND firm name variants
queries = [
    '"B R Krishna" "Patan"',
    '"B.R. Krishna"',
    'from:krishna "Jain Patan"',       # firm name as user pronounced it
    'from:krishna patan',
    'subject:"Patan"',
    '"Patan Chetty"',                  # user's exact phrasing
    '"Jain Patan Chetty"',
]

seen_ids = set()
for q in queries:
    res = gmail.users().messages().list(userId="me", q=q, maxResults=15).execute()
    for m in res.get("messages", []):
        if m["id"] in seen_ids:
            continue
        seen_ids.add(m["id"])
        # ... process
```

**Try the user's exact phrasing first, then phonetic variants.** A user saying "Jain Patan Chetty" might be mis-hearing "J P A" (J-P-A — common abbreviation for an advocate/CA firm). If the search comes back empty on the literal phrase, search for the actual underlying firm or for the person's name alone.

### Step 2 — Inspect To/From/Cc/Reply-To headers, dedupe by email

For each message found, fetch metadata headers (cheap):

```python
detail = gmail.users().messages().get(
    userId="me", id=m["id"], format="metadata",
    metadataHeaders=["From", "To", "Cc", "Bcc", "Subject", "Date", "Reply-To"]
).execute()
headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
# Then print every email address in From/To/Cc with display name
```

**Extract all unique email addresses and their display names.** This is where you find:
- The person's alternate email aliases (e.g. `br.krishna.advocate@gmail.com` AND `krishna@brklaw.in` — both pointed to the same B R Krishna)
- Their firm association (visible via email domain: `pattanshetti.in`, `brklaw.in`)
- Co-parties on the thread (accountants, partners, paralegals)

### Step 3 — Fetch full message body to find phone numbers

`format="metadata"` only returns headers — you do NOT get the body. To find phone numbers in signatures or footers, fetch `format="full"`:

```python
full = gmail.users().messages().get(userId="me", id=m["id"], format="full").execute()

def decode_body(payload):
    """Recursively walk parts and decode all text/plain and text/html parts."""
    out = []
    if payload.get("body", {}).get("data"):
        import base64
        out.append(base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace"))
    for part in payload.get("parts", []):
        out.append(decode_body(part))
    return "\n".join(out)

body = decode_body(full.get("payload", {}))
```

Then regex-scan for Indian mobile / landline numbers:

```python
import re
# Indian mobile: optional +91 / 91 / 0 prefix, then 10 digits starting 6-9
phones = re.findall(r"(?:\+91[\s-]?|91[\s-]?|0)?[6-9]\d{9}", body)
# Indian landline with STD: 0XX-XXXXXXX or 0XXX-XXXXXX
landlines = re.findall(r"\+?91[\s-]?\d{2,4}[\s-]?\d{6,8}", body)
```

**Phone numbers in Gmail signatures are usually the *firm* number, not the person's direct mobile.** Do not assume `T: +91 80 25593195` in the accountant's footer is the advocate's number. Use it as a fallback if no direct mobile is found, but flag clearly to the user.

### Step 4 — When contact details remain incomplete, ASK the user

After running Steps 1–3 you typically have:
- ✅ Email address(es) — confirmed
- ✅ Firm name + firm email domain — confirmed
- ✅ Co-parties on thread (accountants, partners) — confirmed
- ⚠️ Firm office phone — usually in signature, but NOT the person's direct line
- ❌ Direct mobile — almost never in Gmail unless they personally sent a message

**At this point, stop searching and ask the user.** The user's memory or a single message in the thread is the fastest path forward. Do not invent numbers or use the firm phone as the contact's number.

## Worked Example (June 2026)

User asked for Adv. B R Krishna (associated with "Jain Patan Chetty" — actually **J P A & Associates / Pattanshetti**, a CA + advocate firm in Bengaluru).

| Search hit | Source | What it gave us |
|---|---|---|
| `B.R. Krishna` in fullText | Gmail body of multiple 2025 invoice threads | Confirmed the spelling and that he was on the thread |
| `krishna@brklaw.in` in Cc | Dharmesh's Apr 2026 invoice reply | Personal/work email alias |
| `br.krishna.advocate@gmail.com` in Cc | Pattanshetti accountant's original Nov 2025 invoice | Personal Gmail alias |
| Signature of `accounts@pattanshetti.in` | Same Apr 2026 message full body | Firm office address (#70 Infantry Road, BLR 560 001), firm phone (+91 80 25593195), accountant's mobile (9980530355 — Shobha S, NOT Krishna) |
| Direct mobile | NOT FOUND | Asked user to provide or search WhatsApp history |

The "J P A" mis-hearing trap: the user said "Jain Patan Chetty". The actual firm is **J P A & Associates** (J-P-A — a common abbreviation for Joint Practice of Advocates / Justice, Peace, Advocate patterns in Indian CA firms). Searching for "Jain Patan" alone returns nothing; "J P A Pattanshetti" or just "Pattanshetti" + "Krishna" surfaces the right context.

## Phonetic / Mispronunciation Traps

When the user pronounces a name that returns zero search hits, the cause is often a firm-name or person-name mishearing. Common patterns to try:

| User said | Try these alternatives |
|---|---|
| "Jain Patan Chetty" | "J P A", "JPA", "Pattanshetti", "Patanshetti" |
| "Sundar Iyer" | "Sundaresan Iyer", "Sundararajan Iyer" |
| "Bhandari & Co" | "Bhandary", "Bandari" |
| "Karupaiya" | "Karuppiah", "Karuppan", "Karuppasamy" |

Always run BOTH the literal phrase AND the name alone (`"Krishna"` alone often surfaces a thread where firm context is visible).

## What NOT to Do

- **Do not use `gws_sa`** — the module does not exist (removed 2026-05-08; raises `ModuleNotFoundError`). Use `gws_auth.build_service("gmail", "v1")`.
- **Do not assume firm phone = person's phone.** Always flag in the user-facing reply.
- **Do not synthesize a number from name + city + state.** A wrong number is worse than asking the user.
- **Do not burn a tool call trying `people.people().searchContacts`** if the standard contact-resolver already returned nothing — it means the person is not in the user's Google contacts. The Gmail mining path is the right next step.
- **Do not skip Step 1 if Step 3 returns no phones.** Sometimes headers alone (firm association, co-parties) are enough — the user may recognise the firm and provide the direct number from memory.

## Save the Result to Memory

Once confirmed, add a single-line entry to user memory:

```
• Adv. B R Krishna: krishna@brklaw.in / br.krishna.advocate@gmail.com (BRK Law; w/ J P A Pattanshetti CA, #70 Infantry Rd BLR; office +91-80-25593195).
```

Format per the "Frequently-Contacted Contact Memory Pattern" in the umbrella SKILL.md. Keep the line ≤ 200 chars to stay within memory budget.
