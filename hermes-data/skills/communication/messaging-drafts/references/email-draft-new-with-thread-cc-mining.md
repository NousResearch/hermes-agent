# Email Draft — New Email with Full CC List Mined from Thread

## When to Use

The user has an existing email thread with multiple participants about a topic. They now want to send a **new email** (not a reply) to a **new recipient** about the same subject, and want **everyone from the original thread CC'd** so they stay in the loop.

**Real example (Jun 2026):** Nishant had been emailing Nayana AS, Kalpana M, Mohd Ehtesham, Livish J, Manohar Sur, and Prashant Chaudhari at Bajaj Life Insurance about policy revival (thread Subject: "Re: Revival request Policy No - 0444146783"). A new person (rohit.sundarka@bajajlife.com) asked him to email his clarifications directly. Nishant wanted the email sent to the new person with everyone from the existing thread CC'd.

## Workflow

### Step 1: Present the Opening Brief

Before drafting, provide a concise opening brief covering:
- **Subject/policy matter** (what this is about)
- **Current status** (what's happened so far in the thread)
- **What was requested** (the new recipient asked for this)
- **Key background facts** (relevant dates, amounts, documents submitted)

This brief goes in your Telegram response — NOT in the email itself.

### Step 2: Identify All Thread Participants

```python
results = gmail.users().messages().list(
    userId='me',
    q='"Policy No - XXXX"',  # unique thread identifier
    maxResults=20
).execute()

all_emails = set()
for m in results.get('messages', []):
    msg = gmail.users().messages().get(
        userId='me', id=m['id'], format='metadata',
        metadataHeaders=['From','To','Cc','Subject','Date']
    ).execute()
    headers = {h['name']: h['value'] for h in msg['payload']['headers']}
    
    # Parse From, To, Cc
    for field in ['From','To','Cc']:
        val = headers.get(field, '')
        if val:
            for addr in val.split(','):
                all_emails.add(addr.strip())
```

### Step 3: Build the CC List

- Include everyone from the thread EXCEPT:
  - The new To: recipient (they're already in To)
  - The sender themselves (Nishant)
- Remove duplicates (same person appearing in different roles across messages)
- Format as comma-separated for the CC header

### Step 4: Compose and Save as Draft

Use `gmail.users().drafts().create()` — NOT send. Use the email body text the user provided verbatim, with these adjustments:
- Replace placeholders like "Customer Name" with the actual sender name
- Add identifying info (policy number, reference IDs) to the subject line

```python
message_text = f'''Content-Type: text/plain; charset=utf-8
MIME-Version: 1.0
From: Nishant Ranka <ndr@draas.com>
To: {to_recipient}
Cc: {cc_list}
Subject: {subject}

{body}'''

encoded = base64.urlsafe_b64encode(message_text.encode('utf-8')).decode('utf-8')
draft = gmail.users().drafts().create(
    userId='me',
    body={'message': {'raw': encoded}}
).execute()
```

### Step 5: Report Back

Tell the user:
- Draft saved (Draft ID)
- To: recipient
- CC: list (all names)
- Subject
- Where to find it (Gmail → Drafts folder)

## Variant — REPLY-ALL on an existing thread when the last message is narrow

Verified 2026-09-21 (Ranka Oasis / "DRA - Jiraaf" thread, external investor + advisor).

Do NOT take the To/Cc of only the LAST message in the thread. On external threads the last
message is often a one-to-one reply (e.g. Prakash → Vishwas + NDR only), and replying to that
silently drops the rest of the deal team. Instead:

1. `threads().get(userId='me', id=THREAD_ID, format='metadata', metadataHeaders=['From','To','Cc','Date'])`
   → walk EVERY message, union the To+Cc addresses. Exclude NDR's own aliases
   (ndr@draas.com, ndr@drahomes.in) and any no-reply addresses.
2. Take `Message-ID` from the message you are replying to, and `References` from that same
   message plus its own Message-ID appended — this is what keeps the draft correctly threaded.
3. Create with BOTH `raw` and `threadId`:
   `drafts().create(userId='me', body={'message': {'raw': raw, 'threadId': THREAD_ID}})`
   The `threadId` in the message body is what lands the draft in the right conversation.
4. Verify with `drafts().get(format='full')`: To, Cc, In-Reply-To, and `threadId` == original.

Recipient carrying: the user says "reply all" — that means union the thread, then state the
final To/Cc back to them explicitly so they can eyeball who is on it before sending.

## Pitfalls

- **Same person, different email aliases:** Nishant may appear as ndr@draas.com and ndr@drahomes.in in the same thread. Only include others, not his aliases.
- **No-reply addresses:** Skip DONOTREPLY@... addresses — they're not human recipients.
- **Thread is long (6+ months):** Check the most recent 5-10 messages for active participants. Old participants from 2023 may no longer be relevant.
- **New recipient may already be in the thread:** If the new recipient was CC'd on an earlier exchange, they might not need to be in the To field — a Reply All would work instead. Check before drafting a new email.
- **User provides body text with placeholders:** Always replace "Customer Name", "[Name]", etc. with the actual person's name as they sign their emails.
