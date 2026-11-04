# Legal Case Follow-Up — Drafting Pattern

When the user has sent a detailed email about a legal case (RERA, civil, compliance) to a professional (advocate, consultant, auditor) and later wants a follow-up with case developments, strategy changes, and new document attachments.

## Trigger phrases

- "Send a follow-up email to [counsel] — unless he has already responded"
- "Follow up with [advocate] about [case] — share the new orders"
- "Draft a follow-up to [lawyer] — tell them about the HC development"

## Workflow

### 1. Find the original sent email

Search sent items for the specific email thread:

```python
res = gmail.users().messages().list(
    userId='me',
    q='in:sent ([counsel_name] OR [case_number] OR [project_name]) after:[date]'
).execute()
```

Get full content:

```python
msg = gmail.users().messages().get(userId='me', id=msg_id, format='full').execute()
headers = {h['name']: h['value'] for h in msg['payload']['headers']}
```

### 2. Check for replies in the thread

```python
thread = gmail.users().threads().get(userId='me', id=thread_id).execute()
for m in thread['messages']:
    h = {n['name']: n['value'] for n in m['payload']['headers']}
    from_addr = h.get('From', '').lower()
    if 'counsel' in from_addr or 'advocate' in from_addr:
        # Reply exists — report to user
```

The thread may have only 1 message (the user's sent email) — that means no reply.

### 3. Extract the full body text

Sent emails from Gmail often have multi-part MIME structure. The text/plain body is nested inside `multipart/mixed → multipart/alternative → text/plain`. Traverse properly:

```python
import base64

def extract_text(part):
    mime = part.get('mimeType', '')
    body_data = part.get('body', {}).get('data', '')
    if body_data and 'text' in mime:
        return base64.urlsafe_b64decode(body_data).decode('utf-8', errors='replace')
    if 'parts' in part:
        for p in part['parts']:
            result = extract_text(p)
            if result:
                return result
    return ''
```

### 4. Preserve the original To/Cc list

The follow-up should go to the SAME recipients as the original. Extract them:

```python
orig_to = headers.get('To', '')
orig_cc = headers.get('Cc', '')
```

### 5. Compose the follow-up draft

Structure of a legal case follow-up:

| Element | Content |
|---------|---------|
| **Salutation** | "Dear [Name]," |
| **Acknowledge prior** | "Following up on my email of [date] regarding [case name/case number]." |
| **New development** | "Since then, [new event: HC mandamus, Thasildar action, etc.]" |
| **Strategy update** | User's proposed strategy: "I believe we should first [step A], then [step B]." |
| **Call to action** | Request to review new docs + schedule call to discuss |
| **Attachments** | New order copies being sent |
| **Sign-off** | "Regards," / "Warmly," |

Tone: professional, urgent but respectful. Acknowledge the counsel may have been busy. Include a clear request for a call/meeting to discuss strategy verbally.

### 6. Attach new documents

Use the direct MIME multipart pattern to attach PDFs:

```python
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

msg = MIMEMultipart()
msg['To'] = orig_to
msg['Cc'] = orig_cc
msg['Subject'] = f"Re: {orig_subject}"
msg['In-Reply-To'] = orig_message_id
msg['References'] = orig_message_id

body = MIMEText(followup_body_text)
msg.attach(body)

for pdf_path in pdf_paths:
    with open(pdf_path, 'rb') as f:
        part = MIMEBase('application', 'pdf')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(pdf_path))
        msg.attach(part)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = gmail.users().drafts().create(
    userId='me',
    body={'message': {'raw': raw, 'threadId': thread_id}}
).execute()
```

### 7. Verify

Check the draft landed in Gmail Drafts (not Sent) with correct To/Cc:

```python
draft_msg = gmail.users().drafts().get(userId='me', id=draft['id'], format='full').execute()
draft_headers = {h['name']: h['value'] for h in draft_msg['message']['payload']['headers']}
print(f"To: {draft_headers.get('To')}")
print(f"Cc: {draft_headers.get('Cc')}")
```

## Worked Example: Sabyasachi RERA Follow-Up to Vinay (Sep 2026)

**Original email sent 7 Sep 2026:**
- To: Vinay T <vinay@vnv.ca>
- Cc: Prakash Singh, Manish Ranka, Mamta Rathod
- Subject: "Sabyasachi Behera - RERA Case CMP/200628/0006093: Need Advice on Appeal & Stay"
- Body: Detailed background (Mirabilis/KPDL/Dinesh Ranka), RERA judgment, execution notice, Tahsildar recovery notice for Rs 26,63,278/-

**No reply received as of 8 Sep 2026**

**Follow-up needed with:**
1. New development: High Court issued a writ of mandamus because the Thasildar sat on the attachment proceedings
2. Proposed strategy from NDR:
   - First go to court and get a stay
   - Then challenge the entire order before the RERA Appellate Tribunal
   - They never had a chance to represent themselves (post-Dinesh Ranka, no one appeared)
3. New documents: Kannada order copies (to be translated/attached)
4. Request for a call to discuss strategy

## Pitfalls

- **Body under nested MIME parts**: The `payload.body.data` at the top level is often empty. Traverse `payload.parts` → find `text/plain` inside `multipart/alternative`.
- **Thread contains only the sent message**: No reply yet — report "No reply from [counsel]"
- **draft_reply_create vs direct MIME**: When the follow-up needs to go to the SAME To/Cc as the original (NOT just the last sender), use the direct MIME pattern with explicit To/Cc headers. `draft_reply_create` auto-populates To from the message's sender field alone.
- **Attachments already sent**: Don't re-attach docs already sent in the original email. Only attach NEW documents provided now.