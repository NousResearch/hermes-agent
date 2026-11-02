# Legal Case Email Thread Tracing

Trace a specific legal case's email thread by case reference number, identify which law firm member has/hasn't responded, and report the latest status. Commonly requested for cases like CMA No. 742/2026, WP No. 20411/2023, etc.

## Trigger

User says something like:
- "Find the email chain for [case number / party name] and check if [person] has responded"
- "I sent an email to [firm] about [matter] — did they reply?"
- "Follow up on the [case name] matter — has there been any update from [lawyer's name]?"

## Stage 1 — Identify the Case and Participants

From the user's voice, identify:
- **Case reference**: CMA No. 742, WP 20411, etc. — use this as the primary search term
- **Law firm / counsel name**: CMS IndusLaw, Ashlar Law, etc.
- **Specific person to check**: "Jee Vivekananda" (G Vivekanand / Vivek G), "Apsaraa", etc.
- **Opposing party**: "Pawan Kumar", "Saveganapalli Land Partners", etc.

The user often garbles names in voice. Known mapping:
| Voice | Actual |
|-------|--------|
| "Jee Vivekananda" | G Vivekanand / Vivek G — vivek.g@cms-induslaw.com |
| "Rite petition" | Writ petition |
| "Indus Law" | CMS IndusLaw (cms-induslaw.com domain) |

## Stage 2 — Search Gmail

Search broadly with the case reference number and party name:

```python
from tools.gws_auth import build_service

gmail = build_service('gmail', 'v1', service_name='google-draas')

# Primary search — case number + party
results = gmail.users().messages().list(
    userId='me',
    q='CMA No. 742 Pawan Kumar',  # adapt per case
    maxResults=10
).execute()
```

If that returns nothing, broaden:
- Search just the case number: `'CMA 742'`
- Search the law firm domain: `'cms-induslaw.com CMA'`
- Search the lawyer name: `'Vivekanand'` or `'vivek.g@'`

## Stage 3 — Map All Messages in the Thread

Find the thread ID from any returned message, then get all messages:

```python
thread = gmail.users().threads().get(
    userId='me', id=THREAD_ID,
    format='metadata',
    metadataHeaders=['From','To','Cc','Subject','Date']
).execute()

for msg in thread['messages']:
    hdrs = {h['name']: h['value'] for h in msg['payload']['headers']}
```

Present a chronological list with sender, date, and subject.

## Stage 4 — Identify Response Status per Participant

For each named participant (e.g. "G Vivekanand", "Apsaraa"):
- Did they send any message in the thread?
- When was their last message?
- Has the user sent a follow-up that went unanswered?

**Common pattern:** The user may have CC'd a senior partner (Vivekanand) but only received a reply from an associate (Apsaraa). Report this distinction clearly:
> "Apsaraa Sridhar (Associate) responded on 10 July with a status update. G Vivekanand (Partner) has not personally responded since 3 June 2026."

## Stage 5 — Extract the Latest Substantive Update

If a response exists, extract the key information from the body:

```python
def extract_plain_text(msg):
    """Walk payload parts to find text/plain body."""
    def _walk(parts):
        for p in parts:
            if p.get('mimeType') == 'text/plain' and p.get('body', {}).get('data'):
                import base64
                return base64.urlsafe_b64decode(
                    p['body']['data'] + '=='
                ).decode('utf-8', errors='replace')
            if 'parts' in p:
                result = _walk(p['parts'])
                if result:
                    return result
        return None

if 'parts' in msg['payload']:
    return _walk(msg['payload']['parts'])
elif msg['payload'].get('body', {}).get('data'):
    import base64
    return base64.urlsafe_b64decode(
        msg['payload']['body']['data'] + '=='
    ).decode('utf-8', errors='replace')
```

Summarise the update concisely — bullet points, one per key fact.

## Stage 6 — Report to User

Format:
1. **Thread found** — subject line, date range, participant list
2. **Response status per person** — who responded, who didn't
3. **Latest substantive update** — bullet points from the last response
4. **Gap / next step** — what's still pending (e.g. "They said they're working on strategy but haven't shared it yet")

## RERA Karnataka (K-RERA) Complaint Tracing — Special Pattern

Karnataka RERA (K-RERA) cases follow a different email pattern than litigation threads. The complainant files under the CMP/XXXXXXXX/XXXXXXX format, and all communication is **system-generated** from `info.rera@karnataka.gov.in` (or `krera-chairman@karnataka.gov.in` for execution notices) — not from an advocate.

### Key Differences from Litigation Threads

| Feature | Litigation (advocate) | K-RERA Complaint |
|---------|----------------------|------------------|
| Emails from | Individual advocates at law firms | `info.rera@karnataka.gov.in` (system auto-generated) |
| Case format | CMA No. 742/2026, WP 20411/2023 | CMP/200628/0006093 |
| Attachments | IA applications, affidavits as PDF | Judgement PDF only (others are body-text orders) |
| Court orders | In attached documents | **Embedded in email body** as plain text |
| Multiple cases | Separate threads | Multiple complaint numbers heard together (e.g., 5528, 4973, 5887, 6093) |
| Response | Advocate replies in thread | System email — do NOT reply (stated in every email footer) |

### Trigger Phrases

User says something like:
- "RERA related emails for client X"
- "Find RERA notices and court orders for [name]"
- "Collect all documents for this RERA case"
- "[name] RERA complaint"

### Stage 1 — Voice-to-Name Resolution

The user's voice transcription often mangles the client name badly. Do NOT search the verbatim spoken name first — resolve it first:

```
contact_resolver(query="Subya Sachi")  → returns "Sabyasachi Behera"
```

Known K-RERA mappings encountered:
| Voice Says | Actual Name |
|------------|-------------|
| "Subya Sachi" / "Subha Sachi" | Sabyasachi Behera (sabyasachi.basis@gmail.com) |

Once you have the correct name, search Gmail with the corrected spelling AND the phonetic voice variants (Gmail's search seems to handle some variants independently).

### Stage 2 — Search Gmail by Multiple Axes

K-RERA emails are NOT threaded together into a single thread — each hearing notice/order is its own **separate thread** with a single message. You must find ALL of them:

```python
from tools.gws_auth import build_service
gmail = build_service('gmail', 'v1', service_name='google-draas')

# Primary search — complaint number
results = gmail.users().messages().list(
    userId='me',
    q='CMP/200628/0006093',  # the complaint number
    maxResults=25
).execute()
```

**Axes to search** (try all, deduplicate):
1. Complaint number: `CMP/200628/0006093`
2. Party name (correct spelling): `Sabyasachi`
3. Party email domain: `sabyasachi.basis@gmail.com`
4. Project name: `Mirabilis`
5. Phone variant: `Subya`, `Sachi`, `Subhya`, etc.

K-RERA sends from two addresses:
- `info.rera@karnataka.gov.in` — standard hearing notices, orders, compliance requests
- `krera-chairman@karnataka.gov.in` — execution notices (escalated non-compliance)

### Stage 3 — List All Emails Chronologically

Since each email is its own thread, iterate through all returned messages and order by date:

```python
for m in msgs.get('messages', []):
    msg = gmail.users().messages().get(
        userId='me', id=m['id'],
        format='metadata',
        metadataHeaders=['From','Subject','Date']
    ).execute()
    hdrs = {h['name']: h['value'] for h in msg['payload']['headers']}
```

The email **Subject line encodes the document type** — no need to open the body to categorize:
- `Orders of the Court for Complaint number CMP/...dated DD/MM/YYYY` → hearing outcome
- `RERA Karnataka : Enquiry on Complaint Application No ... Dated : YYYY-MM-DD Time : HH:MM` → hearing notice
- `RERA Karnataka : Judgement for complaint no CMP/...` → **judgement PDF attached** (key document)
- `Submit your Compliance Report for Complaint Number CMP/...` → post-judgement compliance demand
- `Execution Notice on CMP/...` → escalation to Chairman level
- `Submit your willingness to refer your Complaint Number ... to Lok-Adalat` → settlement opportunity

### Stage 4 — Extract Court Orders from Email Body

Court orders are in the **email body as plain text**, not in attached PDFs (except the final judgement). Extract with:

```python
import base64

def get_text(payload):
    text = ""
    if isinstance(payload, dict):
        if payload.get('mimeType') == 'text/plain' and payload.get('body',{}).get('data'):
            text = base64.urlsafe_b64decode(payload['body']['data'] + '==').decode('utf-8',errors='replace')
        if 'parts' in payload:
            for sub in payload['parts']:
                text += get_text(sub)
    return text

body = get_text(msg['payload'])
```

Key fields to extract from each order:
- **Attendance**: who was present/absent (complainant, respondent, respondent advocate)
- **Prayer**: what the complainant is asking for (e.g., "execution of sale deed and interest on delay period")
- **Status**: "Posted for orders", "Final chance given", "Memo of calculations filed"
- **Financial details**: apartment cost, amount paid (extract exact figures)
- **Direction**: what the court ordered the respondent to do and by when
- **Related complaint numbers**: noted in the same order (multiple cases often heard together)

### Stage 5 — Download Judgement PDF

Only the final judgement email has a PDF attachment. Check for attachments:

```python
def find_attachments(payload, results=None):
    if results is None:
        results = []
    if isinstance(payload, dict):
        fn = payload.get('filename', '')
        if fn and payload.get('body', {}).get('attachmentId'):
            results.append({
                'filename': fn,
                'mimeType': payload.get('mimeType', ''),
                'attachmentId': payload['body']['attachmentId'],
                'size': payload['body'].get('size', 0),
            })
        if 'parts' in payload:
            for sub in payload['parts']:
                find_attachments(sub, results)
    return results
```

### Stage 6 — Search Drive

The shared Drive folder for K-RERA documents is typically **"RERA Complaints & Orders"**. Search Drive for case-specific files:

```python
for q in ["Sabyasachi", "6093", "Mirabilis"]:
    results = drive.files().list(
        q=f"name contains '{q}' and trashed=false",
        fields="files(id, name, parents)",
        pageSize=10
    ).execute()
```

Drive naming convention for RERA case files:
`YYYYMMDD Project Tahsildar/Recovery/Notice PartyName_English/original.pdf`
- `20260601 Mirabilis KPDL Tahsildar Recovery Notice SabyasachiBehera_English.pdf`
- `20260601 Mirabilis KPDL Tahsildar Recovery Notice SabyasachiBehera_original.pdf`

### Stage 7 — Report Structure

Present the complete timeline in chronological order. Group by document type:

1. **Case Overview**: complaint number, project, parties, apartment/unit details
2. **Timeline** — table of date, event, key observation from each order
3. **Status** — latest order, what was directed, whether complied
4. **Documents Available** — both from email (judgement PDF) and from Drive (tahsildar notices, etc.)
5. **Gap / Exposure** — execution notice pending, non-compliance risk, next hearing date

### Pitfalls

### P10. Voice transcription may produce ZERO Gmail results initially
The user said "Subya Sachi" — searching that exact phrase returned only old unrelated emails (2015-era). The correct spelling "Sabyasachi Behera" returned 20+ RERA-related results. Always resolve the contact name through `contact_resolver` FIRST, then search by the resolved name.

### P11. K-RERA emails are NOT threaded together
Unlike a litigation thread where all messages share one threadId, K-RERA sends each hearing notice/order as its own singleton thread. You must query all messages matching the complaint number and sort them manually by date. Do not try to follow a single thread.

### P12. Subject lines are very long and contain the hearing date twice
K-RERA subjects follow a pattern like `Orders of the Court for Complaint number CMP/200628/0006093 dated 30/08/2022` where the date in the subject is the hearing date cited, not the email send date. Always compare the email's own `Date:` header with the subject date — they often differ (e.g., order for 30/08/2022 sent on 05/09/2022). Report the send date, reference the order date.

### P13. Multiple complaint numbers are heard in the same hearing
Orders frequently cover 3-7 complaint numbers together (e.g., "Complainants in 7738, 5853, 5887 and 6093 are present"). Present all related complaint numbers in your report — the user may not know which others are linked.

### P14. The complainant's own emails (to CRM) are on separate threads
The complainant may have emailed your CRM team directly about possession delay, sale deed drafts, and registration (often CC'd to NDR). These are on entirely separate threads with subject lines like "Registration B1 203" or "Possession or Handover of Flat B1-203" — search for these too to get the full background story. They pre-date the RERA complaint and show the failed attempts at resolution before the case was filed.

## Known Participants (CMS IndusLaw — Saveganapalli/Pawan Kumar matter)

| Role | Name | Email |
|------|------|-------|
| Partner | G Vivekanand (Vivek G) | vivek.g@cms-induslaw.com |
| Associate | Apsaraa Sridhar | apsaraa.sridhar@cms-induslaw.com |

## Pitfalls

### P1. The user says "Jee" but the actual name is "G Vivekanand"

The user pronounces "G Vivekanand" as "Jee Vivekananda" — both the initial "G" (hard to hear) and the spellings differ. Search for both `vivek.g@` and the full name `Vivekanand` across cms-induslaw.com domain.

### P2. Case participants may have moved firms

The engagement may have started with one lawyer at one firm, then shifted to another firm or another lawyer at the same firm. Check multiple threads:
- Thread 1: Initial engagement (e.g. with G Vivekanand in June)
- Thread 2: Active handling (e.g. with Apsaraa in July)

The user's understanding of "who is handling this" may be outdated — the thread will show the actual current handler.

### P3. "Rite petition" = "Writ petition"

The user's voice transcription regularly renders "writ petition" as "rite petition" or "right petition". Always search both.

### P4. Voice says "Echo Green" → actual is "Aqua Green"

The user's voice regularly renders "Aqua" (as in Ranka Aqua Green) as "Echo" or "Aqua Green" → "Echo Green". Always search for both `Aqua` and `Echo` variants when the user references a project name containing "Aqua" via voice memo.

## Stage 7 — Download Attachments from the Latest Email

When the thread contains a new court filing (IA, application, affidavit) as an attachment:

```python
# Find the latest message with an attachment
msg = gmail.users().messages().get(userId='me', id=MSG_ID, format='full').execute()

def find_attachment_part(payload, target_name=None):
    """Walk payload parts to find PDF attachments."""
    if 'parts' in payload:
        for p in payload['parts']:
            fn = p.get('filename', '')
            if fn and p.get('body', {}).get('attachmentId'):
                if not target_name or target_name.lower() in fn.lower():
                    return p
            r = find_attachment_part(p, target_name)
            if r:
                return r
    return None

att_part = find_attachment_part(msg['payload'])
if att_part:
    att_id = att_part['body']['attachmentId']
    fn = att_part['filename']
    attachment = gmail.users().messages().attachments().get(
        userId='me', messageId=MSG_ID, id=att_id).execute()
    data = base64.urlsafe_b64decode(attachment['data'] + '==')
    with open(f'/tmp/{fn}', 'wb') as f:
        f.write(data)
```

Check for PDF attachments in ALL messages, not just the latest — sometimes the attachment email is the second-to-last (forwarded within the thread). Use `format='full'` and walk `parts` recursively.

## Stage 8 — OCR Scanned PDFs

Many court filings are scanned images, not text-layer PDFs:

```bash
# Check page count and if text exists
pdfinfo <file.pdf> | grep Pages
pdftotext <file.pdf> - | head -20  # empty output = scanned

# OCR key pages (first 10 for structure, last 10 for prayer/order)
for p in $(seq 1 10); do
    pdftoppm -f $p -l $p -r 200 -png <file.pdf> /tmp/ocr/page
    tesseract /tmp/ocr/page-$p.png /tmp/ocr/page$p 2>/dev/null
    echo "=== PAGE $p ==="
    cat /tmp/ocr/page$p.txt
done
```

For 92-page-plus documents, OCR the opening pages (title, background, prayer) and the last pages (prayer, affidavits in support). The middle pages are usually annexures/exhibits and can be sampled selectively.

## Stage 9 — Identify the Correct Drive Folder and File

DRAAS litigation documents follow two filing conventions depending on the folder:

| Folder Type | Convention | Example |
|---|---|---|
| **Litigation case folder** (e.g. "RRP vs SPV", OS 553) | Numbered-prefix `NN_Description.pdf` | `32_Orders_IA_No5to7_15Feb2025.pdf` |
| **Consumer Forum Cases / NCDRC folder** (RAQ matters) | YYYYMMDD naming | `20260819_Ranka_Aqua_Green_IA_9783_2026_Complainant_Amendment.pdf` |

**Folder discovery ladder:**
1. Search Drive for the case number: `"CC. No. 1230/2016"` or `"CCNo 1230-2016"`
2. Check under `Legal > Consumer Forum Cases > NCDRC` for NCDRC matters
3. Check under `Current Properties > RAQ > RankaAquagreens > RAQ Legal` for property-level RAQ legal docs
4. For a new IA/application filed at NCDRC, the **NCDRC folder** under Consumer Forum Cases is the correct location

**Document naming for NCDRC filings:**
`YYYYMMDD_Project_IA_No_Year_Description.pdf`
- `20260819_Ranka_Aqua_Green_IA_9783_2026_Complainant_Amendment.pdf`

Upload via Drive API with `MediaFileUpload` and verify the parent chain after upload.

## Stage 10 — Analyze the Legal Document and Draft Strategy Email

After OCR'ing the document, produce a structured analysis covering:

1. **What the opponent is saying** — their claims, arguments, and legal basis
2. **What the court has already observed** — extract from the latest hearing report email (not just the IA)
3. **Our rebuttal positions** — identify the key grounds of attack
4. **Gaps / questions for the advocate** — specific technical legal questions (estoppel, limitation, res judicata, maintainability)

Then draft a threaded email reply to the advocate with:
- **To:** the handling advocate (e.g. Harshavardhan Kotla)
- **CC:** the associate + other counsel on the thread
- **Subject:** `Re: [original subject]` (preserves threading)
- **Body:** structured sections: analysis of the IA, key observations, specific questions on rebuttal strategy, legal pathway, next steps, costs

Build the email as a MIME draft (NEVER send directly):

```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

msg = MIMEMultipart('alternative')
msg['To'] = 'Advocate <advocate@lawfirm.com>'
msg['Subject'] = f'Re: {orig_subject}'
msg['In-Reply-To'] = orig_msg_id
msg['References'] = orig_msg_id
msg['Cc'] = 'associate@lawfirm.com, other.counsel@lawfirm.com'
msg.attach(MIMEText(body, 'plain'))

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = gmail.users().drafts().create(
    userId='me',
    body={'message': {'raw': raw, 'threadId': thread_id}}
).execute()
```

Verify the draft's To/Cc/Subject by reading it back with `drafts().get(format='full')`.

## Pitfalls — Stages 7-10

### P5. The attachment filename from Gmail is often garbage
Gmail attachment filenames often have leading spaces, odd characters, or are truncated (` I.A. 9783:2026pdf.pdf`). Always rename the file to the proper YYYYMMDD convention before uploading to Drive — do NOT use the Gmail filename as-is.

### P6. Scanned PDFs can have blank first pages
Many scanned court filings start with a blank page (the unprinted cover sheet). If OCR returns empty for page 1, check page 2+. The title page may be page 3 onwards.

### P7. The email body (hearing report) is often more insightful than the IA itself
The advocate's covering email (Stage 7) contains the court's oral observations and the actual hearing outcome — it may be more strategically valuable than the IA document. Always extract and read the email body text from the same message AND the previous messages in the thread. The hearing report is typically in the second-to-last message (sent before the IA attachment message).

### P8. Thread is long (25-30+ messages) — read the tail, not the head
For legal threads spanning 2+ years, the key messages are the last 5-10 (latest hearing reports, latest IA filings) and the first 2-3 (original complaint, initial strategy). Middle messages are procedural (affidavit drafts, notary instructions, cost discussions). Don't waste context reading every message.

### P9. NDR prefers concise call-focused strategy emails, not detailed questionnaires
When drafting a legal strategy email to counsel, NDR's preference (corrected Aug 2026 on the Ranka Aqua Green matter) is:
- **Short body** — 2-3 paragraphs: confirm the IA was reviewed, note the key court observation, confirm the IA is filed on Drive, then ask for a call
- **No detailed bullet-point questionnaires** — don't list 10+ specific legal questions in the email body
- **Save the deep dive for the call** — the detailed questions (rebuttal strategy, probability assessment, costs, etc.) should be discussed on the phone, not written into the email
- **Template:** "I reviewed [doc]. Key observation: [one-liner]. Filed on Drive. Would like to get on a call to discuss the plan of action going forward. Kindly let me know a convenient time."

This applies to ALL legal strategy emails to advocates — the email is for scheduling, not for litigating the case in writing.