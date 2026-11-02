# School Absence — Medical Certificate Required (SOP Insistence)

When the school insists on a doctor's certificate per their SOP for Grade 9+
(board-registered students, 3+ day absences), and you already have a known
doctor who can provide the certificate without an in-person visit.

## Workflow overview

1. WhatsApp the known family doctor with a polite request for the certificate
2. Doctor provides the certificate (PDF scan)
3. File the certificate on Drive
4. Reply-all to the school thread with the certificate attached
5. BCC any school admin the user names

## Step 1 — WhatsApp the doctor

Use the "Personal doctor / known medical professional" tone (see `messaging-drafts` → `references/messaging-links-workflow.md`
skill → NDR-specific tone patterns). Structure:

- Warm greeting ("Hey doctor, how is it going?")
- Context: known patient, known condition, current episode
- Situation: school SOP requires a certificate; want to avoid hospital visit
- Specific ask: what the cert needs to state (bullet points)
- Urgency: test tomorrow, need it quickly
- Offer to draft it for them (with smiley 😉)
- Warm close

## Step 2 — File the certificate on Drive

Upload to the child's Medical folder on Drive (e.g. `Ruhaan Medical` folder
inside the main `Ruhaan` folder). Name format: `YYYYMMDD_ChildName_MedicalCertificate_HospitalName.pdf`.

**Pitfall — stale folder IDs for Ruhaan Medical subfolder (Aug 2026):** The
`Ruhaan Medical` folder (ID `0B1Oc8cSaJXPGaEhnaDg1Wjl0Qk0`) returned 404 when
uploading via Drive API. The folder exists in the Drive UI but its API-resolved
ID may differ from what was listed. Fall back to the parent `Ruhaan` folder
(ID `0B1Oc8cSaJXPGbl9VMEZBdE04Z28`) if the subfolder fails. Verify upload with
`drive_search` after creating the file — confirm it's in the expected parent.

## Step 3 — Reply to the school email

Since this is a forwarded-email scenario (original thread in Roshini's mailbox,
not NDR's — see the "forwarded email pitfall" in the SKILL.md), compose as a
**fresh email** (draft_create, not draft_reply_create):

### Recipients

- **To:** Class teachers who sent the SOP reply
- **Cc:** Any other teachers on the thread, plus parent (Roshini) and child
- **Bcc:** School administrator (Joel Kribairaj), ONLY if user explicitly names them

### Tone — polite compliance with a gentle nudge

The school has explained their position (SOP for Grade 9). You have complied
with the requirement. The email should acknowledge this while gently noting
the inconvenience of the hospital visit. Key principles:

1. **Thank them** for explaining the SOP requirement — shows you read and accept it
2. **State compliance** — "Since the school requires it, we have taken Ruhaan to the doctor"
3. **Provide the certificate** as attachment
4. **Gentle nudge** — "We hope this meets the school's requirements" (not confrontational)
5. **Keep it brief** — no debate about policy at this stage

Full example body (validated Aug 2026, Ruhaan @ Aditi Gr9):

```
Dear Ranjitha and Priya,

Thank you for your thoughtful response and for explaining the SOP requirements
for Grade 9. We completely understand the school's need to maintain proper
documentation.

Since the school requires a medical certificate as part of its established
procedure, we have taken Ruhaan to the doctor. Dr. Nishanth Hiremath at
Bhagwan Mahaveer Jain Hospital has examined him and issued the attached
medical certificate.

We hope this meets the school's requirements. Please find the certificate
attached for your records.

We appreciate your continued support and understanding of Ruhaan's medical
condition. Please do let us know if there is anything else required from
our end.

Warm regards,
Roshini Ranka & Nishant Ranka
Parents of Ruhaan Ranka
```

### BCC via raw Gmail API

The bridge does NOT support BCC. Use the raw Gmail API with
`email.message.EmailMessage` (or `MIMEMultipart` when attachments are involved):

```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import base64

msg = MIMEMultipart('mixed')
msg['To'] = 'Primary <primary@example.com>'
msg['Cc'] = 'Cc1 <cc1@example.com>, Cc2 <cc2@example.com>'
msg['Bcc'] = 'Blind <blind@example.com>'
msg['Subject'] = 'Re: Original Subject'

# Attach body + files
# ... (see templates/draft-with-attachments.py for full recipe)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode('ascii')
draft = service.users().drafts().create(userId='me', body={
    'message': {'raw': raw, 'threadId': THREAD_ID}
}).execute()
```

**Note on BCC visibility:** Gmail API strips Bcc from the returned headers
when you verify with `drafts().get()` — this is correct behaviour. The Bcc IS
present in the MIME and will be delivered correctly.

## Related

- `email-drafter` SKILL.md → "School / authority communication tone" — for
  emails that challenge/query a school policy before the cert is obtained
- `school-absence-exam-accommodation.md` — simpler notification-only pattern
- `messaging-drafts` → `references/messaging-links-workflow.md` → "Personal doctor / known medical professional" tone