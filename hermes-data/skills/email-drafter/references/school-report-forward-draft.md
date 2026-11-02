# School Term Report → Forward with Analysis (Aditi/MAIS Reports)

Recurring pattern: NDR's children's school (Mallya Aditi International School / MAIS) sends term reports via email, and NDR asks to forward the report to the child with "Anjali's analysis" (Hermes's analysis — NDR calls it "Anjali" as a nickname for the agent's output).

## Triggers

- "Forward Ruhaan's report that came in"
- "Send the report card to [child] with analysis"
- Any Aditi/MAIS term report email sent to ndr@drahomes.in

## Sender identity

| Sender | Email | Purpose |
|---|---|---|
| Mallya Aditi (MAIS) | mais@aditi.edu.in | Official term reports (text-layer PDFs) |
| Joyce Jose | messenger@plusportals.com | School notices, PTC schedules |
| Teachers | *.gsuite.aditi.edu.in | Subject-specific communication |

## Subject patterns

- `09 IGCSE first term Report - September (2026)` — IGCSE Std. 9 first term
- `Std-08 second term report April- 2026` — Std. 8 second term
- `Std. 08 first term report - October- 2025`
- `Std-06 second term reports April- 2024`
- Various Std. 06/07/08/09 patterns

## Step-by-step workflow

### 1. Locate the report email

Gmail query on `google-draas` (ndr@draas.com — the default account, since school sends to ndr@drahomes.in):

```
q='subject:("IGCSE" OR "report") AND ("term" OR "Std")'
```

Or search by child's name + grade: `q="IGCSE Ruhaan"` or `q="09 IGCSE first term"`.

The report is cc'd to both ndr@drahomes.in and rmurjani@gmail.com (Roshni, NDR's wife).

### 2. Download the PDF attachment

```python
msg = gmail.users().messages().get(userId="me", id=msg_id, format="full").execute()
payload = msg.get("payload", {})

def find_attachment(payload):
    parts = payload.get("parts", [])
    for part in parts:
        if part.get("filename") and part.get("mimeType") == "application/pdf":
            return part
        if part.get("parts"):
            result = find_attachment(part)
            if result:
                return result
    return None

att = find_attachment(payload)
att_data = gmail.users().messages().attachments().get(
    userId="me", messageId=msg_id, id=att["body"]["attachmentId"]
).execute()
pdf_bytes = base64.urlsafe_b64decode(att_data["data"])
with open("/tmp/report.pdf", "wb") as f:
    f.write(pdf_bytes)
```

### 3. Extract text from the PDF

**These are text-layer PDFs** — `pymupdf` works directly. Do NOT pass to `vision_analyze` (it rejects PDFs with "Only real image files are supported").

```python
import pymupdf
doc = pymupdf.open("/tmp/report.pdf")
text = "\n".join(page.get_text() for page in doc)
```

### 4. Compose "Anjali's analysis"

The analysis is authored as "Anjali" (NDR's nickname for Hermes agent output). Structure:

**Subject | Grade | Effort** table — parse the Term Grade and Effort lines from each subject block.

**Key Strengths** — quote teacher praise verbatim (e.g., "sets the standards for attention, participation, quality of work, and demeanour").

**Areas to Focus** — quote next-step suggestions from each teacher's comments.

**Bottom line** — 1-2 sentence summary of overall performance and key advice.

Style: warm, encouraging, direct to the child. Use bullet points and bold for readability.

### 5. Draft the email to the child

**Recipients:**
- Ruhaan Ranka → pebblyshark69@gmail.com
- Rivaan Ranka → rankarivaan@gmail.com

**DRAFT ONLY — never send.** Use `gmail.users().drafts().create()`.

**MIME structure:**
```python
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

message = MIMEMultipart()
message["To"] = child_email
message["Subject"] = f"{child_name} — Your {report_title} + Analysis"

# Body — the analysis text (plain text)
message.attach(MIMEText(analysis, "plain"))

# Attachment — the original report PDF
attachment = MIMEBase("application", "pdf")
attachment.set_payload(pdf_bytes)
encoders.encode_base64(attachment)
attachment.add_header("Content-Disposition", "attachment", filename=original_filename)
message.attach(attachment)

raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
draft = gmail.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

**Account:** `google-draas` (default — ndr@draas.com). The school report goes to ndr@drahomes.in but the Gmail API is accessed via the draas vault account.

### 6. Confirm delivery

Tell NDR: "Draft created in your Gmail Drafts folder" + link to Drafts: `https://mail.google.com/mail/u/0/#drafts`

## Common pitfalls

- **"taxa report card"** is a voice artifact of "the report card" — they mean the PDF attachment. Always include the original PDF.
- **NDR's wife (rmurjani@gmail.com) is cc'd on the original school email** — when NDR says "what I forwarded to my wife, forward it to him as well + include the report card PDF if not included," the deliverable to the child is the full report + analysis.
- **The school sends from `mais@aditi.edu.in` through the `google-draas` vault** — the report goes to ndr@drahomes.in which is a forward/alias accessible via the google-draas vault account. Do NOT search google-gmail or google-ahfl for school emails.
- **Previous term reports are in the same thread** — Gmail threads them by subject, so the latest report may be in a thread with older reports. Use `q='subject:("IGCSE" OR "Std. 09")'` to isolate the current term's email.
- **"forward it to him as well"** — the raw email was already sent to the wife. The child gets a new email with analysis + original PDF, not a forward of the raw school email.