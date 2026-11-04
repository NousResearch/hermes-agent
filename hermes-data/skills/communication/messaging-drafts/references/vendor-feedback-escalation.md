# Vendor-Feedback Escalation — Dual-Channel Workflow

## When to use

The user (NDR) asks to send formal feedback to a vendor/service provider whose product underperformed. Typically involves:
- Multi-person test campaign (employees test the product)
- Comprehensive analysis document (Google Doc)
- Cover email with attachments + follow-up WhatsApp group message

## Channel 1: Google Doc Feedback Report

1. **Gather all analysis data** — per-call transcripts, consolidated notes, employee feedback, compliance scores. Source from local analysis cache (`/data/hermes/cache/analysis/`) or session history.
2. **Build a Google Doc** via HTML import (see `google-doc-formatting-template` skill):
   - Write structured HTML with H1/H2 sections, tables, callout boxes, bullet lists
   - Upload to Drive via `build_service('drive', 'v3', service_name='google-draas')` with `MediaFileUpload(html_path, mimetype='text/html')`
   - **ALWAYS stage in NDR's TMP folder** (Drive ID: `18p74II2uL32sNDzDDwXzmlOUdJJOTmE-`) — never Drive root
   - Title format: `YYYYMMDD_Project_VoiceAgent_Vendor_Feedback_Name`
3. **Verify the doc** via Docs API: check `namedStyleType` (should show HEADING_1/2/3), count tables, check text length
4. **Export to PDF** for email attachment: `drive.files().export(fileId=DOC_ID, mimeType='application/pdf')`
5. **Export any data workbooks** to xlsx for attachment: `drive.files().export(fileId=SHEET_ID, mimeType='...spreadsheetml.sheet')` via `MediaIoBaseDownload`

## Channel 2: Email Draft (never auto-send)

Follow `email-drafter` skill's draft creation workflow, with these additional steps:

1. **Verify recipients** via `find_contact.py` from `messaging-drafts` → `references/personal-messaging-workflow.md`. If vendor addresses are not in contacts, note that the user provided them directly (from chat/forwarded messages) — flag it but proceed since the user is the source.
2. **Use the vendor-feedback tone** (see `email-drafter` SKILL.md → Stage 2 — Draft → Vendor-feedback / frank escalation tone):
   - State situation factually
   - Name failures with numbers
   - Mention alternatives (other provider tested)
   - Express disappointment factually
   - Set consequence as fact, not threat
   - Close with action request
3. **Build the MIME message manually** via raw Gmail API — the bridge `draft_create` doesn't support attachments:
   ```python
   from tools.gws_auth import build_service
   from email.mime.multipart import MIMEMultipart
   from email.mime.text import MIMEText
   from email.mime.application import MIMEApplication
   import base64

   gmail = build_service('gmail', 'v1', service_name='google-draas')
   # whoami check first
   who = gmail.users().getProfile(userId='me').execute()['emailAddress']
   assert who == 'ndr@draas.com', f'WRONG MAILBOX: {who}'

   msg = MIMEMultipart()
   msg['To'] = ', '.join(TO)
   msg['Cc'] = ', '.join(CC)
   msg['Subject'] = SUBJECT
   msg.attach(MIMEText(body, 'plain', 'utf-8'))
   # Attach exported PDF + xlsx
   for path, fname, mime in attachments:
       with open(path, 'rb') as f:
           part = MIMEApplication(f.read(), _subtype=mime.split('/')[-1])
       part.add_header('Content-Disposition', 'attachment', filename=fname)
       msg.attach(part)
   raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
   draft = gmail.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
   ```
4. **Verify** via `drafts().list()` + `drafts().get()` — check To, Cc, Subject, attachments, DRAFT label
5. Remind the user: "It's a draft only — go to Gmail Drafts to review and send"

## Channel 3: WhatsApp Group Follow-Up

After the email draft is created, NDR typically asks to notify the vendor's coordination group on WhatsApp:

1. **Use the `whatsapp_link` tool** — this is the ONLY sanctioned way to create WhatsApp deep links
2. **No phone number** — user picks the group after tapping the link
3. **Use `platform='telegram'`** for safe MarkdownV2 rendering
4. **Message structure** (frank but polite, same tone as email):
   - Email sent to [address list] with feedback + FAQ attached
   - Request urgent action to enable retesting
   - Another provider's first cut was brilliant (Indian accent, facts correct, conversation handled naturally)
   - Very disappointed — vendor product experience is poor, feels half-baked
   - This is the last effort — if it fails, we withdraw
   - Appalling that basic tests aren't being run
   - As polite as possible, no threats, but very clear on consequences
5. **Deliver the display_link** to the user — they tap, WhatsApp opens, pick the group, send

## Pitfalls

- **Whoami check required** — terminal subprocesses can inherit wrong `HERMES_SESSION_USER_ID`. Always assert mailbox == `ndr@draas.com` before creating/verifying drafts
- **Draft attachment limit** — the `gws_skill_bridge.draft_create` does NOT support attachments. Must use raw Gmail API with `MIMEMultipart`
- **Draft update = delete + recreate** — there is no Gmail API for modifying draft body. Delete by draft resource ID (`drafts().delete(id=draft_resource_id)`) and recreate
- **Vendor addresses not in contacts** — common for vendor/vendor emails. Flag to the user that the addresses aren't in contacts, but proceed since the user provided them directly
- **Long WhatsApp messages** — if the message exceeds Telegram's ~4k char limit, the tool auto-splits into parts. Deliver each part as a separate Telegram message

## Example: JOYZ AI Vendor Feedback (2026-08-17)

Validated end-to-end with:
- 9 test calls by 7 employees → Google Doc feedback (10 sections, 3 tables, ~16k chars)
- Google Doc staged in TMP, exported to PDF (187 KB)
- FAQ workbook (42 Qs) exported to xlsx (73 KB)
- Email draft to akash.deep@joyz.ai, a@joyz.ai, s@joyz.ai, help@joyz.ai (CC: sales1.blr@draas.com)
- WhatsApp group message (no phone, user picks group)