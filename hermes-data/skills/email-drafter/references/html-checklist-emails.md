# HTML checklist emails (NDR preference, 2026-09-04)

When the email is an **action checklist** asking a reportee/vendor to gather
documents — title-opinion caveats, bank APF lists, legal DD requisitions (e.g.
the Gunjur Sy 40 P-119 list) — draft it in **HTML**, not plain text, so the
checklist is clearly visible to the recipient.

## Rules
1. Render every requirement as a `☐` checkbox item, grouped under bold section
   headers (A/B/C/…), NOT a numbered wall of prose.
2. Put each hard rule in a **callout box** (`border:2px solid #b8860b; background:#fff8e1`)
   — e.g. the "either the document OR a department endorsement that no record
   exists" rule, and a separate deadline box.
3. Make the **deadline explicit and prominent**: "no later than Friday, 11
   September 2026 (next Friday)" — a named day, never a vague "ASAP".
4. Require a **per-item commitment date response**: "Please reply with a
   commitment date against every single item."
5. Sign-off short. Mirror the same checklist content as the companion WhatsApp
   link (same items, same order) so both messages agree.

## Mechanics (verified 2026-09-04, Gunjur P-119 draft r2170091947646310728)
```python
from email.mime.text import MIMEText
msg = MIMEText(html, "html", "utf-8")   # NOT text/plain
msg["To"] = 'Name <email>'
msg["Subject"] = "..."
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
gmail.users().drafts().update(userId="me", id=DRAFT_ID, body={"message": {"raw": raw}}).execute()
```
Verify after update: `payload.mimeType == 'text/html'`, and
`body.count("☐") == <checklist length>` (22 for the Gunjur list).