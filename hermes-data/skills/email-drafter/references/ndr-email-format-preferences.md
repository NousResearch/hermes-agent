# NDR Email Format & Structure Preferences (verified 2026-09-13)

When drafting task-briefing / assignment emails for reportees (Sarthak, Anbu, Prakash etc.):

## Format: HTML + CSS, never plain text with hard line breaks

NDR is explicit: *"there is a lot of line termination mid-sentence... that's not how I want my emails to be drafted."* 

- Build multipart/alternative MIME (text/plain fallback + text/html primary)
- Paragraphs flow as single logical lines — do NOT insert `\r\n` every 8-12 words
- Inline CSS: Arial/Helvetica 14px, line-height 1.65, colour #222; section headers in dark navy `#1F3864` background, white bold font, padded + rounded; sub-headers in navy `#1F3864` bold; proper `<ul>`/`<ol>` spacing; `max-width: 860px` wrapper

## Structure: one email per task

When the instruction covers multiple distinct tasks for one recipient, split into separate email drafts:
- Each gets subject `Task: <Project/Entity> — <task description>`
- Each body is self-contained (intro → task detail → closing)

## Link placement

Drive links go inline under the relevant section as `<a href="...">link</a>` — NOT in a separate "References" appendix at the bottom (NDR may delete that block).

## Validation/flag items

Place as a numbered "Things to validate / flag to me" section at the end — NOT scattered inline.

## Example HTML pattern used successfully (2026-09-13)

```python
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64, re

html_body = """<!DOCTYPE html><html><head><meta charset="utf-8"><style>
body { font-family: Arial, Helvetica, sans-serif; font-size: 14px; line-height: 1.65; color: #222; margin: 0; }
.wrap { max-width: 860px; margin: 0 auto; padding: 8px 0; }
p { margin: 0 0 12px 0; }
h2 { font-size: 15px; color: #ffffff; background: #1F3864; padding: 8px 12px; margin: 22px 0 10px 0; border-radius: 4px; }
h3 { font-size: 14px; color: #1F3864; margin: 16px 0 6px 0; }
ul, ol { margin: 0 0 12px 0; padding-left: 22px; }
li { margin-bottom: 5px; }
</style></head><body><div class="wrap">%s</div></body></html>"""

msg = MIMEMultipart("alternative")
msg["To"] = to; msg["Subject"] = subject
plain = re.sub(r"<br\s*/?>", "\n", html_body)
plain = re.sub(r"</h[23]>", "\n", plain)
plain = re.sub(r"<li>", "\n- ", plain)
plain = re.sub(r"<[^>]+>", " ", plain)
msg.attach(MIMEText(plain, "plain", "utf-8"))
msg.attach(MIMEText(html_body, "html", "utf-8"))
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
gmail.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

The plain-text fallback is serviceable (no one reads it on mobile/Gmail). The key: paragraphs are single lines, no mid-sentence breaks.