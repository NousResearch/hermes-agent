# Revising an existing Gmail draft that already carries attachments

Scenario: a draft reply to the principal was written days ago with 7 site photographs attached. New source
documents arrive and the body has to be redrafted. **Do not delete the draft and create a new one** — that
loses the attachments (they were uploaded from a local cache that may be gone) and breaks the thread position.

Verified 2026-10-22 on `sales1.blr@draas.com`: draft `r6676034812374149192`, 7 JPEG attachments, body rebuilt
from 13 KB to 14 KB of HTML, two PDFs added, all 9 attachments present on verification.

## Recipe

```python
import sys, base64, email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.utils import formatdate, make_msgid
sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import build_service

svc = build_service("gmail", "v1", service_name="google-draas")     # vault key, resolved via gws_resolve_account

# 1. Pull the existing draft as RAW bytes and keep the original on disk — this is your only copy of the attachments
d = svc.users().drafts().get(userId="me", id=DRAFT_ID, format="raw").execute()
raw = base64.urlsafe_b64decode(d["message"]["raw"])
open("/tmp/draft_raw.eml", "wb").write(raw)
orig = email.message_from_bytes(raw)
oh = {k: orig.get(k) for k in ["Subject", "To", "Cc", "From", "In-Reply-To", "References"]}

# 2. Rebuild: mixed -> alternative(plain, html) -> original attachments re-attached from the parsed MIME
out = MIMEMultipart("mixed")
for k, v in {"Subject": NEW_SUBJECT, "From": oh["From"], "To": oh["To"]}.items():
    out[k] = v
if oh.get("Cc"):        out["Cc"] = oh["Cc"]          # preserve routing verbatim
if oh.get("In-Reply-To"): out["In-Reply-To"] = oh["In-Reply-To"]
if oh.get("References"):  out["References"] = oh["References"]
out["Date"] = formatdate(localtime=False)
out["Message-ID"] = make_msgid(domain="draas.com")

alt = MIMEMultipart("alternative")
alt.attach(MIMEText(PLAIN, "plain", "utf-8"))
alt.attach(MIMEText(HTML, "html", "utf-8"))
out.attach(alt)

for p in orig.walk():                                  # re-attach everything that was attached before
    if (p.get("Content-Disposition") or "").startswith("attachment"):
        att = MIMEApplication(p.get_payload(decode=True), _subtype="jpeg")
        att.add_header("Content-Disposition", "attachment", filename=p.get_filename())
        out.attach(att)

for path, name in NEW_FILES:                           # then add the new documents
    att = MIMEApplication(open(path, "rb").read(), _subtype="pdf")
    att.add_header("Content-Disposition", "attachment", filename=name)
    out.attach(att)

# 3. Update IN PLACE on the same draft id — thread position and draft id survive
svc.users().drafts().update(userId="me", id=DRAFT_ID,
    body={"message": {"raw": base64.urlsafe_b64encode(out.as_bytes()).decode()}}).execute()
```

## Always verify after the update

Re-fetch `drafts().get(..., format="raw")` — do not trust the update response — and assert:

```python
atts = [p.get_filename() for p in m.walk() if (p.get("Content-Disposition") or "").startswith("attachment")]
assert len(atts) == expected_count          # e.g. 7 old photos + 2 new PDFs = 9
assert "2. Cost sheet" in html              # a distinctive string from the NEW section
```

Print the attachment filenames and the subject line back; that is what proves nothing was silently dropped.

## Pitfalls

- `drafts().get` with `format="raw"` returns the whole message including attachment payloads (2.8 MB here) —
  fine, but write it to disk once and work from the file rather than re-fetching per iteration.
- `_subtype` must match the real content (`jpeg`, `pdf`); guessing `octet-stream` loses inline preview.
- Keep the cwd for `python3` as `/opt/hermes` (or `sys.path.insert` first) or `tools.gws_auth` will not import.
- When the new material changes only PART of the body, keep the untouched sections byte-for-byte — the principal
  has usually already read them, and gratuitous rewording forces a re-read.
- This is still **draft-only**. `drafts().update` never sends; do not follow it with any send call.
