# Revising an existing draft, and request-tone rules

## 1. Tone rule — never put a scope disclaimer in a request (NDR/Bharat, Oct 2026)

When a draft *asks* another team or person to produce something, do **not** open with
"this was not our scope / we prepared it anyway / treat it as reference only". The
recipient reads it as a complaint attached to a favour, and it lands as rude.

Bharat's correction, verbatim: *"you cannot keep the same thing in the draft, it looks
rude… we are requesting her, can't be so rude."*

Rewrite the same facts as a courteous request that credits the recipient:

> Before: "To be clear on the sequence - preparing these posters was not our scope, but we
> have prepared them anyway… Please treat the attached posters as a reference only."

> After: "A quick note before the details. We have prepared a direction poster for each of
> the 13 locations ourselves, purely so that the distance, the drive time and the arrow
> direction are settled before design begins. Please treat these as a working reference -
> **the design itself is entirely in your hands**."

Plus a closing courtesy line before the sign-off: *"Happy to walk you through any of the
locations on a call before you begin."*

Generalisation: state who owns the final deliverable (them), keep the sequence/context
(ours), drop the grievance. Never argue scope inside an ask.

## 2. Revising a draft that already carries attachments

`drafts().update` on a hand-edited raw MIME is fragile (quoted-printable re-encoding,
payload/CTE mismatches). The reliable path — rebuild and swap:

```python
svc = build_service("gmail", "v1", service_name="google-draas")
d = svc.users().drafts().get(userId="me", id=OLD_DRAFT).execute()
msg = svc.users().messages().get(userId="me", id=d["message"]["id"], format="full").execute()
# ⚠️ pass format="full" explicitly — without it payload can come back None

def flatten(part, out):
    out.append(part)
    for p in (part.get("parts") or []):
        flatten(p, out)
    return out          # ⚠️ return the list — a walker that returns None and is iterated
                        # as `for p in walk(...)` raises "'NoneType' object is not iterable"

body, atts = None, []
for p in flatten(msg["payload"], []):
    if p.get("filename"):
        atts.append((p["filename"], p["mimeType"], base64.urlsafe_b64decode(
            svc.users().messages().attachments().get(
                userId="me", messageId=msg["id"], id=p["body"]["attachmentId"]
            ).execute()["data"])))
    elif p["mimeType"] == "text/plain" and body is None:
        body = base64.urlsafe_b64decode(p["body"]["data"]).decode("utf-8")

new_body = body.replace(OLD_PARA, NEW_PARA)          # exact-string replace on the decoded text
m = MIMEMultipart(); m["To"] = to; m["Subject"] = subject
m.attach(MIMEText(new_body, "plain", "utf-8"))
for fname, mime, data in atts:
    part = MIMEApplication(data, _subtype="pdf")     # or the file's real subtype
    part.add_header("Content-Disposition", "attachment", filename=fname)
    m.attach(part)
raw = base64.urlsafe_b64encode(m.as_bytes()).decode()
nd = svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
svc.users().drafts().delete(userId="me", id=OLD_DRAFT).execute()   # exactly one draft left
```

Then **verify by reading the new draft back** (`messages().get(format="full")`): confirm the
new paragraph is present, the old one is gone, and the attachment filename still lists.

Reporting shape that worked: show the removed line, show the replacement, list what is
unchanged (subject, tables, attachments), state that it is a draft only and the user sends it.
