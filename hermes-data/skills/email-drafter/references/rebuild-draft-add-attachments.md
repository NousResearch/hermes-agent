# Rebuilding an existing Gmail draft (adding attachments or rewriting the body)

The `draft_create` bridge call takes text and no attachments. When a draft already exists and the user then
sends files with "add these to the email" / "these images also added in all of the emails", the wrong move is
to call `draft_create` a second time — that leaves **two drafts with the same subject** in the mailbox and the
user may send the stale one. Rebuild in place: create the replacement, verify it, then delete the superseded
draft.

Proven 17-Sep-2026 on ndr@draas.com (Kotak fraud lock-down letter + five WhatsApp screenshots).

## Mechanics

Run via `terminal` with `/opt/hermes/.venv/bin/python3` and
`HERMES_SESSION_USER_ID=<telegram id>` when the target mailbox is not the session default. Do **not** use
`execute_code` — its sandbox lacks `gws_fetch_token`, so the vault credential loader fails with
`ImportError: cannot import name 'gws_fetch_token'`.

Write the script to a file with `write_file` and run the file. An `&` inside a bash heredoc is rejected by the
backgrounding guard.

```python
import base64, mimetypes
from email.message import EmailMessage
from tools.gws_auth import build_service

svc = build_service("gmail", "v1", service_name="google-draas")
assert svc.users().getProfile(userId="me").execute()["emailAddress"] == "ndr@draas.com"   # ALWAYS assert

# 1. read the draft you are replacing
old = svc.users().drafts().get(userId="me", id=OLD_DRAFT_ID, format="full").execute()
# walk old["message"]["payload"] for the text/plain part; save it to a file and edit THAT

# 2. build the replacement
msg = EmailMessage()
msg["To"], msg["Cc"], msg["Subject"] = TO, CC, SUBJECT
msg["From"] = "Nishant Ranka <ndr@draas.com>"
msg.set_content(open("/tmp/body.txt", encoding="utf-8").read())
for path, name in ATTACHMENTS:
    maintype, subtype = (mimetypes.guess_type(name)[0] or "image/jpeg").split("/", 1)
    with open(path, "rb") as fh:
        msg.add_attachment(fh.read(), maintype=maintype, subtype=subtype, filename=name)
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
new = svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

## Ordering rules that matter

1. **Read the old draft with `format='full'`.** `format='metadata'` returns headers only — no body. Dump the
   `text/plain` part to a file and compose the new body from that file, so nothing is silently dropped in the
   rebuild.
2. **Create the replacement before deleting anything.** If creation or verification fails, the original draft
   is still intact — a half-migrated letter is worse than a stale one.
3. **Verify by reading the new draft back with `format='full'`** and recursively walking `payload` collecting
   `(filename, body.size)`. Assert: expected attachment count, every size is non-null, and a couple of sentinel
   strings from the body are present. Only then `drafts().delete(userId='me', id=OLD_DRAFT_ID)`.
4. **Confirm nothing was sent**, and put the number in your reply:
   `messages().list(userId='me', q='in:sent subject:"<fragment>"')` must return `0`.
5. Attachments must be readable **before** they are added. When files arrive via web-chat they sit in
   `/mnt/uploads/<file_id>_<original_name>`; match on the id prefix, since the original name may repeat
   across uploads. Rename to a descriptive `filename=` on the way in — the recipient sees that, not WhatsApp's.

## Pitfalls

- `drafts().list()` will show both drafts until the delete succeeds; check the count after, and report the
  surviving draft id.
- Body char count is a cheap sanity check to include in your log line — a rebuild that silently drops
  half the letter shows up immediately.
- Do not re-derive the To/Cc from memory or from your own earlier message. Read them off the draft you are
  replacing (`format='full'` → headers) — that is the version the user already saw and approved.
- Attachment sizes come back as bytes in `body.size` after read-back; if they are null the payload walk is
  wrong (usually you matched on a nested multipart container rather than the leaf part).
