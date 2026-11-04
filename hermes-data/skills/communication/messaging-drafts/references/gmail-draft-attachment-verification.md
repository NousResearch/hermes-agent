# Verifying a Gmail draft that carries ATTACHMENTS

Verified 2026-09-04 (Arvind Krishna succession-docs draft: 2 .docx attachments
on Dharmesh's "Succession Certificate" thread).

## The trap

After `drafts().create(...)` with a multipart raw MIME containing attachments,
verifying with:

```python
d = gmail.users().drafts().get(userId='me', id=draft_id, format='metadata').execute()
```

**reports `0` attachments** — the `payload` returned in `metadata` format has
NO `parts[]` list with the attachment parts (metadata format strips body and
attachment content; only top-level headers survive). Counting filenames in
`payload.parts` there yields zero even when the draft genuinely carries the
files. This is NOT a failed draft — it's a verification-format problem.

## The fix: verify with format='full' and walk the payload

```python
import time
time.sleep(2)  # transient 404 right after create — see personal-task-tracker pitfall
d = gmail.users().drafts().get(userId='me', id=draft_id, format='full').execute()

def walk(part):
    out = []
    if part.get('filename'):
        out.append((part.get('filename'),
                    part.get('mimeType'),
                    part.get('body', {}).get('size')))
    for sp in part.get('parts', []):
        out.extend(walk(sp))
    return out

atts = walk(d['message']['payload'])
# assert len(atts) == 2 and filenames match expected
```

For `format='full'` the attachment parts appear as child parts with
`filename`, `mimeType` (e.g. `application/vnd.openxmlformats-officedocument.wordprocessingml.document`)
and `body.size` matching the uploaded byte count.

## Body verification

- Check TOP-LEVEL `payload.body.data` FIRST (single-part drafts return the
  whole text there with `parts=[]`), then recurse `parts[]` for text/plain.
- Base64 url-safe decode + startswith assert for the expected opening line.

## Order of operations for threaded drafts with attachments

1. Fetch source message metadata: `Message-ID` (for In-Reply-To/References)
   and `threadId` (from the SAME message record — never the message id).
2. Build `EmailMessage`, set To/From/Cc/Subject/Message-ID/Date/
   In-Reply-To/References, `set_content(body)`.
3. `add_attachment(bytes, maintype=..., subtype=..., filename=...)` per file
   (docx subtype: `vnd.openxmlformats-officedocument.wordprocessingml.document`).
4. `base64.urlsafe_b64encode(em.as_bytes())` → `drafts().create(userId='me',
   body={'message': {'raw': raw, 'threadId': thread_id}})`.
5. `time.sleep(2)` → verify with `format='full'` (attachments) + read body.