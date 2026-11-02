# Gmail draft integrity: "the draft is in the list but won't open"

Verified 2026-09-17 (two drafts, ndr@draas.com). Read this before rebuilding a draft
the user says is broken, and before attaching files to a reply draft.

## Symptom

NDR: *"For some reason I can see draft in the listing but I can't open the draft.
Can you please recheck and maybe redo the draft, delete the existing draft and do a
new one?"*

The draft appears in the Gmail web Drafts list with correct To/Subject/snippet, but
clicking it errors / opens nothing.

## Root cause — malformed raw MIME

The offender was a draft built with Python `email.message.EmailMessage` whose stored
raw (read back via `drafts().get(format='raw')`) was missing the standard headers and
had a mangled `References`:

```
From: Nishant Ranka <ndr@draas.com>
To: Anbarasan M <pm2.blr@draas.com>
Subject: RE: Bux ranka quatation
In-Reply-To:
 <CAPDkZRavs22oGKQszkZ5NcykFGSbpZGdKsABsBkRdJ=Y-5j6ZQ@mail.gmail.com>
References:
 <…> <…> <…>
 =?utf-8?q?=3CMA5PR01MB11265CDB819F58D2D58DD151DEBB82=40MA5PR01MB11265=2EIND?
 =?utf-8?q?PRD01=2EPROD=2EOUTLOOK=2ECOM=3E?=      <-- RFC2047 encoded-word fold
```

Three compounding problems:
1. **No `MIME-Version`, no `Content-Type`, no `Content-Transfer-Encoding`** in the
   stored draft.
2. **No `Message-Id`.**
3. The long `References` chain contained a **third-party Outlook Message-ID with `+`
   and `=` characters**; Gmail folded it and re-encoded the fragment as
   `=?utf-8?q?…?=` encoded-words. A `References` header carrying encoded-words is
   what breaks the web UI's draft renderer.

A sibling draft created in the same session (HDFC CKYCR, different code path) had
`MIME-Version`, `Content-Type`, `Content-Transfer-Encoding: 7bit`, a `Date` and a
`Message-Id` — and opened fine. Use that as the shape reference.

## Diagnosis (do this BEFORE deleting anything)

```python
# 1. what drafts exist, and are they really drafts?
svc.users().messages().list(userId='me', q='is:draft').execute()

# 2. raw headers of each suspect draft
raw = svc.users().drafts().get(userId='me', id=DRAFT_ID, format='raw').execute()
print(base64.urlsafe_b64decode(raw['message']['raw']).decode('utf-8','ignore')[:800])
```

Red flags: missing `MIME-Version` / `Content-Type` / `Message-Id`; any `=?utf-8?q?`
in a header; `References` longer than 2–3 ids.

## ⚠️ Check `labelIds` before "redoing" a draft — it may already be SENT

In this session the Jasleen draft (id `r4516369503362493299`) still resolved via
`drafts().get()`, but its `labelIds` had flipped from `['DRAFT']` to `['SENT']` —
i.e. **NDR had already sent it from his phone while the web view wouldn't open it.**
Its `Message-ID` had been regenerated at send time, and it sat in the thread as an
11th message.

So: `drafts().get()` succeeding does **not** mean the draft is still sendable. Always
inspect `labelIds` (and re-list `is:draft`) and **report the send** rather than
silently recreating a duplicate. State plainly that the agent only ever creates
drafts and cannot send, then ask the user to confirm the send was theirs.

## Fix — hand-built raw MIME (do NOT use EmailMessage for threaded drafts)

Short `References` (immediate parent + grandparent only), explicit `Message-Id`,
explicit MIME headers, explicit CRLF:

```python
msgid = '<%s@mail.gmail.com>' % uuid.uuid4().hex.upper()[:24]
raw_txt = (
 'MIME-Version: 1.0\r\n'
 'From: Nishant Ranka <ndr@draas.com>\r\n'
 'To: Anbarasan M <pm2.blr@draas.com>\r\n'
 'Subject: RE: <same subject so it threads>\r\n'
 f'In-Reply-To: {PARENT_MSGID}\r\n'
 f'References: {GRANDPARENT_MSGID} {PARENT_MSGID}\r\n'
 f'Message-Id: {msgid}\r\n'
 'Content-Type: text/plain; charset="UTF-8"\r\n'
 'Content-Transfer-Encoding: 7bit\r\n'
 '\r\n' + body.replace('\n', '\r\n')
)
raw = base64.urlsafe_b64encode(raw_txt.encode('utf-8')).decode()
d = svc.users().drafts().create(
        userId='me',
        body={'message': {'raw': raw, 'threadId': THREAD_ID}}).execute()
```

Verify after writing: re-fetch with `format='raw'` and confirm the head shows
`MIME-Version` → `From` → `To` → `Subject` → `In-Reply-To` → `References` (short) →
`Message-Id` → `Content-Type` → `Content-Transfer-Encoding`. Gmail substitutes its own
`Message-Id` on write — that's fine and expected.

**Do not delete the working draft first and then fail.** Delete, write, verify, and
only then report (this session: delete + recreate in one script, verified in the same
run).

## Re-attaching the original files to a reply draft

"Reply all / forward this **and make sure the attachments are visible**" means the
recipients must see the files as real attachments, not just as in-thread history.

**⚠️ Never copy an `attachmentId` out of a tool result into a script (hit 2026-09-25).**
Attachment ids are long base64-ish tokens (often 300+ chars) and the `gws_gmail_get` /
search tool output carries them **truncated** — pasting one back in fails with:

```
HttpError 400 ... returned "Invalid attachment token".
Details: "[{'message': 'Invalid attachment token', 'reason': 'invalidArgument'}]"
```

That error means *your id is truncated*, not that the attachment is gone. Re-derive the id
from the message itself — the walk below returns the **full** token and is the only
trustworthy source. Cheap sanity check: the token you are about to use should match the one
the walk produced character-for-character.

1. Pull every attachment off the source message:
   ```python
   def walk(p, out):
       if p.get('filename') and p.get('body', {}).get('attachmentId'):
           out.append((p['filename'], p['body']['attachmentId']))
       for c in p.get('parts', []) or []: walk(c, out)
   out = []; walk(msg['payload'], out)          # msg = messages().get(format='full')
   for fn, aid in out:
       d = svc.users().messages().attachments().get(
               userId='me', messageId=SRC_MID, id=aid).execute()
       open('/tmp/att_' + fn, 'wb').write(base64.urlsafe_b64decode(d['data']))
   ```
   Print `(filename, mimeType, size, attachmentId)` for each part before downloading — the
   sizes let you confirm later that a round-trip did not alter the file.
2. Build `multipart/mixed` by hand (body part first, then one part per file,
   `base64.encodebytes(...).decode().replace('\n','\r\n')`), plus any extra file you
   add yourself (e.g. a checklist PDF).
3. Verify by re-fetching the created draft with `format='full'` and walking the
   payload for `filename` — assert every expected filename is present. Do not assume
   the create call attached them.

Re-runnable: `scripts/create-draft-with-attachments.py`.

## Related

- Threaded-reply mechanics and per-user mailbox override: `references/draft-create-verify-api.md`
- Which projects/recipients to use: `references/voice-dictation-existing-thread.md`
- **Employee session + native `gws_gmail_*` tools** (drafting as a DRA colleague rather than
  ndr@draas.com): `references/native-gws-draft-tools-pitfalls.md` — the native tool does NOT
  HTML-escape the body (so `&amp;` is stored literally), `gws_gmail_get` OMITS the Cc header,
  `gws_resolve_account` decides which mailbox `google-draas` means, and `contact_resolver`
  needs an explicit `account_email`. Re-runnable verifier:
  `scripts/verify_draft_headers_raw.py`.
