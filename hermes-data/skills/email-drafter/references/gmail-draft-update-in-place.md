# Updating an existing Gmail DRAFT in place (rewrite feedback emails, corrected subjects)

Pattern: NDR asks to "completely update the draft email" — the draft already exists in Drafts
(e.g. `r-7568781667180755847`), is threaded to a prior message, and must keep that thread.

## Rewriting an EXISTING draft that CARRIES AN ATTACHMENT (verified 2026-10-16)

Situation: the draft already has the right recipients and the right attachment, but the
body/subject must be rewritten. Do NOT delete+recreate — you would have to hunt down the
source file, which is usually already gone from `/tmp`. Pull the attachment OUT of the
draft, rebuild the raw with the new body and the SAME bytes, and `drafts().update()` in place.

1. Read the draft `format='full'`, walk `payload['parts']` for the part whose `filename`
   ends in the expected extension, then
   `users().messages().attachments().get(userId='me', messageId=<msg id>, id=part['body']['attachmentId'])`
   → `base64.urlsafe_b64decode(att['data'])`. Save it to disk as well.
2. Rebuild with `email.message.EmailMessage` (default policy RFC2047-encodes the Subject
   itself — **do NOT assign `Header(subj,'utf-8').encode()`**: that value carries fold
   newlines and `msg['Subject'] = ...` raises
   `ValueError: Header values may not contain linefeed or carriage return characters`).
   Then `set_content(plain)`, `add_alternative(html, subtype='html')`,
   `add_attachment(bytes, maintype='application', subtype='pdf', filename=<same name>)`.
3. **Re-set the ORIGINAL `Message-ID` header AND pass `threadId` explicitly.** Doing both
   kept `threadId` stable on the FIRST update (verified: `1a14376692e44ddb` unchanged
   before and after). The silent re-thread described above is what happens when the raw is
   rebuilt WITHOUT the original Message-ID. Still read the response's `threadId` and re-run
   the update once if it moved.
4. Byte-verify: sha256 of the attachment read back out of the UPDATED draft == sha256 of the
   local copy, filename unchanged, and `labelIds == ['DRAFT']`.

**The draft's MESSAGE-ID CHANGES on every in-place update — stale ids 404.** A message id
captured from a search result or an earlier read can be dead minutes later:
`messages().get(id=<old>)` → `404 Requested entity was not found`. That is NOT a deleted
draft and NOT a reason to recreate it. Re-find it by enumerating
`drafts().list(userId='me', maxResults=200)` UNFILTERED and matching on To + Subject (see the
`q=` warning above). The DRAFT id (`r…`) and the `threadId` are the stable handles; the
message id is not. Print all three on every read so you can tell which one you are holding.

## Why not just create a new draft?
Creating a fresh draft orphans the thread. The threaded reply draft carries References /
In-Reply-To headers pointing at the parent message; a brand-new draft loses them. The
recipient's client will show a NEW thread instead of the continued conversation — wrong
for feedback/checklist rewrites on an active thread.

## Correct mechanics (Gmail API via tools.gws_auth.build_service)

1. **GET the draft, keep its identity headers:**
   ```python
   d = gmail.users().drafts().get(userId='me', id=DRAFT_ID, format='full').execute()
   msg = d['message']
   headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}
   thread_id = msg.get('threadId')          # preserve
   refs = headers.get('References', '')     # preserve
   in_reply = headers.get('In-Reply-To', '')# preserve
   ```

2. **RFC 2047-encode the SUBJECT — plain UTF-8 in the raw MIME line mojibakes!**
   An em dash / ₹ / non-ASCII in `Subject: ... — ...` written raw into the MIME string
   comes back as `â€”` in Gmail. Always:
   ```python
   from email.header import Header
   subj = Header('Re: Byadarhalli (Ren-Kaj Lands) — ...', 'utf-8').encode()
   # -> =?utf-8?q?Re=3A_Byadarhalli_=28Ren-Kaj...?=
   ```

3. **Rebuild the raw message** with From/To/Subject/References/In-Reply-To + MIME-Version +
   Content-Type: text/html; charset=UTF-8 + the HTML body, encode with
   `base64.urlsafe_b64encode(mime_msg.encode('utf-8')).decode('utf-8')`.

4. **Update, not create:**
   ```python
   body = {'message': {'raw': raw, 'threadId': thread_id}}
   gmail.users().drafts().update(userId='me', id=DRAFT_ID, body=body).execute()
   ```

5. **VERIFY read-back** — `drafts().get(format='full')` again and print the Subject header;
   confirm em dashes are intact (not `â€”`) and threadId unchanged.

## API pitfalls hit (Sep-2026)

- **`metadataHeaders=[...]` filtering is CASE-SENSITIVE and silently drops headers
  (hit 2026-09-19, Deloitte/Amit Jain thread).** Requesting
  `metadataHeaders=["Message-ID"]` returns the header block WITHOUT any `Message-ID`
  key, so `h["Message-ID"]` raises `KeyError: 'Message-ID'` even though the message
  clearly has one. Gmail's actual header name is `Message-Id` (lowercase `d`), and the
  filter does not fall back to a case-insensitive match. The robust pattern is to
  request NO filter (or ignore what you asked for) and **normalise keys to lowercase**
  before lookup:
  ```python
  anch = svc.users().messages().get(userId='me', id=ANCHOR_MSG,
      format='metadata', metadataHeaders=['Message-ID','References']).execute()
  h = {x['name'].lower(): x['value'] for x in anch['payload']['headers']}
  parent = re.sub(r'\s+', ' ', h['message-id']).strip()
  ```
  Note `'references'` really is all-lowercase, so only `Message-ID` bites — which makes
  it look intermittent. Lowercase-normalise EVERY header dict you build from a payload.

- **`drafts().list()` with a `q=` search does NOT find a draft that plainly exists
  (hit 2026-09-19).** Listing with no query returned the draft; every
  `drafts().list(q='subject:"..."')` / `q='"Pvt Ltd to LLP"'` / `q='<sender>'` variant
  returned `0` for that same draft. The draft search index lags or does not index
  draft bodies/subjects reliably. **Never conclude "there is no draft" from a `q=`
  search** — enumerate `drafts().list(userId='me', maxResults=100)` unfiltered, then
  match on the message's own headers/threadId in Python:
  ```python
  for d in svc.users().drafts().list(userId='me', maxResults=100).execute().get('drafts', []):
      m = svc.users().messages().get(userId='me', id=d['message']['id'], format='metadata').execute()
      if m.get('threadId') == THREAD_ID: ...
  ```
  Corollary: the whole mailbox may hold only a handful of drafts while the
  user believes several exist — the unfiltered list is cheap, so always run it before
  reporting draft state, and before creating a new draft on a thread (prevents stacking
  two drafts on the same thread, which the user sees as a duplicate).

- **drafts().update with a MIMEText-rebuilt raw message can SILENTLY RE-THREAD the draft
  (hit 2026-09-09, Raghu SSA reply; hit AGAIN 2026-09-17, DRA wind-up reply).** Rebuilding via `MIMEText` + adding only
  To/Cc/Subject/In-Reply-To/References does NOT carry the ORIGINAL `Message-ID` header.
  Gmail then treats the updated raw as a fresh message: the returned (and subsequent GET)
  `threadId` changes to a NEW thread (`19ed0777ef8c10f9` → `1a08686cd402bc90`) and the
  draft is orphaned from the conversation. This defeats the whole purpose of an
  in-place update. ALWAYS verify `threadId` in the response equals
  the original thread before declaring success.

- **FIX CONFIRMED 2026-09-17 — you do NOT have to delete + recreate. Re-run the SAME
  `drafts().update` with an explicit `threadId` in the body and the draft re-pins to the
  original thread.** This is the cheaper and now-preferred recovery; delete+recreate
  (see `draft-update-delete-recreate.md`) is the fallback only if the re-pin fails.

  Sequence that worked (DRA wind-up reply, `1a089af60712afd3` → orphaned to
  `1a0a39075b416313` → re-pinned back to `1a089af60712afd3`):
  ```python
  # 1st update — re-threads, threadId now WRONG
  svc.users().drafts().update(userId='me', id=DRAFT_ID,
      body={'message': {'raw': raw, 'threadId': ORIGINAL_THREAD_ID}}).execute()
  # -> returned threadId is a NEW thread. Do not panic, do not delete yet.

  # 2nd update — same raw, same explicit threadId. Re-pins.
  d = svc.users().drafts().update(userId='me', id=DRAFT_ID,
      body={'message': {'raw': raw, 'threadId': ORIGINAL_THREAD_ID}}).execute()
  assert d['message']['threadId'] == ORIGINAL_THREAD_ID   # now True
  ```
  Always run BOTH the update and a follow-up `drafts().get(format='full')` to prove the
  threadId is stable before telling the user it's done. A single update is not proof.

- **Build `References` as the FULL chain, parent Message-ID added LAST (2026-09-17).**
  When the reply target is itself a reply (the common case), Gmail needs the whole chain,
  not just the immediate parent. Read the target message's own `References`, append its
  `Message-ID`, and set that as `References`; set `In-Reply-To` to the target's
  `Message-ID` alone:
  ```python
  target = svc.users().messages().get(userId='me', id=TARGET_MSG_ID,
      format='metadata', metadataHeaders=['Message-ID','References']).execute()
  h = {x['name']: x['value'] for x in target['payload']['headers']}
  parent_msgid  = h['Message-ID'].strip()
  prior_refs    = h.get('References', '').strip()
  msg['In-Reply-To'] = parent_msgid
  msg['References']  = f'{prior_refs} {parent_msgid}'.strip()
  ```
  Ordering matters — `References` is oldest-to-newest and the parent goes last. Getting
  this wrong is what produced the re-thread above. Sanitize with
  `' '.join(re.findall(r'<[^>]+>', refs))` before assigning.

- **`drafts().get(format='metadata', metadataHeaders=[...])` raises TypeError** — the
  googleapiclient discovery method for drafts().get does NOT accept a `metadataHeaders`
  kwarg (messages().get does). Use `format='full'` (or 'metadata') and parse headers from
  `payload.headers` yourself.
- **Mirror the latest sent mail's To/Cc when drafting a reply (2026-09-09, Raghu SSA).**
  Before creating a threaded reply, GET the most recent message in the thread
  (`messages().get(format='metadata', metadataHeaders=['To','Cc'])`) and replicate its
  recipient set — long-running threads often carry a standing CC (Raghu thread: every NDR
  mail had `Cc: Manohar Singh <msingh@redsoul.co.in>`). `draft_reply_create` does NOT
  auto-inherit CC; pass `cc=` explicitly. A reply missing the standing CC starts a
  recipient drift the user notices immediately.
- **build_service needs the right service_name** — `build_service('gmail','v1')` without
  `service_name=` defaults to vault key `'google'` which has NO token and fails with
  "No google token for user ndr-7449813913". For NDR's DRA account always pass
  `service_name='google-draas'` (resolve via gws_resolve_account for others). The sandbox
  DOES route through the vault properly — the failure was the wrong service key, not the sandbox.
- **Use `/opt/hermes/.venv/bin/python` — the bare `python3` on PATH does not have the
  Google libraries (2026-09-17).** `python3 -c "from tools.gws_auth import build_service"`
  dies with `ModuleNotFoundError: No module named 'googleapiclient'` because the session
  default interpreter resolves to `/opt/data/.venv` (playwright/numpy only). The Hermes venv
  that owns `googleapiclient` is `/opt/hermes/.venv`. Two things must both be set: the
  interpreter AND the import path:
  ```bash
  /opt/hermes/.venv/bin/python -c "import sys; sys.path.insert(0,'/opt/hermes'); ..."
  ```
  Quick probe before any Gmail work if unsure:
  ```bash
  for v in /opt/hermes/.venv /opt/data/.venv*; do echo -n "$v: "; \
    $v/bin/python -c "import googleapiclient" 2>/dev/null && echo OK || echo missing; done
  ```
  Same applies to the `whatsapp_link_tool` import below.
- Subject verification after update: the update response headers list may come back empty /
  trivially different; always do a second GET to verify.

## Editing only the BODY TEXT of an existing multi-part draft (attachments must survive)

Different sub-case from a full rebuild: the draft is already correct (recipients, thread,
attachments) and the user asks for **one sentence/clause change** — e.g. "make the email say
we filed it under DR Estate Account, not Dad's account". Do NOT rebuild the MIME from scratch;
that risks the re-thread trap above and the attachments. Edit the text parts in place.

### ⚠️ Trap 1 — a raw-bytes regex/replace on the MIME SILENTLY edits only SOME parts (hit 2026-09-25)

Gmail stores each part with its own transfer encoding, and they differ **within one message**:
observed on a single draft — `text/plain` was **base64** while `text/html` was
**quoted-printable**. A `re.sub` over `drafts().get(format='raw')` decoded bytes therefore
matched the QP HTML occurrence but **not** the base64 plain-text one, producing a draft whose
HTML and plain-text parts disagreed — a *half-applied* edit that still reports success.

**Always decode each part, edit the decoded text, then re-encode:**

```python
em = email.message_from_bytes(base64.urlsafe_b64decode(raw))
P_IN  = "filed this in Kelsa against the PO/WO already raised for Smriti Legal"
P_OUT = "filed this in Kelsa under DR Estate Account, against the PO/WO already raised for Smriti Legal"
for part in em.walk():
    if part.get_content_type() in ('text/plain', 'text/html'):
        txt = part.get_payload(decode=True).decode('utf-8', 'replace')
        txt = txt.replace(P_IN, P_OUT)
        txt = re.sub(r"it is to be paid out of Dad.{0,5}s account",
                     "it is to be paid out of that account", txt)
        # see Trap 2 for the correct write-back
```

Print which parts changed (`print(ct, 'changed' if txt != before else 'no-change')`) — if only
one of the two reports `changed`, you have the half-edit and must not proceed.

### ⚠️ Trap 2 — `set_payload(..., charset=...)` + `encode_quopri` DOUBLE-ENCODES the part (hit 2026-09-25)

`part.set_payload(txt, charset='utf-8')` **already** sets `Content-Transfer-Encoding: base64`
and encodes the payload. If you then `del part['Content-Transfer-Encoding']` and call
`email.encoders.encode_quopri(part)`, you re-encode the **already-base64 string**, and the part
decodes to binary garbage — surfaced as:

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb4 in position 2: invalid start byte
```

The part header still *looks* fine (a valid `Content-Transfer-Encoding`), so this is easy to
miss. **Write back by stripping ONLY `Content-Transfer-Encoding` — keep `Content-Type`, then
set the payload once and stop:**

```python
# ✅ CORRECT — declared type survives, encoded exactly once
if 'Content-Transfer-Encoding' in part:
    del part['Content-Transfer-Encoding']
part.set_payload(txt, charset='utf-8')   # encodes once; do NOT follow with encode_quopri
```

**🚨 DO NOT `del part['Content-Type']` — that single line is what causes Trap 2b below.**
An earlier revision of this very file recommended stripping BOTH headers, and that advice is
what produced the raw-HTML-tags bug the user reported on 2026-09-25. Strip only the transfer
encoding; the declared `Content-Type` must stay with the part.

Then `em.as_bytes()` → `base64.urlsafe_b64encode(...)` → `drafts().update(...)`.

### ⚠️ Trap 2b — deleting `Content-Type` DEMOTES the HTML part to `text/plain` → the user sees RAW HTML TAGS (user-reported, 2026-09-25)

The `del part['Content-Type']` in Trap 2's write-back is the dangerous half. A stored part carries
its declared type as a header, so once it is stripped `set_payload(txt, charset='utf-8')` re-adds a
**default `text/plain`** header for a payload that is actually an HTML document. The draft still
creates and still returns success — and the recipient's client renders the recipient-side body as a
wall of literal markup.

**The user-visible symptom (quote it back, it is the tell):** *"The email body consists of HTML code
rather than rendered HTML. So something is wrong in the way the email was generated."*

This is NOT caused by a missing `html=True` flag — that is the *bridge* pitfall for a NEW draft. On
an in-place edit the cause is a lost/incorrect `Content-Type` on the stored part.

**Detection — parse EVERY text part and print both its declared type and its opening characters:**

```python
for part in em.walk():
    ct = part.get_content_type()
    if ct in ('text/plain', 'text/html'):
        t = part.get_payload(decode=True).decode('utf-8', 'replace')
        print(ct, '|', repr(t[:60]))
```

`text/plain | '<div style="font-family:Arial...'` is the bug: an HTML document wearing a
`text/plain` label.

**Recovery — do NOT patch the corrupted draft. Rebuild from the pristine snapshot** with explicit
subtypes, then delete the broken draft:

```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

msg = MIMEMultipart('mixed')
msg['To'], msg['From'], msg['Subject'] = TO, FROM, SUBJECT
alt = MIMEMultipart('alternative')
alt.attach(MIMEText(plain_new, 'plain', 'utf-8'))   # subtypes declared EXPLICITLY
alt.attach(MIMEText(html_new,  'html',  'utf-8'))
msg.attach(alt)
for name, data in atts:                              # re-read from the snapshot, byte-exact
    a = MIMEApplication(data, _subtype='pdf')
    a.add_header('Content-Disposition', 'attachment', filename=name)
    msg.attach(a)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
d = svc.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
svc.users().drafts().delete(userId='me', id=BROKEN_DRAFT_ID).execute()
```

**Verify the structure, not just the text — this is the assertion that would have caught it:**

```python
# expect: multipart/mixed → multipart/alternative → text/plain(base64) + text/html(base64) → attachments
assert (em.get_content_type(), len(em.get_payload())) == ('multipart/mixed', 3)
parts = [(p.get_content_type(), p.get_charset(), p.get_filename()) for p in em.walk()]
assert ('text/html', 'utf-8', None) in parts
assert ('text/plain', 'utf-8', None) in parts
```

**Rule:** never `del part['Content-Type']` on a part you intend to keep typed. Strip only
`Content-Transfer-Encoding`, then let `set_payload(txt, charset=...)` re-encode the SAME declared
type. If a rebuild is needed at all, declare `MIMEText(txt, 'plain'|'html', 'utf-8')` explicitly.

**✅ VALIDATED RECOVERY — the correct rebuild, run end-to-end 2026-09-25 and accepted.** When the
draft is already broken, this sequence fixes it in one pass (it is the version that actually worked,
superseding the sketch above):

```python
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

orig = email.message_from_bytes(open('/tmp/draft_raw.eml','rb').read())   # pristine snapshot
plain_txt = html_txt = None
atts = []
for part in orig.walk():
    ct = part.get_content_type()
    if ct == 'text/plain' and part.get_filename() is None:
        plain_txt = part.get_payload(decode=True).decode('utf-8', 'replace')
    elif ct == 'text/html':
        html_txt = part.get_payload(decode=True).decode('utf-8', 'replace')
    elif part.get_filename():
        atts.append((part.get_filename(), part.get_payload(decode=True)))

msg = MIMEMultipart('mixed')
msg['To'], msg['From'], msg['Subject'] = TO, FROM, SUBJECT
alt = MIMEMultipart('alternative')
alt.attach(MIMEText(plain_new, 'plain', 'utf-8'))    # subtype declared EXPLICITLY
alt.attach(MIMEText(html_new,  'html',  'utf-8'))
msg.attach(alt)
for name, data in atts:                               # byte-exact from the snapshot
    a = MIMEApplication(data, _subtype='pdf')
    a.add_header('Content-Disposition', 'attachment', filename=name)
    msg.attach(a)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
d = svc.users().drafts().create(userId='me', body={'message': {'raw': raw}}).execute()
svc.users().drafts().delete(userId='me', id=BROKEN_DRAFT_ID).execute()
print('new draft', d['id'], '| msg', d['message']['id'])
```

Notes that mattered in practice:
- `MIMEMultipart('mixed')` + a nested `multipart/alternative` is what produces the Gmail-correct
  shape; do not attach the two text parts directly to the `mixed` container.
- The two attachments were `application/pdf` and survived byte-exact (`351130` / `120338`).
- **Delete the broken draft AFTER the new one is created and verified** — never before, so a failed
  rebuild cannot leave the user with no draft at all.
- **Report the NEW draft id and message id to the user**; the ids change and any earlier id you
  quoted is dead. Also tell them the broken one was removed, so an old link doesn't confuse them.

**Verify the structure, not just the text — this is the assertion that would have caught it:**

```python
# expect: multipart/mixed → multipart/alternative → text/plain(base64) + text/html(base64) → attachments
assert (em.get_content_type(), len(em.get_payload())) == ('multipart/mixed', 3)
parts = [(p.get_content_type(), p.get_charset(), p.get_filename()) for p in em.walk()]
assert ('text/html', 'utf-8', None) in parts
assert ('text/plain', 'utf-8', None) in parts
```

**Also: a rejected recipient-set edit leaves a working draft intact.** When a change request is
about NAMES/addresses in the body ("say DR Estate Account, not Dad's account") rather than encoding,
the earlier revision may already be fine — check whether the body change is even needed on the part
that failed before rebuilding. Only the encoding was broken; the wording edits had landed in both
parts. Read the parts and decide per-part rather than assuming the whole draft is void.

### ⚠️ Trap 3 — updating a draft CHANGES its MESSAGE id (the draft id stays the same) (hit 2026-09-25)

`drafts().update(id=DRAFT_ID, ...)` returns the same draft resource id
(`r-1645639718900550561`) but mints a **new message id** underneath:

| | before | after update 1 | after update 2 |
|---|---|---|---|
| draft id | `r-1645639718900550561` | same | same |
| message id | `1a0d89cb64af0914` | `1a0d8ab283584e0d` | `1a0d8ac18a05e931` |

So re-reading the draft by its old message id fails:

```
HttpError 404 ... returned "Requested entity was not found."   # "notFound"
```

That 404 means *the message id is stale*, NOT that the update failed or the draft is gone.
Re-resolve by **unfiltered `drafts().list(userId='me')` + match on the Subject header**, then
read back through `format='raw'`/`'full'`. Always address the draft by its **draft id**, never
by a message id captured before an update.

### Verification that actually proves the edit landed AND the attachments are intact

Parse the read-back raw MIME and assert on **every** axis:

```python
em2 = email.message_from_bytes(base64.urlsafe_b64decode(new_raw))
for part in em2.walk():
    ct = part.get_content_type()
    if ct in ('text/plain', 'text/html'):
        print(ct, P_OUT in part.get_payload(decode=True).decode('utf-8','replace'))
    else:
        print(ct, part.get_filename(), len(part.get_payload(decode=True) or b''))
```

Expected: **both** text parts show the new text, and each attachment's decoded byte length
matches its pre-edit value exactly (this session: `351130` and `120338`). A text change that
shifts an attachment's size by even one byte means the re-serialisation damaged it — revert and
rebuild from the pristine snapshot.

**Snapshot the pristine raw BEFORE the first edit** (`open('/tmp/draft_raw.eml','wb').write(...)`).
When an attempt half-applies or double-encodes, that snapshot is what you rebuild from — do not
try to unpick the corrupted message.


## Calling the whatsapp_link tool from terminal (2026-09-17)
The module is `tools.whatsapp_link_tool` (NOT `tools.whatsapp_link`). The function takes a
**single positional dict**, not kwargs:
```python
from tools import whatsapp_link_tool as w
r = w.whatsapp_link_tool({'phone': '+919704404000', 'text': '...'})
# -> {phone, text, delivered, mode, url, note}
```
Calling it as `whatsapp_link_tool(phone=..., message=...)` raises
`TypeError: whatsapp_link_tool() missing 1 required positional argument: 'args'`.
The `message` field is named `text` in the dict form.

**The return value is a JSON STRING, not a dict (hit 2026-09-19).** `r['url']` raises
`TypeError: string indices must be integers, not 'str'`. Parse it first:
```python
import json
r = w.whatsapp_link_tool({'phone': '+919704404000', 'text': '...'})
data = json.loads(r) if isinstance(r, str) else r
print(data['url'])
```
Print the whole result on the first call so the shape is visible rather than guessed. Report the returned `url` as a
deep link the user taps — `delivered` is always `false`; nothing is ever sent.
