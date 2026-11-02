# Creating and verifying a Gmail DRAFT (raw MIME) — API gotchas

> **Drafting through the NATIVE tools (`gws_gmail_draft_create` / `gws_gmail_get`) instead, or
> drafting in a DRA employee's session rather than ndr@draas.com?** Read
> `references/native-gws-draft-tools-pitfalls.md` first — the native tool does not HTML-escape
> the body (`&amp;` gets stored literally), `gws_gmail_get` omits the Cc header entirely, and
> `google-draas` may resolve to the session user's mailbox. Re-runnable verifier:
> `scripts/verify_draft_headers_raw.py`.

Verified 2026-09-09 (Aditi speaker-session email to Soni Thomas, Bcc Joel Kribairaj).

## Working recipe

```python
import base64
from email.mime.text import MIMEText
from tools.gws_auth import build_service

gmail = build_service("gmail", "v1", service_name="google-draas")
# sanity: gmail.users().getProfile(userId='me').execute()['emailAddress'] must == ndr@draas.com

msg = MIMEText(body, "plain", "utf-8")
msg["To"] = "Name <name@example.com>"
msg["Bcc"] = "Name2 <name2@example.com>"
msg["Subject"] = "..."
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = gmail.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

Returns `draft["id"]`, `draft["message"]["id"]`, `draft["message"]["threadId"]`.

## Pitfall — hand-rolling `payload` instead of `raw` → `HttpError 400 "Missing draft message"`

Verified 2026-09-19 (AR Portable Cabins reply-all draft). Attempting to build the draft from an
explicit Gmail API message resource rather than encoded MIME bytes:

```python
# WRONG — 400 "Missing draft message" (domain global / invalidArgument):
draft = gmail.users().drafts().create(userId="me", body={"message": {
    "threadId": TID,
    "payload": {
        "mimeType": "text/plain",
        "headers": [{"name": "To", "value": "..."}, {"name": "Subject", "value": "..."}],
        "body": {"data": base64.urlsafe_b64encode(body.encode()).decode()},
    },
}}).execute()
```

`drafts().create` wants `message.raw` (a base64url-encoded RFC-2822 byte stream). The
`payload` message-resource shape is what `messages().get` RETURNS, not what `create` ACCEPTS —
handing it back is rejected outright because the server sees no draft message body it can parse.
The error text `"Missing draft message"` does not hint at the cause.

**Fix:** always go through `EmailMessage` (or `MIMEText`) → `raw`:

```python
from email.message import EmailMessage
import base64

msg = EmailMessage()
msg["To"] = "info@vendor.com"
msg["Cc"] = "sayed@vendor.com, pm2.blr@draas.com"
msg["Subject"] = "Re: ..."
msg["In-Reply-To"] = PARENT_MSGID          # threading header
msg["References"] = PARENT_MSGID
msg.set_content(body)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = gmail.users().drafts().create(userId="me",
    body={"message": {"threadId": TID, "raw": raw}}).execute()
```

The `raw` path also gets you the correct MIME headers (`MIME-Version`, `Content-Type`,
`Content-Transfer-Encoding`) for free — the hand-rolled `payload` variant omits them, which is
the second cause of the "draft lists but won't open" failure documented further below.

`Cc` in the `EmailMessage` path is comma-separated with bare addresses
(`"a@x.com, b@y.com"`) — display names are optional and Gmail resolves them fine.

## Pitfall — drafts().get() has NO metadataHeaders kwarg

`gmail.users().messages().get(format='metadata', metadataHeaders=[...])` works for
messages. `gmail.users().drafts().get(...)` does NOT accept `metadataHeaders` —
it raises `TypeError: Got an unexpected keyword argument metadataHeaders`
(googleapiclient wraps it as a generic TypeError, not an HttpError).

**This trips up even experienced use — re-confirmed 2026-09-18 (Behera/Jasleen reply-all
draft) after `drafts().create()` had ALREADY SUCCEEDED.** The habit of pairing
`format='metadata'` with `metadataHeaders=[...]` is strong because that is the correct
`messages().get()` idiom, so it gets copied across to drafts().get() reflexively.

**When you hit this TypeError immediately after a successful `drafts().create()`, the draft
exists and is fine.** Do NOT delete and recreate it — just re-verify without the kwarg. The
create call printed the draft id before the traceback; use that id.

```python
# WRONG — TypeError on drafts():
v = svc.users().drafts().get(userId='me', id=DID, format='metadata',
                             metadataHeaders=['To','Cc','From','Subject','In-Reply-To'])
# RIGHT — format='metadata' already returns the full header list; drop the kwarg:
v = svc.users().drafts().get(userId='me', id=DID, format='metadata').execute()
hd = {x['name']: x['value'] for x in v['message']['payload']['headers']}
print(v['message'].get('labelIds'))   # must be ['DRAFT']
```

To make the create-then-verify flow atomic (so a verify-side TypeError can never cast doubt on
whether the create worked), capture the id first and wrap the verify in its own try:

```python
draft = svc.users().drafts().create(userId='me', body={'message': {'raw': raw, 'threadId': T}}).execute()
print("DRAFT ID:", draft['id'], "| THREAD:", draft['message']['threadId'])   # create is DONE
v = svc.users().drafts().get(userId='me', id=draft['id'], format='metadata').execute()
```

**Verification pattern that works for drafts:**

```python
check = gmail.users().drafts().get(userId="me", id=draft["id"]).execute()
h = {x["name"].lower(): x["value"] for x in check["message"]["payload"]["headers"]}
print(h.get("to"), h.get("bcc"), h.get("subject"))
```

Read From/To/Bcc/Subject straight out of `payload.headers` — no `format` or
`metadataHeaders` args. Note `format='metadata'` IS accepted on drafts(); only the
`metadataHeaders` param is rejected.

## Pitfall — truncated message IDs 404 on re-fetch

Gmail message IDs are **16-char hex strings**. If search/list output truncates them for display
(e.g. `m['id'][:14]`), re-fetching the printed short string fails with
`HttpError 404 "Requested entity was not found"` — even in the same account, with no
account-scoping involved (real case 2026-09-09: printed `1a083fdc5f3376`, real ID
`1a083fdc5f33767f`). Always carry the FULL ID from the list response into the follow-up
`messages().get()` / `threads().get()`; when printing IDs for your own later use, print them
untruncated.

## Threaded reply draft (In-Reply-To/References + threadId)

Verified 2026-09-09 (Anbu replying to NDR's 9-Sep follow-up mail, thread 1a05cd25ac603c96).

To make a draft that lands in the existing thread (so NDR sees it as "Re:" in the same chain):

```python
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
import base64

gmail = build_service("gmail", "v1", service_name="google-draas")
# get the message you are replying to, fetch threading headers:
msg = gmail.users().messages().get(userId="me", id=ORIG_MID, format="metadata",
    metadataHeaders=["From","To","Subject","Date","Message-ID","References","In-Reply-To"]).execute()
h = {x["name"]: x["value"] for x in msg["payload"]["headers"]}
orig_msgid = h.get("Message-ID", "")
refs = (h.get("References") or "") + (" " + orig_msgid if orig_msgid else "")

m = EmailMessage()
m["To"] = h.get("From").split("<")[-1].rstrip(">") if h.get("From") else ""
# Subject prefix stripping must be REGEX, never lstrip() — lstrip strips CHARACTERS,
# not a prefix. On "RE: Succession Certificate" it leaves "E: Succession Certificate"
# (verified 2026-09-16 — the draft subject became "Re: E: Succession Certificate").
import re
_raw_subj = h.get("Subject", "")
_clean = re.sub(r"^(?:Re|RE|Fw|FW|Fwd|FWD|E)\s*:\s*", "", _raw_subj, flags=re.IGNORECASE).strip()
m["Subject"] = ("Re: " + _clean) if _clean else ""
m["In-Reply-To"] = orig_msgid
if refs.strip():
    m["References"] = refs.strip()
m["Date"] = formatdate(localtime=True)
m["Message-ID"] = make_msgid("draas")
m.set_content(body)

draft = gmail.users().drafts().create(userId="me",
    body={"message": {"raw": base64.urlsafe_b64encode(m.as_bytes()).decode(),
                      "threadId": msg["threadId"]}}).execute()
```

- Passing `threadId` to drafts().create keeps the draft IN the thread (Gmail shows it as a draft reply).
- `EmailMessage` works fine with utf-8 subjects containing en-dashes (—) without any encoding juggling; raw base64 encode `m.as_bytes()`.
- Verify with `drafts().get` (no metadataHeaders) — see pitfall below.

## Pitfall — `lstrip('Re: Fwd: ')` corrupts Outlook subjects ("Re: E: ...")

Verified 2026-09-16 (both the Amit Jain Deloitte follow-up and the Keytruda Ticket 7027 chaser).
The older recipes above use `h.get('Subject').lstrip('Re: Fwd: ').strip()` — but `str.lstrip`
strips CHARACTERS, not a prefix. Outlook subjects arrive as `RE: Succession Certificate`
(uppercase, sometimes `RE : ` with a space before the colon); lstrip removes `R` then stops at
uppercase `E` (not in the char set), leaving `E: Succession Certificate`, so the draft subject
becomes `Re: E: Succession Certificate`. The draft is still in the right thread but the subject
is visibly wrong and NDR notices.

Correct prefix-strip with regex, and handle both `RE:` and `RE :` (space-before-colon) forms:

```python
import re
raw = h.get('Subject', '')
clean = re.sub(r'^(RE|Re|re|E|FWD|FW|Fwd)\s*:\s*', '', raw).strip()
clean = re.sub(r'^[Ee]\s*:\s*', '', clean).strip()   # second pass for "E:" remnants
subject = clean if clean.lower().startswith('re:') else 'Re: ' + clean
```

ALWAYS verify the draft's subject via `drafts().get` → `payload.headers` after creating —
a subject that reads `Re: E:` / `Re: Fwd:` is the signature of the character-strip bug.

## Attachments in a threaded reply draft

Verified 2026-09-11 (Charitra Keytruda infusion cycle, Ticket 7027, ndr@ahfl.in).

### Native draft tools have NO attachment parameter

The bridge/native tools (`gws_skill_bridge draft_create`, `gws_gmail_draft_create`,
`gws_gmail_reply_draft`) only take to/cc/bcc/body — no attachments. Verified 2026-09-13
(Succession Certificate thread, deed to Maneesha Kongovi / CMS IndusLaw): when a drafted reply must
carry a file, either

1. Draft the text and tell the user exactly what to attach and where it lives on Drive
   (folder name + exact filename — e.g. `Family Settlement - Final - 2025 / 20250806 Family
   Arrangement Deed.pdf`); they attach from the Gmail drive picker before sending, or
2. Use the raw-MIME `EmailMessage` path below, pulling the file bytes first with the standard
   Drive download: `build_service('drive','v3', service_name=...).files().get_media(fileId=...).execute()`.

Don't claim the attachment is in the draft when using the native tools — be explicit that
attaching is the user's one manual step.

### NEVER write body copy claiming attachments unless they are actually in the message

Real case 2026-09-14 (914 Embassy Habitat Astra design-brief email): the drafted body opened
with *"together with the floor plan, reference photos and videos attached to this mail"* but the
message was ultimately sent/delivered with **ZERO attachments** — the floor plan did not exist on
Drive or anywhere else at drafting time. The recipient was left expecting files that were never
there, and NDR had to chase it with a follow-up correction email.

Rules:
1. **Ship-truth the attachments BEFORE writing "attached"/"PFA" language.** If the file isn't in
   the draft (or on Drive with a link), the body must NOT claim it is attached. Say "I'll share
   the plan separately" or link the Drive file instead.
2. **Verify a draft's parts before handing it over**: recurse `payload.parts` on
   `drafts().get(...)` for `filename` entries — each attached file shows as a part with a
   filename; if the promised files aren't there, fix the body or the attachment, not the copy.
3. **Verify a SENT message the same way**: `gmail.users().messages().get(userId='me', id=...,
   format='full')` and walk `payload.parts` (and nested parts) for `filename`/`attachmentId`.
   Do this whenever a previous turn's message "had attachments" per its text — text claims are
   not evidence (real case: promised attachments, actual parts contained only the text part).
4. If the collateral genuinely doesn't exist (e.g. no floor plan anywhere on Drive/in mail),
   say so plainly in the follow-up instead of inventing a link — NDR prefers the honest
   correction to a fabricated reference.

`EmailMessage` supports attachments directly — build the same threaded message as above, then:

```python
with open('/path/file.pdf', 'rb') as f:
    m.add_attachment(f.read(), maintype='application', subtype='pdf',
                     filename='20260902 Charitra Keytruda Infusion Confirmation MSD.pdf')
```

- Attach BEFORE base64-encoding: `raw = base64.urlsafe_b64encode(m.as_bytes()).decode()`.
- To/Cc are NOT auto-derived by the raw-MIME path — re-specify the full reply-all set explicitly in `m['Cc']` (Gmail's reply-all doesn't apply to raw drafts).
- Verify attachments in the saved draft by recursing `payload.parts` for `filename` entries on `drafts().get(...)` — each attached PDF shows as a part with a filename.

## Correction workflow — wrong render/design was already sent (verified 2026-09-14, Moduli office draft)

When NDR says "we sent the wrong office design / render in that email — update it with the correct one":

1. **Delete the OLD draft first** (`gmail.users().drafts().delete(userId="me", id=<draft_id>)`) so the stale draft can't be sent by accident. Then create a new draft; don't try to edit in place.
2. Body must **explicitly say the earlier visual was the wrong design** ("the image we shared earlier belonged to a different project of ours — please disregard it") — do NOT just silently replace the attachment; the recipient must know the first render is invalid.
3. Subject should carry a correction marker, e.g. `CORRECTION: prefab site/marketing office for Ranka Udaya — revised design render attached`.
4. Reuse the same raw-MIME `EmailMessage` + `add_attachment` path; **verify the attachment part exists** by walking `payload.parts` for `filename` on `drafts().get(...)` before reporting success.
5. If the user attached the corrected render in chat, read the bytes from `/mnt/uploads/<file_id>_<original_name>` (web chat attachments mount there) rather than searching Drive for it — the file on disk is the authoritative copy.
6. Tell NDR explicitly which file is attached and that the old draft was deleted — he pulls the render from the Kelsa WO himself when he wants to double-check.

A binary upload into Drive also needs the raw `MediaIoBaseUpload` path, not `gws_ops.drive_create_file(..., content=...)` — that helper encodes content as UTF-8 text and corrupts JPG/PDF bytes. Use `build_service("drive","v3")` + `MediaIoBaseUpload(BytesIO(bytes), mimetype=..., resumable=False)`.

## Multi-thread chase batch — CC comes from EACH original thread, not from memory (verified 2026-09-18)

Trigger: NDR names several sent emails and wants a follow-up on **each**: *"I want a follow-up
email on each of them asking them to please reply urgently on the progress made on each."*

When drafting N threaded replies to N different threads (often to the same person), the CC set
must be read **per-thread from that thread's own headers**. Verified case: three chase replies
were needed — two carried `Cc: Prakash, Roshni` (landowner-facing matters), one carried
`Cc: Prakash` only (the Sevaganapalli stamp-duty matter). A single "usual CC set" applied across
all three would have silently added or dropped recipients.

```python
def thread_info(tid):
    t = gmail.users().threads().get(userId='me', id=tid, format='metadata').execute()
    m = t['messages'][-1]                      # reply to the LATEST message in the thread
    h = {x['name'].lower(): x['value'] for x in m['payload']['headers']}
    return h.get('message-id'), h.get('subject'), h.get('references', ''), h.get('cc', '')
```

Rules:
1. **Reply to the LAST message in the thread**, not the first — the `Message-ID` you thread
   against should be the newest one, and its `References` already carries the chain.
2. **Use that thread's own `Cc`** — it encodes who was deliberately kept on the matter. Do not
   infer CC from NDR's address book or from a sibling thread.
3. Pass `threadId` to `drafts().create` regardless (belt and braces alongside In-Reply-To).
4. Prefix `Re:` only when the subject doesn't already start with it.
5. Verify `getProfile()` == `ndr@draas.com` once before the batch, not per-draft.
6. The email layer has NO one-matter-per-message limit — sending N replies to the same person is
   the expected shape for a chase. (The WhatsApp side does have that limit; see `messaging-drafts` → `references/messaging-links-workflow.md`
   P14.)

## Explicit reply-all with Cc reconstruction

When the reply must go to multiple recipients matching the original thread's Cc list (reply-all), fetch the original message's headers and replicate To/Cc explicitly:

```python
import base64, re
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

gmail = build_service('gmail', 'v1', service_name='google-draas')

# Fetch the LATEST message in the thread you are replying to
orig = gmail.users().messages().get(userId='me', id=LATEST_MID,
    format='metadata', metadataHeaders=['From','To','Cc','Subject','Message-ID','References']).execute()
h = {x['name']: x['value'] for x in orig['payload']['headers']}

# Cc list from original — preserve ALL recipients (may include people NDR wants in the loop)
orig_cc = h.get('Cc', '')

# Build the reply-all message
m = EmailMessage()
m['To'] = h['From']  # the sender becomes the To
if orig_cc:
    m['Cc'] = orig_cc  # preserve the full Cc list
# subject prefix stripping must be regex, NOT lstrip (see pitfall below — lstrip
# strips chars, so "RE: ..." → "E: ..."; verified 2026-09-16)
import re
_clean_subj = re.sub(r'^(?:Re|RE|Fw|FW|Fwd|FWD|E)\s*:\s*', '', h.get('Subject', ''), flags=re.IGNORECASE).strip()
m['Subject'] = 'Re: ' + _clean_subj if _clean_subj else ''
m['In-Reply-To'] = h['Message-ID']
refs = (h.get('References', '') + ' ' + h['Message-ID']).strip()
if refs:
    m['References'] = refs
m['Date'] = formatdate(localtime=True)
m['Message-ID'] = make_msgid('draas')
m.set_content(body)

raw = base64.urlsafe_b64encode(m.as_bytes()).decode()
draft = gmail.users().drafts().create(userId='me',
    body={'message': {'raw': raw, 'threadId': orig['threadId']}}).execute()
```

Key points:
- `m['To'] = h['From']` — the original sender becomes To.
- `m['Cc'] = orig_cc` — preserve the full Cc chain (Prakash, Manish, Mamta, etc.).
- The MIMEText variant (older) works too but EmailMessage handles headers more cleanly.

## Recovering Bcc recipients from a previously SENT message (reply drafts)

Verified 2026-09-15 (Manish's "Re: Winding up matters" reply; NDR: "add Roshni and Shruti to
the BCC — the address is in the first email we sent out"). When NDR says the Bcc addresses
are in a mail we already sent, the SENT copy in Gmail is the source of truth:

```python
res = gmail.users().messages().list(userId="me", q='subject:"Winding up matters"', maxResults=30).execute()
for m in res.get("messages", []):
    full = gmail.users().messages().get(userId="me", id=m["id"], format="metadata",
        metadataHeaders=["From","To","Cc","Bcc","Subject","Date"]).execute()
    h = {x["name"]: x["value"] for x in full["payload"]["headers"]}
    if "SENT" in full.get("labelIds", []) and h.get("Bcc"):
        bcc = h["Bcc"]   # e.g. 'Roshini Ranka <rnr@draas.com>, Shruti Chhajer <Shrutichhajer@gmail.com>'
```

Rules:
1. Match on `labelIds` containing `SENT` AND a populated `Bcc` header — drafts and received
   copies have no Bcc, so the sent copy is unambiguous. `q='subject:"..."'` (quoted) is
   enough to find the thread; walk all hits and pick the sent one.
2. Put the recovered addresses on the reply via `m['Bcc'] = ...` on the threaded
   EmailMessage (same To/Cc/In-Reply-To/References pattern as above).
3. Verify the DRAFT's Bcc after creation: `drafts().get` → walk `payload.headers` for
   `Bcc` — drafts().get returns headers even though it has no `raw` envelope.
4. Cross-check the recovered address against the contacts sheet (contact_resolver). If the
   Bcc person is phone-only in the sheet (real case: Shruti Chajjer row 3420 had numbers,
   NO email), enrich the row with the recovered email + real-world research (org, title,
   Instagram, website, LinkedIn — NDR asked for this explicitly for Shruti Shushma) using
   the contact-phone-lookup column map; Notes = idx14/col15.
5. Name-spelling drift: mail display names may spell a surname differently than the sheet
   ("Chhajer" in Gmail vs "Chajjer" in the sheet). Resolve against authoritative web
   sources (LinkedIn/brand site) and flag the correction to NDR rather than silently
   choosing.

## Revising an existing UNSENT draft in place (drafts().update)

Verified 2026-09-15 (NDR's "Re: Winding up matters" reply to Manish — second version, HTML bullets,
shorter, regrouped assets). When NDR asks to change a draft that has NOT been sent, update it in
place — do NOT delete-and-recreate. Keep the SAME draft id and thread.

1. Locate the draft: `drafts().list` then `drafts().get` per draft; match on subject/thread.
   Read headers from `payload.headers` (To, Cc, Subject, In-Reply-To, References) so the revised
   message preserves the full threading chain.
2. Rebuild the message with `EmailMessage` — same To/Cc/Subject/In-Reply-To/References, new
   `Message-ID` and `Date` — then:

```python
m.set_content(plain_text)
m.add_alternative(html, subtype="html")          # multipart/alternative
raw = base64.urlsafe_b64encode(m.as_bytes()).decode()
res = gmail.users().drafts().update(userId="me", id=DRAFT_ID,
    body={"message": {"raw": raw, "threadId": THREAD_ID}}).execute()
```

3. Verify with `drafts().get` (remember: drafts().get accepts NO metadataHeaders kwarg) — check
   headers + walk payload parts for the text/html body; confirm no duplicate drafts remain.
4. There is no `messages().send` here — draft only, same hard rule as creation.

### Pitfall — drafts().update RE-THREADS the draft unless threadId is passed explicitly

Verified 2026-09-15 (Winding-up reply to Manish, second revision): the first in-place update
called `drafts().update` with ONLY `{"message": {"raw": new_raw}}` — no `threadId` — and Gmail
silently moved the draft OUT of the original thread (draft's `threadId` changed from
`1a089af60712afd3` → `1a0a39075b416313`) even though In-Reply-To/References headers were intact.
If it had been sent, the reply would have arrived as a BRAND-NEW detached thread, not in the
conversation chain NDR expected. Unlike `drafts().create`, rebuild-on-update does NOT reliably
inherit the thread from the threading headers.

Rules:
1. **ALWAYS pass `threadId` in the update body** — `body={"message": {"raw": raw, "threadId": THREAD_ID}}`
   where THREAD_ID is the original thread fetched BEFORE the update (record it from the last
   `drafts().get` / `messages().get` of the original email). Never omit it, even when the headers
   look right.
2. **Verify threadId AFTER every update**: `drafts().get(...)` → `draft["message"]["threadId"]`
   must equal the original thread. The diagnostic signature is a stray `labelIds: ['DRAFT']`
   message that shows up as its OWN thread in subject searches and no longer appears in the
   original thread.
3. If the draft did re-thread, re-run the update with `threadId` pinned — same draft id, same
   raw — and re-verify. No need to delete/create.
4. When iterating on a long draft (second/third revision from NDR), do a content check by
   walking the saved HTML part for the corrected strings (`base64.urlsafe_b64decode` on the
   text/html part's body data) AND assert the threadId in the same pass — one script, both
   assertions, before reporting success.

## Pitfall — verifying a draft's BODY content: drafts().get has NO raw envelope

Verified 2026-09-15 (Ranka Udaya strategy email). After `drafts().get`, checking
markers against `d["message"].get("raw")` returns `False` for EVERYTHING — Gmail's
`drafts().get` does not include the `raw` field (unlike `messages().get`), so any
`if marker in raw_all` check silently fails even though the draft is perfectly fine.

**BUT `format='raw'` IS accepted and DOES return the raw MIME (verified 2026-09-24).**
The "no raw envelope" behaviour above is for the DEFAULT call. When you need true
byte-level verification — e.g. proving an embedded attachment matches a local file —
call `drafts().get(userId='me', id=..., format='raw')`, decode `message['raw']`
(base64url, pad with `=`), parse with `email.message_from_bytes(..., policy=policy.default)`,
walk parts, `part.get_content()` and sha256-compare against the source file. Presence of a
`filename` in `payload.parts` is NOT proof of intact bytes; sha256 match is.

Working verification (walk parts, decode the text/html body data):

```python
d = gmail.users().drafts().get(userId="me", id=draft_id).execute()
def collect(part, acc):
    if part.get("mimeType") == "text/html":
        acc.append(base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", "replace"))
    for p in part.get("parts", []):
        collect(p, acc)
    return acc
html = collect(d["message"]["payload"], [])[0]
for marker in ['width="100%"', "Rs 1.2 crore", "psingh@draas.com"]:
    print(marker, "->", marker in html)
```

Also assert `payload.parts` walk turns up at least one text/html part — a 4 KB payload
when 11+ KB of HTML was sent means the HTML was dropped (see draft-create bridge
`html=` kwarg quirk — it's a boolean flag, the string goes in `body=`).

## HTML bodies for NDR's drafts (multipart/alternative)

NDR explicitly wants HTML/CSS-formatted drafts when the email is structured (bulleted asset lists,
bold section headers, highlighted key numbers, step boxes). Gmail renders INLINE styles only —
do NOT rely on `<style>` blocks. Working pattern (2026-09-15 Winding up email):

- container: `<div style="max-width:640px;margin:0 auto;background:#ffffff;padding:28px 32px;font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.55;color:#222222;">`
- section headers: `<p style="margin:16px 0 6px 0;padding-bottom:4px;border-bottom:2px solid #1a4d7c;color:#1a4d7c;font-weight:bold;font-size:15px;">Section title</p>`
- bullets: plain `<ul style="margin:0 0 10px 20px;padding:0;">` + `<li style="margin:0 0 4px 0;">`
- highlight the ONE critical number/date: `<strong style="color:#b03a2e;">June 2027, the fifth year since Dad</strong>`
- step emphasis: `<p style="margin:14px 0 6px 0;background:#eef4fa;border-left:4px solid #1a4d7c;padding:8px 12px;font-weight:bold;color:#1a4d7c;">`
- use HTML entities (`&mdash;`, `&ldquo;`, `&rsquo;`) inside the HTML part; keep `m.set_content(plain)` fallback so the draft also reads fine as plain text.

## Explicit word-bans — mid-dictation "don't use word X" is a HARD content constraint

Verified 2026-09-16 (Amit Jain / Deloitte follow-up to Dharmesh + Manish). NDR dictated a follow-up
for the LLP-conversion thread, then caught himself mid-memo: *"not liquidation, sorry, don't use the
word liquidation."* Even though his OWN earlier email on the parallel "Winding up matters" thread
(15-Sep-2026 asset map) uses "liquidation" liberally, the follow-up draft must NOT contain it.

Rules:
1. A mid-dictation "don't use the word X" is a hard content constraint on the email body — do not
   reintroduce the word anywhere, even when the same thread / parallel threads use it freely.
2. Substitute the framing NDR himself reached for when he corrected: "simplifying and winding down
   the DRA entities and making them more manageable as a private family-held business."
3. In the delivery reply, explicitly confirm the banned word is absent ("'liquidation' kept out as
   you asked") and offer to reword sibling terms (e.g. keep "winding down" or drop it) — NDR decides.
4. Same class as other content/tonal bans: no verbatim "shut shop", no accusations, no invented facts.

## DRA family restructuring — live thread map (Sep 2026)

Two interlocked threads in ndr@draas.com about winding down / restructuring DRA entities; future
sessions on this matter should resolve threads BEFORE drafting (same rule as voice-dictation-existing-thread):

- **"Deloitte - Amit Jain: Pvt Ltd to LLP conversion for non-operating entities"** — threadId
  `1a05bdbce4e46b9c`. Started 1-Sep-2026 (NDR intro: Amit Jain, Partner M&A Deloitte Hyderabad; LLP
  conversion route for non-operating, asset-heavy, liquidity-heavy entities). Recipients: Dharmesh
  (dharmesh.ranka1@gmail.com), Manish (mdr@draas.com) — NDR's messages To both; reply-all means
  To = latest From (Dharmesh) / Cc = latest Cc (Manish). Last msg before the 16-Sep follow-up draft:
  Dharmesh 8-Sep "I am not in Bangalore on Friday" (the 11-Sep invite died there).
- **"Winding up matters"** — threadId `1a089af60712afd3`. Manish 10-Sep "WE NEED TO SHUT SHOP!"; NDR
  15-Sep asset map (Gandhinagar Mamta building, AT Edifice, Queens Corner, Ranka Chambers office,
  Trafalgar/Dunkeld, three Goa units, Mantri Infotech, Gunjur Sy 40 (rev clean-up; Sy 39/40/41 for CLU/JDA/A-Kharab regularization; Kusuma Muniraj spelling confirmed), Merces Goa Prestige JDA, Century
  Regalia South City inventory, Kothanur, Eastern Farmland, corporate compliance); Manish 15-Sep
  "we need to meet. There is a systematic process...". Manish's 16-Sep follow-up invite context.
- **Address gotcha:** NDR's 15-Sep Winding-up msg Cc'd `dharmesh.ranka@gmail.com` (NO "1"), while the
  Deloitte thread consistently uses `dharmesh.ranka1@gmail.com`. Always verify the address per-thread
  from Gmail headers (To/From/Cc) before drafting — don't copy from the other thread or from memory.

## Pitfall — dictated YEAR may already be in the past; sanity-check deadlines

NDR dictates by voice; "by no later than June ... this year" produced "June 2026" in a conversation
dated Sep 2026 — a deadline in the PAST, impossible to send. Resolve dictated deadline years against
today's date and the referenced anniversary (here: fifth year after a 2022 event = June 2027), use the
corrected date in the draft, and flag the change explicitly in the handoff note. Same class of check:
survey numbers, office names, and spellings ("Kusuma Muniraj", "Eastern Farmland", unparseable phrases
like "a Dunkle the state") — list them as 'to confirm' items rather than silently choosing.

## Pitfall — "delete all the drafts, I'll prepare fresh" is LITERAL and destroys an unsent payment confirmation (verified 2026-10-08, Behera/Jasleen)

NDR's instruction *"Delete all of the drafts, take factor into consideration these additional
WhatsApp messages ... and prepare a fresh new reply all draft"* means delete the drafts he has
been shown — but a mailbox can hold other live drafts from adjacent work, and a blanket
`drafts().list` + delete-all takes them with it.

Real failure: a **payment-confirmation draft** (Smriti Legal fee ₹76,500 / UTR KKBKH26279677292
+ three case documents) was swept up in a "delete all drafts" pass. Jasleen then wrote *"The
Appeal is ready to be filed. Please have the professional fees of Rs.85000/- remitted.
Resharing the invoice sent earlier on 25.09.2026."* — and the reply had vanished; the mailbox
had **no draft at all**. It had also never been sent (no UTR anywhere in Sent).

Rules:
1. **Enumerate before deleting.** `drafts().list(userId='me', maxResults=50)` and print each
   draft's To + Subject + threadId. Delete only the drafts in the matter under discussion; leave
   anything that reads like a different workstream untouched, and SAY what you left.
2. **After any bulk delete, report what was removed by subject** so the user can object.
3. **When the user later says "you were supposed to send X" — verify by state, not memory.**
   Check Sent (`messages().list(q='<distinctive phrase>', includeSpamTrash=True)` → `labelIds`)
   AND Drafts. "Never sent and no longer a draft" is a real, reportable outcome: rebuild it
   fresh and say plainly that it had been deleted in the cleanup pass.
4. A deleted draft is unrecoverable — rebuilding is the only repair, so capture the content
   (body text, attachments, recipients, threading anchor) from the earlier turns or the thread
   before deleting next time.

**This pitfall RECURRED on 2026-10-09 and the prediction held exactly.** NDR asked *"did we ever
send that email because it doesn't seem from the sent mail that that email was sent neither is it
sitting in draft"* — about the same Smriti Legal payment-confirmation draft (₹76,500 / UTR
KKBKH26279677292 + three case documents) that the earlier bulk-delete had swept away. It had never
been sent (no UTR anywhere in Sent) and the Drafts folder held ZERO matching items. The repair that
worked, in order: (1) prove the state — search Sent for the UTR string with `includeSpamTrash=True`
AND list all drafts, report "never sent and no longer a draft"; (2) find the trigger — Jasleen's
latest mail (*"The Appeal is ready to be filed. Please have the professional fees of Rs.85000/-
remitted"*); (3) rebuild fresh from the evidence (payment details + attachments pulled from the
pack, byte-verified), threaded to HER latest Message-ID; (4) state plainly it had been deleted in
the cleanup pass. Do not apologise vaguely or imply the mail was sent — name the state.
**Corollary:** when a matter has an unsent-but-important draft (a payment confirmation, a filing
cover note), say so explicitly in the handoff note after any draft cleanup, so its absence is a
known fact rather than something the user discovers days later.

## Pitfall — running the raw-MIME recipes from anywhere but /opt/hermes

`from tools.gws_auth import build_service` resolves ONLY when running from `/opt/hermes` with
`PYTHONPATH=/opt/hermes`. A plain `python3 script.py` from $HOME fails with
`ModuleNotFoundError: No module named 'tools'` (real case 2026-09-15). Run:
`cd /opt/hermes && PYTHONPATH=/opt/hermes python3 /path/script.py`.
Write the script via write_file, not heredoc, when the body contains `&` (terminal tool rejects
inline heredocs containing `&` even inside Python strings).
- Always pass `threadId` so the draft lands in the existing conversation.

In a session that belongs to a DRA employee (e.g. Anbu), `build_service(..., service_name="google-draas")`
may resolve to **ndr@draas.com** (the vault owner), NOT the session user's own mailbox. Verified 2026-09-09:
Anbu's session printed `WHOAMI: ndr@draas.com` with no override, then `pm2.blr@draas.com` with
`HERMES_SESSION_USER_ID=pm2_blr` (also accepts `pm2.blr@draas.com`; NOT `anbarasan@draas.com` — that
has no vault mapping).

Always:
1. Check `build_service(...).users().getProfile(userId="me").execute()["emailAddress"]` BEFORE writing a draft.
2. If wrong mailbox, run the whole script with `HERMES_SESSION_USER_ID=<slug>` env prefix (e.g. `pm2_blr`),
   never forge via any tool arg — the env var is the supported selector.
3. If a message ID from another mailbox 404s ("Requested entity was not found"), it is almost certainly
   account-scoped: the ID belongs to a different account than the one you're currently authenticated as
   (real case 2026-09-09: NDR-sent ID fetched while authenticated as ndr, then 404 under Anbu). Re-check
   WHOAMI rather than assuming the message disappeared.

## Pitfall — terminal guard trips on '&' inside inline Python heredocs

Sep 2026 real case: `python3 - <<'PY'` heredoc containing a subject `"Re: 18 & Oak …"` was rejected by
the terminal tool with `Foreground command uses '&' backgrounding …` even though the '&' was inside a
Python string, not a shell background. Workaround: write the Python to a file with write_file() and run
`python3 /path/script.py` — no shell sees the '&'. Keep subjects/drafts with '&' out of inline `-c`
one-liners and heredocs entirely.

## Pitfall — Gmail attachment ID goes STALE after the first message fetch ("Invalid attachment token")

Verified 2026-09-14 while reading Vishwas Rao's "Discussion Note - Sevaganapalli.docx" from the
DRA - Jiraaf thread: an attachmentId captured during an EARLIER `messages().get(format='full')`
call fails when used in the follow-up `attachments().get(...)` call:

```
HttpError 400 "Invalid attachment token"
```

The attachmentId returned by Gmail is only valid while the message payload it came from is
"current"; re-using a token you printed/stored from a previous fetch (even minutes earlier, same
account) is rejected. **Fix:** re-fetch `messages().get(userId='me', id=MSG_ID, format='full')`
fresh in the SAME script that will download, walk `payload` recursively for the part whose
`filename` matches, take `body.attachmentId` from THAT walk, and immediately call
`attachments().get(...)` with it. One fetch → one download, never reuse a stored token.

## Pitfall — reading a .docx attachment without python-docx

python-docx is NOT installed in the sandbox by default (`No module named 'docx'`). For a quick
text read of an advisor's note / attached .docx, use the zipfile fallback — a .docx is a zip
with `word/document.xml`:

```python
import zipfile, re, html
z = zipfile.ZipFile("/tmp/note.docx")
xml = z.read("word/document.xml").decode("utf-8", "replace")
text = re.sub(r"<[^>]+>", "", xml)         # strip all tags
text = re.sub(r"<w:p [^>]*>", "\n", xml)   # (if needed, before stripping) keep paragraph breaks
text = html.unescape(text)
```

Preserve paragraph structure with `<w:p` → newline BEFORE stripping tags (see comment). If the
docx has tables, either `pip install python-docx` into the venv (`uv pip install --python
/opt/hermes/.venv/bin/python3 python-docx`) or parse `word/document.xml` `<w:tr>/<w:tc>` cells.
This beats OCR for native .docx files; OCR remains for scanned PDFs.

## Pitfall — 'UID' is a readonly bash variable

Do not use `UID` as a loop variable name (`for UID in ...` → `readonly variable` error). Use `WHO`,
`IDX`, or a per-item names.

## Pitfall — Outlook-style subject prefixes mangle threaded-draft subjects (verified 2026-09-16)

`'Re: ' + h.get('Subject').lstrip('Re: Fwd: ').strip()` is WRONG for Outlook/Exchange-originated threads. `str.lstrip` strips individual CHARS, not a prefix:
- `RE: Succession Certificate` (Outlook uppercase RE:) → `E: Succession Certificate` → draft subject `Re: E: Succession Certificate`
- `RE : [Ticket No:7027] Infusion Confirmation Form and Prescription` (space before colon) → `Re: E : [Ticket No:7027] ...`

Hit twice in one session (CMS IndusLaw Succession thread + Medybiz Ticket-7027 thread). Use a regex prefix strip instead:

```python
import re
raw_subject = h.get('Subject', '')
clean = re.sub(r'^(?:RE|Re|re|Fwd|FW|E)\s*:\s*', '', raw_subject).strip()
subject = clean if clean.lower().startswith('re:') else 'Re: ' + clean
```

Handles `RE:` / `RE :` / `Fwd:` / leading `E:` (Outlook "EXTERNAL EMAIL" marker) in one pass. If the stored subject is already `Re: RE : ...` (Gmail keeps the nested Outlook prefix), leave it as-is — matching the thread's own subject is correct. Fix an already-created draft in place with `drafts().update` (pin `threadId`!) — see the "Revising an existing UNSENT draft" section. The two recipe blocks above that still show `.lstrip('Re: Fwd: ')` should be read with this correction.

## Pitfall — a DRAFT inside a thread LOOKS like a sent message in thread listings (verified 2026-09-17)

`threads().get(...)` and `messages().list(...)` return a draft sitting in a thread as a
normal-looking message: same subject, `From:` = the account, and a `Date` header equal to its
**creation** time — so it appears at the bottom of the thread looking exactly like the newest
sent mail. The only tell is `labelIds: ['DRAFT']` (a sent message carries `['SENT']`).

Rules:
1. Before answering "did that go out?" or reporting a reply as sent, check `labelIds`. Never
   claim a message was sent without seeing `SENT`.
2. When enumerating a thread to find the message you are replying to, SKIP entries whose
   `labelIds` contain `DRAFT` — a draft has no `Message-ID` header, so using it for
   `In-Reply-To` yields a detached draft.
3. When superseding an earlier draft of the same reply: `drafts().list(userId='me', q='<matter>')`,
   read each draft's headers to confirm its thread, `drafts().delete(...)` the stale one, then
   create the replacement — and report which `draft_id` survives. Two live drafts of one reply
   is a real "user sends the wrong version" hazard. (Real case: a prior turn's Ranka Amber SSA
   draft had to be deleted before the corrected one was created in the ICICI thread.)

## Pitfall — a draft that LISTS but will not OPEN in the Gmail web UI (malformed MIME headers)

Verified 2026-09-17 (NDR: *"I can see draft in the listing but I can't open the draft"*).

Cause: the stored draft carried an over-long, badly-folded `References` header whose Outlook
`Message-ID` Gmail had re-encoded into RFC-2047 encoded-words (`=?utf-8?q?=3CMA5PR01MB…?=`), and
the message carried **no `Message-Id`, no `MIME-Version` and no `Content-Type`/_Transfer-Encoding_
header at all**. That combination renders in the drafts LIST but fails to open in the composer.

Diagnose by reading the draft's STORED raw envelope —
`drafts().get(userId='me', id=..., format='raw')` → base64-decode → inspect the HEADER BLOCK:
a healthy draft has `MIME-Version`, `Content-Type`, `Content-Transfer-Encoding` and `Message-Id`;
the broken one had none of them, and its `References` contained `=?utf-8?q?…?=` encoded-words.

Fix — delete and rebuild with a minimal, clean header set:

1. `drafts().delete(userId='me', id=<broken_draft_id>)` — do not try to patch around it.
2. Build the raw MIME **by hand as a string** (not `EmailMessage`, whose folding is what produced
   the encoded-word mess) with an explicit header block, CRLF throughout:
   `MIME-Version` · `From` · `To` · `Subject` · `In-Reply-To` · `References` · `Message-Id` ·
   `Content-Type: text/plain; charset="UTF-8"` · `Content-Transfer-Encoding: 7bit` · blank line · body.
3. **Keep `References` SHORT — parent + grandparent only.** Never paste the whole thread chain;
   long chains fold, and folding is what Gmail mangles. `threadId` on `drafts().create` does the
   real threading work — a 2-ID `References` still lands the draft in the right conversation.
4. Supply your own `Message-Id` — `'<%s@mail.gmail.com>' % uuid.uuid4().hex.upper()[:24]`. Gmail
   substitutes its own on save, but its presence keeps the header block well-formed.
5. Verify by reading `format='raw'` back and confirming the header block is unfolded, carries no
   encoded-words, and still shows the intended `To` / `In-Reply-To` / `threadId`.

## Pitfall — a draft the user "can't open" may ALREADY HAVE BEEN SENT

Verified 2026-09-17 (Behera / Adv. Jasleen Kaur reminder). NDR reported the draft would not open
and asked for it to be deleted and redone. Re-checking state FIRST showed
`drafts().get(userId='me', id=<draft_id>)` now returning a message with `labelIds: ['SENT']`, and
the thread had gained a new `SENT` message whose body was verbatim the drafted text — it had been
sent from another client (phone) while the web view was stuck. Deleting and "redoing" it would have
created a duplicate draft for an email already delivered.

Rules:
1. Before deleting/redoing ANY draft the user reports as broken, re-read state:
   - `drafts().list(userId='me')` — is the artifact still a draft at all?
   - `drafts().get(userId='me', id=<old_draft_id>)` — which `labelIds` come back?
   - `threads().get(userId='me', id=<thread>, format='full')` — did a new `SENT` message appear
     after you created the draft? Diff its body/`Message-ID`/`References` against your draft.
2. **If it was sent: report that plainly with the send timestamp and DO NOT recreate.** Ask the
   user to confirm the send was theirs. Never silently reissue a message that already went out —
   and never claim you sent it: state that nothing in the drafting workflow can send.
3. Only rebuild when the artifact is genuinely still a draft. Report which `draft_id` is live
   afterwards.
4. When several drafts exist in the mailbox, list ALL of them with To/Subject/thread before
   concluding which one the user means — one of them may belong to a different session/matter
   entirely and must be left untouched.

## Pitfall — verify the thread has not advanced since NDR dictated the instruction

Verified 2026-09-19 (AR Portable Cabins apology reply-all). NDR dictated by voice:
*"I have just sent them an email saying no quotation was attached … tell them my apologies, I
found the quotation."* Taken literally that is a plain apology to the vendor's 17-Sep mail. But
the thread had actually advanced:

- 17-Sep 18:26 vendor: *"Attached quote with all details."*
- **19-Sep 19:31 (2 h before the dictation) NDR himself: *"No quotation was attached to this
  email … please attach the quotation again and send it."*

The message NDR was apologising for was HIS OWN from two hours earlier — not the one he was
telling me to reply to. Drafting naively against the 17-Sep vendor mail would have produced an
unexplained apology, and if the vendor had already re-sent the quote on Saturday evening, the
apology would have read as confused. Catching this produced a materially better draft: it
explicitly references *"my last email"*, so the recipient joins the dots.

**Rule — before drafting any reply NDR dictated by voice, re-read the thread tail and confirm
which message is actually being answered:**

```python
t = gmail.users().threads().get(userId='me', id=TID, format='metadata').execute()
for m in t['messages']:
    h = {x['name']: x['value'] for x in m['payload']['headers']}
    print(m['id'], [l for l in m.get('labelIds', []) if l in ('SENT', 'DRAFT')],
          '|', h.get('From'), '|', h.get('Date')[:31], '|', m.get('snippet', '')[:90])
```

1. **Thread against / reply-to the LAST message**, but name in the BODY whichever earlier message
   the instruction actually refers to. Those are often different messages.
2. If NDR's own recent message is what he is apologising for or correcting, say so in the body
   ("right after sending my last email I found the quotation — please disregard it") rather than
   leaving the recipient to infer it.
3. **Do not ask the vendor to redo an action NDR has already retracted.** When the parent message
   is NDR's own "please resend", the reply must explicitly cancel that request — a draft that
   repeats the ask contradicts the apology sitting directly above it.
4. **Flag the timing collision to NDR before creating the draft when it changes the framing**
   (e.g. vendor may have re-sent inside the gap). State the two timestamps and what you propose;
   one clarifying question is cheaper than an email that reads as confused. Note this is a
   FRAMING check, not a substitute for drafting — if the framing is unambiguous, draft.
5. A voice message saying *"just sent"* or *"I have just sent"* is a cue to search the thread for
   NDR's newest SENT message in the last few hours, not an assumption that the last thread message
   is the vendor's.

## Reply-all + injecting a DRA colleague into the Cc

Verified 2026-09-19 (AR Portable Cabins). NDR: *"reply all email. Making sure that Anbarasan, my
PM2, is also added on the CC on this."*

1. Read the thread head for the full recipient set — vendors typically Cc their own second
   contact (`Sayed T <sayed@arportablecabins.com>`) plus any DRA colleague already on the chain.
2. Resolve the colleague through `contact_resolver` — NEVER hand-type the domain. Anbarasan
   resolved to **`pm2.blr@draas.com`** (work), with `pm2.blr@drahomes.in` as a third address and
   `anbarasandraass@gmail.com` as home. DRA staff carry several addresses across
   draas.com / drahomes.in; picking the wrong one silently drops them from the matter.
   **NDR settled this on 2026-10-09, verbatim: *"For all my employees and colleagues we use their
   draas.com email address and not drahomes.in."* → for every DRA employee/colleague, the
   `@draas.com` address is the ONLY correct one.** Real correction in that session: Bharat Hawaldar
   was Cc'd as `sales1.blr@draas.com` (correct) while the contacts sheet listed
   `sales1.blr@drahomes.in` under "work" — NDR overrode the sheet. When the contacts sheet offers a
   `drahomes.in` address for a DRA staffer, use the `@draas.com` one and say so in the handoff note
   so he can confirm. `@drahomes.in` remains valid for third parties and for the company's external
   identity (e.g. the drahomes.in site), not for addressing staff.
3. **Do not duplicate a Cc** — if the colleague is already on the chain (Anbarasan had been Cc'd
   since NDR's original 14-Sep mail), keep him on and say so in the delivery note rather than
   listing him twice.
4. Reply-all means the vendor's `From` becomes `To`, and the thread's own `Cc` set is preserved
   plus the injected colleague. Re-specify `To`/`Cc` explicitly — the raw-MIME path applies no
   reply-all logic of its own.
5. In the delivery note, name every recipient and state who was added, so NDR can eyeball the
   audience before sending.

## Pitfall — a mail reported as "draft / unsent" may ALREADY HAVE BEEN SENT

Verified 2026-09-19. A prior turn's summary recorded the DRA liquidation-inventory email as
"draft live and unsent". A subject search showed it had actually gone out that afternoon
(`labelIds: ['SENT']`, 19-Sep-2026 16:56, to Prakash + Cc Bharat). The prior turn's own
verification had never re-checked `labelIds` after the user reviewed it.

**State carried in a context summary (or in your own earlier prose) is a claim, not evidence.**
Before describing any artifact as unsent — and before drafting a replacement or a chaser that
assumes it never went out — re-check:

```python
res = gmail.users().messages().list(userId='me', q='"<distinctive subject phrase>"',
                                    maxResults=20, includeSpamTrash=True).execute()
for m in res.get('messages', []):
    full = gmail.users().messages().get(userId='me', id=m['id'], format='metadata',
        metadataHeaders=['Subject','From','To','Cc','Date']).execute()
    h = {x['name']: x['value'] for x in full['payload']['headers']}
    print(full.get('labelIds'), h.get('Date'), h.get('To'), h.get('Cc'))
```

`includeSpamTrash=True` matters — a sent mail that got auto-archived or filed will otherwise
be invisible. `['SENT']` = went out; `['DRAFT']` = still yours to review. This is the inverse
of the "DRAFT inside a thread LOOKS like a sent message" pitfall above; both are cured by the
same one-field check.

## Pitfall — `%`-formatting an HTML email body breaks on literal `%` in the content (hit twice: 2026-09-14, 2026-10-05)

Building the HTML part with Python `%`-formatting (`html = """...%s...""" % (LINK,)`) fails the
moment the body contains a literal percent sign — which real business email almost always does:

```
TypeError: must be real number, not str
```

Real cases in this mailbox: *"GST 18% extra"*, *"almost 80% more"*, and `width:100%` inside a
style attribute. The `%` operator reads `18%` / `80%` as a conversion specifier and raises (or,
with a tuple of the wrong arity, silently mis-substitutes later in the string — worse, because
the draft is created and looks plausible). Escaping to `%%` technically works but is fragile:
the next person to edit the copy has to remember it.

**Fix — named sentinel tokens + `.replace()`, never `%`-formatting, for email bodies:**

```python
LINK = "https://content.ahfl.in/a/DRA-ART-2026-000047"
html = """<div>...<a href="__LINK__">__LINK__</a>...
<p>GST 18% extra &mdash; almost 80% more than target.</p></div>"""
html = html.replace("__LINK__", LINK)
```

Same rule for `str.format()` — a literal `{` (inline CSS, `@media` blocks, JS in an embed)
raises `KeyError`/`IndexError`. `.replace()` on sentinel tokens has no metacharacters at all, so
it is the only interpolation method that is safe on real email copy. Keep the plain-text
alternative (`m.set_content(plain)`) token-free, or substitute it too.

Verify after drafting: walk the saved text/html part and assert the substituted value is present
AND the sentinel string is gone.

## Recipient-audience check — internal pricing must NOT reply-all into a thread carrying the counterparty

Verified 2026-10-05 (Ranka Udaya site office; Gowri Singh coordinating with vendor Sayed / AR
Portable). NDR dictated *"reply all to her email"* for a note containing his **internal cost
position**: his independent benchmark of the vendor's plumbing schedule, his target build-up
(₹8–10 lakh tops), the vendor's current ~₹18 lakh figure, and the line *"almost 80% more than I
think this should sit"*. Landing that on the vendor thread would have handed Sayed NDR's
walk-away number and his view that the quoted furniture line is padded.

It didn't, because Gowri's clarification mail was a **separate thread she started herself**
(subject `Clarifications on the Site Office Quotation...`, `threadId == messageId` — it has no
parent) addressed to `ndr@draas.com` only. Reply-all on it resolves to Gowri alone. The vendor
was on a *different* thread (`Re: Ranka Udaya Site Marketing Office…`) where Gowri and Sayed are
both participants.

Rules:

1. **Before honouring a "reply all", resolve which thread and enumerate its participants.**
   `threads().get(...)` and collect every `From` / `To` / `Cc` across the thread's messages.
   Never assume "reply all" is safe merely because NDR said it — he is describing the audience he
   has in mind, not auditing the header for you.
2. **Classify the body first: internal-analysis or counterparty-facing?** Cost benchmarks, target
   or walk-away prices, margin/deal analysis, "their quote looks padded", negotiating positions,
   and colleague-to-colleague instructions are internal. If the body is internal and ANY
   participant is the counterparty (vendor, consultant, landowner, bank, tenant), that thread is
   the wrong target — even when NDR named it. Use the internal-only thread, or start a new mail.
3. **An adjacent thread is not the same thread.** Two threads on one matter — one with the vendor,
   one without — is the normal shape once a colleague is coordinating. Pick the one without the
   counterparty, and thread the draft onto that one (correct `In-Reply-To`/`threadId`).
4. **`draft_reply_create` fills `To` from the parent's sender only** — it performs no reply-all
   reconstruction. Genuine reply-all needs the raw-MIME path with `To`/`Cc` re-specified
   explicitly. When the parent's sender is the only non-you participant, surface that in the
   delivery note so NDR knows reply-all resolved to one person.
5. **State the audience in the delivery note** — name every recipient, and when a counterparty was
   deliberately excluded, say so and why ("that thread is just you and Gowri; Sayed is on the
   vendor thread, so reply-all resolves to Gowri alone"). This is how NDR catches a mis-targeted
   draft before he sends it.
6. This is the content-side mirror of the address-side rule in *"Reply-all + injecting a DRA
   colleague into the Cc"* below: that one guards **who** is on the draft, this one guards **what**
   the draft is allowed to say to them.

## Proactively SHORTEN `References` on long reply-all chains (verified 2026-10-05)

The "draft lists but will not OPEN" pitfall (above) is caused by header folding. On a long-running
matter the parent's `References` can carry 15+ message IDs (~1.5 KB) — reply-all on top of that is
the highest-risk shape in this mailbox. Real case: the Succession Certificate thread's parent had
**16 IDs**; the first draft built from the full chain carried a 1.4 KB `References`.

**Rule: never inherit the full chain.** Take grandparent + parent only — `threadId` on
`drafts().create`/`update` does the real threading work, and a 3-ID `References` still lands in the
right conversation:

```python
ids = re.findall(r"<[^>]+>", h.get("References", ""))
m["References"] = " ".join(ids[-2:] + [orig_msgid])   # 3 IDs, ~220 chars
print("References length:", len(m["References"]))      # assert < 400
```

Rebuild-and-`drafts().update` is fine for this (pin `threadId`, then re-verify). Assert the
shortened length and re-check `threadId` in the same pass — the update re-threads silently
otherwise (see the drafts().update pitfall below).

## Thread map — "Succession Certificate" (Ranka family P&SC)

threadId `1a0680657add515f`. Adult family, advocates (CMS IndusLaw + Law Square), one long chain.
Drafting here means reply-ALL to a large audience — get the audience right or the family sees a
half-addressed mail.

- Advocates: Maneesha Kongovi (CMS IndusLaw, `maneesha.kongovi@cms-induslaw.com`), Arvind Krishna G
  (CMS, early msgs only), Sanjay Sethiya (`sanjay@lawsquare.in`), Nishanth Adv
  (`nishanth@lawsquare.in`).
- Family set (NOT the same in every message — read the parent's own Cc): Mamta Rathod
  `mamatadr@gmail.com` (2nd address `maadi32@gmail.com`), Manish Ranka `mdr@draas.com`,
  Dharmesh Ranka `dharmesh.ranka1@gmail.com` (note: the *Winding up* thread uses
  `dharmesh.ranka@gmail.com` — no "1"; always verify per-thread).
- **Substance:** succession certificate for late Dinesh D. Ranka (d. 20.07.2022). The pivot is
  **who gives the GPA to Rajendra** — final position: Nishant, Mamata Ranjeeth Rathod and Kanta
  Ranka only; Dharmesh and Manish represent themselves. Documents in play: P&SC Petition,
  Order III Rule 2 application + affidavit, and the GPA. Marked-up copies sit in Drive
  (Family > Final Settlement Docs) as `20261002 … (NDR markup).docx`.
- **Known open defect:** the Order III Rule 2 affidavit cites the GPA as dated 01.10.2026 while the
  GPA is executed 03.10.2026. Flagged to NDR + in the 5-Oct draft; confirm before printing.
- Petitioners: No. 1 Nishant, Nos. 2 and 3 Dharmesh and Manish (self-represented), Nos. 4 and 5
  Mamata and Kanta (GPA-represented).

## Notes

- This creates a DRAFT only (`drafts().create`) — never `messages().send()`.
  Bcc shows up in the draft's own headers; that is expected and fine.
- `canonical_uid` warning on stderr ("vault has no identity mapping … using raw
  id as fallback key") is benign when the profile check prints the right mailbox.
- The gws_skill_bridge `draft_reply_create` / `draft_create` are the sanctioned entry points;
  raw-MIME drafts().create is for cases that need custom threading headers or per-user env override.
- **BRIDGE DRAFT FUNCTIONS CAN RETURN PLAUSIBLE SUCCESS WITH NO DRAFT CREATED (verified 2026-10-06).**
  Through the skill bridge, `draft_reply_create`/`draft_create` returned success-like strings
  ("Draft created successfully", IDs like `r-4233332030418157975`) but NOTHING landed in the
  Gmail Drafts folder — the user could not see the draft at all, and `drafts().list` showed the
  mailbox empty of them. Some calls even produced a real message-id dressed up as a draft id.
  After this failure mode, NDR's firm instruction was: **"You need to use the GWS client only to
  create draft and nothing else."** The raw-MIME `build_service('gmail','v1', service_name='google-draas')`
  + `drafts().create(userId='me', body={'message': {'raw': raw, 'threadId': TID}})` path (sections
  above) is the RELIABLE route — verified end-to-end by reading the draft back from Gmail
  (`drafts().get` → confirm `labelIds: ['DRAFT']`, correct `threadId`, parts carry the attachment).
  RULE: **never report a draft as created on the bridge's return string alone** — list drafts and
  read the object back; if the bridge says success but nothing is in Drafts, rebuild via raw MIME.
- **Draft `threadId` placement — INSIDE the message object, not at the draft level (verified
  2026-10-06).** `drafts().create(body={'message': {'raw': raw, 'threadId': TID}})` lands the draft in
  the intended thread. Passing threadId higher in the payload body, or relying on In-Reply-To headers
  alone after a bridge call, produces a draft whose `threadId` is a NEW thread — the user looks in the
  existing conversation and sees nothing. Always read the CREATED draft's `message.threadId` and assert
  it equals the master thread (e.g. `1a07b0f2821d61f6`) before reporting.
- **Sending without threading headers creates a detached thread (verified 2026-10-06, re-sent Jasleen
  email).** A message sent with NO `In-Reply-To`/`References` lands as a brand-new conversation even
  when the user believed they replied to the chain. When NDR says "resend but on the same thread",
  rebuild the raw MIME with `In-Reply-To` = the anchor message's OWN Message-ID (not its
  In-Reply-To), carry `References`, pass `threadId`, and also open the draft with a one-line note
  ("Resending in the same thread — please ignore the earlier copy").
## Pitfall — a threaded draft that "won't join the thread": the In-Reply-To anchor Message-ID must EXIST in the target thread (verified 2026-10-06/07, Jasleen Behera thread)

Symptom: after `drafts().create(..., threadId=<master>)` the draft's `message.threadId` comes back a NEW thread id, or the user sends it and it lands as a detached conversation. The usual suspects (missing `threadId`, wrong header names) were already covered — but there is a subtler cause found at the very end of the Behera saga:

**The `In-Reply-To` value must be a Message-ID that ACTUALLY EXISTS as a message inside the thread you want.** If you anchor to an ID that is not present in the master thread (a stale/guessed ID, an ID from a different thread, or the anchor's own `In-Reply-To` instead of its own `Message-ID`), Gmail cannot place the draft in the intended conversation — it creates a fresh thread even though your To/Cc/Subject/References look perfect.

Diagnose in two steps:

```python
# 1. does the proposed anchor Message-ID exist anywhere? q may need no angle brackets
res = gmail.users().messages().list(userId='me', q='"<anchor-id> OR <anchor-id-no-brackets>"', maxResults=5).execute()
# 0 messages  ==> the anchor is not in the mailbox; the draft will go to a new thread

# 2. list the master thread's messages and find the REAL anchor (usually the LAST message)
t = gmail.users().threads().get(userId='me', id=MASTER_TID, format='full').execute()
for m in t['messages']:
    h = {x['name']: x['value'] for x in m['payload']['headers']}
    print(m['id'], h.get('Message-ID'), h.get('From','')[:30], h.get('Date','')[:25])
```

Rules:
1. **Anchor `In-Reply-To` to a message that is VERIFIED inside the target thread** — fetch the thread, take the desired anchor message's own `Message-ID` header, and pass that exact value. The anchor is normally the latest message (or the latest message from the counterparty). An anchor that merely "looks like it belongs" (similar subject, same people) is not enough — Gmail matches the exact ID.
2. **After ANY draft create/update, re-read the draft's `message.threadId`** and — when the draft's threadId is a NEW id — run the q-search above before concluding the send will thread right. Gmail shows a provisional threadId on drafts; the send-time placement is computed from In-Reply-To/References, so the draft's provisional id alone is NOT proof. The q-search proving the anchor exists in the master thread IS the proof.
3. **A provisional/odd draft threadId is normal for drafts** — do not panic-delete on it. Verify the header chain (anchor exists + References carry the chain) and report: "draft is in master thread; Gmail computes placement at send from these headers."
4. This is the endpoint of the Behera saga lesson: proper threading = anchor Message-ID exists in the thread + References chain + `threadId` passed inside `message` + verification that the anchor is resolved. Skimp on the existence check and a "perfect-looking" draft re-creates the detached-thread failure the user has now hit multiple times.

## NDR's counsel-facing tone discipline — SHORT, cooperative, suggestive (verified 2026-10-07, Jasleen/Behera)

When drafting to external legal counsel (advocates, law firms) on an active matter, honor these hard constraints NDR stated explicitly:

- **Never say the draft/work "is not filing-ready", "has defects", or otherwise question the lawyer's professional authority.** Frame every review point as: "these are our thoughts / areas we felt could be strengthened — ultimately your professional judgment decides."
- **Keep it SHORT.** "That email is too elaborate, way too much content" — a wall of analysis overwhelms the lawyer. Lead with the one-sentence ask, then the few points that matter, then the specific request for their view. Detailed line-by-line feedback belongs in the attached commented document (Word docx with native comments / anchored HTML), NOT in the email body.
- **Structure with clear HTML/CSS** (short header band, one or two callout boxes, a single REQUEST box) so the ask is instantly visible.
- **End with a concrete ask**: their status update + their plan of action + a timeline they can commit to (or their professional view on which of two strategy tracks to pursue).
- When the user shares strategy input from a third party (e.g. a fellow customer's advice), present it as "input we received / he advises / we would welcome your view" — never as a directive to the lawyer.

- **Bridge parameter names are literal, and a wrong key fails SILENTLY.** `draft_reply_create`
  takes **`to`** (not `reply_to`) and **`message_id`** (not `thread_id`). Verified 2026-09-05
  (Ranka Amber follow-up to Vardhan Ventures): passing `reply_to=` raised no error — the bridge
  simply ignored it, fell back to deriving the recipient from the thread, and produced a draft
  addressed to **NDR himself** instead of the counterparty. A mis-addressed draft sitting in the
  mailbox is one keystroke away from being sent to the wrong person.
  **Rule:** after every `draft_reply_create` / `draft_create`, read back the saved draft's headers
  (`drafts().get` → `payload.headers`, remembering `drafts().get` accepts no `metadataHeaders`)
  and assert `To` is the counterparty, plus that `In-Reply-To`/`References` are intact. If `To` is
  wrong, `gmail.users().drafts().delete(userId="me", id=<draft_id>)` and recreate with `to=` —
  never hand over a draft you have not read back.

## Shortcut that works: `tools.gws_ops` + `load_credentials` (verified 2026-10-09)

`gws_ops` already implements correct reply-drafting. Use it instead of hand-rolling raw MIME when
you do NOT need custom headers:

```python
import sys; sys.path.insert(0, "/opt/hermes")
from tools.gws_auth import load_credentials          # NOT build_service — see trap below
from tools import gws_ops

creds = load_credentials("google-draas")             # raw Credentials
res = gws_ops.gmail_reply_draft(creds, message_id=PARENT_MID,
                                body=html_string, html=True,
                                cc="Prakash Singh <psingh@draas.com>, Sarthak Sharma <admin3.blr@draas.com>")
# -> {'draft_id': 'r7010525338867523716', 'message_id': '...', 'thread_id': '1a10c71519ac4760'}
```

`gmail_reply_draft` does the whole job in one call: `To` ⇐ the parent's `From`, subject gets a `Re:`
prefix, `In-Reply-To`/`References` are chained, and `threadId` is passed to `drafts().create` — so
the draft lands in the right conversation. It also sets a real `Message-ID` and uses
`MIMEMultipart("alternative")` with a `text/html` part, which avoids the malformed-header
"lists but won't open" failure above. **Draft-only — it can never send.** (Same generation as
`gmail_draft_create`, which takes `to=`/`subject=`/`body=`/`cc=`/`html=`.)

**Trap — `build_service()` returns a Resource, not credentials.** `gws_ops` functions expect raw
`Credentials`; passing `build_service("gmail","v1",...)` raises
`AttributeError: 'Resource' object has no attribute 'before_request'` (or similar) deep inside
`.execute()`. Use `load_credentials(service_name)` for anything you hand to `gws_ops`.

### Pitfall — `gws_ops` writes LOWERCASE header names, so a case-sensitive verify reads `None`

Verified 2026-10-09 (Ranka Oasis / Jiraaf round-3 draft). `gws_ops._build_message` sets
`msg["to"] = to`, `msg["subject"] = subject`, `msg["cc"] = cc` for the multipart/HTML branch — so
the stored headers come back as `to`, `cc`, `subject`, `in-reply-to`, `references`. A verification
dict built with capitalised keys:

```python
h = {x["name"]: x["value"] for x in m["payload"]["headers"]}
h.get("To")       # -> None   <-- WRONG, looks like the draft has NO recipient
```

returns `None` for every field and reads exactly like a broken draft. The draft is fine.

**Always lowercase the header name when building the lookup dict** (this is the correct idiom for
every Gmail verification, bridge-built or raw-MIME):

```python
h = {x["name"].lower(): x["value"] for x in m["payload"]["headers"]}
for k in ("from","to","cc","subject","in-reply-to","references","message-id"):
    print(k + ":", h.get(k))
```

Assert on the lowercased dict, then also confirm `labelIds == ['DRAFT']` and that
`message.threadId` equals the target thread. Printing through `threads().get(..., format='metadata')`
and listing each message's `Date | From | Subject | labelIds` is the fastest way to see the draft
sitting as the newest entry in the right thread.