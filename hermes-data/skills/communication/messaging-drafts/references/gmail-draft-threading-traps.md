# Gmail reply-all threading traps (verified Sep-Oct 2026, Behera RERA pack sessions)

> Sibling ref: `dictated-confirmation-email-minimalism.md` — when NDR/Sarthak dictates a 2–3 point
> confirmation mail after a call, build ONLY those points: no dated document inventory, no
> "items still pending" block, and never upgrade "he will confirm next week" into "confirmed".
>
> Sibling ref: `draft-revision-and-internal-request-tone.md` — revising an existing draft in
> place (rebuild the MIME, keep the attachment, delete the superseded draft so exactly ONE live
> draft exists), the internal-request **tone** rule (never leave "not our scope" language in a
> mail where we are the ones asking another team for work), and the anonymous public-link probe
> to run before any collateral link goes to a customer.

User repeatedly reported reply-all drafts landing in the WRONG thread or not appearing
at all. All root causes below were hit in one week; each has a stable fix. Read this
before building ANY reply-all with attachments.

## 1. threadId goes INSIDE the `message` object

`drafts().create(body={'message': {'raw': raw, 'threadId': MASTER_THREAD}})` — NOT
`body={'message': {'raw': raw}, 'threadId': MASTER_THREAD}`. At draft top level the
field is IGNORED and Gmail mints a new thread even when In-Reply-To/References look
correct. Always verify with `drafts().get(...)` → `message.threadId == MASTER_THREAD`.

## 2. Skill-bridge draft_reply_create is for simple replies only

For a true reply-all WITH attachments (To: specific recipient, Cc: the crew):
- bridge sets To = anchor message's From (WRONG for reply-all)
- bridge drops the Cc list
- bridge cannot attach files
Build the raw MIME manually instead:
```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import base64
from tools.gws_auth import build_service
svc = build_service('gmail', 'v1', service_name='google-draas')

msg = MIMEMultipart('mixed')          # alternative part for plain+HTML body
msg['To'] = 'Lawyer <l@firm.com>'
msg['Cc'] = 'crew1@x.com, crew2@y.com'
msg['Subject'] = 'Re: ...'
msg['In-Reply-To'] = anchor_msg_id
msg['References'] = anchor_references_chain + ' ' + anchor_msg_id
# attachments: MIMEBase('application','pdf') + set_payload(f.read()) + encode_base64 +
#   add_header('Content-Disposition','attachment',filename=...)
raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
svc.users().drafts().create(userId='me', body={'message': {'raw': raw, 'threadId': MASTER}})
```

## 2b. Native gws_gmail_reply_draft ALSO overrides explicit To/Cc (verified 2026-10-10)

The trap in §2 is NOT bridge-only. Calling the NATIVE tool with explicit params —
`gws_gmail_reply_draft(message_id=ANCHOR, to="Dharmesh <x@gmail.com>", cc="Manish <mdr@draas.com>, ...", ...)`
— the draft it mints has **To = anchor message's From** and **NO Cc at all**, even though
the tool reports success and a correct `thread_id`. The draft matches the thread UI, so it
*looks* fine; only header read-back reveals the recipients were silently replaced.

This makes raw-MIME-via-build_service the ONLY reliable path for ANY reply-all with
non-trivial recipients, native tool or bridge. Same recipe as §2, plus:
- `To`/`Cc` set explicitly as headers on the MIMEMultipart
- `In-Reply-To` / `References` built from the anchor's metadata headers
- `threadId` INSIDE the `message` object on drafts().create
- Delete the botched native draft first (gws_gmail_draft_delete) so the user's Drafts
  folder doesn't show two replies.

## 2c. Read-back detail: drafts().get rejects metadataHeaders

`svc.users().drafts().get(userId='me', id=DRAFT_ID, format='metadata', metadataHeaders=[...])`
raises `TypeError: Got an unexpected keyword argument metadataHeaders` (gmail v1 client).
Use `format='full'` and parse `message['payload']['headers']` the same way. This is the
verification step that §5 demands — never skip it, but use the right call shape.

## 2d. Recipient emails: trust the THREAD, not the contacts sheet

For a reply-all, resolve recipient addresses from the actual thread's To/Cc headers where
possible. The contacts sheet can be stale — e.g. Manish Ranka's sheet work email is
`mdr@drahomes.in`, but every current thread uses `mdr@draas.com`. Dharmesh's Gmail appears
in contacts as `dharmesh.ranka1@gmail.com` and the live threads confirm it. When a voice-
dictated recipient ("Manish as NBR at dras") is garbled, cross-check it against the thread
participants (Manish Dinesh Ranka = MDR = mdr@draas.com) before drafting.

### 2e. The dictated participant list is a RECOLLECTION, not the record (verified 2026-10-10)

When NDR dictates a follow-up he often hedges about the crew: *"There is an earlier email
I have sent to Dharmesh, Manish, everything regarding Gunjur, whoever, **Mamta also may be
on that mail**, I have given them all an update."* Mamta (Mamta Rathod, a family contact,
not staff) was **not** on that thread at all — the 22 Sep mail went to Dharmesh + Manish
only, and a "Gunjur" search across both ndr@draas.com and ndr@ahfl.in confirmed she never
appears on any Gunjur thread.

Rules:
- Build To/Cc from the **thread's own headers** — union of To+Cc across every message
  (`threads().get(userId='me', id=THREAD, format='metadata')`), minus the user themselves.
  The simplest and strongest signal is the **anchor message's own headings**: NDR's 24 Sep
  note read `To: Dharmesh Ranka` / `Cc: Manish Ranka`, so reply-all = exactly that pairing.
- Treat any name the user only *tentatively* attaches ("may also be on that mail",
  "whoever") as **unverified**. Never add a tentative name to the Cc on your own
  initiative — that silently widens distribution to someone who was never in the loop.
- Draft with the verified participants and **flag the discrepancy back in the reply**
  ("Mamta was not on that thread — add her to the Cc?"). The user decides; you don't guess.
- Only the two/three names NDR states flatly ("I have sent to Dharmesh, Manish") are safe
  to carry across unverified — and even then, confirm against the thread headers.

### 2f. Working assembly for a reply-all whose anchor is the user's OWN sent mail

Confirmed end-to-end 2026-10-10 (Gunjur follow-up, thread `1a0c814b15776596`). This is the
reliable recipe — do NOT reach for `gws_gmail_reply_draft` (§2b, it replaces To/Cc) or the
skill bridge (§2):

```python
em = EmailMessage()
em['To'] = 'Dharmesh Ranka <dharmesh.ranka1@gmail.com>'
em['Cc'] = 'Manish Ranka <mdr@draas.com>'
em['Subject'] = 'Re: ' + original_subject
em['In-Reply-To'] = anchor_mid                      # anchor's Message-ID
em['References']  = anchor_refs + ' ' + anchor_mid  # full chain + anchor
em.set_content(plain_text_fallback)
em.add_alternative(open('/path/body.html').read(), subtype='html')
raw = base64.urlsafe_b64encode(em.as_bytes()).decode()
d = g.users().drafts().create(userId='me',
        body={'message': {'raw': raw, 'threadId': THREAD}}).execute()   # §1: INSIDE message
```

Read back with `drafts().get(id=DRAFT, format='raw')` (§2c — `format='metadata'` with
`metadataHeaders=` raises TypeError) and assert: `To`, `Cc`, `Subject`, `In-Reply-To`,
`References`, `message.threadId == THREAD`, exactly one `text/plain` + one `text/html`
part. For an HTML body also re-run the full-width greps **on the stored HTML part**:
`max-width` count == 0, `width:100%` count >= 1 (NDR's standing rule — see
`html-body-email-drafting.md`). `EmailMessage` + `add_alternative` is cleaner than
`MIMEMultipart('mixed')` and produces the same two-part shape.

### 2g. Anchor is a THIRD PARTY's message — read its own To/Cc, don't rebuild from the dictation

Confirmed end-to-end 2026-10-16 (Prestige Hermitage khata reply-all, thread `1a0c935faedae159`).
This is the most common reply-all shape in business threads: the user is replying to
*someone else's* message, not to his own sent mail (contrast §2f).

- **Reply-all = To the anchor's `From`, Cc the anchor's `Cc`** — minus the user himself.
  One metadata read gives everything needed in a single call:
  ```python
  m = g.users().messages().get(userId='me', id=ANCHOR, format='metadata',
          metadataHeaders=['Message-ID','References','Subject','From','To','Cc']).execute()
  h = {x['name']: x['value'] for x in m['payload']['headers']}
  ```
  `Message-ID` → `In-Reply-To`; `References` + ` ' ' ` + `Message-ID` → `References`;
  `Subject` is already `Re:`-prefixed, reuse it verbatim rather than re-prefixing.
- **The anchor's Cc is frequently the whole answer.** On Hermitage the anchor
  (Manish, 13 Oct 15:37) already carried `Cc: Dharmesh Ranka <dharmesh.ranka1@gmail.com>`,
  which is exactly the pairing NDR dictated ("reply to Manish's email marking Dharmesh").
  Read it and confirm the match instead of assembling recipients from memory — and if the
  dictated crew is *narrower* than the anchor's Cc, say so rather than silently dropping
  someone who was in the loop.
- **Resolve addresses from the live thread, not the contacts sheet** (§2d). Dharmesh
  Ranka exists as both `dharmesh.ranka1@gmail.com` and `ddr@draas.com` in the sheet; every
  current thread uses the Gmail one for him and `mdr@draas.com` for Manish (never the
  stale `mdr@drahomes.in`).
- The §2b trap **re-fired here** — `gws_gmail_reply_draft` was not attempted; the raw-MIME
  path was used from the outset. Keep doing that for any reply-all with a real recipient
  list; the native tool replaces To with the anchor's From and emits no Cc at all.

### 2h. List the Drafts folder BEFORE creating — check for a leftover in the same thread

```python
ds = g.users().drafts().list(userId='me', maxResults=25).execute().get('drafts', [])
```
For each, read `format='metadata'` with `metadataHeaders=['To','Cc','Subject']` and print
`threadId`. Verified 2026-10-16: two unrelated drafts were sitting in the folder, neither in
the target thread — so the new reply was safe to create. Had an earlier session left a draft
in the same thread (easy to do when a prior turn was told "send this and I'll review"), the
user's Drafts folder would show **two replies to the same mail** and he has to guess which to
send. One cheap call removes that failure mode; do it as a matter of course.

### 2i. The skill bridge DOES honour an explicit `to=` / `cc=` — but only via KEYWORD args (verified 2026-10-16)

§2's blanket "bridge drops the Cc list" needs a nuance. `tools.gws_skill_bridge.call()` is
declared `call(operation, service_name="google-draas", **kwargs)` — **every argument after the
operation name must be passed as a KEYWORD.** Passing the payload dict positionally silently
lands it in `service_name` and the call dies with a badly misleading error:

```
call("draft_reply_create", {"message_id": ..., "to": ..., "cc": ..., "body": ...})
→ RuntimeError: Failed to load {'message_id': '...', 'to': '...'} credentials:
  Invalid or missing service name: "{'message_id': '...', ...}"
```

That message looks like a vault/credential problem. It is not — it is a positional-argument bug.

Correct form (worked end-to-end on the Prestige Hermitage khata reply-all, thread
`1a0c935faedae159`, 2026-10-16):

```python
import sys; sys.path.insert(0, "/opt/hermes")
from tools.gws_skill_bridge import call
call("draft_reply_create",
     service_name="google-draas",
     message_id=ANCHOR_ID,
     to="mdr@draas.com",                      # explicit override — honoured
     cc="dharmesh.ranka1@gmail.com",          # explicit override — see the verification gap below
     body=HTML_BODY,
     html=True)
```

Why this matters: `draft_reply_create` sets `message["To"] = args.to or <anchor From>`, so an
explicit `to=` is the **lightweight fix for the §3b bug** (anchor = your own sent mail → To
becomes yourself). When the anchor is the user's own outbound message and the recipients are
simple (no attachment), this is one call instead of hand-rolled raw MIME.

**Verification gap to close yourself:** `draft_get` prints only `to`, `subject`, `body` — it does
NOT print `Cc`. So a bridge-created reply-all is only *half* verified by the standard read-back.
Either read the draft with `format='raw'` / `format='full'` and parse the headers, or state
plainly to the user that To is confirmed and Cc should be eyeballed before sending. Never report
"Cc verified" off a `draft_get`.

Still reach for the §2/§2f raw-MIME path when you need an **attachment**, a full
`References` chain, or `To`+`Cc` you can assert in one read-back.

### 2i-bis. Native `gws_gmail_draft_create` DOES honour `cc=` — but `gws_gmail_get` will not show it (verified 2026-10-31)

For a **new** (non-reply) draft, `gws_gmail_draft_create(to=..., cc=..., subject=..., body=...)`
carries the Cc through correctly — unlike `gws_gmail_reply_draft` (§2b, which drops it). The trap
is the verification step: `gws_gmail_get` on the returned `message_id` prints `from` / `to` /
`subject` / `body` and **silently omits `cc`**, so a Cc that WAS set looks unconfirmable and a Cc
that was dropped looks identical.

```python
import base64
from tools.gws_auth import build_service
g = build_service('gmail', 'v1', service_name='google-draas')
d = g.users().drafts().get(userId='me', id=DRAFT_ID, format='raw').execute()
raw = base64.urlsafe_b64decode(d['message']['raw'] + '==').decode('utf-8', 'replace')
for line in raw.split('\r\n')[:40]:
    if line.split(':')[0] in ('To', 'Cc', 'Bcc', 'From', 'Subject'):
        print(line)
```

Never report "Cc verified" off `gws_gmail_get` — it cannot see the header. Also read back the
draft's own `draft_id` (not just the message id): the account + draft id + message id belong in
the same sentence when you report it (§6).

### 2j. Attachments cannot be ADDED to an existing draft — recreate it

There is no "attach a file to draft X" call. If NDR supplies the file *after* the draft was
created (the common shape: the body is dictated first, the screenshot arrives later), the draft
must be **deleted and rebuilt** as a multipart MIME with the file part attached, keeping the same
`threadId` + `In-Reply-To`/`References`. Do not tell the user to "just drag it in" unless you are
also willing to hand back the exact file path.

Corollary: when the source image is on **WhatsApp**, it may be unreachable (inbound media is not
stored as bytes — see the `whatsapp-reply` skill). Then the honest move is: create the draft with
the complete body, tell the user the attachment could not be retrieved, and ask him to forward
the image into the chat so the draft can be rebuilt with it. Say that up front rather than
promising a draft "ready with the attachment".

### 2k. `gws_gmail_reply_draft` has NO `to` parameter — and the bridge path REFLOWS your body (verified 2026-10-24)

Three separate traps hit while drafting a short scheduling reply to Dharmesh + Manish
(thread `1a1178935be29c75`, "Re: Call for a meeting"):

1. **The native tool cannot be told who to send to.** Its schema is
   `gws_gmail_reply_draft(message_id, body, cc, bcc, html, service_name)` — there is **no `to=`**.
   Anchored on NDR's own latest mail in the thread it threaded correctly and minted
   `To: Nishant Ranka <ndr@draas.com>` (§3b) — with nothing to override it. The only fixes are
   re-anchoring on the counterparty's message (§3b, "better still") or the bridge (§2i).
2. **The bridge honours `to=`/`cc=`… and re-wraps the body at ~72 columns with hard CRLFs.** The
   dictated text came back as
   `"…stay as they are for whichever\r\nday we fix, so we come in with a list\r\nand leave with decisions."`
   — mid-sentence breaks that post as visibly broken lines in the recipient's client.
   **Keep every body line under ~60 characters** (short paragraphs, break list items yourself)
   and re-read the raw to confirm no sentence straddles a break. Raw MIME via
   `EmailMessage`/`set_content` (§2f) does not do this to you.
3. **A draft's message id 404s on `messages().get`.** Verifying a just-created draft with
   `gws_gmail_get(message_id=<draft msg id>)` returned
   `HttpError 404 … Requested entity was not found`. Use
   `drafts().get(userId='me', id=DRAFT_ID, format='raw')`, base64-decode, and print the whole MIME
   head — that is also the only read-back that shows `Cc` honestly (§2i, §6).

**Run the builder script ONCE.** In `execute_code`, never call `subprocess.run(...)` a second time
just to print `stderr`: the second run re-executes every side effect. It left **two live reply
drafts** in the folder this session and burned the 300 s script budget. Capture both streams from
the single run — `r = subprocess.run(cmd, ...); print(r.stdout[-2000:], "ERR:", r.stderr[-600:])`
— then sweep `drafts().list()` and delete strays (§2h).

## 3. Anchor = latest message ALREADY in the master thread

Find via `threads().get(userId='me', id=MASTER_THREAD, format='metadata')`; use the
last message's Message-ID as In-Reply-To and its References chain (+ own ID) as
References. A sent email WITHOUT In-Reply-To/References becomes a standalone thread —
recreate the draft, don't try to re-parent it.

## 3b. Reply-draft on a message YOU sent addresses it back to YOU

`gws_gmail_reply_draft(message_id=<a message the user sent>)` sets **To = the anchor
message's From** — and when the anchor is the user's *own sent* mail, From is the user.
Verified 2026-10-09: a follow-up to Anbarasan M was meant for `pm2.blr@draas.com`, but
the draft came back with `To: Nishant Ranka <ndr@draas.com>`. Had the user simply hit
send, the follow-up would have gone to himself.

This is the most dangerous draft bug in the set, because the draft *looks* completely
fine in the Drafts list and in the thread UI — only the To header is wrong.

**Fix — correct the recipient and keep the thread.** Update the draft in place with a
raw MIME that sets To explicitly:
```python
# read the anchor's Message-ID first — you need it for threading
m = svc.users().messages().get(userId='me', id=ANCHOR_ID, format='metadata',
                               metadataHeaders=['Message-ID','Subject']).execute()
orig = {h['name'].lower(): h['value'] for h in m['payload']['headers']}['message-id']

em = EmailMessage()
em['To'] = 'Anbarasan M <pm2.blr@draas.com>'   # THE FIX
em['Subject'] = 'Re: ' + original_subject
em['In-Reply-To'] = orig
em['References'] = orig
em.set_content(body)
raw = base64.urlsafe_b64encode(em.as_bytes()).decode()
svc.users().drafts().update(userId='me', id=DRAFT_ID,
                            body={'message': {'raw': raw, 'threadId': THREAD}})
```
Then read the draft back with `format='metadata'` and assert **To**, **In-Reply-To** and
**threadId**. The write returns success even when the header is wrong, so the read-back
is the only proof.

**Rule:** whenever the anchor message is the user's own outbound mail, assume the
bridge's To is wrong and set it explicitly.

**Better still — pick a DIFFERENT anchor rather than repairing the To (verified 2026-10-17).** When
the newest message in a thread is the user's own sent mail but the reply belongs to the
counterparty, do not anchor on the user's message at all. Anchor on the **counterparty's most recent
message**, even though it is not the latest in the thread. `gws_gmail_reply_draft(message_id=<counterparty's
message>)` then sets `To` = that counterparty directly and returns the **same master threadId**.

Verified: replying to Bhavik's 16-Oct 18:26 mail — NDR's own 19:39 reply sat *later* in the same
conversation — produced `thread_id=1a138babcbde9755` (the correct conversation) with
`To: bhavik@draas.com`, and needed no post-hoc To repair and no raw-MIME rebuild. Gmail threads by
`References` / `threadId`, not by "newest message", so an anchor chosen for the recipient is still
the right thread. Reach for §3b's repair path only when there is no counterparty message to anchor
on (e.g. a follow-up on a thread where the user has been the only sender).

### 3b-bis. Anchor is an INBOUND mail TO the user — never copy its `To` header (verified 2026-10-23)

Replying on a colleague's behalf (Anbu/pm2.blr@draas.com answering NDR's mail), the anchor was
NDR's message **sent to Anbu**. Copying `h['To']` off the anchor — the natural move in §2g —
produced `To: Anbarasan M <pm2.blr@draas.com>`: a reply addressed **to the sender himself**.
Gmail accepted it, `threadId` matched, subject was right, and the draft looked perfect in the
thread. Only the To header was wrong (§3b is the same class of silent bug, opposite trigger).

**Rule:** the recipient of a reply is the anchor's **`From`**, never its `To`. The anchor's `To`
is only worth reading to know whose mailbox it landed in. The anchor's `Cc` is still the right
Cc source (§2g) — but treat it as the *third-party* crew only.

```python
msg['To'] = h['From'] if <anchor is inbound> else <explicit counterparty>
msg['Cc'] = h.get('Cc','')          # anchor's Cc, minus the user themselves
```

**Verification that catches it:** assert `To` does not contain the drafting account's own address.
Do this read-back on every draft built for someone else's mailbox — the account is not the user's
own, so "it looks fine in the thread" proves nothing about where a Send would go.

**Anbu's dictated replies/positions follow this shape** (his inbox, To = NDR, Cc = whoever NDR had
on the original): build raw MIME via `build_service(..., HERMES_SESSION_USER_ID='pm2_blr')`, keep
`h['Cc']` from the anchor, and sign as *Anbu / Anbarasan M / VP, DRA Realty Private Limited /
+91 81500 29900*. Plain text is fine — these are internal working replies, not HTML collateral.

## 3c. NEVER invent the anchor's Message-ID — fetch it (verified 2026-10-14)

> **RE-FIRED 2026-10-30** (VGRO / Sudheer draft): a hand-typed
> `In-Reply-To: <CANvXoQ6Qd9J5vWz3mKp2sT1bY8hL4nR7fG6dE5cB4aZ3yX2wVu1@mail.gmail.com>` was written
> straight into `MIMEText`. Gmail accepted it, returned a draft id and the right `threadId`, and it
> was caught only on read-back. This is the single most repeatable error in the whole class — if you
> are about to type a Message-ID from memory, **stop and fetch it**. Read this section before
> writing any `In-Reply-To`.
>
> Two extra guards on the read-back: (a) **string-compare** the draft's `In-Reply-To` against the
> anchor's `Message-ID` (`assert draft_irt == orig_mid`) rather than eyeballing it; (b) also assert a
> handful of **fact strings** you know must be in the body (names, percentages, rupee totals, dates) —
> cheap proof the body wasn't truncated or a section silently dropped.


Building the raw MIME for a reply, the `In-Reply-To` / `References` values **must** come
from a live read of the anchor message. Do not pattern-match a plausible-looking
`<CAH3aL+...@mail.gmail.com>` from memory, from another thread, or from the shape of
neighbouring IDs. Verified 2026-10-14: a draft was built with a **fabricated** Message-ID
that looked exactly like a real Gmail one. Gmail accepted the create call, returned a
draft id and a correct `threadId` — nothing errored — and the draft would silently have
landed as a NEW conversation on the recipient's side, because the References chain pointed
at a message that does not exist.

```python
# the ONLY correct source for the anchor id
m = svc.users().messages().get(userId='me', id=ANCHOR_ID, format='metadata',
                               metadataHeaders=['Message-ID','Subject']).execute()
orig = {h['name']: h['value'] for h in m['payload']['headers']}['Message-ID']
```

`threadId` on `drafts().create` does NOT save you here — it places the draft in the right
conversation in the *sender's* Drafts list, but recipient-side threading follows
`References`, so a fabricated id still strands the reply for everyone else.

This is a data-integrity error, not just a formatting one: a made-up header is fabricated
output. If the real id cannot be read, say so and stop — do not substitute a placeholder
that looks real. Fix path when it has already happened: delete the bogus draft
(`drafts().delete` → 204 with an EMPTY body, so don't `json.loads()` it) and rebuild with
the fetched id, then read the draft back and assert `To`, `In-Reply-To`, `threadId`.

## 4. In-place draft update can DETACH the thread

Partial-MIME updates regenerate the Message-ID and Gmail fragments the thread. Prefer
delete + recreate with full correct headers, then re-verify via drafts().get.

**Nuance (2026-10-09):** an update that supplies a *complete* raw MIME **and** passes
`threadId` inside the `message` object DID hold the thread — `drafts().update()` returned
the original `threadId` and `drafts().get()` confirmed it unchanged. So the distinction
is: a *partial* MIME update detaches; a *full raw-MIME* update with `threadId` inside
`message` does not. Still verify every time — §3b is precisely the case where you are
updating in place, and a silent detach would strand the reply.

## 5. User's hard rule (stated after this saga)

"Use the GWS client only to create drafts, nothing else." This means build_service +
raw-MIME drafts with proper threading headers, verified by reading the draft back from
Gmail. The user checks the Drafts folder / thread UI himself and will flag any draft
that shows up outside the conversation.

## 6. Name the ACCOUNT in the same breath as the draft id — and never speculate about one (2026-10-16)

Twice in one session NDR came back with *"I am not able to see the draft, so I am not sure
in which email have you prepared the draft"*, and then *"Why ahfl.in account? This should
be in the draas.com account … Can you please make sure it moves back to draas.com
account?"* Two separate failures, both avoidable:

- **Don't make him ask.** Every time you report a draft, name the account in the same
  sentence as the id: *"ndr@draas.com → Drafts, draft id r9104262728390347381, message id
  1a144f87c8392927"*. He runs several Google accounts in the Gmail app and Drafts is
  per-account, so a draft that genuinely exists is invisible if he is looking at another
  one. State the account, the draft id, the message id, and the thread it sits in.
- **Never speculate about an account you did not use.** The ahfl.in mention was pure
  speculation in prose ("if you were looking at ndr@ahfl.in you wouldn't have seen it") and
  NDR read it as a fact — *"Why ahfl.in account?"*. When a draft appears to be missing,
  verify and report **only verified state**: the draft count in the account you used, and the
  draft count in the account he might be looking at. *"Drafts in ndr@draas.com: 1. Drafts in
  ndr@ahfl.in: 0."* Nothing to move, nothing to explain away.
- **Corollary — a re-do makes exactly one draft.** If he can't find an existing draft, delete
  it and create one fresh one (§2h), then report the new id. Never leave two candidates in
  the folder for him to guess between.

Read-back that produces those sentences — note `drafts().get` rejects `metadataHeaders=`
(§2c) but `messages().get` accepts it:

```python
m = g.users().messages().get(userId='me', id=DRAFT_MESSAGE_ID, format='metadata',
        metadataHeaders=['From','To','Cc','Subject','Date']).execute()
h = {x['name']: x['value'] for x in m['payload']['headers']}
print(h['From'], '|', h['To'], '|', h.get('Cc'), '|', m['labelIds'])
```

Assert `labelIds == ['DRAFT']`, and take **Cc from this read-back** — a draft can look
correct in the thread UI and still carry the wrong recipients (§2b), so "Cc confirmed" is
only sayable off a header read, never off a `draft_get` or the thread view.

## 7. A draft you have not re-read does not exist (verified 2026-10-22, Ayana Woods report)

A prior turn reported *"the mail is redrafted — sitting in your Drafts folder, 9 attachments
verified"*. It was not there: `search('in:drafts Ayana')` returned 0, the full
`drafts().list()` had no matching subject, and there was no sent copy either — the draft had
been deleted (or never persisted) and nothing errored at the time.

- **Never claim a draft exists off the write call's success.** In the turn where you report it,
  re-read it: `drafts().get(id=DRAFT, format='raw')` (or `format='full'` + header parse), and
  name the account + draft id + message id + thread in the same sentence (§6).
- **When it has gone, say so plainly** — *"the earlier draft is no longer in the mailbox; I
  rebuilt it from the saved copy and re-attached everything"* — then create ONE fresh draft.
  Do not silently recreate it and let him believe it is the same one.
- **Rebuild from the artefact on disk, not from the conversation.** Drafts built by raw-MIME
  scripts usually leave `/tmp/<x>.eml` (full multipart, body + attachments) and/or
  `/tmp/<x>.html` (body only) behind. Reading the `.eml` reproduced the 9-attachment report
  byte-faithfully in one pass; re-deriving the body from memory would have lost the
  photographs' filenames and the Section 2 cost-sheet block entirely.

## 8. Gmail NORMALISES HTML entities on save — patch the live draft, not the entity form

Verified 2026-10-22: a draft whose HTML was written with `&#8377;`, `&mdash;`, `&ldquo;` read
back with literal `₹`, `—`, `“` characters, and with `\r\n` line endings inside the HTML. A
follow-up patch script built on the entity strings died on
`assert '<li>…&#8377;24' in html` — twice, on two different bullets.

- Read the draft, then patch with `re.search(r"<li><strong>Density contradiction\.</strong>.*?</li>", html, re.S)`
  — a regex tolerant of whitespace/`\r\n` and of either glyph form — instead of asserting an
  exact literal.
- Only after such a patch can you assert on the *output* (`"Note on provenance" not in html`)
  and on the presence of the new content.
- Same applies to a body you wrote yourself earlier in the session: the stored copy is
  Gmail's normalisation of it, not the string you passed in.

## 9. "Reply to the same email where he sent about X" may be TWO separate threads

Same subject, sent hours apart, different `threadId`s. Ayana Woods: the visit brief (00:17) and
the follow-up checklist (10:40, `Re: …`) were **separate conversations** in Gmail. Rules:

- Reply to the message that carries the **queries/checklist** — that is "the same email" in the
  user's mind — and tell him the other thread exists and did not get the reply. Never silently
  pick one and let him discover the split later.
- **When the reply must answer a numbered checklist, prepend a status block** ("Your seven
  questions — where we stand": one paragraph per question, each labelled answered / partly
  answered / still to be obtained) BEFORE the existing report sections, rather than
  restructuring or renumbering the report. It is the part he looks for first.
- **Never park an item as "still to be obtained" when the record is already in hand.** The
  unit-mix answer had to be corrected back to the RERA plot schedule after the draft said it
  was pending. Before sending, re-read each "pending" line and ask whether a document already
  on hand settles it — and say in the mail *where* each figure came from.

## 7b. "My Drafts folder is empty" — check SENT before concluding anything (verified 2026-10-23)

The §7 failure recurred in its worst form. NDR said *"Please send a follow up email to Jasleen
because I have not heard back from my last email"*. The agent answered **"the draft is still
sitting in your Drafts, unsent — that's why she never replied"**. NDR checked and pushed back:
*"Are you sure about that? … My drafts are empty … So you are just hallucinating. Please actually
check and update your memory."*

He was right. The draft had already been **SENT** — the content was the 9-Oct 16:02 outbound
(`Image`-free, subject "…Evidence Pack and Defense Strategy"), and the thread's true last outbound
was his own 20-Oct 01:13 chaser. Drafts across all three accounts were genuinely empty because
everything had gone out.

**Diagnostic order — run all three before saying anything about a missing draft:**

```python
# 1. drafts in EVERY vault account (not just the one you think you used)
for svc_name in ('google-draas','google-ahfl','google-gmail'):
    g = build_service('gmail','v1', service_name=svc_name)
    print(svc_name, len(g.users().drafts().list(userId='me').execute().get('drafts',[])))

# 2. SENT for the subject/thread — this is the step that was missing
g.users().messages().list(userId='me', q='in:sent to:<recipient> newer_than:14d',
                          maxResults=25).execute()

# 3. the thread itself: newest message, who sent it, when
g.users().threads().get(userId='me', id=THREAD, format='metadata').execute()
```

- **"Not in Drafts" has three causes and they need different answers**: never created; created then
  deleted; **already sent**. Only the third means the user genuinely owes the counterparty a chaser.
  Guessing "still unsent" when it was sent makes him doubt the whole report — and correctly so.
- **State the truth as verified counts**, not narration: *"Drafts in ndr@draas.com: 0. Last outbound
  to her: 20-Oct 01:13, no reply since."* (§6). Then say what you are doing about it.
- **Never re-describe a draft you have not re-read in this turn.** If the previous turn claimed a
  draft exists, that claim is stale by definition — re-verify or withdraw it explicitly
  (*"I was wrong — that draft is not there; it was already sent"*).
- Check the user's `in:sent` **and** the reference chain, because a "draft" the user sent himself
  carries no draft id at all — the artefact you were tracking stopped existing the moment he sent it.

## 7c. Participant set DRIFTS across a long thread — "reply all" = union, not latest message

Same session, same thread (`1a07b0f2821d61f6`, 25+ messages over 6 weeks), three different Cc sets:

| Message | Cc |
|---|---|
| 9-Oct inbound (Jasleen) | Manish `mdr@draas.com`, Mamta `maadi32@gmail.com`, Pradyumna, Prakash |
| 9-Oct outbound (NDR) | + Suhail `e.suhail.ahmed@gmail.com`, Vinay `vinay@vnv.ca`, `info.smritilegal@gmail.com` |
| 20-Oct outbound (NDR) | Mamta **`mamatadr@gmail.com`**, Manish, **Roshini `rnr@draas.com`** |

Blindly taking the *latest* message's Cc would have dropped the whole legal team; blindly taking the
9-Oct set would have dropped Roshini and used Mamta's older address. NDR's instruction was explicit:
*"making sure that as a reply all everybody in the chain is marked."*

- **When the user says "everybody in the chain", build the UNION of To+Cc across the last ~5
  messages**, minus the user themselves. Do not pick one message.
- **A person can legitimately have TWO live addresses on one chain** (Mamta: `mamatadr@gmail.com`
  from the family/heir side, `maadi32@gmail.com` from the earlier counsel side). If both appear in
  the recent chain, include both — on an estate matter a missed heir is far worse than a duplicate
  address. Say which you included and why.
- **New participants get ADDED over time** (Roshini Ranka appeared only on the 20-Oct mail, when the
  succession/family side widened). Diff the last message's Cc against the one before it before
  finalising; a newly-added name is usually deliberate.
- Verify the final set with a header read-back (§2c/§6) and print To + the full Cc in your report so
  he can eyeball it without opening Gmail.

## 7d. A chaser must carry the last ELAPSED hard date, not just "any update?"

When the last outbound is the user's own and the counterparty has gone quiet, the cheapest way to
make the chaser land is to name the dated thing that has already passed with no report. Here the
20-Oct mail had asked for status on four items; the sharper ask three days later was
*"CCC No. 478/2026 was listed on 12 October 2026 — we would be grateful to know what transpired."*

- Mine the thread + case notes for **dates that elapsed since the last outbound** (hearing dates,
  filing deadlines, listing dates) and lead the ask with them. A generic "any update?" gets ignored;
  "the 12-Oct listing has passed — what happened?" does not.
- Re-state the four outstanding asks as a **compact table** (item | status requested) rather than
  prose — that is what got answered last time and it makes silence expensive to maintain.
- Keep the tone deferential and give the counterparty an out (*"I do appreciate you may have been
  occupied"*), then the elapsed date, then the table. Short — see the counsel-email style rules in
  `legal-matter-coordination`.
