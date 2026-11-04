# NEW (non-threaded) HTML + attachment draft — ASCII subject, base64 text parts

Verified 2026-10-14 (Shankar/Sankar Electricals draft tax invoice .docx → Anbu, Cc Bhavik).
Use this when NDR says *"attach <file> in an email to <person>"* and there is **no parent
thread to reply on**. For threading onto an existing thread with an attachment, use
`html-reply-with-attachments-threaded.md` instead. For **forwarding an already-SENT mail to a new
person** (full attachment set carried, original body quoted, delivered-vs-bounced list in the cover
note), use `forward-existing-sent-mail.md` — it carries the `%`-formatting/CSS pitfall that breaks
the HTML build, and the one-address-per-person rule.

## 1. Resolve the recipient's SENDING address — never trust the contacts sheet

The contacts sheet holds the *mailbox*; Gmail headers hold the address the person actually
mails **from**. Verified 2026-10-14:

- **Anbu = Anbarasan M. — he sends from BOTH `anbarasan@draas.com` AND `pm2.blr@draas.com`.**
  Verified again 2026-10-27: on the "Servant room – tile repair work at 2401" thread he used
  `anbarasan@draas.com`, while on Ranka Amber/Iris mail the same day he used `pm2.blr@draas.com`.
  So do not treat either as "the older mailbox" and do not pick blind — **read the `From` header of
  the thread you are replying into and use that address.** For a brand-new mail, `anbarasan@draas.com`.

Verified from `from:` headers again on 2026-10-21 (a five-colleague internal announcement):

| Person | Address to use | Note |
|---|---|---|
| Anbarasan M (Anbu) | `anbarasan@draas.com` | `pm2.blr@draas.com` is an older mailbox he also sent from |
| Sinchana Gowda | `sgowda@draas.com` | `sinchana@kelsa.io` is a **different person** on the Kelsa sync — never substitute |
| Gowri Singh | `gsingh@draas.com` | content + creative approvals |
| Bharat Hawaldar | `sales1.blr@draas.com` | the sheet is a muddle here (`sales1.blr@drahomes.in` + a colon-joined cell of obsolete addresses). Every mail he sends is `BHARAT H <sales1.blr@draas.com>`, and NDR's replies go to that address |
| Roshni Ranka | `rnr@draas.com` | `rnr@drahomes.in` and `rmurjani@gmail.com` only — nothing else |

**Internal team announcement (one brief to 5+ colleagues)** is a distinct variant of a new email:
To-list only with **no Cc**, navy `#1F3864` header band + amber `#f5a623` rule, and one tinted box
carrying the concrete thing being asked for (day / time / place) so it survives a phone-glance. The
same content must also exist as the `text/plain` alternative — plain-text `*bold*` markdown reads as
one continuous running block in Gmail and is wrong for this class. Always verify the draft carries
`labelIds == ['DRAFT']`, all five `To` names, and the key phrases in **both** parts before reporting.

When the same brief also goes out on WhatsApp to the same people: draft the email first, then stage
the WhatsApp drafts (one per person, individually addressed), and report the email as *in Drafts*
— never as sent.
- **Bharat Hawaldar** = `sales1.blr@draas.com` (display name `BHARAT H`). He sends as the shared
  sales mailbox, so that is the correct personal To — not an alias to avoid. His contacts-sheet row
  is a mash-up of five unrelated `@draas.com` addresses; never paste from it.
- **Sinchana Gowda** = `sgowda@draas.com`. Do **not** use `sinchana@kelsa.io` — that is "Sinchana
  Kelsa", a different person.
- **Gowri Singh** = `gsingh@draas.com`; **Roshni Ranka** = `rnr@draas.com`.

The full verified table for the DRA team, plus the header-scanning method that produced it, is in
`references/dra-team-sending-addresses.md`.

Recipe: `gws_gmail_search(query="from:<candidate-address>")` for each candidate and use the
one that appears as a *sender*. Confirm with the recipient name in the header
(`Anbarasan M <anbarasan@draas.com>`).

## 1b. Before creating a NEW mail, check whether the matter already has a thread

NDR says *"send an email to <person> with this attached"* — he does not say "new" or "reply". If a
live thread on the same matter exists, **reply into it**; a fresh mail orphans the history and, on
vendor work, separates the quotation from the work order that has to follow it.

Find it by searching the **job**, not the person:

```python
gws_gmail_search(query='subject:("Servant room") 2401')
```

then `threads().get(id=…)` for the tail (NDR answers the other side's latest message, not the head).
Thread with `In-Reply-To` = the latest message's `Message-ID`, `References` = its `References` +
that ID, and `threadId=` on `drafts().create`. Set `To` from the `From` of that latest message.

**When the parent subject contains a non-ASCII character, keep your own subject ASCII.** The
Prestige Hermitage thread's subject carries an en-dash (`Servant room – tile repair work at 2401`).
Replying with it verbatim forces an RFC2047 encoded-word in the header — the exact thing §3 says
breaks Gmail's draft renderer. Threading is decided by `threadId` + `References`, **not** by subject
string equality, so send `Re: Servant room - tile repair work at 2401, Prestige Hermitage` with a
plain hyphen. The draft nests correctly and no encoded-word appears anywhere in the MIME.

## 2. Import path for `build_service`

```python
import sys
sys.path.insert(0, "/opt/hermes")          # NOT "/opt/data" — that raises
from tools.gws_auth import build_service    # ModuleNotFoundError: No module named 'tools'
svc = build_service("gmail", "v1", service_name="google-draas")
```

`service_name` comes from `gws_resolve_account` (`google-draas` = ndr@draas.com,
`google-ahfl`, `google-gmail`, `google-o3infotech`). Company/vendor mail → `google-draas`.

**Name throwaway scripts carefully.** A script that imports `tools.gws_auth` must NOT be called
`inspect.py` (nor `types.py`, `email.py`, `json.py`, `code.py`, `re.py` …). The script's own directory
is first on `sys.path`, so `import inspect` deep inside `importlib.metadata` resolves to *your* file and
the chain dies with:

    ImportError: cannot import name 'build_service' from partially initialized module
    'tools.gws_auth' (most likely due to a circular import)

That message blames a circular import in the framework and is **wrong** — it is your filename. Rename
the script and re-run.

## 3. Keep the SUBJECT pure ASCII

Gmail's web draft renderer is known to break on RFC2047 encoded-words in headers
(`gmail-draft-integrity-and-attachments.md`). Any non-ASCII in a Subject forces
`=?UTF-8?B?...?=`. So:

- Subject: write **`Rs 2,00,000`**, **`-`** (hyphen), plain words. No `₹`, no `—`, no `·`.
- Body parts: base64-encode **both** `text/plain` and `text/html`
  (`Content-Transfer-Encoding: base64`) and then `₹`, `—`, `·`, `&` travel perfectly with
  **zero encoded-words anywhere in the MIME**.
- HTML **entities** (`&mdash;`, `&amp;`) are plain ASCII and are safe as-is in the HTML part —
  the trap is *raw* non-ASCII characters, not entities.

## 4. MIME shape (hand-built by string concatenation, explicit CRLF)

```
MIME-Version: 1.0
From: Nishant Ranka <ndr@draas.com>
To: <recipient sending address>
Cc: <the people NDR said to "mark">
Subject: <ASCII only>
Date: <email.utils.formatdate(localtime=True)>
Message-Id: <uuid hex upper @mail.gmail.com>
Content-Type: multipart/mixed; boundary="MIX"

--MIX
Content-Type: multipart/alternative; boundary="ALT"

--ALT
Content-Type: text/plain; charset="UTF-8"
Content-Transfer-Encoding: base64
<base64 of plain body>
--ALT
Content-Type: text/html; charset="UTF-8"
Content-Transfer-Encoding: base64
<base64 of html body>
--ALT--

--MIX
Content-Type: application/vnd.openxmlformats-officedocument.wordprocessingml.document; name="<file>.docx"
Content-Transfer-Encoding: base64
Content-Disposition: attachment; filename="<file>.docx"
<base64 of the file>
--MIX--
```

Then `base64.urlsafe_b64encode(raw.encode('utf-8'))` and
`drafts().create(userId='me', body={'message': {'raw': raw}})`.
No `threadId` — this is a new thread by design.

## 4b. Simpler alternative — stdlib `EmailMessage` + `add_attachment` (verified 2026-10-14)

The hand-built MIME above is one way; the stdlib builder produces the same shape with far less
rope to hang yourself on, and it is **verified working on `drafts().create`** (NDR → Kishan
Murjani Nair, two JPEG attachments, subject `My profile headshot`):

```python
import os, base64, mimetypes
from email.message import EmailMessage

msg = EmailMessage()
msg["To"] = TO
msg["Subject"] = SUBJECT                 # still ASCII-only — see §3
msg.set_content(plain_text)              # -> text/plain
msg.add_alternative(html, subtype="html")  # -> multipart/alternative
for p in ATTACH:                         # -> multipart/mixed
    ctype, _ = mimetypes.guess_type(p)
    main, sub = (ctype or "application/octet-stream").split("/", 1)
    with open(p, "rb") as f:
        msg.add_attachment(f.read(), maintype=main, subtype=sub,
                           filename=os.path.basename(p))

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
svc.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

- What actually 400s with `"Missing draft message"` is a hand-made **dict** payload — not
  `EmailMessage`. Prefer `EmailMessage` unless you need byte-exact control of the MIME.
- Multiple attachments are fine; `add_attachment` appends each one.
- **Neither `gws_skill_bridge.call("draft_create", …)` nor the native `gws_gmail_draft_create` has an
  attachment parameter.** When NDR says *"email <person> with the sketch attached"*, this block is the
  only path: build with `EmailMessage`, then `drafts().create` directly. A plain `draft_create` cannot be
  "topped up" afterwards — re-issuing it mints a SECOND draft rather than adding the file, so decide the
  attachment set before the first call and delete the empty draft if you got the order wrong.

## 4c. Photos INLINE in the body — `multipart/related` wrapping `multipart/alternative`

Verified 2026-10-24 (Ranka Oasis QA review → Anbu, Cc Kanthesh, 10 phone photos inline + attached).
Use this when the email is *about* the photographs and they must be visible in the body, not filed
away at the bottom of the mail.

`<img src="cid:photo01">` in the HTML, and the image parts carry a matching `Content-ID`:

```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.utils import formatdate
import uuid

root = MIMEMultipart("related")
root["From"] = "Nishant Ranka <ndr@draas.com>"
root["To"], root["Cc"], root["Subject"] = TO, CC, SUBJECT      # subject still ASCII-only
root["Date"] = formatdate(localtime=True)
root["Message-Id"] = "<%s@mail.gmail.com>" % uuid.uuid4().hex.upper()

alt = MIMEMultipart("alternative")
alt.attach(MIMEText(PLAIN, "plain", "utf-8"))     # utf-8 -> base64, so no encoded-words
alt.attach(MIMEText(HTML, "html", "utf-8"))
root.attach(alt)

for i, path in enumerate(paths, start=1):
    part = MIMEImage(open(path, "rb").read(), "jpeg")
    part.add_header("Content-ID", "<photo%02d>" % i)
    part.add_header("Content-Disposition", "inline", filename="Photo-%02d.jpg" % i)
    root.attach(part)

raw = base64.urlsafe_b64encode(root.as_bytes()).decode()
```

- **TRAP — `EmailMessage.make_related()` cannot do this.** On a message that is already
  `multipart/alternative` it dies with
  `ValueError: Cannot convert alternative to related`, and there is no argument that fixes it.
  The stdlib will happily go alternative→mixed but never alternative→related. Build the `related`
  wrapper **by hand** as above; do not spend a round trying to coax `EmailMessage` into it.
- The same parts that render inline also show up as attachments in the recipient's mail, so there is
  **no separate `add_attachment` pass** — do not add the files twice.
- Inline `cid:` images survive Gmail's renderer and the `text/plain` alternative is still present for
  clients that strip HTML. Verify both.
- **Portrait phone photos need a height cap or they are unusable.** A 718×1600 frame at
  `width:100%` inside an 880 px card renders ~1830 px tall. Branch on orientation:

  ```python
  if h > w * 1.1:
      style = "display:block;height:520px;width:auto;max-width:100%;margin:0 auto;border:0;"
  else:
      style = "display:block;width:100%;max-width:820px;height:auto;border:0;"
  ```

  Resize before attaching (max long edge 1600, JPEG q84, `optimize=True, progressive=True`): ten
  phone photos land at ~3.5 MB, ~4.8 MB of raw MIME — well inside Gmail's 25 MB, and small enough
  that the recipient's phone can open it.

### TRAP — re-running the build script mints a SECOND draft

The script's last step is `drafts().create`, so any re-run "just to look at it" silently produces a
duplicate in Drafts. Run the builder **once**. If it happens, delete the extra:

```python
svc.users().drafts().delete(userId="me", id=dup_draft_id).execute()
```

To preview the layout without creating anything, read the draft that already exists back and point
its `cid:` refs at local files, then open it in the browser:

```python
d = svc.users().drafts().get(userId="me", id=draft_id, format="raw").execute()
p = message_from_bytes(base64.urlsafe_b64decode(d["message"]["raw"]), policy=policy.default)
html = p.get_body(preferencelist=("html",)).get_content()
for i in range(1, 11):
    html = html.replace("cid:photo%02d" % i, "send/Photo-%02d.jpg" % i)
open("preview.html", "w").write(html)
```

Then `browser_navigate("file:///…/preview.html")` + `browser_vision` asking specifically about the
header band, the rating-table colours, and whether every photo and caption rendered. That catches
`%`-formatting breakage, unescaped CSS and broken image refs before NDR ever opens the draft. Send
him the resulting screenshot as `MEDIA:`.

### Verify the inline set, not just the filenames

For each image part assert `Content-ID` is present and `sha256(decoded) == sha256(local file)`; then
count `html.count('src="cid:photo')` and require it to equal the number of image parts. A caption
under a missing image looks like a rendering glitch to the recipient and a pass to a filename-only
check.

## 5. Verify before reporting (mandatory)

1. `drafts().list(userId='me')` **first** — if a draft for the same recipient/topic already
   exists, delete it (`gws_gmail_draft_delete`) so Drafts carries exactly one.
2. `drafts().get(format='full')` on the new draft and assert:
   - `labelIds == ['DRAFT']` (not `['SENT']` — see the send-detection trap in
     `gmail-draft-integrity-and-attachments.md`)
   - `To` / `Cc` / `Subject` exactly as intended
   - walk `payload.parts[]` and assert the attachment filename is present
3. **Byte-verify the attachment**: re-download it via
   `messages().attachments().get(userId='me', messageId=m['id'], id=aid)` and compare
   `sha256` to the local file. Byte-identity is the only proof the recipient receives the
   same document — a filename in the parts metadata is not.
4. Extract the `text/plain` and `text/html` parts (base64-decode `body.data`) and run a
   keyword check on **both** — e.g. amount, recipient's own name, each condition heading.
5. Plain-text check: every paragraph must be one logical line (no mid-sentence wrapping);
   only the greeting and sign-off are short lines.
6. **If the body links to a Drive file, prove the recipient can open it BEFORE you
   create the draft.** A link pasted into an email is worthless if the file is
   private to a colleague — and the failure is silent, because the draft looks fine.

   Verified 2026-10-21 (Ranka Amber Investment Guide → an external channel partner):

   - "It's in our Drive" is often false. A document a colleague circulated on
     WhatsApp lived in **their** folder; `gws_drive_list` on ndr@draas.com returned
     **0 results** for it. Search the owning colleague's Drive with
     `gws_dwd_drive_list(target=...)` — legitimate here because the user named the
     source and asked for that specific document to be linked.
   - Then grant two permissions via DWD on the owner, and list them back to prove it:
     the **external recipient → `reader`**, and **ndr@draas.com → `writer`** (so the
     user can open the link he is about to send).
   - Prefer granting the named external address over "anyone with link". Widening
     sharing on a colleague's file to public is not something to do unprompted.
   - Link form: `https://drive.google.com/file/d/<fileId>/view`.
   - Re-read with `gws_drive_get` and check `permissions[]` contains both grants
     before reporting the draft as ready.

### `get_body()` raises AttributeError unless you parse with `policy=email.policy.default`

Plain `message_from_bytes(raw)` returns a legacy `Message`, which has **no `.get_body()`**. A
verification read-back dies with:

    AttributeError: 'Message' object has no attribute 'get_body'

**That is the parser default, not a broken draft — the draft was created fine. Do NOT re-create
it.** Parse with the policy to get a real `EmailMessage`:

```python
import email
from email import message_from_bytes
p = message_from_bytes(base64.urlsafe_b64decode(d["message"]["raw"]),
                       policy=email.policy.default)
plain = p.get_body(preferencelist=("plain",)).get_content()
html  = p.get_body(preferencelist=("html",)).get_content()
```

Then run the keyword check on **both** parts. On a long multi-section briefing this is the cheapest
proof that (a) the HTML part did not demote to literal markup, (b) a `text/plain` alternative is
actually present, and (c) every required section heading survived. The same trap bites any
`get_body`/`iter_parts` call in this skill's verification recipes.

### Cheap raw read-back when you already hold the `message_id`

```python
from email import message_from_bytes
from email.header import decode_header, make_header
m = svc.users().messages().get(userId="me", id=message_id, format="raw").execute()
parsed = message_from_bytes(base64.urlsafe_b64decode(m["raw"]))
print("LABELS:", m.get("labelIds"))                 # ['DRAFT'] == unsent
print("To     :", make_header(decode_header(parsed["To"] or "")))
print("Cc     :", make_header(decode_header(parsed["Cc"] or "")) or "(none)")
print("Subject:", make_header(decode_header(parsed["Subject"] or "")))
for part in parsed.walk():
    fn = part.get_filename()
    if fn:
        print("  attach:", make_header(decode_header(fn)),
              part.get_content_type(), len(part.get_payload(decode=True)) // 1024, "KB")
```

`Cc` is invisible in the JSON draft summaries, so this read-back is the only way to prove a Cc
list actually landed.

**TRAP — `message_from_bytes(...)` alone returns a legacy `Message`, not an `EmailMessage`, and it
has no `.get_body()`.** A verify script that does `parsed.get_body(preferencelist=("plain",))`
dies with `AttributeError: 'Message' object has no attribute 'get_body'`. Pass the policy:

```python
import email
parsed = message_from_bytes(raw_bytes, policy=email.policy.default)   # -> EmailMessage
plain = parsed.get_body(preferencelist=("plain",)).get_content()
html  = parsed.get_body(preferencelist=("html",)).get_content()
```

Verified 2026-10-19: the draft was created perfectly; only the verification step crashed. **Do not
read that traceback as "the draft was not created"** and do not rebuild — re-run the read-back with
the policy set. With `policy=default` the headers also come back already decoded, so `parsed["To"]`
prints `Sarthak Sharma <admin3.blr@draas.com>` directly and you can drop
`make_header(decode_header(...))` on the read side.

### TRAP — the `message_id` you hold goes STALE the moment the draft is opened in Gmail

Verified 2026-10-16 (Flameback/Kishan letter). Gmail **re-saves** a draft when the user opens it in the
web UI, and the re-save **mints a brand-new message id**. The **draft id stays the same**; only the
message id changes. So the `messages().get(id=<message_id>)` call above later returns

    HttpError 404 ... "Requested entity was not found."

which is indistinguishable from *"the draft was deleted"* — and it has **not** been deleted. Do not
rebuild it, do not report a deletion.

**Read drafts back through the DRAFT endpoint, never the message endpoint:**

```python
d = svc.users().drafts().get(userId="me", id=draft_id, format="raw").execute()
p = message_from_bytes(base64.urlsafe_b64decode(d["message"]["raw"]))
```

Second false alarm from the same re-save: `d["message"]["threadId"]` stays pinned to the **original**
message id, so a naive `threadId == message.id` test for "is this a NEW thread?" reports REPLY after a
re-save. Judge new-thread-ness from the moment of creation (threadId == messageId then) or from
`drafts().list()` showing no prior thread — not from a later read.

And re-verify by **diffing the live body against the text you authored**: if the user has been editing,
what is in the draft is no longer your build. See
`references/long-advisory-letter-html-formatting.md`.

### Pitfall — the byte-check flag

Do **not** accumulate the comparison into a flag initialised to `False`:

```python
att_ok = False
att_ok = att_ok and (sha == local)     # prints False even when EVERY file matched
```

Print the per-file comparison (`f"{fn}: sha256 match: {sha == local}"`) and let that be the
evidence. The accumulated-flag version cost a false-negative scare in this session.

## See also

- `references/voice-dictated-proper-nouns.md` — **voice-dictated names, case names, projects and
  entities**: verify each one against the thread NDR is answering + `contact_resolver`, adopt the
  thread's spelling, and flag every change (and every unverifiable name) in the summary. Covers
  "Kushal" → Khushroo Engineer and "heritage and equity" → "Hermitage and Aquagreen", the
  Bcc-not-Cc rule for family/board vote mails, and the silent no-op `str.replace` trap when you
  post-process the body before creating the draft.
- `references/qa-review-email-to-site-team.md` — turning a `construction-qa-photo-review` verdict into
  the internal issue-by-issue email to the site team: verified addresses (Anbu `anbarasan@draas.com`,
  Kanthesh `kanteshbgme@gmail.com` with no `@draas.com` identity), the 7-section body, the rating
  table colours, the separate safety box, and the "study it point by point / reply in three working
  days / bring these three documents" ask block. Photos go inline via §4c.
- `references/intro-connection-email.md` + `templates/intro-connection-email.html` — the **intro /
  connection** variant of a new email: NDR hands a live external relationship to a colleague. To =
  the acting colleague, Cc = the external person + everyone NDR said to "mark"; five-block body
  (who + origin of intro, concrete demos, link + contact card, why-now projects, the ask); plus the
  rule that a voice-dictated project name is kept as dictated and **flagged**, never silently
  corrected.
- `references/forward-sent-mail-with-attachments.md` — forwarding an already-**SENT** mail WITH all
  its attachments plus a covering note. It is a NEW thread (not a reply), so `draft_reply_create` is
  the wrong tool. Carries the `%`-in-CSS formatting pitfall (never `"""…""" % html`), delete-and-
  recreate when only the Cc changes, the five mandatory verification checks, and the recipient-
  address rule that Manohar Singh is `msingh@redsoul.co.in` with **no Cc** to his kelsa/o3 mailboxes.

## 6. Worked example

Shankar/Sankar Electricals draft tax invoice → Anbu, Cc Bhavik, 2026-10-14:
To `anbarasan@draas.com`, Cc `bhavik@draas.com`,
Subject `Ranka Iris: Shankar Electrical Invoice for Rs 2,00,000 - Raise on Their Letterhead & Send Today`,
attachment `Draft Tax Invoice - Sankar Electricals - Ranka Iris Meters - Rs 2,00,000 (DRAFT).docx`
(40,267 bytes, sha256 verified identical off the created draft).
Body: navy `#1F3864` header band, amber `#f5a623` rules, four numbered sections, an
`.fyi` box for the cc'd approver carrying the payment-release instruction, an amber
`.flag` box for the "treat as priority today" ask.

## 7. NDR dictates a message to a PERSON but names no channel — resolve the channel first

Verified 2026-10-30 ("wishing you a very happy Diwali, Peron … I'm also sharing the final
elevations with you"). A dictated greeting/relationship message never says *email* or *WhatsApp*,
and guessing wrong means the message goes nowhere or to the wrong place. Order of work:

1. **Resolve the person, not the first name.** `contact_resolver(query="Peron")` returned two
   candidates ("P Gopinath", "Peron Roach"). Pick on organisation/email domain, and say in the
   report which one you picked and why — a bare first name is a coin-flip otherwise.
2. **Then test for a WhatsApp chat.** `wa_list_chats(query=<name>)`, `wa_search_messages(query=<name>)`,
   and once more on the firm/company word and the deal keyword ("Roach", "Clover"). Three misses is
   proof, not bad luck — a chat that does not exist cannot be messaged, and there is no
   "create chat by phone number" shortcut to reach for.
3. **Fall back to email, evidenced by history.** `gws_gmail_search(query="<name> OR <company> OR
   <domain>")` on `google-draas`. If the whole relationship lives there (Peron: the Clover Greens
   MoU and the bulk-membership offer thread), email is the channel — and the search doubles as the
   context that tells you what "our arrangement" and "the membership arrangement" refer to.
4. **State the routing decision in the report**, including the negative evidence ("no WhatsApp chat
   under name / firm / deal keyword — so this went to email"). NDR should never have to ask
   "why did this come by mail?".

Do not ask him which channel he meant when steps 1–3 answer it. Do not send anything anywhere.

### Assembling the artefact set for a relationship mail (greeting card + technical set)

Same session, worth copying as a pattern:

- **The card is rebuilt, not recycled.** "A greeting image created in the same style as the one for
  the Ranka Oasis investors" means a NEW card in that style — not the investor card forwarded to a
  third party with someone else's audience copy on it. Reuse the previous build script
  (`build_card.py` + `fonts/` + `dra-logo-w.svg` from the earlier job dir), swap only the `COPY`
  dict (headline / addressee / lead / body / wish / sign), and rebuild on the same 4K art. Keep the
  investor copy out of it.
- **OCR will miss gold-on-sky type.** A single vision pass on the full card returned only the body
  paragraphs. Crop the top band (headline + addressee + logo + website) and the bottom band
  (blessing + sign-off) at native resolution and read each separately — that is how the headline,
  "for Peron" and "Warm wishes, Nishant Ranka" got verified.
- **Attach a drawing set as a compact PDF, not a page lifted out of a deck** — see
  `document-dossier-compilation` → `references/single-page-extract-and-size-trap.md`.
- **Never attach a document whose own footer says INTERNAL / DO NOT SHARE.** The Ranka Oasis kit is
  Part A (partner-facing) + Part B (INTERNAL · DO NOT SHARE); only the extracted Part A page went out,
  and the report said so.
