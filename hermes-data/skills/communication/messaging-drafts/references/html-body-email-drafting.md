# HTML-body email drafting (full document embedded, not attached)

For MASS INDIVIDUAL drafts from a spreadsheet (one draft PER recipient, personalized
salutation, CC ndr@draas.com, 1,000+ drafts, in-place mass UPDATE via drafts().update,
the NO-PRICE-FIGURES email rule, and the spam/deliverability DNS-over-HTTPS checks) →
`references/mass-individual-draft-campaign.md` in this skill. This file's BCC-blast and
single-draft recipes do NOT apply there.

Related: partner/consultant intro email for the Ranka Udaya RG scheme (terms, creatives, structure) → `references/ranka-udaya-rg-partner-intro-email.md` in this skill.

When the user says "send the entire thing in HTML format to the email" /
"embed the HTML in the email" / "that HTML report should be the email body",
build a Gmail DRAFT whose body IS the full HTML document (not a PDF/HTML
attachment). Verified pattern, 2026-09-21 (Ranka Oasis RERA villa comparison
→ Anbu).

## Why not an attachment

NDR reads the email body directly; an embedded HTML section renders in Gmail
and is forwardable without the recipient opening a file. The attachment habit
fails this request — the deliverable IS the body.

## Recipe (works, ran from /opt/hermes venv via terminal)

```python
import sys, base64
sys.path.insert(0, '/opt/hermes')
from tools.gws_auth import build_service
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

gmail = build_service('gmail', 'v1', service_name='google-draas')
profile = gmail.users().getProfile(userId='me').execute()
# WHOAMI check: profile['emailAddress'].lower() == 'ndr@draas.com' else abort

html_body = open('/tmp/rera/oasis_villa_comparison.html', encoding='utf-8').read()
# short plain-text/briefing intro goes ABOVE the HTML document in the body:
intro_html = '<div ...>brief note...</div>'   # plain-text part also carries the gist

msg = MIMEMultipart('alternative')
msg['To'] = 'Anbarasan <pm2.blr@draas.com>'
msg['Cc'] = 'Vinay T <vinay@vnv.ca>'
msg['Subject'] = '...'
msg.attach(MIMEText('plain fallback', 'plain'))
msg.attach(MIMEText(intro_html + html_body, 'html'))

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
draft = gmail.users().drafts().create(userId='me',
    body={'message': {'raw': raw}}).execute()
```

## Verification (always do after create)

1. Read back the draft: `drafts().get(userId='me', id=<draft_id>, format='full')`.
   Confirm To / Cc / Subject header values.
2. Walk the payload recursively collecting `body.data`, base64-decode each part:
   expect exactly 2 parts (plain + html) roughly the size of the HTML file.
3. Case-insensitive content checks on the HTML part for the KEY markers
   (registration numbers, names, formcqr URLs, Drive links). Case matters —
   check with `.lower()` in `.lower()` so "Rhythm by DAC" passes even if the
   report renders "RHYTHM BY DAC".
4. Do NOT check recipient emails against the HTML body — they live in headers,
   not the body; a "MISSING" there is expected.

## FULL-WIDTH IS MANDATORY — NDR's standing rule for HTML email bodies

Stated explicitly by NDR (2026-09, restated verbatim): *"make sure that the width
of the HTML occupies entire width always, never limit the width, never put a
width element in HTML CSS for the entire page."*

Rules for every HTML email body:

- **Never set `max-width` on the page/body/outer container.** No `max-width`
  anywhere on a wrapper element. The body must stretch to the full mail-client
  viewport.
- Use `width:100%` (not a fixed px width) on the outer table/wrapper.
- Individual inner elements (text blocks, images) may still have sensible sizing,
  but the PAGE shell itself must not be width-capped.
- Do NOT reach for the common responsive-email pattern of a centred 600px
  container — it is exactly what he is rejecting.

Verify before creating the draft:

```bash
grep -ci 'max-width' file.html          # must be 0
grep -c  'width:100%' file.html         # must be >= 1
```

Then confirm on the stored draft by decoding the HTML part back out of the MIME
and re-running those two greps — do not trust the local file alone.

## Pitfalls

- **Submission is a draft, never auto-send.** Delivery is always the user
  pressing send from Drafts. State that plainly in your reply.
- **Recipients must be contact-verified.** Check the contacts sheet / People
  API before putting any address on To/Cc. If the user names someone with NO
  verifiable email (e.g. "Mark" in the licensing loop), do NOT guess — leave
  them off, mention in the body as dictated, and ask the user for the address
  explicitly.
- HTML with `<!DOCTYPE html>`+`<html>` root works fine as an email body part;
  keep `mimeType='html'` implicit via MIMEText(..., 'html').
- The file was written to /tmp/rera — write deliverables to a stable path
  under /tmp or /opt/data if you need to re-verify after the session.

## Native tool path — no python needed (2026-10-01, Vishwas Jiraaf email)

`gws_gmail_draft_create(to=..., subject=..., body=<full html>, html=true)` (or the
skill-bridge `draft_create`) creates the same HTML-body draft directly — no
MIMEMultipart assembly. Verified: body stored as a single `text/html` part; fine
in Gmail.

### Verification pitfall — read headers with `format='raw'`, not `format='full'`

`drafts().get(id=..., format='full')` returns `payload.headers` that can omit
`To`/`Subject` entirely (only `From` present) — do NOT conclude the draft is
malformed. Read `format='raw'`, base64-decode, and check the raw MIME lines;
the Subject will be RFC2047-encoded (`=?utf-8?q?...?=`) — decode or match the
visible subject loosely. Confirm To/Subject/From/Date from raw lines.

Also re-verify the HTML body on the stored draft: walk payload parts, decode the
`text/html` part, re-run the `max-width`/`width:100%` greps and key content
markers — don't trust the local file alone.

### The FORWARDABLE OPENING NOTE — when the email must be sent ONWARDS (verified 2026-10-08)

NDR's shape: *"send an email to Roshni marking Aamir … and put the opening note so that even if
they forward to any architect, they can just use the same opening note and forward it to an
architect."* The email is addressed to a trusted intermediary, but the document is meant to reach
a THIRD party the sender has no relationship with. A link alone does not achieve that.

Structure that worked — one email, To the relationship-owner, Cc the second person, four visually
distinct full-width blocks:

1. A boxed link block — the ONE canonical link, its reference ID, and any "print to PDF gives
   clean A4" note. One link only; the point is that the linked document carries the rest.
2. A three-row "what is inside the link" table (the document / what is linked FROM that document /
   what is linked from THAT). This is what makes a single-link email read as complete.
3. A numbered `WHAT I NEED FROM YOU` block — the actual asks, one line each.
4. A banner-labelled `OPENING NOTE — FORWARD THIS TEXT TO ANY ARCHITECT, AS IS` block holding a
   self-contained letter: who we are, the first project, what we are looking for, the link, the
   "read this as intent, not as finished design" caveat, a closing ask and signature. Write it so
   it survives being pasted with no surrounding context — no "as discussed", no references to the
   recipient or the conversation.

Give each block its own heading banner (navy `#0f4c81` for the ask block, a second accent colour
for the forwardable banner) so the sections are scannable without reading the prose.

**Then say the limitation out loud.** The forwardable note travels, but a link into an
access-controlled system does not — the third party's click hits sign-in and is refused until they
are granted access BY NAME. Ask for those addresses in the same reply. An email promising "forward
this to anyone" over a permissioned link is a silent failure waiting to happen.

Build the body as a file on disk and let the script read it (`open(path).read()`), never
hand-transcribe multi-KB HTML into a tool argument. Verify on the STORED draft after read-back:
decode the `text/html` part out of the MIME and re-run the full-width greps there plus a marker
grep for the forwardable-banner heading — not just the local file.

## The ARTIFACT + PROMPT HANDOFF email — pass work on for independent cross-check (verified 2026-10-08)

NDR shape: *"send a detailed email to Sarthak explaining that here there is an LLM prompt, I've also
modelled the spreadsheet based on the LLM prompt ... give him editor access to the spreadsheet and to
the prompt."* The recipient is a peer modeller who will re-run the work through an external LLM. The
email is a **handoff pack**, not a notification — it must be self-sufficient enough that he can act
without opening anything else.

Rules that worked:

- **Grant access BEFORE you claim it in the body, and verify by readback.** `drive.permissions.create`
  then `drive.permissions.list` — confirm `role: writer` for the recipient on EACH file. Only after
  readback say "editor access granted to you". Never assert access you have not confirmed.
- **Subject line is dictated — use it verbatim** ("One more cash flow modelling exercise — lease land,
  lease commercial property"). Do not polish or shorten it.
- **Body shape that worked — 12 numbered sections, one idea per section, full-width.** Masthead → a
  "what you have access to" link table → one section per topic (transaction summary, derivation table,
  cost heads, payout heads, revenue heads, timeline/phasing, funding, valuation methods, model
  architecture, indicative output with numbers) → an orange **WHAT I NEED FROM YOU** numbered-ask block
  → a green-bordered monospace copy-paste box holding the full prompt → sign-off. NDR's reason: *"it's
  a lot of content, it becomes difficult — it should read step by step."* Sequential numbered sections,
  not one wall of prose.
- **The prompt goes IN the body, in a monospace box** (`font-family:'Courier New';white-space:pre-wrap`)
  so the recipient can copy-paste straight into Gemini/Claude. It also lives as a Doc — reference both.
- **Build the body from a file, always.** The prompt was a 33.9 KB Google Doc: export it with
  `drive.files().export(fileId=..., mimeType='text/plain')`, `html.escape()` the text, then string-replace
  a `__PROMPT__` placeholder in the HTML template on disk. **Passing a multi-KB body as a tool argument
  is exactly where this goes wrong** — a draft was created once with the literal string
  `PLACEHOLDER_PROMPT` still in it, because the body was hand-passed to `gws_gmail_draft_create` instead
  of file-read. It had to be deleted and recreated via `MIMEMultipart`. Never hand-pass; always
  `open(path).read()`.
- **Use `MIMEMultipart('alternative')` with a real plain-text fallback** carrying the same gist (the key
  numbers, both links, the asks), hand-built — not the native single-string tool — whenever the body is
  large or needs both parts.
- Close by stating plainly the draft is in Drafts awaiting send.

### Pitfall — MIME walk decode with `policy=policy.default`

`email.message_from_bytes(raw, policy=policy.default)` returns parts whose `get_content()` is already a
**str**. Calling `.decode('utf-8')` on it raises `AttributeError: 'str' object has no attribute 'decode'`.
Guard both ways: `c = part.get_content(); c = c if isinstance(c, str) else c.decode('utf-8','replace')`.

### Verification markers for a handoff email

Read the draft back with `format='raw'`, decode, then assert: exactly one `text/plain` + one `text/html`
part; `max-width` count == 0 and `width:100%` count >= 1 on the STORED html; the placeholder string is
ABSENT; both Drive/Sheet links present; and a couple of fragments from deep inside the embedded prompt
(e.g. a heading from the doc's later sections) present — that proves the whole prompt landed, not just
the first paragraph. Do NOT expect recipient emails in the HTML body — they live in headers.

## Block layout for advisor position emails

When reworking a dense dictated WhatsApp message into an email (NDR: "use HTML,
CSS so it's easier to read each point, each ask, each constraint is very clear"),
structure it as one visually distinct block per numbered point: a navy header bar
(`background:#0f4c81; color:#fff; font-weight:bold`) with the point title
("Ask — …", "Constraint — …"), a bordered body below it, bullets with `<b>`
highlighting key terms, and a boxed closing ask. Full-width outer wrapper per the
standing rule (no max-width).

## Base64 data URI image embedding — CID-free alternative (verified 2026-10-03)

When the user says *"encode the image as base64 in the HTML itself"* / *"embed the image in the HTML so it renders even without external loading"* — build the email as a single `text/html` MIME part with the image encoded as a `data:image/jpeg;base64,...` URI inside an `<img>` tag. No multipart/related, no CID, no Drive URL. The whole email is one HTML string.

### When to use base64 vs CID

| Approach | Image is... | Pros | Cons |
|---|---|---|---|
| **CID** (`multipart/related`) | Sent as a separate MIME part, referenced by `cid:xyz` | Images cached by the mail client; offline-viewable | Assembly is more complex; Gmail's 25 MB limit includes all parts |
| **Drive URL** | Fetched live from Drive at open time | No email size bloat; easy to update | Requires `anyoneWithLink`; blocked by strict mail clients; no offline view |
| **Base64 data URI** | Embedded directly in the HTML as a data URL | Single flat MIME; simplest assembly; renders everywhere HTML is processed | Increases HTML size by ~33% vs binary; Gmail may compress long data URIs |

Base64 embedding is the **user's explicit preference** when they say "make the image part of the email so it opens" — use it unless the image would push the raw MIME past ~20 MB (ceiling for Gmail draft creation).

### Recipe — compress + embed as base64 data URI

```python
import base64
from PIL import Image

# 1. Compress — always downscale for email (800px width is ample)
img = Image.open("/path/to/original.png")  # may be 2.4 MB PNG
w, h = img.size
if w > 800:
    img = img.resize((800, int(h * 800 / w)), Image.LANCZOS)
compressed_path = "/tmp/hero_compressed.jpg"
img.save(compressed_path, "JPEG", quality=85, optimize=True)
# Result: ~115 KB from a 2.4 MB source (95% reduction)

# 2. Base64 encode
with open(compressed_path, "rb") as f:
    b64 = base64.b64encode(f.read()).decode("ascii")
data_uri = f"data:image/jpeg;base64,{b64}"

# 3. Use in HTML body
html = f"""<img src="{data_uri}" alt="Description" style="width:100%;height:auto;display:block;border:0;">"""
```

### Build the draft (no attachments, base64 image only)

```python
raw_parts = [
    'From: "Prakash Singh" <psingh@draas.com>',
    f'To: {orig_to}',
    f'Subject: {subject}',
    'MIME-Version: 1.0',
    'Content-Type: text/html; charset="utf-8"',
    'Content-Transfer-Encoding: base64',
    '',
    base64.b64encode(EMAIL_HTML.encode("utf-8")).decode("ascii"),
]
raw_email = '\r\n'.join(raw_parts)
encoded_raw = base64.urlsafe_b64encode(raw_email.encode('utf-8')).decode('ascii')
draft = gmail.users().drafts().create(userId='me',
    body={'message': {'raw': encoded_raw, 'threadId': thread_id}}).execute()
```

### Threading for reply drafts

When replying to a sent email (Prakash's common pattern — "redo the drafts as replies to the original emails"):
- `In-Reply-To` = the original message's `Message-ID` header
- `References` = original `References` + `\r\n\t` + original `Message-ID`
- `threadId` = the original message's `threadId` (add to the draft body: `{'raw': ..., 'threadId': thread_id}`)
- `To` = the original message's `To` header (NOT `From` — the `From` is the user themselves for a sent email!)

### Image compression guidelines

| Target use | Width | JPEG quality | Typical size |
|---|---|---|---|
| Hero / hero shot (email header) | 800–1200 px | 84–85 | 115–230 KB |
| Inline strip / secondary image | 620 px | 78 | 45–55 KB |
| Thumbnail / spot image | 400 px | 75 | 20–30 KB |

Always use `Image.LANCZOS` for downscaling (sharpest), and `optimize=True` in `save()` for smaller files. JPEG is preferred over PNG for photographic renders; PNG is better for screenshots with text/diagrams.

### Pitfalls

- **Raw MIME size limit.** Gmail's API rejects drafts where the base64url-encoded raw exceeds ~25 MB. A 115 KB image produces ~153 KB of base64 — negligible. But a 2.4 MB PNG produces 3.2 MB of base64 text, and a 6.6 MB PDF adds another ~9 MB. Keep images under 150 KB each via compression.
- **CID vs data URI confusion.** If the user said "encode as base64", use data URI — do NOT use CID. If they said "add the image" without specifying, CID is the standard approach (image stays as a separate part, smaller total payload). Base64 is what they specifically asked for after a Drive URL didn't render.
- **Double encoding.** The raw MIME is already base64-encoded for the Gmail API's `raw` field. Ensure your HTML body is base64-encoded only once (inside the `Content-Transfer-Encoding: base64` part), NOT pre-encoded before that step.
- **The `To:` trap on replies.** On a SENT message, `hdr('From')` returns the user's own address. Setting `To: {hdr('From')}` sends the draft back to yourself. The correct recipient is `hdr('To')` — the original recipients.

## Stored-draft verification when the gateway rewrites images (verified 2026-10-06)

Creating a draft whose HTML embeds `data:image/jpeg;base64,...` URIs: on read-back the stored draft may NO LONGER contain the data URIs — the gateway (gws-vault / trusted process) strips them into separate `image/jpeg` MIME parts and rewrites the `<img>` tags to `src="cid:ii_..."`, each part carrying a matching `Content-ID: <ii_...>` and `Content-Disposition: inline`. This is a GOOD outcome (images render inline in Gmail), not a failure.

Verify correctly:
1. Walk `m.walk()` and confirm: one `text/html` part + N `image/jpeg` parts, each with a `Content-ID`.
2. Regex the HTML for `<img src="cid:` — every cid must match a part's Content-ID (strip `< >`).
3. Do NOT grep the stored HTML for `data:image/` — it will legitimately be absent.
4. Re-run the full-width greps (`max-width` = 0, `width:100%` ≥ 1) and content markers on the STORED html part.
5. The image parts' filenames come from the `<img alt="...">` text — keep alts descriptive.

## Downloading source images / PDFs from Drive (verified 2026-10-06)

`drive.files().get(fileId=..., alt='media').execute()` on a binary file returns a DICT, not bytes ("a bytes-like object is required, not 'dict'"). Use MediaIoBaseDownload:

```python
from googleapiclient.http import MediaIoBaseDownload
import io
request = drive.files().get_media(fileId=FILE_ID)
fh = io.BytesIO()
dl = MediaIoBaseDownload(fh, request, chunksize=1024*1024)
done = False
while not done:
    _, done = dl.next_chunk()
open(path, 'wb').write(fh.getvalue())
```

Run under the session user's slug (`HERMES_SESSION_USER_ID=psingh`) with the `/opt/hermes` venv (has googleapiclient + PIL; `/opt/data/.venv` may lack both). Then downscale to 800px/q85 JPEG per Prakash's preference. Always `vision_analyze` candidate creatives BEFORE embedding so captions describe what the image actually says (e.g. RG hero vs explainer vs flyer — they differ).

## Replacing existing drafts with improved versions (verified 2026-10-03)

When the user wants to redo drafts that already exist (e.g. "the image is too large, redo all 10 drafts with a compressed version"):

1. **Find the draft message IDs.** Search for the sent/replied messages with `gmail.users().messages().list(userId='me', q='query')` — each reply-to-sent campaign creates a new message in the thread. Get its `id`.

2. **Check draft vs sent status.** Gmail labels tell you whether a message is still a draft or was already sent:
   ```python
   labels = gmail.users().labels().list(userId='me').execute()
   label_map = {lbl['id']: lbl['name'] for lbl in labels['labels']}
   msg = gmail.users().messages().get(userId='me', id=MSG_ID, format='minimal').execute()
   is_draft = 'DRAFT' in [label_map.get(l, l) for l in msg.get('labelIds', [])]
   ```
   A message with `SENT` label was already delivered — skip it. A `DRAFT` label means the draft exists but wasn't sent.

3. **Delete old drafts (trash via API).** Only trash the draft messages (those with `DRAFT` label). Do NOT trash the original thread-starter email:
   ```python
   gmail.users().messages().trash(userId='me', id=DRAFT_MSG_ID).execute()
   ```

4. **Recreate drafts.** Target the ORIGINAL thread-starter message (the first email in the thread), not the draft you just deleted. The original message ID is stable — get it from `msg['threadId']` and querying the thread, or from the stored mapping. The new draft will thread correctly because the thread still exists.

5. **Do NOT include `To: orig_from`.** The original thread-starter's `From` is the user — `To` must be the original recipients from the `To` header.

### Verification checklist after replacement

- [ ] Drafts list shows the correct count (original count − sent count)
- [ ] Each draft's `To` matches the channel partner, not the user
- [ ] Each draft's `threadId` matches the original thread
- [ ] Label check confirms the new messages are `DRAFT`, not `SENT`
- [ ] Image decoded from the draft's HTML body matches the compressed version (size, dimensions)

## Inline project images (CID) + a PDF attachment — verified 2026-10-03

When the user says "add the project image in the email" / "add the attachment",
the image must be EMBEDDED INLINE in the body (renders in Gmail), not sent as an
attachment, and the document rides alongside it as a normal attachment. The native
`gws_gmail_draft_create` tool CANNOT do this — it accepts a single HTML string,
with no image and no attachment parameters. Assemble `MIMEMultipart` by hand, same
as the recipe above.

Structure: `multipart/mixed` → [`multipart/related` → (text/plain, text/html,
image parts)] + [application/pdf attachment].

```python
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication

msg = MIMEMultipart("mixed")
msg["Subject"] = SUBJECT

alt = MIMEMultipart("related")
alt.attach(MIMEText(BODY_INTRO, "plain"))   # plain fallback carries the key facts
alt.attach(MIMEText(HTML, "html"))          # body uses <img src="cid:hero">
for cid, path in (("hero", hero_jpg), ("ext3", ext3_jpg), ("lobby", lobby_jpg)):
    img = MIMEImage(open(path, "rb").read(), _subtype="jpeg")
    img.add_header("Content-ID", "<%s>" % cid)
    img.add_header("Content-Disposition", "inline", filename=os.path.basename(path))
    alt.attach(img)
msg.attach(alt)

pdf = MIMEApplication(open(pdf_path, "rb").read(), _subtype="pdf")
pdf.add_header("Content-Disposition", "attachment", filename="... .pdf")
msg.attach(pdf)

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
gmail.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()
```

Rules that matter:

- CID must match EXACTLY: `cid:hero` in the HTML ↔ `<hero>` in the Content-ID header.
- **Downscale and JPEG-compress the renders first.** Drive PNG renders run
  2.4–2.6 MB each. PIL `Image.LANCZOS` → 1200 px / q84 gives a ~230 KB hero;
  620 px / q78 gives ~50 KB inline strips. Three raw PNGs plus a 6.6 MB PDF would
  have pushed the raw MIME past Gmail's 25 MB message ceiling.
- Keep the full-width rule on every image: `width="100%"` plus
  `style="display:block;width:100%;height:auto;border:0"`.
- A 6.6 MB source PDF produced a ~12 MB raw MIME after base64 — fine, but check the
  ceiling if you also attach the editable HTML/Doc siblings.

Verification — read the draft back with `format='raw'` (NOT `'full'`, which can omit
To/Subject), decode via `email.message_from_bytes(raw, policy=policy.default)`,
`m.walk()`, and assert the part inventory: one text/plain, one text/html, one
image/jpeg per CID, and the application/pdf with its filename. Then re-run the
`max-width` (must be 0) / `width:100%` (>=1) greps on the STORED html part, not the
local file.

## "Draft a mail to the IDs mentioned in this spreadsheet" — verified 2026-10-03 (Ranka Amber → 36 Whitefield channel partners)

Recurring Prakash shape: he shares a Google Sheet of agents/consultants/vendors and
says *"draft a mail to the email IDs mentioned in this spreadsheet — add a brief of
<project> with RERA number and other relevant details, attach the attachment, add the
project image."* Treat it as ONE campaign draft, not N drafts.

1. **Read the sheet** with the session user's own token — `build_service("sheets",
   "v4", service_name="google-draas")` then `spreadsheets().values().get(...,
   range="A1:Z200")`. Grab the title and ALL rows; the header row matters for
   interpreting columns.
2. **Harvest addresses with one regex over every cell of every row** —
   `[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}` — then lowercase-dedupe.
   Do NOT assume the email column index; several sheets carry "Not publicly
   available" in the Email column and real addresses in the next one along
   (e.g. a "Contact Details" column that holds plain phone numbers).
   Multiple addresses per firm are normal (sales@ + service@, or the principal's
   direct address on a second domain) — keep them all.
3. **Put every address in `Bcc`**, `To` empty. There is no mail-merge; a flat BCC
   blast is the intended model for a channel-partner announcement. Say plainly in
   your reply that BCC was used so no partner sees the others, and OFFER to split
   into a smaller To: list if he prefers.
4. **Report coverage, don't silently drop firms.** Name the firms that had NO
   published address so he knows the blast is partial and phone/portal outreach is
   still owed — he tracks reach, and a silent omission reads as "all 36 contacted".
5. **Assemble MIME by hand** (mixed → related + pdf attachment, inline CID images)
   exactly as in the section above; the native `gws_gmail_draft_create` cannot carry
   images or attachments.
6. **Message shape that works**: image-led body — hero render across the full width,
   masthead, a short letter opening ("Greetings from ..."), then a **Project at a
   Glance** table carrying the RERA registration number, sanctioned-plan number and
   date, location, unit mix, price, bank/APF reference, possession date, promoter
   and GPS; then a short "why this project" block; then project images; then an
   explicit "Attached — <document> (version, pages, what it contains)" call-out;
   then the next-step ask and the signature block. Body in HTML, plain-text part
   carries the same key facts (RERA no., price band, attachment, contact).
7. **Draft only, never send.** Close the reply by stating the draft is in Drafts and
   he presses send.

RERA/statutory identifiers in the body come from the project's own approved source
documents, not from the sheet — for Ranka Amber the authoritative set is
`PRM/KA/RERA/1251/446/PR/050826/008859`, plan `GBA/BECC/0540/25-26` (07-05-2026),
ICICI APF `KA/BAN/26/61572`. Pull them from the project's Investment Guide / approved
documents rather than retyping from memory, and grep the built HTML to confirm each
one landed.

**Ready-to-adapt script:** `templates/campaign-email-builder.py` in this skill
implements the whole chain — sheet harvest (regex over every cell + dedupe), PIL
downscale, full-width HTML build with a Glance table and cid: images, hand-assembled
MIMEMultipart with BCC blast and PDF attachment, draft creation, and the stored-draft
verification pass. Copy it, fill the REPLACE markers, and run it under the session
user's own slug. It contains no send call by design — do not add one.