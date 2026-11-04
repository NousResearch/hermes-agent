# Stage 2 — Draft: full detail

> Extracted from `email-drafter` SKILL.md on 2026-09-20 (the SKILL.md was at the 100 KB hard limit).
> Everything about composing the draft body, formatting, attachments, threading, and the per-case
> worked examples lives here. Load it whenever you are actually building the draft.

## 3. Stage 2 — Draft

### Work email tone
- No greeting, go straight to the point
- Numbered tasks if there are asks, deadlines in bold if HTML
- No boilerplate ("Hope you're well", "Dear [name]") unless explicitly asked
- Subject: `[Project/Entity Name]: [one-line description]`

### Personal / casual tone
- Warmer, no subject prefix
- Plain text is fine
- **Roshni Ranka / "RO":** Always personal tone
- **Family (kids, spouse, parents):** Always personal tone; same content can go to multiple family members in ONE email using To + CC — do NOT create a separate draft per recipient (user correction 2026-07-13: "I need only one draft email. Why four drafts? This is one email.")
- **Extended family (brother-in-law, sister, sister-in-law):** Personal tone and family address ("Dear Jai ji") even when the content is business/legal — use warm opening but structured body with numbered items. Sender is ndr@draas.com when the content relates to DRA governance/estate matters (validated 2026-08-27, Ranjeeth/Mamata/Roshni succession petition email). See `references/family-legal-document-sharing.md` for the full pattern.

### School / authority communication tone (polite-cooperative, 2026-08-27)
Used when emailing teachers, school administrators, or any institutional authority (government offices, regulatory bodies, RERA authorities). The goal is to express a concern or push back politely while remaining cooperative and seeking understanding — NOT confrontational or demanding.

Structure and language principles:
- **Open with appreciation:** Thank them first — acknowledge their response, accommodation, or effort before raising the question. This sets a collaborative tone.
- **State the medical/known context factually:** "Ruhaan has an established medical pattern — whenever he contracts a viral infection, it triggers an asthmatic flare-up resulting in a persistent cough. This is a known, recurring issue." Frame it as context-sharing, not excuse-making.
- **Explain the action taken (and why no doctor visit):** "Following our doctor's standing advice for this known pattern, we have started him on azithromycin without a hospital visit this time. There may be no clinical benefit in taking him to a doctor unless his condition deteriorates."
- **Acknowledge their requirement/position:** "We fully understand the school's need to maintain proper records."
- **Express the concern as a question, not an objection:** Frame it as seeking understanding: "We would like to understand the basis for this requirement — is it mandated by IGCSE/board rules, or is it a school policy for record-keeping?" Never say "this is unnecessary" — say "an unnecessary visit would put him under additional strain and risk of infection for a known medical issue."
- **Offer alternatives proactively:** "We are happy to share a video or any other form of proof. If there is an alternative medium that satisfies your requirement without a medical visit, we would be grateful for your guidance."
- Close with warmth and cooperation: "We would be grateful for your guidance so we can find the best way forward for Ruhaan."

**Related:**  `references/school-absence-medical-cert-required.md` — full workflow when the school insists on a certificate as SOP (Grade 9+ board requirements, known chronic condition, WhatsApp the family doctor, reply with attached cert + BCC admin).
**Related:** For a simpler notification-only absence + exam-accommodation email (not challenging a policy, just informing and requesting), see `references/school-absence-exam-accommodation.md`.

**Discovery tip — school Welcome PDFs are the authoritative source for teacher roles:** When emailing a school and unsure who the actual class teachers are (versus who handles attendance), search Gmail for the school's welcome/introductory email at the start of the academic year — `subject:"Welcome" from:<school-domain>`. The attached PDF usually lists class teachers + teaching team explicitly. Run `pdftotext` or `pymupdf` on the attachment to extract the names. Example: Aditi Std 07 SS Welcome PDF showed Neetu Shrivastava + Subath Senan as class teachers, while the person handling the leave note (Ranjitha Tikandar) was not a class teacher at all. This avoids the common trap of assuming the attendance-contact person is a class teacher.

Key distinctions from vendor-feedback tone:
- ❌ No competitive pressure ("commissioned a parallel test")
- ❌ No disappointment framing ("we are extremely disappointed")
- ❌ No consequences set ("this is our last effort")
- ✅ Positive, cooperative, appreciative throughout
- ✅ Frame the underlying question as seeking understanding
- ✅ Offer alternatives before asking for exceptions

### Vendor-feedback / frank escalation tone (user preference, 2026-08-17)
Used when delivering formal feedback to a vendor/service provider whose product has underperformed. NDR's explicit direction: "as frankly as possible without doing any threats."

Structure:
1. **State the situation factually** — what was tested, how many calls/transcripts, what methodology (recorded → transcribed → cross-checked against verified data).
2. **Name the failures concretely** — data errors, latency, missing features. Quote specific numbers (percentages, call counts) so the feedback is undeniable.
3. **Mention alternatives if they exist** — "we have commissioned a parallel test with another provider. In the very first cut, it was absolutely brilliant — correct accent, facts right, conversation handled naturally." This frames the conversation as competitive, not remedial.
4. **Express disappointment factually** — "we are extremely disappointed that your product experience is so poor, especially for a team that advertises heavily. Right now it feels half-baked — not even an alpha."
5. **Set a clear consequence as a fact, not a threat** — "This is our last effort to get a product that meets basic requirements. If this still fails, we will withdraw from this pilot." NEVER say "we may be forced to reconsider" or "we have limited bandwidth" — that's hedging, not frankness. State the consequence plainly.
6. **Close with an action request** — "kindly do the needful urgently so we can test again." Professional, no anger, just clarity.

Key rules:
- No threats, no aggression, no rhetorical questions. The evidence does the work.
- Mentioning another provider is acceptable and adds competitive pressure without threatening language.
- If the vendor advertises heavily, it's fair to note the gap between marketing and delivered product — this is a factual observation, not a personal attack.
- Keep attachment evidence (PDF feedback, data sheets) bundled in the email so they can verify every claim.

### One draft, multiple recipients (To + CC)
When the same content goes to several people, create a single draft with multiple To addresses and CC the rest. Do NOT spawn one draft per recipient.

`google_workspace_manager` supports this directly:
```
tool: google_workspace_manager
input:
  command: "gmail messages send --to a@example.com,b@example.com --cc c@example.com,d@example.com --subject '...' --body '...'"
  account_email: "nishantranka@gmail.com"
```

For HTML: add `--bodyHtml '<p>...</p>'` alongside `--body`. Confirm the full recipient list (To + CC) with the user before drafting.

If using the underlying `gws_skill_bridge` Python module instead (e.g. from a venv script), `draft_create` accepts flat `to=`, `cc=`, `subject=`, `body=` kwargs:
```python
from tools.gws_skill_bridge import call
call("draft_create", service_name="google-gmail",
     to="rankarivaan@gmail.com,pebblyshark69@gmail.com",
     cc="rnr@draas.com,Rmurjani@gmail.com,nishantranka@gmail.com",
     subject="⚽ FIFA World Cup 2026 — Polymarket & Kalshi odds",
     body=html_body,
     html=True)   # <-- required when body is HTML; without it the draft is plain-text and your markup is sent as raw literal characters
```
The bridge handles MIME multipart and base64 internally — no need to build MIMEMultipart yourself. Use comma-separated addresses (no `Name <addr>` formatting needed; the bridge adds display names from contacts if you pass bare emails).

**Full workflow reference:** `references/drive-to-draft-pipeline.md` — covers finding files on Drive, downloading, categorizing, building the MIME message, and deduplicating. Use it whenever the draft needs multiple Drive-sourced attachments.

**New-draft-from-Gmail-attachments:** `references/gmail-to-drive-to-draft.md` — covers extracting attachments from prior Gmail emails, renaming per NDR naming convention (YYYYMMDD_Entity_Description), uploading to the right Drive folder for archival, then creating a fresh draft to a NEW recipient with those Drive files attached. Use this when the user says "find the floor plans I sent to [person], rename and file them, and send to [new recipient]."

**Repurposing a prior chain to a new recipient** ("send the same briefing + attachments to advocate Y"): `references/repurpose-email-chain.md` — search multi-thread by subject keyword, read all bodies, collect attachments across ALL messages/threads (petition may be on one thread, supporting docs on another), rebuild MIME with raw Gmail API, verify via `drafts().list()`. Session-validated Aug 2026 (succession certificate → Sanjay Sethia).

**Pitfall — `threads().get(format='raw')` is NOT a valid parameter (confirmed 2026-08-02):**
The Gmail API `threads().get()` only accepts `format` values `['full', 'metadata', 'minimal']` — passing `'raw'` raises `TypeError: Parameter "format" value "raw" is not an allowed value`. This traps you when building a threaded reply draft that needs the latest message's `Message-ID`/`References` headers. The fix is a two-step fetch:
```python
t = service.users().threads().get(userId='me', id=THREAD_ID, format='metadata').execute()
latest_msg_id = t['messages'][-1]['id']
latest = service.users().messages().get(userId='me', id=latest_msg_id, format='raw').execute()
raw_bytes = base64.urlsafe_b64decode(latest['raw'].encode('ascii'))
# then regex Message-ID / References out of the raw text
```
`messages().get()` DOES support `format='raw'`; only the thread-level call rejects it.

**Pitfall — raw-regex Message-ID extraction is fragile; use metadataHeaders instead (2026-08-25):** `re.search(rb'Message-ID:\s*(\S+)', raw_bytes)` on the raw MIME once captured garbage (`subject:date:mime-version:from`) instead of the real Message-ID, silently producing a malformed In-Reply-To/References on a threaded draft. Prefer fetching headers directly — no raw decode, no regex:

```python
latest = service.users().messages().get(userId='me', id=latest_msg_id, format='metadata',
    metadataHeaders=['Message-ID','References','In-Reply-To']).execute()
heads = {x['name'].lower(): x['value'] for x in latest['payload']['headers']}
src_mid = heads.get('message-id','').strip()
assert src_mid.startswith('<') and src_mid.endswith('>'), f"BAD MID: {src_mid!r}"
```

**Pitfall — `draft_reply_create`/`draft_create` do NOT support BCC (2026-08-27, Ruhaan school email):**
Neither operation in `gws_skill_bridge` accepts a `bcc` parameter. If the user wants someone blind-copied, you MUST fall back to the raw Gmail API using `email.message.EmailMessage`:

```python
from email.message import EmailMessage
import base64
from tools.gws_auth import build_service

svc = build_service('gmail', 'v1', service_name='google-draas')

msg = EmailMessage()
msg.set_content('Email body text here')
msg['To'] = 'Primary <primary@example.com>'
msg['Cc'] = 'Cc1 <cc1@example.com>, Cc2 <cc2@example.com>'
msg['Bcc'] = 'Blind <blind@example.com>'
msg['Subject'] = 'Re: Original Subject'

raw = base64.urlsafe_b64encode(msg.as_bytes()).decode('ascii')
draft = svc.users().drafts().create(userId='me', body={
    'message': {'raw': raw}
}).execute()

# Verify
verify = svc.users().drafts().get(userId='me', id=draft['id'], format='full').execute()
headers = {h['name']: h['value'] for h in verify['message']['payload']['headers']}
assert 'Bcc' in headers, 'BCC header missing in draft — use raw API, not bridge'
assert 'DRAFT' in verify['message'].get('labelIds', []), 'Not a draft!'
```

The Bcc header MUST be set on the `EmailMessage` object before encoding — it will NOT appear in the `drafts().get()` returned headers on Gmail's side (Gmail strips Bcc from API responses for security), but it IS present in the MIME and will be sent correctly. Verify by checking that the draft shows up in the correct mailbox with correct To/Cc visible.

**Pitfall — `draft_reply_create`/`draft_create` do NOT support attachments (2026-07-15):**
Neither operation in the `gws_skill_bridge` accepts file attachments. The MIMEText-based body construction has no provision for multipart/mixed. If your draft needs a PDF, image, or any file, you MUST fall back to the raw Gmail API:

1. Build a `MIMEMultipart("mixed")` message manually (HTML body part + MIMEBase attachment parts)
2. Base64-encode the whole message and call `service.users().drafts().create()` directly with `threadId`
3. Use `tools.gws_auth.build_service("gmail", "v1", service_name=...)` — NOT `gws_skill_bridge._build_service()` (see next pitfall)

See `templates/draft-with-attachments.py` in this skill for the full working recipe.

**Pitfall — Gmail attachment cap is on the ENCODED size, not the raw file (2026-08-25):**
`draft_create`/`draft_reply_create` build MIME internally — the size that decides
sendability is the base64url-encoded `raw` field, which is ~16/9 ≈ 1.78× the raw
PDF (base64 inside MIME ×4/3, then the MIME itself base64url-encoded ×4/3 for the
API). The Gmail API cap is **35 MB encoded**, so the attachment PDF must be
≤ ~20 MB — do NOT treat "25 MB" as the raw-file limit, and do NOT trust a
36 MB → 25 MB /ebook compression as "done" (observed: 36 MB pack → 25.18 MB with
/ebook → `len(raw)` = 43.3 MB = would FAIL on send).

Always print the encoded size from the build script
(`print(round(len(raw)/1048576, 1))`) and confirm ≤ 35 MB before declaring the
draft done. If you already created the draft with an oversized attachment:
`drafts().delete(userId='me', id=<draft resource id, NOT message id>)`, then
recreate with the compressed file (same mailbox, verified via drafts().get walk
of payload.parts for the attachment filename + size).

**Recipe that actually got a 225-page / 36.4 MB compiled pack to 19.2 MB legible (2026-08-25, Ranka Udaya pack):**
`/ebook` alone only reached 25.2 MB → still 43.3 MB encoded (FAIL). The pass that
worked combined `/screen` with explicit image downsampling:

```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen \
  -dColorImageDownsampleType=/Bicubic -dColorImageResolution=90 \
  -dGrayImageDownsampleType=/Bicubic -dGrayImageResolution=115 \
  -dMonoImageResolution=160 -dNOPAUSE -dBATCH -dQUIET \
  -sOutputFile=out.pdf in.pdf
```

19,156,889 bytes (19.2 MB) → `len(raw)` = 32.9 MB → SAFE. Verify legibility before
attaching: render 2–3 sample pages with `pdftoppm -png -r 100 -f N -l N` and OCR/
vision-check that text still extracts. 100–130 DPI grayscale keeps scanned deed
text readable; below ~90 DPI it degrades.

For a 30–40 MB multi-page scanned legal pack, `/ebook` is insufficient — use
`/screen` with explicit downsampling (observed: 36 MB → 19.2 MB → encoded
32.9 MB = PASS, 225 pages preserved):

```bash
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/screen \
  -dColorImageDownsampleType=/Bicubic -dColorImageResolution=90 \
  -dGrayImageDownsampleType=/Bicubic -dGrayImageResolution=115 \
  -dMonoImageResolution=160 -dNOPAUSE -dBATCH -dQUIET \
  -sOutputFile=out_email.pdf in.pdf
pdfinfo out_email.pdf | grep Pages   # must match pdfinfo in.pdf
```

After aggressive compression, spot-check legibility before attaching:
`pdftoppm -png -r 100 -f <p> -l <p> out_email.pdf /tmp/p` on 2–3 pages (title +
a scanned deed page + one mid-pack page) and OCR/vision them — scanned deeds
degrade fast below ~115 DPI grayscale; if unreadable, raise GrayImageResolution
and re-verify.

Attach the compressed copy but keep the ORIGINAL customer-facing name in the
MIME display filename (local file can be `*_email.pdf`, display as e.g.
`Ranka Udaya - Legal Document Pack.pdf`). Observed (Gunjur Will chain email):
29 MB 12-page 1990 sale deed → 2.7 MB, 12 pages preserved, sent fine.

**Pitfall — attaching a Google-native file (Sheets/Docs/Slides) requires export FIRST (2026-08-01):**
A Google-native spreadsheet has no binary payload to attach — `MIMEBase` needs a real file on disk. Export it via the Drive API to xlsx before building the MIME message:

```python
from googleapiclient.http import MediaIoBaseDownload
import io

req = drive.files().export(
    fileId=SHEET_ID,
    mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)
fh = io.BytesIO()
dl = MediaIoBaseDownload(fh, req)
done = False
while not done:
    status, done = dl.next_chunk()
open('/tmp/charges.xlsx', 'wb').write(fh.getvalue())
```

Then attach `/tmp/charges.xlsx` as a MIMEBase part with `Content-Disposition: attachment` and the human-friendly filename (spaces/dashes OK). This is the pattern for "attach this Google Sheet to my reply" requests — the recipient receives a real .xlsx, not a link.

**Pitfall — Gmail 25 MB attachment cap: compress scanned PDFs before MIME build (2026-08-25).** Certified-deed scans routinely exceed the cap (observed: 1990 Doddaballapur sale deed = 29 MB / 12 pages → draft rejected on send). Compress first with `gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4 -dPDFSETTINGS=/ebook -dNOPAUSE -dBATCH -dQUIET -sOutputFile=out.pdf in.pdf` (29 MB → 2.7 MB on a 12-page scan), then verify `pdfinfo` page count matches the original BEFORE building the MIME message — never attach an un-checked compressed copy. Keep the original Drive file as the source of truth; the compressed copy is for sending only.

**Pitfall — `_build_service` vs `build_service` account resolution (2026-07-15):**
`gws_skill_bridge._build_service(api, version)` does NOT accept a `service_name` parameter and defaults to `google-draas` (ndr@draas.com). If your thread lives in a different account (e.g. `google-ahfl` for ndr@ahfl.in), the API call for `messages().get()` will return HttpError 404 because the message doesn't exist in the default account.

Fix: import from `tools.gws_auth` directly:
```python
from tools.gws_auth import build_service
service = build_service("gmail", "v1", service_name="google-ahfl")
```
Resolve the correct service_name first via `gws_resolve_account(account="ahfl")`. Never use the bridge's internal `_build_service` for cross-account operations.

### Counterparty replied with a NEW email, not a threaded reply (2026-08-13)

Law firms / advocates often reply from a DIFFERENT person's mailbox with a NEW subject and no In-Reply-To (observed 2026-08-12: original went to Sanjay Sethia sanjay@lawsquare.in; reply came from Nishanth Adv nishanth@lawsquare.in — same domain — with subject changed from "Priority: Succession Certificate ..." to "Succession Certificate (Dinesh Ranka)", In-Reply-To: None). When the user says "reply in the same email" after such a reply:

1. Find the LATEST counterparty message by **domain**, not just name: `q='lawsquare.in newer_than:7d'` across ALL accounts (google-draas, google-gmail, google-ahfl). The replier may be a colleague of the original addressee, so a name search for the original recipient misses it.
2. Check its `In-Reply-To` / `threadId`. If it's a fresh thread (In-Reply-To: None), the correct "same email" is the counterparty's NEW message — reply to THAT message (`draft_reply_create` with that message_id), not the original sent thread. This threads the reply from their side and keeps the rest of the firm on Cc.
3. Derive To/Cc from the new message: To = its `From`; Cc = its `To` (minus self) + its `Cc` — reproduces Gmail's reply-all for the actual message being answered. Verify the draft (drafts().get: Subject `Re: <their subject>`, In-Reply-To set, DRAFT label, present in drafts().list()).

### Email body claimed attachments ≠ actually attached (2026-08-13)

Before answering "was the petition/documents actually attached to our email?", do NOT trust the body's attachment list. Walk the real MIME parts:

```python
def walk(payload):
    if payload.get('filename'):
        print('ATTACH:', payload['filename'], payload.get('body', {}).get('size'))
    for p in payload.get('parts', []):
        walk(p)
```

Observed: sent email body listed 4 docs ("Draft petition, Death certificate, Family tree, Aadhaar cards") but only 3 were actually attached — the Aadhaar cards were listed yet never attached. When the counterparty says "documents were not attached", report the discrepancy precisely (which attachments exist, which claimed one is missing) so the reply states facts accurately (here: petition WAS attached; only Aadhaar cards missing — so the reply corrected them while confirming we'd send the missing set).

### Threaded reply — last message is from you

When replying in a thread where the **most recent message was sent by you** (the user), the `draft_reply_create` helper defaults its `To` to the `From` of the last message — which is the user themselves. This creates a self-reply draft.

**Fix:** Scan the thread to find the last message FROM the other participant (not the user), and set `To` explicitly:
```python
thread = service.users().threads().get(userId='me', id=THREAD_ID, format='metadata').execute()
reply_to = None
for msg in reversed(thread['messages']):
    h = {hdr['name'].lower(): hdr['value'] for hdr in msg['payload']['headers']}
    frm = h.get('from', '')
    if 'ndr@' not in frm and 'nishantranka' not in frm:
        reply_to = frm
        break
# Now use reply_to as the To address
```

### Reply-all — derive recipients from the LATEST message (2026-08-01)

For a "reply all in this thread" request, derive To/Cc from the **most recent message in the thread** (the one you're actually answering), not the thread's first message or the draft helper's default:

```python
latest = sorted(thread['messages'], key=lambda m: m.get('internalDate', 0))[-1]
h = {x['name'].lower(): x['value'] for x in latest['payload']['headers']}
me = 'ndr@draas.com'
def split(addrs): return [a.strip() for a in (addrs or '').split(',') if a.strip()]
# Standard Gmail reply-all: original SENDER goes to To; original To (minus me)
# + original Cc (minus me) go to Cc. Verified 2026-08-11 (Bajaj Life reply:
# To=Rohit.Sundarka@bajajlife.com, Cc=rnr@draas.com, echamundeshwari@draas.com).
to = [h['from']] if h.get('from') and me not in h['from'] else [a for a in split(h.get('to','')) if me not in a]
cc = [a for a in split(h.get('to','')) if me not in a] + [a for a in split(h.get('cc','')) if me not in a]
cc = [a for a in cc if a not in to]   # don't duplicate the sender between To and Cc
```

- The latest-message `To` field still contains the user's own address (they were a recipient) — **strip self** from To before drafting.
- The original sender goes to **To** (that's who you're answering); everyone else from the latest To/Cc (minus self) goes to **Cc** — this matches Gmail's reply-all.
- Preserve display-name form (`Name <email>`) — don't strip it; it keeps the draft readable in Gmail.
- Verify the final draft with `drafts().get()`: To/Cc should contain every intended participant exactly once, and no excluded party should appear.

### Reply-all — "everyone from all the email chains" (2026-08-12)

When the user says to include everyone who appeared across the different emails (e.g. "his PA was in earlier emails… make sure everyone who was there in the email chains across the different emails are all added"), deriving from the LATEST message alone is NOT enough — a PA/assistant may have been on older threads but dropped from the most recent message (real case: Millers Road — Atheeq padirector@ahindia.com was Akber's PA/accounts and had to be added even though the latest revert email listed only Akber + Aamir).

1. Search Gmail for ALL threads matching the person/subject keyword (e.g. `q='(from:akber OR to:akber) Millers'`), and collect every To/Cc across every message in every matching thread.
2. Dedupe by person; keep ONE address per person — prefer the address used in the MOST RECENT thread. People change addresses mid-deal (Aamir: aamirkhan@me.com in Jun–Jul → khan.hussain.aamir@gmail.com in Aug 2026). Do not CC both old and new.
3. To = the person the user names ("we send it to Akbar"); Cc = everyone else in the union (PA + Aamir).
4. Reply in the LATEST thread (where the counterparty most recently reverted), with proper In-Reply-To/References.
5. Contact rule still applies: every address used must be verifiable in the thread history / NDR's contacts — never invent one.

### Voice-transcription disambiguation before drafting (2026-08-17)

NDR dictates drafts by voice; the transcript garbles names and terms. Observed same-session examples:
- "Aamir" → meant BOTH the lessor **Akber Hussain** (akber@ahindia.com, To) AND advisor **Aamir Khan** (khan.hussain.aamir@gmail.com, Cc) — both were on the thread, and "reply all" resolved the ambiguity
- "online visa agreement" → lease agreement; "the laser" → the lessor; "ETO" → ETA; "FR" → FAR; "Ajayarchitect" → A.J. Architects / Arvind Jain; "7 above 14 bar 1" → Sy. No. 7/1 & 14/1
- **Money figures render from lakh/crore phrasing, not the digit garble (2026-08-20):** "one fifty from GB Jat" → ₹1.5 crore; "a thirty" → ₹30 lakh; "twenty five lakh rops" → ₹25,00,000. DRAAS/RSFC deals talk in lakhs/crores, so "one fifty"/"one point five" = ₹1.5 Cr and "a thirty" = ₹30 L. Cross-check against the actual transaction figures (refund amount on the thread) before locking them in.
- **Entities garble as phonetically-different words (2026-08-20):** "regsorformers collective" → **Red Soul Farmers Collective**; "DRA Realty Private Limited" (user dictation) vs the thread's "DRA Thindulu Land Partners" — when the user names an entity that partially matches the thread scope, flag the difference and offer to broaden (don't silently pick one).
- **"Bucks Ranka" → BuxRanka (2026-08-30, corrected twice in one session):** Do NOT resolve "Bucks Ranka" as "Ranka North Star" — "Bucks" is the phonetic garble for **Buxani** (Haresh Buxani), so "Bucks Ranka" / "BuxRanka" = the **BuxRanka Hudson Project** (Buxani + Ranka JV, BRDPL, GFM/Godrej Fund Management deal). The written guideline-value confirmation precedent lives with BuxRanka — when NDR says "like the confirmation obtained for Bucks Ranka, Rahul knows about it", the reference project is BuxRanka, and the message must say "BuxRanka" verbatim. Context that disambiguates: the BuxRanka → Godrej thread is about Premium FAR vs TDR procurement (Mar–Jun 2026, Viraj/Harsimran at godrejventure.com). Ranka North Star is the Allalasandra project — a DIFFERENT entity; confusing the two sends the wrong precedent to colleagues.
- **"Bhavish Bhavna" → "Bhavesh Bafna" (2026-08-21):** Voice transcription can garble BOTH first and last names simultaneously. The contact sheet has Bhavesh Bafna (bvbafna@yahoo.co.in / bvbafna@gmail.com) — "Bhavish" is the garbled first name and "Bhavna" is the garbled last name. When both names come out wrong, search the contact sheet by first-name-only substrings ("bhav" → "Bhavesh"), cross-check with Gmail thread history, and surface the candidate to the user rather than guessing.
- **Recipient "Saurabh VK the CA" / "accounting group where Nitin is part"** resolve via contact sheet entries carrying job-tag suffixes (e.g. `Saurabh CA Ref VK`, `Nitin Osthwal Saurabh CA`) — these exact tags disambiguate the CA from the many other Saurabhs/Nitins. Pull the phone from the tagged entry, not a generic name match.

- **Legal/estate voice garble (2026-08-25):** "debt certificate" → death certificate; "sevdhanapalli" / "SEV Ganapalli" → **Sevaganapalli** (Ranka Oasis village, Hosur Taluk); "Tushas" / "Vishwas" → resolve via find_contact (Vishwas Rao, Vantage Point Advisors, vishwas@vantagepointadvisors.in); "Giraffe Capital" → Jiraaf Capital. When the user names a clause to remove from a deal doc ("profit of 15% to Nishant Prakash"), read the ACTUAL latest doc first — the clause text may differ from the dictation, and verifying the real content before editing is a hard user expectation on deal docs.
- **DR Family succession voice garble (2026-09-01):** "Hermit" / "Damesh" / "Manesh" → **Dharmesh Ranka** (dharmesh.ranka1@gmail.com — the Gmail header-verified active address; the contacts sheet also carries dharmesh.ranka@gmail.com / ddr@draas.com, prefer the most recent Gmail To-header form); "Manish, Manesh" → **Manish Ranka** (mdr@draas.com); "vocala" → **vakalat** (vakalatnama); "GPI" → **GPA** (General Power of Attorney); "Mamtha" / "Mamata" → **Mamata Rathod** (mamatadr@gmail.com, sister in Chennai, ready to sign). When the dictation says "email to both X and Y telling them..." the recipients are Dharmesh + Manish (brothers) as TO — do not draft the external-CC variant; the voice describing an email TO two family members about a filing means To both.
- **Family-succession voice garble (2026-09-01, DR Family succession email):** "Hermit" / "Damesh" / "Damesh" → **Dharmesh Ranka** (dharmesh.ranka1@gmail.com — use the gmail1 address from Gmail headers, not the sheet's ddr@draas.com); "Manish / Manesh" → **Manish Ranka** (mdr@draas.com); "Mamtha / Mantha" → **Mamata Rathod** (mamatadr@gmail.com); "Sanjay Setia" → **Sanjay Sethia** (sanjay@lawsquare.in, Law Square); "vocala / vakalat" → **vakalatnama**; "GPI / GPA / GPI" → **GPA** (General Power of Attorney); "Rajendra" = the family GPA holder already named in the petition. Do NOT invent the trust name ("Gauravar Bhavan Jain" was garbled → write "recently handled trust-related work" generically).
- **Email address merging from voice dictation (2026-08-27, Kanta Ranka TPA query reply):** When the user says something like "Mark Sarthak at admin3.blr@draas.com" during voice dictation, the transcript produces a single concatenated string: "Sarthakadmin3.blr@draas.com". This is NOT a valid email — the name ("Sarthak") merged with the email prefix ("admin3") because the speaker paused in the wrong place. Do NOT use the merged string. Resolve each recipient's actual email from:
   1. Gmail thread headers (the person was already on the CC of a prior message in the thread)
   2. The `contact_resolver` tool
   3. Prior drafts on the same thread
  In this case, `admin3.blr@draas.com` is the correct email for Sarthak Sharma — never construct a combined form.
- **Removing a money term from a deal doc — flag the economic side-effect (2026-08-25).** When the user says "remove the ₹2 Cr goodwill / the 15% profit clause" from a term sheet, the headline numbers move (net consideration ₹33.34 Cr → ₹35.34 Cr). State the new headline in your reply so the user can confirm before the doc goes out to a consultant — do not let a clause removal silently change the deal economics.

**Rules when drafting from a voice note:**
1. Resolve every person/entity name against the ACTUAL thread headers (From/To/Cc of the latest + earlier messages) before choosing recipients — never trust the transcript name alone.
2. When the spoken name could match two people (Akber vs Aamir Khan), prefer **reply-all to the thread** — it covers both interpretations.
3. Numbers/technical terms ("7/1 & 14/1", "1.75 FAR", "25% setback relaxation") — render from the deal context in the thread, not the garble.
4. Before finalizing, surface the uncertain items to the user: "Survey numbers: I rendered 7/1 and 14/1 — correct?" / "The ADTP name came through unclear — I wrote just ADTP." Do NOT silently guess.

### Reminder / follow-up emails — anchor on the ACTUAL latest reply (2026-09-01, Raghu SSA)

When NDR says "send one more follow-up email... if he's not replied, remind him; if he's replied,
let me know" — ALWAYS read the live thread BEFORE drafting:
1. Search by subject keyword + from address (`q='supplementary sharing agreement'`) and read the
   last 2 messages' full bodies. The counterparty may ALREADY have replied ("will review over the
   weekend and revert") — that's a reply but no commitment, so the reminder is still warranted but
   must be re-framed: reference their pending reply ("you mentioned you'd review... just checking
   in"), never draft as if they were silent.
2. Report the reply state to NDR in your response — he explicitly asks to be told when the
   counterparty has replied, even if the reply is only "will revert".
3. Reply to the counterparty's LATEST message id via `draft_reply_create(message_id=...)` so
   In-Reply-To/References thread to their message, not NDR's own send. Set `to=` explicitly.
4. When the same instruction also closes a task and opens a follow-on, run the sheet update FIRST,
   then draft — see personal-task-tracker skill → references/task-close-followon-and-email.md.

### Updating an existing draft's body content

The Gmail API has no direct `draft.update()` endpoint for modifying body text. To update an existing draft, you must **delete and recreate**:

```python
from tools.gws_auth import build_service
import base64

service = build_service('gmail', 'v1', service_name='google-draas')

# Step 1: List drafts to find the correct draft resource ID (≠ message ID)
result = service.users().drafts().list(userId='me').execute()
target_draft = None
for d in result.get('drafts', []):
    msg = service.users().messages().get(userId='me', id=d['message']['id'],
                                          format='metadata',
                                          metadataHeaders=['Subject', 'To']).execute()
    h = {hdr['name']: hdr['value'] for hdr in msg['payload']['headers']}
    if 'target-subject' in h.get('Subject', '') and 'target@email' in h.get('To', ''):
        target_draft = d  # d['id'] is the draft resource ID, d['message']['id'] is the message ID
        break

# Step 2: Delete old draft using draft resource ID (NOT message ID)
service.users().drafts().delete(userId='me', id=target_draft['id']).execute()

# Step 3: Recreate with updated body
message = {
    'raw': base64.urlsafe_b64encode(
        f"Subject: Re: Original Subject\r\n"
        f"To: recipient@example.com\r\n"
        f"MIME-Version: 1.0\r\n"
        f"Content-Type: text/plain; charset=\"UTF-8\"\r\n"
        f"\r\n"
        f"{updated_body_text}".encode('utf-8')
    ).decode('utf-8')
}
created = service.users().drafts().create(userId='me', body={'message': message}).execute()
```

**Pitfall — draft resource ID vs message ID:** When listing drafts from `users().messages().list(q='in:drafts')`, the returned `id` is the **message ID**, not the draft resource ID. Using it in `drafts().delete()` returns `HttpError 404`. Always use `users().drafts().list()` to get the draft resource ID (`d['id']`). The message ID is at `d['message']['id']`.

**Pitfall — `draft_delete` in `gws_skill_bridge`:** The bridge exposes `draft_delete(draft_id=..., service_name=...)` — if using the bridge, pass the draft resource ID, not the message ID.

### Forwarding emails with recipient exclusions

When the user asks to forward an existing email to a subset of the original recipients (e.g. "forward to Aamir and CC Prasanna, but remove Sangam"):

1. **Fetch the original message** with `format='raw'` to get the full MIME content, thread ID, and Message-ID header.

2. **Build the new message** by writing a **fresh script to `/tmp/` and running via terminal** (do NOT use `execute_code` — the sandbox lacks `gws_fetch_token`):
   ```python
   import base64, email.mime.text
   from tools.gws_auth import build_service
   service = build_service('gmail', 'v1', service_name='google-draas')

   # Get original message
   orig = service.users().messages().get(userId='me', id=MESSAGE_ID, format='raw').execute()
   raw_bytes = base64.urlsafe_b64decode(orig['raw'])

   # Build new MIME message
   msg = email.mime.text.MIMEText(new_commentary + forwarded_original, 'plain', 'utf-8')
   msg['To'] = 'Aamir Khan <aamirkhan@icloud.com>'
   msg['Cc'] = 'Prasanna Swaminathan <prasannaswaminathan91@gmail.com>'
   msg['Subject'] = 'Re: ' + original_subject  # keep on same thread
   msg['In-Reply-To'] = original_message_id
   msg['References'] = original_message_id

   draft_body = {
       'message': {
           'raw': base64.urlsafe_b64encode(msg.as_bytes()).decode('ascii'),
           'threadId': original_thread_id
       }
   }
   service.users().drafts().create(userId='me', body=draft_body).execute()
   ```

3. **Key rules for forwarding:**
   - **Use the same thread ID** so Gmail nests it correctly (`threadId` in the draft body)
   - **Set `In-Reply-To` and `References`** to the original Message-ID — both are required for proper threading
   - **Exclude unwanted recipients** by simply not including them in the new message's `To`/`Cc` headers. The forwarded email text will reference their names (that's fine) — what matters is who receives the new message
   - **Write the new message content ABOVE** the forwarded original (not inline reply)
   - **Include the forwarded email as a "---------- Forwarded message ---------" block** in the body text

4. **Verify the draft** — call `service.users().drafts().get(id=draft_id)` and check:
   - `To` and `Cc` headers contain only the intended recipients
   - No excluded recipient email appears in `To` or `Cc`
   - `threadId` matches the original thread
   - `In-Reply-To` is set

5. **Pitfall — forward vs reply semantics:** A forward is NOT a reply-all with recipients removed. The user wants a NEW message on the same thread that introduces the original email to a new audience, with their own commentary. The forwarded content is context, not the primary message body.

### Forwarding an OWN sent email as FYI (new audience, no send to originals)

When the user says "forward the email I sent to X to Y & Z, FYI" — the source is the user's OWN sent message, and the point is to give Y/Z visibility into that decision/transaction, NOT to elicit a reply. Steps (validated 2026-08-20, Red Soul Farmers Collective refund):
1. Locate the original by subject (`q='subject:"..."'`) — report its To/Cc separately from each forwarded copy; users often forward the same mail to multiple people one-by-one, so each forward is its own message with its own Cc.
2. Fetch the original with `format='raw'`, extract its Message-ID + subject.
3. Build a fresh `EmailMessage`: From = ndr@draas.com, To = the FYI audience, Subject = `Fwd: <orig subject>`, In-Reply-To + References = the original's Message-ID. Body = short "FYI — please find below the email I sent regarding …" note, then the full "---------- Forwarded message ---------" block (From/Date/Subject/To/Cc + body).
4. Create as DRAFT only (always), verify recipients + body, then tell the user it's ready in Drafts. Never auto-send.
Tip: users often ask "who was marked on that email?" first, then "forward it to Y & Z FYI" — run the recipient trace before drafting so you can confirm exactly who the FYI gain should copy and that you're forwarding the RIGHT one (original vs a forward).

### Find email with attachment → download → forward to new recipient (2026-08-21)

When the user says "find the latest email from [person] about [topic] — there's a PDF/presentation attached — send it to [someone else]":

1. **Locate the sender by email domain/address** — search with `q='from:arch_arvind2000@yahoo.co.in'` or the sender's domain. If the user says "AJ Architect" or a short name, resolve it via entity_resolver/contact_resolver first (e.g. "AJ Architect" → A.J. Architects / Arvind Jain, arch_arvind2000@yahoo.co.in).
2. **Identify the right thread by topic keywords** — scan subject lines for the topic the user named (e.g. "farm home design" → "Proposed farmhouse presentation"). There may be multiple threads from the same sender; pick the LATEST one that matches the topic.
3. **Check for attachments** — fetch the thread with `format='full'` and walk `payload.parts[].filename` for each message. The attachment may be on a forwarded/resent version (e.g. your Fwd: of their original), not the original sender's message. Check the latest message in the thread first.
4. **Download the attachment** — use `messages().attachments().get(attachmentId=..., messageId=...)`. Verify with a magic-bytes check: `data[:4] == b'%PDF'` (or the appropriate MIME header for the file type).
5. **Forward to new recipient** — either:
   - **Create a fresh email draft** with the PDF as a MIME attachment (via raw Gmail API with `MIMEMultipart("mixed")`), body describing the attachment and asking for feedback. The user's spoken message becomes the email body.
   - **Forward the original email thread** (see "Forwarding emails with recipient exclusions" above) if the user wants the full context.
6. **Key difference from the FYI-forward pattern**: the source is a THIRD-PARTY email (not the user's own sent message), and the user wants to share it with a new audience for feedback/review — not as a FYI broadcast. The body should introduce the scheme/design and ask for their opinion.

### SharePoint/OneDrive link extraction from Gmail

When the user needs a sharing link from a vendor's email (e.g. SharePoint invite from Godrej, Microsoft 365 file share), the link lives in the HTML body, NOT the plain-text part. Navigate the MIME `multipart/alternative` structure, decode the `text/html` part, and regex for `sharepoint.com` hrefs. Full recipe in `references/gmail-sharepoint-link-extraction.md`.

### Read-only email forensics: "who was marked / does it have the attachment?"

Recurring for transaction/refund emails (investor transfers, reimbursements). Two read-only checks via `build_service('gmail','v1',service_name='google-draas')`:
- **Recipient trace**: search by subject, then for EACH message in the result print `From`/`To`/`Cc`/`Date` separately (`format='metadata'`, metadataHeaders=['From','To','Cc','Subject','Date']). Originals and forwards are separate messages — report each. A forward sent to just one person has `Cc: None` even if the original had a Cc list.
- **Attachment presence/type**: walk `payload.parts[].filename` + `.mimeType` + body `.size` to confirm an email has (e.g.) an `.xlsx` with investor details. Do NOT trust the body prose ("please find attached…") — enumerate the real parts. Fetch with `format='full'` (not 'metadata' — metadata drops parts).
The email body usually states what the attachment contains ("amount to be returned to each investor + account number, IFSC, branch") — quote that alongside the filename so the user knows it matches their need before you download/parse it.

**Pitfall — `drafts().get(format='full')` returns NO `raw` key (2026-08-20)**

After creating a draft, `service.users().drafts().get(userId='me', id=..., format='full').execute()` gives you `message.payload` but does NOT include `message['raw']` — trying `base64.urlsafe_b64decode(msg['raw'])` raises `KeyError: 'raw'`. To verify the draft BODY text, read it from the payload with the recursive text walker (the same `extract_text(payload)` you use for messages). Headers are verifiable from `payload.headers`; only the body needs the walker. (If you must decode raw MIME, pass `format='raw'` explicitly to `drafts().get()`.)

**Pitfall — `drafts().create()` may assign the WRONG threadId even with correct In-Reply-To/References (2026-08-20).** When building a threaded reply-all draft (ITR follow-up), I set `In-Reply-To` + `References` to the source message's Message-ID correctly, but Gmail auto-assigned `threadId: 1a01de64f31d5b61` while the source thread was `1a0193d4df586045` — a mismatch that would have sent the reply as a disconnected new thread. Fix: pass the source thread's `threadId` EXPLICITLY in the draft body and verify it on the returned resource:

```python
draft = service.users().drafts().create(userId='me', body={
    'message': {'raw': raw, 'threadId': source_thread_id}
}).execute()
assert draft['message']['threadId'] == source_thread_id
```

Verify after creation with `drafts().get(id=...)` — the returned `message.threadId` must equal the source thread, not a fresh id. If you already created (and confirmed mismatched) a draft, delete it (`drafts().delete`) and recreate with the explicit `threadId`. This is NOT the same as the SENT-label problem above (that's a malformed-References issue); this is purely the thread-join determination.

**Pitfall — user says "the draft isn't part of the thread, make it reply-all in the same thread" (confirmed 2026-08-27, Ranka Aqua Green NCDRC):** A draft created from scratch (plain `draft_create` / `drafts().create()` with no threadId) with subject `Re: <original>` does NOT join the original conversation — Gmail shows it in Drafts as its own separate thread, even though the subject looks like a reply. The user noticed this and asked to keep it in the same thread. Fix (delete + recreate as a THREADED reply):
1. `drafts().delete(userId='me', id=<draft resource id>)` — draft resource id from `drafts().list()`, NOT the message id.
2. Find the LATEST counterparty (non-self) message in the target thread: `threads().get(threadId=..., format='metadata')`, walk `t['messages'][-1]`, check the From header isn't ndr@/nishantranka. Take that message's id.
3. Fetch `messages().get(userId='me', id=<that id>, format='metadata', metadataHeaders=['Message-ID','References'])`; set `In-Reply-To` = its Message-ID, `References` = existing References + Message-ID (dedupe).
4. `drafts().create(userId='me', body={'message': {'raw': raw, 'threadId': <source thread id>}})` — **explicit threadId is mandatory** (without it Gmail may spawn a new thread).
5. Verify: returned `draft['message']['threadId'] == <source thread id>` AND the draft shows under the original thread in Gmail UI. Report the drafts-inbox link. This is the same pattern as "Pitfall — drafts().create() may assign the WRONG threadId", but triggered by the user-visible symptom "this draft is separate from the thread — reply-all it".

### Legal / advocate consultation emails — "in between" structure (NDR preference, validated 2026-08-27)

When NDR asks to email his counsel / advocate about a live matter (call request, strategy discussion, case status), his preferred structure sits between the two failure modes:
- ❌ **Too basic** — "I've reviewed the IA, let's have a call, tell me a time." Reads dismissive; gives the counsel nothing to prepare on.
- ❌ **Too detailed** — a multi-part questionnaire with 10+ legal sub-questions and cited case law in every line. Reads like homework; NDR himself called the earlier 5-part version too much and asked for a replacement.

✅ **The "in between" structure that landed (Ranka Aqua Green IA 9783/2026 call request to Harshavardhan Kotla):**
1. **Open** — one line: reviewed the filing (name the document) + the latest hearing update, want a call to discuss strategy.
2. **"Our analysis of the current position"** — 2–3 short paragraphs: what the Bench observed (quote the operative direction + next date), what the opposing side's new filing appears to be attempting (1–2 sentences), and any estoppel/annexure claims they raised. Plain account of the state of play — this is what tells the counsel you've done your homework.
3. **"Our rebuttal stance / key legal points"** — 3–5 labelled bullets, each one argument with the supporting authority inline (e.g. `a) Section 12(1)(b) — maintainability; Brigade Enterprises v. Anil Kumar Virmani (2021)`). One line per argument; no mini-essays. Correct the user's own stated timeline if wrong (user said "filed 10 years after handover" — the 10 years was case PENDENCY 2016–2026, the OC-to-complaint gap was ~2 years; state the corrected figure in the email so the counsel works from accurate facts).
4. **"Key information we need from you on the call"** — 5–6 numbered concrete questions, each one sentence: strength of our primary argument at the next date, how to counter their estoppel argument, whether res judicata from individual cases is pleadable, probability/advisory opinion, settlement consideration given pendency, strategic-delay advice while the opposing side decides whether individuals step forward.
5. **Close** — document filed on Drive for reference (one line), ask for a convenient time this week. No boilerplate pleasantries beyond the initial "I hope this email finds you well."

Recipients: derive To/Cc from the latest thread message (reply-all; NDR may have added family members — e.g. Manish/MDR, Dharmesh — into the Cc; preserve whatever recipients are already on the draft/thread).
