# WhatsApp / SMS deep-link generation (absorbed)

> **⚠️ This file is at the size cap.** New content on invoking the sanctioned tool when
> `whatsapp_link` is absent from the live tool list lives in
> `references/whatsapp-link-tool-module-from-execute-code.md` (verified 2026-09-23).

> From the `messaging-drafts` → `references/messaging-links-workflow.md` (archived 2026-09-20). Its references and templates were
> re-homed into this skill's `references/` and `templates/` directories.

name: messaging-links
description: |
  Generate WhatsApp deep links (api.whatsapp.com/send — never wa.me) and similar pre-filled messaging links.

  ## Formatting — Nishant's preference

  When generating a WhatsApp message for Nishant use bullet points (•) for lists, break into short sections, put each thought on its own line, address people by name, use bold for key items, and end with a clear call-to-action. Never one wall-of-text paragraph.
  (sms:, mailto:) for NDR. WhatsApp is the dominant channel for Indian real-estate
  follow-ups, and WhatsApp deep links are how Hermes "sends" WhatsApp messages — Hermes
  cannot programmatically post to WhatsApp, so it produces a link the user taps
  to open WhatsApp with the message pre-filled. Covers individual contacts and
  group messages (no phone, just text). Trigger: "WhatsApp [name]", "message
  [name] on WhatsApp", "WhatsApp the group". Also covers sms: and mailto:.
  Named-group broadcasts: see references/group-broadcast-to-named-group.md.
metadata:
  hermes:
    tags: [messaging, whatsapp, wa-me, communication, outreach, follow-up]
category: communication
version: 1.0.0
author: ndr@draas.com
---

# Messaging Links (WhatsApp / SMS / mailto)

## CRITICAL: How to call the tool

**Caption rule for colleague messages (NDR rule):** WhatsApp messages to DRA colleagues
(Anbu, Rahul, Sinchana, Prakash, Sarthak, Bharat…) ALWAYS open with a bold caption
`*Project — Subject*` (e.g. `*Ranka Udaya — IRR Model: Build on Actual Tally Costs &
Timed Cash Flows*`). NEVER add a caption for vendors / third parties unless explicitly
asked. Log the caption + recipient in the task tracker if the message closes a task.

There is no manual `wa.me` URL construction. There is one and only one sanctioned
way to produce a WhatsApp link, and that is the `whatsapp_link` Hermes tool. The
tool has **no safe ampersand handling** — it converts raw `&` characters in the
source text to a **fullwidth ampersand** (U+FF06, URL-encoded as `%EF%BC%86`).
This produces `R＆D` instead of `R&D` in the WhatsApp compose box, which reads
as a broken character on some mobile WhatsApp clients. **The rule is absolute:
rewrite EVERY `&` to "and", "plus", "and/or", or an em-dash in the source text
BEFORE passing it to the tool.** Do not assume the tool's handling is safe — it
is not in this case. Treat the fullwidth conversion as data corruption that you
must prevent at the source level.

ALWAYS call `whatsapp_link` — never hand-construct wa.me URLs in
`execute_code`/`urllib.parse`/string concatenation. Manual encoding has been
observed to break on mobile WhatsApp clients.

### Terminal fallback when tool isn't registered

The `whatsapp_link` tool belongs to the **messaging** toolset, which may not be
loaded in every session (especially Telegram gateway sessions where the toolset
is restricted). If the tool isn't in your available tools list, do NOT
hand-encode the URL. Instead, call the tool function directly via the Hermes
venv:

```python
# From terminal() or execute_code:
result = terminal(""cd /opt/hermes && .venv/bin/python3 -c '
import sys, json
sys.path.insert(0, \".\")
from tools.whatsapp_link_tool import whatsapp_link_tool

args = {
    \"phone\": \"+919999673483\",
    \"text\": \"Your message here — no raw & allowed\",
    \"platform\": \"telegram\"
}
result = json.loads(whatsapp_link_tool(args))
print(\"URL:\", result[\"url\"])
print(\"Display:\", result.get(\"display_link\", \"\"))
'"", timeout=15)
```

The handler takes an `args` dict (phone, text, platform) and returns a JSON
string with `url` and optionally `display_link` (for Telegram). This is the
**only** acceptable fallback — it uses the same encoding logic as the
registered tool, so the ampersand workaround and all other encoding rules are
preserved. Never bypass to urllib.parse or string concat.

## 1. Trigger Conditions

Activate when the user says anything like:
- "WhatsApp Sunny Sadhwani — [content]"
- "Send a message to [name] on WhatsApp"
- "WhatsApp the group with [names]"
- "Draft a WhatsApp message for [recipient(s)] about [topic]"
- "Open a chat for [name] with this message"
- "SMS [name] [content]" (use sms: fallback)
- "Send a text to [number] saying [content]"
- "Give all the landowners an update" / "update the landowner group" / "message
  the Northstar group" → landowner-group broadcast: load
  `references/landowner-group-milestone-update.md` and read any uploaded
  WhatsApp export before drafting

**Channel default for personal follow-ups / apologies to third parties (corrected 2026-09-05, Pankaj Kumar / Blue Hat Solutions):** when NDR dictates a personal apology / catch-up / re-engage message ("write back to him with sincere apologies…") and does NOT name a channel, the default is the WhatsApp link, NOT an email draft — he corrected an email draft with "Not email I need a WhatsApp message." Email stays the default only for formal content (documents, terms, governance, project briefs to professionals). For "when did I last meet X" context: the calendar lookup (`calendar.events().list(q='<name>', timeMin/timeMax)` across the relevant months) gives the last meeting date to reference as the opening anchor ("since our meeting in June") plus the venue — use it to ground the apology.

## 2. Stage 1 — Resolve the recipient

### For an individual contact

Use Google Contacts lookup — but search **ALL three vault accounts**, not just
google-draas. Government / phone-added / spouse contacts can live in
`google-ahfl` (confirmed 2026-08-14: **Nagarajappa JDTP North exists ONLY in
google-ahfl**, and Lokesh Gandhi / Nachiketh Gowda in google-draas). The People API path is
`people.people().searchContacts(query=..., readMask=...)` — never a flat
`connections().list()` (see MEMORY).

```python
from tools.gws_auth import build_service
service = build_service('people', 'v1', service_name='google-draas')
results = service.people().searchContacts(
    query='Sadhwani',
    readMask='names,emailAddresses,phoneNumbers,organizations,biographies,nicknames'
).execute()
```

When a name has aliases (e.g. "Sunny Sadhwani, also known as Rajesh Sadhwani"),
the search returns one contact with both names — treat them as the same person,
do not ask the user to disambiguate. Present the canonical display name + the
email + the E.164 phone.

### Fallback: Contact not in Google Contacts — check the NDR DRAAS Contact Sheet

For NDR, the **NDR DRAAS Contact Sheet** (Google Sheets ID `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`)
often has phone numbers & labels that aren't returned by the People API
search (contacts may be in the sheet but not synced to Google Contacts, or
stored with different spellings). If `searchContacts` returns zero results,
**always query this sheet before falling through to Gmail extraction**.

**Important: contact found in sheet but no phone number.** The sheet may
contain a row for the person with only name + email and no phone columns
filled. In that case, do NOT fall through to Gmail extraction yet — first
check honcho memory for a phone number (see below). If honcho also has no
phone, inform the user the contact exists in the sheet (name + email) but
you need their phone number to generate the WhatsApp link. Show the draft
message text so they can copy-paste it, or send it once they provide the
number.

Sheet structure (header row 1):
- Col A: First Name, Col B: Middle Name, Col C: Last Name
- Col Q (index 16): Labels (e.g. "FrndA ::: * myContacts")
- Col AC (index 28): Phone 1 - Value (the phone number)
- Col AB (index 27): Phone 1 - Label (e.g. "Mobile")
- Email columns start at Col R (index 17)

NDR contacts are stored in both Google Contacts and this sheet (always in
tandem). The sheet was the primary storage before Google Contacts sync was
added, so it often holds contacts that predate the sync or were entered
manually by NDR's team.

```python
from tools.gws_auth import build_service
sheets = build_service('sheets', 'v4', service_name='google-draas')
# Use a broad search: scan all rows checking First/Last name match
all_data = sheets.spreadsheets().values().get(
    spreadsheetId='1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g',
    range='A:DA'
).execute()
rows = all_data.get('values', [])
headers = rows[0] if rows else []
for i, row in enumerate(rows[1:], 2):
    first = (row[0] if len(row) > 0 else '').lower().strip()
    last = (row[2] if len(row) > 2 else '').lower().strip()
    if search_term.lower() in first or search_term.lower() in last:
        phone_idx = 28  # Phone 1 - Value column
        label_idx = 16  # Labels column
        phone = row[phone_idx] if len(row) > phone_idx else ''
        labels = row[label_idx] if len(row) > label_idx else ''
        print(f'Sheet row {i}: {row[0]} {row[2] or ""} | Phone: {phone} | Labels: {labels}')
```

**Important**: The sheet is 4000+ rows and the full read is ~2 seconds.
Read the whole column range once and iterate in Python — do NOT make
multiple paginated calls. The range `A:DA` covers all known columns.

**Pitfall**: A name can appear in Google Contacts but not in the sheet,
or vice versa. Always check BOTH sources. When the sheet has the number
but People API didn't return it, use the sheet's number — it's the one
NDR's team entered directly.

### Supplementary check: Honcho memory — before Gmail extraction

After querying both Google Contacts and the NDR DRAAS Contact Sheet (and
before falling through to Gmail signature extraction), always run
`honcho_search` for the person's name. Honcho memory captures:

- **User-confirmed phone numbers** from prior conversations — e.g. Nishant
  may have said "Anbu's DRA number is +918150029900" in a past session,
  which honcho saves but never makes it into Google Contacts or the sheet.
- **Context about who the person is** — their role, project involvement,
  email address, which helps confirm you have the right person when the
  sheet shows multiple matches.
- **Email addresses and alternate spellings** that help refine what you
  search for next.

```python
results = honcho_search(query="Anbarasu Anbu DRAAS phone")
```

**When to skip Gmail entirely:** If honcho returns a phone number that was
explicitly stated and confirmed by the user in a prior conversation, use
it directly. Honcho-confirmed numbers carry the user's own verification —
they are more reliable than a Gmail signature extraction.

**Pattern from this session:** For Anbu/Anbarasu (DRAAS employee involved
in RERA), Google Contacts returned only a "Guru Tiles (Anbu)" entry with
a different number. The Contact Sheet had "Anbarasan" at row 250 but with
a murky phone. Honcho memory returned the exact confirmed DRA number
(+918150029900) that Nishant himself stated in a prior session: *"Anbarasan
(Anbu) has username pm2.blr, phone +918150029900, and is WhatsApp-only for
site."* This saved a Gmail search and provided the correct number directly.

### Fallback: Contact not in Google Contacts or Sheet — extract from Gmail signature

If **both** Google Contacts `searchContacts` and the NDR DRAAS Contact Sheet
return zero results for the person's name (including phonetic variants),
the person may still be in the user's email history with a phone number in
their email signature. Use the Gmail-signature extraction technique:

1. Search Gmail for the person's name (and phonetic/domain variants) via
   `gws_skill_bridge.call('gmail_search', query=..., max=5)`
2. Find an email **FROM** the person (check the `from` field in results)
3. Use `gws_skill_bridge.call('gmail_get', message_id=..., format='full')`
   to read the full body — the signature block usually sits at the bottom
4. Scan for phone numbers (lines starting with `+`) labelled "WhatsApp"
   or "Mobile"
5. Pass the WhatsApp-specific number to `whatsapp_link()`

**Parameter gotcha:** `gmail_get` takes `message_id=`, not `id=`. See
`references/contact-from-gmail-signature.md` for the full procedure,
phone-number selection rules, and all pitfalls.

**Return-shape gotcha:** `gws_skill_bridge.call('gmail_search', ...)` returns a
JSON **string**, not a list — `json.loads()` it before iterating (observed
2026-09-07 while resolving "Dr Nishant Hiramath"; a direct `.get()` on the
string raises `'str' object has no attribute 'get'`).

For phone numbers, prefer `phoneNumbers[].canonicalForm` (already E.164) over
the raw value. The wa.me path needs the country code with no `+`, no spaces, no
dashes (the `whatsapp_link` tool strips non-digits for you, but if you read
the canonical form directly it is cleanest to pass).

### For a group

The user picks the group in WhatsApp after tapping the link. Pass an **empty
phone** to the tool so the URL has no `phone=` parameter. The text is the
message; the group is selected by the user at compose time.

```python
args = {"phone": "", "text": group_message, "platform": "telegram"}
result = json.loads(whatsapp_link_tool(args))
# → url starts with https://api.whatsapp.com/send?text=...
```

### For a cold / forwarded incoming message — generate WITHOUT a phone number (NDR rule, 2026-08-30)

When the user forwards an incoming message (cold approach, referrer-shared contact,
someone not in his contacts) and asks for a reply, the default is
`whatsapp_link(text=...)` with **NO phone** — even when the sender's number is
visible in the forwarded message. NDR verbatim: *"generate the message without the
telephone number, I'll respond to the correct sender."* He picks the recipient
himself in WhatsApp; the number in the forward is not always the chat he wants to
reply from.

Workflow that worked (Harshvadan Guruji cold-approach reply):

1. **Resolution first** — check the sender against ALL sources (contacts sheet +
   People API across all three vault accounts + Gmail) so your reply opens with an
   honest "not in contacts anywhere" note. Do NOT add the sender to contacts or
   save the number unless NDR asks.
2. **Draft the reply** (Pattern B — response to a third-party message; expect
   2–3 refinement rounds).
3. **Present the draft for confirmation BEFORE generating the link**
   (confirm-before-actions rule) — NDR refined the text twice before greenlighting.
4. **After approval**, call `whatsapp_link` with text ONLY — no `phone` —
   `platform='telegram'`, deliver the `display_link`. The URL has no `phone=`
   param (same shape as a group-paste link). Tell the user: "WhatsApp opens —
   pick the sender's chat and send."
5. When the user asks for the link "without the number", do NOT include the
   sender's phone in the tool call, in prose, or anywhere in the reply.

**Polite meeting-request wording (NDR correction, 2026-08-30):** for replies to
respectful/spiritual contacts (Guruji, religious figures), NDR edited
"I would like to meet you at your convenience accordingly" to
"I would be happy to meet you at your convenience accordingly". Use the
happy-to phrasing for any polite meeting-request reply to a senior or spiritual
figure — the like-to version reads as colder and gets corrected.

## 3. Stage 2 — Draft the message (no raw `&`)

Voice messages from the user are messy. The job here is to extract the
intent, organize it into numbered points or natural paragraphs, and rewrite
all `&` characters to a safe alternative.

### User-provided evidence URLs go INSIDE the message body (corrected Sep 2026)

When the user shares a URL (YouTube short, image, Drive link, etc.) as evidence
to share with the recipient — e.g. a video of an AC water leak, a screenshot of
an error, a photo of damage — embed that URL **inside the WhatsApp message text
itself**, not alongside it as separate context.

Common failure: the agent puts the evidence URL in its reply prose ("I'm also
sharing this video link…") but does NOT include it in the message text passed to
`whatsapp_link`. The recipient never sees it in the pre-filled compose box. The
user then corrects: "you haven't given them the link to the video."

Rules:
- The user-provided URL is part of the message content, not metadata about it.
- Pass the URL as plain text inside the `text` parameter to `whatsapp_link`. The
  tool percent-encodes it correctly. Example: "I have recorded a video, please
  watch it: https://youtube.com/shorts/abc123".
- Do NOT separate the evidence URL from the message prose in the `text` argument.
  The entire communication to the recipient goes in one `text` parameter.
- This is distinct from the P9 rule (voice mentions "the URL I'm providing below"
  but it never arrives in the transcript). If the URL is genuinely present in the
  user's message (pasted link, link in voice transcription), embed it. If the user
  says "I'm also sharing the URL" but no link arrives, use P9's placeholder pattern.

### Hinglish / Roman-script Hindi messages for trade/service persons (Sep 2026)

When the user explicitly asks for a message in "Hindi English" / "Hindi English
using English font characters only" / "Hinglish" — meaning Roman-script Hindi
with English words mixed in — this is targeting a recipient who is not fluent
in English (AC technician, plumber, electrician, driver, domestic help).

Rules:
- Write the message in Roman-script Hindi with English terms embedded where the
  recipient uses them (e.g. "cooling", "AC", "water leakage", "curtains",
  "floor", "urgently"). Use these English terms as-is, not translated to Hindi.
- Keep sentence structure Hinglish: Hindi word order + English loanwords.
- Keep the message respectful but warm, not overly formal. *"Sir, please
  urgently come… Bahut zaroori hai."* is the right register.
- Use *aap* (formal "you") — implied through respectful phrasing, not written
  out explicitly unless the user does so.
- Output styling: all in Latin script, no Devanagari, no transliteration marks.
- The `whatsapp_link` tool handles percent-encoding of these mixed-language
  messages correctly — no special pre-processing needed.
- Do NOT translate the user's original dictation into pure Hindi unless they
  explicitly ask for it. The Hinglish register is a deliberate choice.
- These messages also typically involve **evidence URLs** (see above) — a video
  of the problem, a photo. Include the URL inside the message text as plain text.

### Voice → text cleanup checklist

- Replace `&` with one of: `and`, `plus`, `with`, `and/or`, em-dash (`—`).
  Examples: "Premium FAR & TDR" → "Premium FAR plus TDR"; "Amit & Saurabh"
  → "Amit and Saurabh"; "FAR 4.0 & 4.5" → "FAR 4.0 and 4.5".
- Replace `Rs.` / `INR` with `Rs` (no trailing period) for encoding safety.
- Keep currency as `Rs XX Cr` (the rupee symbol `₹` is fine in source — the
  tool URL-encodes it correctly — but `Rs` is universally readable on
  mobile).
- Preserve the user's voice — don't add boilerplate greetings, don't strip
  filler, don't "AI-ify" the phrasing.
- **Technical terms the recipient will ACT on must be corrected, not
  reproduced** — when the user misnames a technical term and the recipient
  needs the right term to act safely (observed 2026-08-14: NDR said "HIPAA
  filters" meaning **HEPA**; the air-purifier cleaning message must say HEPA
  or the cleaning instructions are meaningless), write the CORRECT term in
  the message and tell NDR you corrected it. Reproducing the misnomer costs
  a correction round; the recipient acts on the wrong word.
- **OC workflow terms (corrected 2026-09-07):** For Occupancy Certificate follow-ups with BBMP officials, the user may dictate "AWE" — the correct term is **AEE** (Assistant Executive Engineer). The online BBMP OC portal workflow: apply online → file moves to the AEE in the console → AEE does inspection/verification → recommendation to JDTP for OC issuance. Always correct AWE → AEE before generating the message.
- Address each named recipient in a separate short paragraph (the user
  will paste into a group, so named addressing is helpful for context).
- For voice messages containing multiple distinct topics (the typical
  NDR follow-up pattern), use numbered points (1), 2), 3)) — the user
  mentally tracks items by number.

### NDR-specific tone patterns

- **Festival / seasonal greeting FIRST in business WhatsApp messages (2026-09-14, Bharat Babu reschedule):** around a festival (Ganesh Chaturthi, Diwali, etc.), open the message with the greeting before any business content — user's own words: "first wish for Happy Ganesha, whatever the normal greetings, and then we get into this issue." Shape: greeting line (name + 🙏) → business ask (state what has been done: "I have rescheduled it to 6 PM") → reason → polite close ("Hope that works for you"). Applies to vendors AND contacts alike.
- **Broker / agent / chief style**: open with the nickname or
  "Chief" / "Boss" if the user used it; numbered items; "Awaiting your
  update" / "Please advise" close; no greeting.
- **Godrej Venture / institutional style**: open with "Good morning
  [name], [name], [name]" (voice-message form), then numbered items
  with sub-points; close with "Awaiting your update and feedback".
- **Personal / family**: warmer, no subject prefix, no numbering.
- **Senior government official / Joint Commissioner**: formal and respectful. Open with "Respected Sir," (or "Respected Madam,"). Use full honorifics and deferential phrasing ("Request you to kindly...", "What do you advise, sir?"). Close with "Regards," and full name. Structure the message to acknowledge the official's position and the relationship history before making the ask. The goal is to make the recipient feel inclined to help — not pressured. Use phrases like "under your advice and guidance," "as assured by you," "I hope you understand our position." Tone: cooperative, not demanding; the passage of time is stated as fact, not as accusation. Sign off with company name as "DRA Group" not "DRAAS".
- **Personal doctor / known medical professional** (user-corrected Aug 2026): warm and personal before stating the medical context. Open with a greeting — "Hey doctor, how is it going?" — not straight into business. Reference the prior patient relationship as shared context: "As you know, you have seen [child] before." State the known medical pattern factually (chronic condition, viral → asthmatic flare-up). Explain why the certificate/document is needed — external pressure (school SOP, board requirements), not your preference. Make the specific ask with bullet points. Add urgency if relevant (exam tomorrow). Offer to draft it for them with a 😉 to keep it light and collaborative. Close warmly: "Thank you as always, doctor!" No numbered items, no competitive pressure. Structure: greeting → context → situation → ask with bullets → offer to help → warm close.
- **New contact introduction** (after meeting someone for the first time): direct but warm — open with "Pleasure meeting you." Then share Nishant's contact details in plain structure: Name, Role, Phone, Email. Close with a specific reference to the project discussed (shows you were listening). No numbered items, no greeting boilerplate. Example:

```
Hello [Name],

Pleasure meeting you. Here are my contact details:

Nishant Ranka
CEO — DRA Group

Mobile: +91 98800 55634
Email: ndr@draas.com

Looking forward to working together on the [Project Name].

Thanks,
Nishant
```

The WhatsApp link is generated using the contact's phone number (just looked up from the new contact record). This pattern always follows visiting card processing (see `google-workspace` skill: Visiting Card Import Workflow).

### Message layout preference (USER-CORRECTED)

The user explicitly prefers WhatsApp messages formatted with **visual structure**:
bullet lists for enumerations, line breaks between distinct sections, and clear
separation of topics. A wall-of-text single-paragraph message will be corrected.

**Formatting rules (in priority order):**

1. **Bullet any list of 2+ items** — email addresses, file types, action items,
   people names. Each bullet on its own line.
2. **Blank-line separate each logical section** — don't cram multiple topics into
   one paragraph. A message about "files shared + close out DD + originals pickup"
   should have 3 sections separated by blank lines, not one run-on sentence.
3. **Lead with the key action** in the first line, not a preamble. The user
   reads the first line to decide urgency.
4. **Label each bulleted item** with a short descriptor in parentheses when the
   raw value needs context (e.g. email addresses → "(you)" / "(Advocate Vinod)"),
   so the recipient instantly sees who each line refers to.
5. **Keep the link title concise** — the Telegram display text should be enough
   to convey the message. The detailed formatting (bullets, line breaks, sections)
   matters in the WhatsApp compose box after the user taps the link, not in the
   Telegram link title itself. The `display_link` from the tool already shows the
   full text, so structure it there.

**Worked example of the preferred layout** (from user's explicit correction):

```
Nishant Prakash — I've shared 1-month viewer access to all the Serenity Hillview
legal DD files (43 files) to the following 4 email addresses:

• nishantprakash@theyelloweye.com (you)
• nishantprakash@me.com (you)
• vinod@advocatev.in (Advocate Vinod)
• hemanthpanagar07@gmail.com (Advocate Vinod)

Request you to please close out the entire legal DD today. Everything has been
given and clarified in the email to Vinod as well.

As far as original scrutiny is concerned — I've requested you and Manohar to
follow up with Vikram and pick up all originals ASAP.
```

**When NOT to use this layout:** single-sentence voice-message summaries for
recipients who prefer brevity (e.g. "Chief — no update from Bhuvanesh yet,
following up again"). The structure rule applies to *informational* and
*action-list* messages, not to one-liners.

## 4. Stage 3 — Call the tool and report

### Individual contact

When the tool IS loaded in your toolsets, call it by name. When it is NOT
(terminal fallback — see the Terminal fallback section above):

```python
import sys, json
sys.path.insert(0, "/opt/hermes")
from tools.whatsapp_link_tool import whatsapp_link_tool

args = {
    "phone": "+919999673483",       # E.164 with or without +
    "text": cleaned_message,        # NO raw `&`
    "platform": "telegram"          # optional, for MarkdownV2-safe output
}
result = json.loads(whatsapp_link_tool(args))
url = result["url"]
display_link = result.get("display_link", "")
```

Present to the user as a markdown link: `[Open WhatsApp to Sunny](url)`.
Also include the raw URL in parentheses for users who want to copy it.

### Group

Same reporting pattern, but the link text should make clear that the
**user needs to select the group after tapping**:
> [Open WhatsApp with this message pre-filled — pick the group to send it to](url)

The handler call is identical to the individual contact case — simply omit
the phone number:

```python
args = {"phone": "", "text": cleaned_message, "platform": "telegram"}
result = json.loads(whatsapp_link_tool(args))
```

## 4a. Batch multi-recipient runs — one link per person (NDR pattern, Aug 2026)

When the user asks for the same message to N people ("a separate WhatsApp message for each one so I can send it to each one by clicking"):

1. **Resolve all contacts in ONE script.** Query People API `searchContacts` across BOTH `google-draas` and `google-ahfl` for every name in one pass, print name + phones. This beats running `find_contact.py` N times. (Contacts like "Nagarajappa Jdtp North" live only in google-ahfl; employees in google-draas.)
2. **Two numbers on one contact:** use the first/primary listed, then list ALL used numbers in the final summary reply and flag the alternates so the user can correct (e.g. "Aravind had two numbers — I used X, say the word for the alternate"). Do not silently pick.
3. **Generate ALL links in ONE terminal script** calling `tools.whatsapp_link_tool` directly (the sanctioned fallback — same encoding logic), looping over `(name, phone)` and printing `### Name | url`. This keeps 14 × ~4KB URLs out of context; a parallel 14-call tool-invoke would flood it. Handle `"split": true` per link if it appears.
4. **Deliver each link as its OWN Telegram message** via `send_message` — one bubble per person with a short label: `[Open WhatsApp — <Name> (voice recording)](url)`. NEVER combine N links in a single reply: N × ~4KB URLs blow past Telegram's 4096-char cap and links get truncated. This extends the P5 split rule to multi-link runs (verified Aug 2026: 14-link broadcast delivered as 14 bubbles, all intact).
5. Same message body for all, personalized salutation (first name only). Use the appropriate tone pattern — for internal urging, Pattern D (cooperative-but-firm) from `messaging-drafts` → `references/personal-messaging-workflow.md`.

## 4b. QR Code generation from WhatsApp link

When the user asks for a **QR code image** containing a WhatsApp link (e.g. "generate a QR code for my WhatsApp number with a message"), the workflow is:

1. **Compose the message** following all rules in Stage 2 (no raw `&`, bullet structure, etc.)
2. **Call `whatsapp_link` first** to get the safe, fully-encoded URL — never hand-construct it, even though you're making a QR code, not a clickable link. The `whatsapp_link` tool's encoding is the authoritative source.
3. **Extract the URL** from the tool's JSON result (`result["url"]`)
4. **Generate the QR code** using `qrcode[pil]`:

```python
import qrcode

# URL from whatsapp_link tool
url = result["url"]  # e.g. "https://api.whatsapp.com/send?phone=919880055634&text=..."

qr = qrcode.QRCode(
    version=None,
    error_correction=qrcode.constants.ERROR_CORRECT_H,  # H = ~30% recovery
    box_size=10,
    border=4
)
qr.add_data(url)
qr.make(fit=True)
img = qr.make_image(fill_color='black', back_color='white')
img.save("/path/to/output.png")
```

5. **Deliver** the image via `MEDIA:/path/to/file.png` in your response. The user can save the image from Telegram to their phone and share it.

**Context:** The user specifically asked for "a QR code using a QR code generating Python library where it's a WhatsApp link" — this is a distinct use case from just sharing a clickable wa.me link. The QR code is meant to be printed, displayed on marketing material, or shared as an image that anyone can scan to open WhatsApp.

### Install the library

```bash
uv pip install qrcode[pil]
```

`qrcode[pil]` includes PIL/Pillow for image rendering. Install in the Hermes environment.

### DELIVERY: Must include message text (USER-CORRECTED)

**Always include a pre-filled message in the QR code's WhatsApp link.** A QR code with only a phone number (no `?text=` parameter) is useless — when scanned, it opens a blank chat with no context, and the user will ask you to regenerate it with the message. This was corrected in session.

The one exception is when the user explicitly says "just the number, no message" — and even then, confirm they really mean it, because they often change their mind once they see the bare QR.

### Pitfalls

- **Install issue:** `uv run python3` may fail to build hermes-agent itself. Use `python3` directly with the venv path added to `sys.path`:
  ```python
  import sys
  sys.path.insert(0, '/opt/hermes/.venv/lib/python3.13/site-packages')
  import qrcode
  ```
- **Platform-specific:** The generated image is a standard PNG — works on all platforms Telegram delivers to (iOS, Android, Desktop).

## 4c. Word-document delivery — batch per-recipient links in a .docx (Bharat pattern, Aug 2026)

When the user wants MANY DIFFERENT messages (one per lead/customer, each drafted from its own context) compiled into a single shareable file — "put it in a Word document instead of Excel, I'll click each link" — build a `.docx` with clickable hyperlinks:

- Draft each message from the source record (e.g. the Kelsa lead's last note), one entry per recipient.
- **Address sales/customer messages by honorific, never first name** (Bharat's rule: ladies → "Ma'am", gents → "Sir"). Flag gender guesses (ambiguous names) in the chat reply.
- Generate every link in ONE Python script with the sanctioned terminal fallback (`tools.whatsapp_link_tool`, see the top of this skill) — loop over (name, phone, message), keep the bulky URLs out of context.
- Embed as real hyperlinks: python-docx has NO native hyperlink API — build `w:hyperlink` with a rels relationship (working helper in `templates/whatsapp-links-docx.py`).
- Install python-docx into the Hermes venv when missing: `uv pip install --python /opt/hermes/.venv/bin/python3 python-docx` (the venv often has no pip module; uv is the reliable path).
- Verify before delivery: reopen the doc, count `w:hyperlink` elements, assert every `target_ref` starts with `https://api.whatsapp.com/send?` — 0 hand-built URLs.
- Deliver the file with `MEDIA:</path>` in the reply; tell the user to tap the links on their phone → WhatsApp opens with that number and the text pre-filled.

Reference implementation: `templates/whatsapp-links-docx.py` (copy, fill LEADS, run).

## 4d. Excel (.xlsx) batch links — one workbook, clickable per-row links (Samvatsari pattern, Sep 2026)

When the user asks for a **festival/forgiveness broadcast** ("create a sheet where every contact is listed with their telephone number and a link with the message encoded for each") — same message to every Jain friend/community contact, compiled into an Excel he clicks through — build a 3-tab .xlsx instead of N individual bubbles:

1. **Message** = the user's exact dictated text (WhatsApp markdown fine; verify no raw `&` before link generation).
2. **Contacts mining** — read the FULL contacts sheet in one pass (ID `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`, tab `'NDR DRAAS Google contacts.csv'`, range A1:FX) and surname-match against the Jain-community surname list (jain, shah, mehta, bafna/pafna, kothari, surana, bhandari, oswal, mardia/maridya, rathod, ranka, murjani, doshi, sanghvi, sethia, sisodia, nahar, gala, sancheti, chordia, babel, bhansali, soni, …). Result: ~299 surname hits, ~289 with phones.
3. **Recency filter** — for each candidate with an email, Gmail last-touch query `(from:<em> OR to:<em> OR cc:<em>) after:2025/08/01` tracking latest internalDate; this separates "recently in touch" (~26) from the full list. **Runs on the OWNER's mailbox — see the identity pitfall below.**
4. **Explicit-name expansion** — resolve every dictated name via People API + Gmail + sheet scans (voice variants live in `references/festival-greeting-broadcast-excel.md`). Ambiguous/unresolved names go in flagged rows, never dropped silently.
5. **Relative exclusion** — drop rows whose Labels carry RelA/RelB/family UNLESS NDR explicitly named them (he posts the cousins group himself).
6. **Link generation** — one script loops `tools.whatsapp_link_tool` over (name, phone, same message); keep the bulky URLs out of context.
7. **xlsx build** — tabs: Recommended (Name, Phone, Alt Phones, Email, Why Included, Last Email, WhatsApp Link), All <Surname> Contacts (backup with links), Message (raw text). Hyperlink cells = `cell.hyperlink = url` + blue underline font. openpyxl gotcha: `ws.append({dict})` raises `ValueError: 'name' is not a valid column name` — append header-ordered LISTS.
8. **Verify before delivery** — reload the workbook, iterate the link column, assert every hyperlink target starts with `https://api.whatsapp.com/send?` and contains `phone=` and `&text=` (343/343 passed in this run). Save to `/opt/data/` and deliver with `MEDIA:<path>`.
9. **Report** — counts per tab, every unresolved name called out explicitly, multi-number flags (e.g. Charitra had old + current mobiles — used current, listed old in Alt Phones).

### Identity pitfall — recency analysis must run on the OWNER's mailbox

A Gmail/Sheets token resolved via `build_service(...)` is the **session user's**, not necessarily NDR's. In a colleague's Telegram session (e.g. Prakash Singh, 8502281203), `service_name='google-draas'` resolves to `psingh@draas.com` — recency results come back near-empty and the sheet read still succeeds (shared sheet), silently masking the mismatch. Fix: set `HERMES_SESSION_USER_ID=ndr` in the terminal/script env for any NDR-contact work AND verify identity with `gmail.users().getProfile()['emailAddress'] == ndr@draas.com` before trusting the analysis. (`HERMES_SESSION_USER_ID` mapping: ndr=7449813913, Prakash Singh=8502281203.)

## 5. Pitfalls

### P0b. DELIVER THE LINK ITSELF — never just the message prose (USER-CORRECTED)

The `whatsapp_link` tool returns the URL in `url` (raw) and `display_link`
(markdown-link form for Telegram). The deliverable is **the link**, not a
restatement of the message text. A reply that only shows the formatted message
prose and says "the WhatsApp message is ready" forces the user to ask again for
the actual link. Corrected Aug 2026 — user: *"Somehow, the message for Rahul got
truncated. Also, I need the WhatsApp link... redo the entire message as multiple
links."*

Delivery rules:
0. **A summary of the message contents is NOT delivery — the tappable link MUST be in the reply (corrected 2026-09-02).** After calling `whatsapp_link`, a reply that only says "Ready — [Name]'s link delivered ☝️" and then bullet-points what the message says is a wasted turn: the user must reply "I can't see the WhatsApp message link for [Name]" and you regenerate. The tool call result is the deliverable, not a description of it. If the reply contains no `[text](url)` markdown link (or split parts), it has failed regardless of how much context was included.
1. **NEVER emit a literal placeholder instead of the link (USER-CORRECTED 2026-09-02).** The same "I can't see the WhatsApp link" complaint arises when the reply contains the literal token `{display_link}` / `{url}` / `[link]` — i.e. the assistant wrote the variable name instead of pasting the tool's actual returned value. This happens after LLM compaction drops the real link from context: the model fabricates the placeholder to keep the turn moving. Rule: if the actual `url`/`display_link` string is not present, DO NOT reply with a placeholder — re-call `whatsapp_link` (same text + phone) to regenerate the link, THEN deliver. A visible placeholder is indistinguishable from a broken link to the user and costs a full correction round.
2. When the tool returns a single `display_link`, send it as an inline
   `[text](url)` markdown link in your reply — the user taps it to open
   WhatsApp. Do not paraphrase it away.
3. **Long messages: use a SHORT label, not the full-text display_link (USER-CORRECTED Aug 2026).** For a message of ~500+ chars, the tool's `display_link` labels the link with the ENTIRE message text (MarkdownV2-escaped). Pasted verbatim, Telegram renders the escaped label literally (`\*`, `\.` visible) and it does not read as a tappable link — user: "It's not generated properly as a link. Using the link tool, please." Fix: keep the tool-generated URL but wrap it in a concise label, e.g. `[Open WhatsApp — DRA Realty Group (Robocalling message)](<url>)`. The full text still arrives in the WhatsApp compose box; only the Telegram link title is shortened.
4. When `split: true` is returned, the `parts` array holds one complete
   tappable link per part — deliver EACH part as its own separate Telegram
   message (never combine them into one message; Telegram's 4096-char splitter
   will cut the link). Apply the short-label rule to each part too.
5. After delivery, one short line of context is fine ("folder + 3 drawings
   linked, deviations note at bottom") — but the link(s) themselves must be
   visible in the response.
6. If the user says the link "got truncated" or "didn't come across", re-call
   the tool (same text) and re-send the `display_link`/`parts` — do not just
   paste the prose again.
7. **Never hand-wrap the tool's URL in image-markdown / nested link brackets (`[![text](url)]`).** That construction is malformed markdown — Telegram renders the two split parts as ONE merged, garbled blob and the user reports "part one and part two have got mixed up in a single message" (observed 2026-09-04, Gunjur P-119 2-part link; the LLM was rebuilding links by hand instead of using the tool's output). Use the tool output verbatim: either the `display_link` or a short label around the raw `url` (`[Open WhatsApp — Name (Part N)](<url>)`), one bubble per part via the P5d call shape. If delivery tooling is misbehaving, put each part's link in the reply labeled Part 1/Part 2 — still never re-author the markdown.



When the user says "share document number X with [person]" via WhatsApp, you must resolve which actual Drive files they mean **before** generating the link. The user frequently refers to numbered documents in a series (e.g., Ranka Udaya docs 01–08 in the TMP folder).

**Prerequisite sequence before Stage 1:**

1. **Find the document** — Search the TMP folder (`18p74II2uL32sNDzDDwXzmlOUdJJOTmE-`) for files with `NN_` prefix matching the user's number. Example:
   ```python
   children = drive.files().list(
       q="'18p74II2uL32sNDzDDwXzmlOUdJJOTmE-' in parents and trashed=false",
       fields='files(id,name,webViewLink)', pageSize=100, orderBy='name'
   ).execute()
   ```
2. **Handle missing numbers** — If the user says "07" but no `07_` file exists, the numbering may have shifted. Read document content (export as text/plain, check first 1000 chars) and match by description:
   - "agent briefing" / "tone" / "bucketing users" → `05_FINAL_Agent_Briefing_v3.md`
   - "master brief" / "automation" / "cron jobs" → `08_MASTER_Brief_to_Tech_Agency.md`
3. **Verify access** — Before telling the recipient to "download and share," check the recipient already has writer/editor permission on the doc. Query `drive.files().get(fileId=DOC_ID, fields='permissions')` for their email. If missing, grant it via `permissions().create()` before the WhatsApp message goes out.
4. **Build the message** with numbered doc identifiers matching the user's own numbering, a clear one-line description of each doc, and the Drive link. Example structure:

   > Bharat — please share these 2 documents with [team]. You already have editor access to both.
   >
   > Document 7 — Merged Agent Briefing for Tech Agency: [link]
   > Covers how the AI agent should respond, tone of response, bucketing users.
   >
   > Document 8 — Master Brief to Tech Agency: [link]
   > Covers the total automation stack — cron jobs, enrichment, lead state store.

### P10. Newly-created spreadsheet share → WhatsApp link (2026-09-05, ORR Samudhara model)

When NDR says "I just created a spreadsheet on my Drive and shared it with X — WhatsApp them the link", the file may NOT be on the account you'd expect. Verified flow:

1. **Search Drive across ALL THREE vault services** (`google-draas`, `google-ahfl`, `google-gmail`) with `q="name contains '<keyword>' and trashed=false"` — the file can be owned by `nishantranka@gmail.com` (google-gmail) yet visible from google-draas; a single-account search misses it. Match the exact filename NDR dictated — "Simplified Calculation" vs "Value Calculation" are different sheets; pick the one he named.
2. **Verify the share is actually live BEFORE the message claims it** — `drive.permissions().list(fileId=...)` and confirm the recipient's email appears (writer/editor). The message opens with "I've shared X with you" — that claim must be true when NDR taps send.
3. **Sanity-check the dictated figures against the sheet** — read the summary rows (`sheets.spreadsheets().values().get(range='A1:H30')`) and confirm NDR's numbers match the sheet's computed values (worked: ₹80/SF/mo rental → Land Value per Acre ₹47.85 Cr). Catches voice-transcription errors in numbers before they're baked into a tappable link.
4. **Then draft + `whatsapp_link` with the standard rules (no raw `&`, bullets, named recipient, clear CTA).**

### P10b. Granting a colleague Drive access "for N hours" — check for an EMAIL first, and know the hard expiry limit (2026-09-17)

"Send [person] the link and give them viewer access for 24 hours" is really two asks, and each can fail independently.

1. **A per-person share needs an email address, never a phone number.** Resolve the recipient's email BEFORE promising access. Staff / domestic / vendor contacts frequently have NO email at all — observed on Pawan (Office Boy, contacts sheet row 2417), whose email columns literally contain the strings `Office Boy` and `Work` with no address behind them, and who has no Google account anywhere in the mail history. With no email there is nobody to grant a permission to. Say so plainly and fall back to the already-working anyone-with-link URL embedded in the WhatsApp message rather than silently skipping the request.

2. **Google Drive CANNOT expire a link permission.** `permissions().update(fileId=..., permissionId='anyoneWithLink', body={'role':'reader','expirationTime':...})` fails with **403 "Expiration dates cannot be set on domain or anyone type permissions."** (omit `role` and you instead get 400 *"The permission role field is required."*). `expirationTime` is honoured only for `user` and `group` permissions. So a time-limited *link* is not achievable via the API. The workable substitute is a scheduled job that deletes the `anyoneWithLink` permission after the requested window — and that revokes it for EVERY holder of the link, not just the one person. State that side effect and get a yes before scheduling it.

Also: the deliverable is a deep **link**, not a send. Say "tap it and press send" — never phrase it as though the message went out.

### P11. "Write back to [Name]" in voice is a WHATSAPP request by default (USER-CORRECTED 2026-09-05)

NDR voice-dictates personal/relationship follow-ups as "I want to write back to [Name]…" with the full message content but NO email word. **Default to a WhatsApp deep link, NOT a Gmail draft.** Corrected 2026-09-05: the agent drafted an apology email to Pankaj Kumar (Blue Hat Solutions, pankaj@bluehatsolutions.com) for a ~10-week silence after a 27-Jun-2026 partnership meeting; NDR replied "Not email I need a WhatsApp message" and the Gmail draft had to be deleted.

- "write back to" / "message back" / "reply to [Name]" with no email/address/mail word is a WhatsApp trigger. Only email when he says "email", "draft an email", "mail", or an address is involved.
- WhatsApp links are inert and immediate; if the dictation is ambiguous, ship the link and mention email only if the content needs attachments or a formal record.
### P12. A bare phone number in a NEW message does NOT mean "attach it to the previous dictation" (2026-09-11)

After generating a group WhatsApp link (no phone), NDR sent just a phone
number (`+91 94453 90912`) as the next message. The assistant assumed it
was a Teragren-group recipient and regenerated the SAME Teragren message
link to that number. Wrong: the number belonged to a completely different
thread — Ramesh of HydroTech3D (Bambu A1 printer supplier), discovered only
when the user's next voice message explained who it was ("This is the
telephone number of HydroTech3D Chennai…"). No message was sent (deep link
only), but the wrong-target link had to be explicitly retracted.

Rules:
1. **A bare number with no instruction is a LOOKUP or CONTEXT message, not
   a link-target update.** Before attaching it to any pending draft, run the
   exhaustive identity sweep (People API ×3 accounts + contacts sheet + Gmail
   `from:` search). If the number resolves to a person NOT connected to the
   pending message's thread, do NOT silently reuse the pending text — flag the
   mismatch to NDR and ask what he wants the number used for.
2. **Context bleed across voice messages is the norm.** Phone numbers,
   names, and URLs dictated near each other are often about DIFFERENT
   tasks (here: Teragren group meeting vs Bambu printer supplier vs
   Cheprakal land deal). Treat each bare datum as its own unit until
   NDR wires it into a thread explicitly.
3. When you DO realise the wrong link was generated, say so plainly and
   retract it ("ignore the link to <number> — nothing was sent; it was the
   wrong target") and regenerate only against the correct recipient once
   confirmed.

### P13. Never generate a WhatsApp link to a number that isn't the intended recipient's

Resolved-identity discipline: the phone argument must come from the
recipient's own contact record (or an explicit NDR "message this number"
instruction). A number that fails the two-source phone verification
(contact-phone-lookup skill) and is NOT authoritatively linked to the
intended person must not be baked into a message link. When NDR later
discloses who a number belongs to (e.g. Ramesh HydroTech3D), treat that as
the identity — and if you had generated a link to it under a wrong banner,
retract it (P12).

### P1. Raw `&` in the source text breaks the link (USER-CORRECTED)

This is the single most important rule in this skill. The user explicitly
called this out after the first wa.me link broke:
WhatsApp tool… it is again breaking around the ampersand sign and in the
tool we have already created special handling for ampersand sign.\"

Fix: rewrite `&` to \"and\" / \"plus\" / \"and/or\" / em-dash in the source
text BEFORE calling `whatsapp_link`. Do not try to URL-encode `&` yourself
in the message string — let the tool's special handling do its job, and
that handling assumes no raw `&` is present.

If you must keep `&` for typographic accuracy (e.g. a legal name like
\"Smith & Co.\"), use a workaround: \"Smith and Co.\" in the message and
note the original spelling in a short preamble the user can drop.

### P2.1. Contact name doesn't match — try phonetic variants and clarify

When the user says a name that returns ZERO contact results, the name may be:
- A **phonetic variant** of a common Indian name (e.g. "Gitu" → "Jitu", "Jitendra", "Jitu Mehta"; "Bappi" → "Bapi"; "Shridhar" → "Sridhar"). Try searching the phonetic alternative before reporting "contact not found."
- A **nickname / term of address** the user uses personally that differs from the contact's stored display name (e.g. user says "Sir" or "Boss" for someone stored as "Dr. Venkatesh"; user says "G2" for someone stored as "Jitu Virwani"). Ask: "Is this person also known as [variant] in your contacts?"
- A **role-based reference** (e.g. "the auditor" → search by email domain, "the Godrej guy" → search by organization name). Don't assume the user will use the stored name.
- **Designation-as-name suffix (correction 2026-08-21):** The user's voice often appends a role/designation suffix to the name: "Mohan ADTP", "Mohan Sir JDTP GBA East", "Nagrajappa JDTP", etc. Standard `contact_resolver` fails because the stored name differs from the voice-compound. **Fix:** Strip the suffix (ADTP, JDTP, BBMP, GBA East, Sir, etc.) AND search People API `searchContacts()` directly with just the first name — the role is often part of the stored display name as a single unstructured field like "Mohan Sir JDTP GBA East". The standard partial-name `contact_resolver` doesn't match these compound forms; direct People API does. For "Mohan ADTP" (this session), the actual contact was found via `searchContacts(query='Mohan')` on google-draas returning "Mohan Sir JDTP GBA East" (+91 98868 85455). Checklist: strip suffix → search bare given name across all three vault accounts → cross-reference role context (land, Northstar, airport NOCs) to confirm the right match.

**Correction history**: The user said "Gitu, sir" for a contact involved in NDA/family settlement work. Initial search for "Gitu" returned nothing. The person was likely "Jitu" (a different contact in the same account) — the user's pronunciation merged dental and retroflex consonants. Always try the alternate dental/retroflex or voiced/unvoiced variant when a name search returns zero results.

Search strategy when initial query fails:
1. Try the phonetic variant (swap G↔J, T↔D, S↔Sh, etc.). Common Indian name confusions: "Nisabharaju" → "Narsimharaju", "Gitu" → "Jitu", "Bappi" → "Bapi", "Shridhar" → "Sridhar", "Vineet" → "Vinay", "Bhuvanesh" → "Bhubanesh".
2. Try partial matches (first 3-4 chars of the name).
3. **Check the NDR DRAAS Contact Sheet** — it often has contacts stored under a different spelling or format than Google Contacts. Scan the First Name column (col A, index 0) for partial string matches. The sheet may store the name in a single field like "Narsimharaju Jt Com IT" while the user says "Nisabharaju". Pattern:
   ```python
   sheets = build_service('sheets', 'v4', service_name='google-draas')
   all_data = sheets.spreadsheets().values().get(
       spreadsheetId='1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g',
       range='A:AR'
   ).execute()
   for i, row in enumerate(all_data.get('values', [])[1:], 2):
       first = (row[0] if row else '').lower()
       if search_term.lower() in first or any(term.lower() in first for term in phonetic_variants):
           # row found — report the detected stored name
   ```
   **Important**: Also search with the user's ORIGINAL pronunciation, not just the phonetic variant. The sheet name might match the user's pronunciation better than the Google Contacts name does.
4. If the person was mentioned in this conversation or a recent session, use `session_search` to find their full name from context.
5. If still not found in any People API account, the contacts sheet, or Gmail signature extraction, the contact may only exist in the user's WhatsApp — saved with a role-based name (e.g. "Mohan Sir JDTP GBA East") that doesn't appear in any of NDR's synced directories. **Do NOT pick a different person with the same first name and hope it's correct** — that costs a correction round (observed 2026-09-07: picked Mohan Krishna, BBMP EE, when the real contact was Mohan Sir JDTP GBA East, saved only in WhatsApp). Flag this clearly: "This contact doesn't appear in any of your online directories — it's likely a WhatsApp-only contact. Could you share the phone number?" Show the drafted message text so the user can copy-paste it once they provide the number.

### P2.2. Multiple contacts match — disambiguate with `clarify`

**Before clarify, try Gmail-context disambiguation.** When several same-name candidates exist but the conversation topic is known, run a Gmail query pairing the name with topic keywords (e.g. `Kishan (algo OR AIF OR trading OR Nifty)`). If one candidate's threads are decisive — recurring "Performance Update" emails from their company domain matching the user's story — use that person and skip the clarify round-trip. Worked 2026-08-14: 4 "Kishan" candidates; Kishan Murjani Nair (kishan@flamebackcapital.com) won on thread evidence alone. Fall back to `clarify` only when no candidate has contextual proof.

When a name search returns MULTIPLE contacts and none is an obvious match for the user's intent, use the `clarify` tool to present a shortlist as numbered choices. This is faster than asking open-ended and avoids the user having to read your full prose list.

```python
clarify(
    question="I found several \"Vikram\" contacts — which one?",
    choices=[
        "Vikram Jain Karbawala (+91 98450 70779)",
        "Vikram Talreja / Sparc Studio (+91 98453 99997)",
        "Vikramaditya / Kelsa (+91 77605 03837)",
        ...
    ]
)
```

**When to use this**: you've searched and found 3–10 plausible contacts, none of which has a strong contextual match to the topic the user mentioned (e.g. project name, company name). Do NOT use this when only 1–2 contacts matched — just present the best candidate and ask a yes/no question instead.

**Family-context disambiguation — the `RelA` label wins (2026-09-06, "Jayanth" → Jayanthy Jain):** when the subject is PERSONAL/FAMILY (healthcare contacts, doctor's details, relatives, medical coordination), a candidate whose contacts-sheet Labels column carries `RelA` (relative) beats all business candidates, even if the business ones have more data. Voice said "Jayanth"; candidates were a Nexxoft engineer, an advocate, a Sriram Properties contact, and Jayanthy Jain (RelA, +91 98676 77652). Picked the relative for the Trustwell/Dr. Haldipur family share. Name the resolved person + number in the reply so the user can correct before tapping — do not silently pick.

**Repeat-share flow ("send to X again", "the same contact list"):** when the user says "send the same [contacts/dossier] to [people] again", the previous share usually lives in session history — `session_search` for the entity (e.g. "Haldipur OR Trustwell") to recover the exact list, then RE-VERIFY every number live against People API + the contacts sheet before generating links (dossiers drift — numbers change, staff rotate). Ritesh = Rithesh Jain (MK Silk World) resolved via the Sep-6 chocolates message thread; the Trustwell support team (Sridhar/Charan/Elumali) was re-verified from People API in the same session.

**Phone-saved name mismatch**: the user sometimes says "my address is [name]" — meaning their phone's local address book entry. This phone entry name may differ from the Google Contacts display name (e.g. phone says "Vikram Sa" but Google has "Vikram Jain Karbawala"). Search broadly for the given name and use the clarify tool to list the options, rather than assuming the phone's abbreviation will match Google's stored name.

### P2.3. Voice-transcribed names are often wrong — including project/builder names

User's STT regularly corrupts proper nouns. "Sadhwani" comes back
correctly, but "Byadarhalli" → "Bihar Ali", "Vineet" → "Vinay",
"Jiraff Capital" → "Jiraffe Kaya", and **"Legacy Cataleya" → "Century Katalya"** have all been observed.

**For contact names** — verify against the People API (searchContacts returns canonical spelling) and Gmail Sent. Call out which names were corrected.

**For project / builder / company names** — when the user asks you to research a named project and web search returns nothing, the name may be garbled. Follow this workflow:

1. **Search the garbled name** exactly as transcribed — sometimes the STT match is correct and the project is simply not online.
2. **Try phonetic variants** — swap L↔C, K↔C, T↔D, etc. "Century Katalya" → "Century Cataleya", "Century Catalia", "Century Katalia", then try the first word swap: "Legacy Cataleya" (the actual project was "Legacy Cataleya", not "Century Katalya" — every word was wrong).
3. **Search by landmarks** — the user often provides location context. "Kaningam Road" (→ Cunningham Road), "next to Ranka Chambers" helped narrow Cunningham Road as the location.
4. **Search real estate portals** for projects at that landmark — check 99acres, MagicBricks, Housing.com, Commonfloor for projects matching the location, not the name.
5. **Use `clarify` with the closest candidates** — present 3–4 options from what you found at the location. The user will recognise the right one even if they'd accept the garbled name.
6. **Once the correct name is confirmed**, do full research (RERA number, address, survey number, BBMP sanction, status, unit details). Use the RERA portal (rera.karnataka.gov.in) or aggregator sites like aurumproptech.in for structured project data.
7. **Call out which names were corrected** in your response so the user knows the voice transcription was wrong.

**Session example (2026-07-29):** User said "Century Katalya on Kaningam Road next to Ranka Chambers." Web search returned zero results for "Century Katalya." Through phonetic variants and landmark-based searching on real estate portals, the closest match found was "Legacy Cataleya" on Cunningham Road. The user confirmed via clarify. Full project details were then researched from RERA and brokerage sites.

### P2.4. Doctor contacts stored by ROLE + HOSPITAL — voice gives the surname (2026-09-07, Dr Nishanth Hiremath)

NDR said "Dr. Nishant, Hiramath" — a doctor he addresses as "hey doctor" (snoring/sleep-app follow-up). Nothing matched "Hiramath" in People API or the contacts sheet. Resolution chain that worked:

1. **People API `searchContacts('Nishant')` across all 3 vault accounts** returned "Dr Nishanth ER Mahaveer" (org: Mahaveer hospital, pavlov23@gmail.com) — but no Hiramath/Hiremath.
2. **Contacts sheet sweep** for surname + variants (`hiramath`, `hirmath`, `mahaveer`, `nishant`) found "Dr Nishanth ER Mahaveer" row with +91 99010 99222 — same email address.
3. **Gmail search on `"Dr Nishant" Hiremath` variants** found threads signed "Dr.Nishanth Hiremath.S Sr.Consultant & Chief Department of Emergency Medicine BMJH" from `pavlov23@gmail.com` — the **shared email address is the join key** proving "Dr Nishanth ER Mahaveer" (ER = Emergency dept, Mahaveer = Bhagwan Mahaveer Jain Hospital) and "Nishanth Hiremath" are the SAME person.
4. Voice "Hiramath" → **Hiremath** (common Karnataka surname; also "Hirmath").

Pattern: medical contacts are often saved by role/hospital breakdown (ER, specialty, hospital name) while NDR says the surname. When a surname search fails, query Gmail for `"Dr <First>" <surname-variant>` + `"Dr <surname>"` and join on the shared email address. Confirm the resolved name + number aloud before generating the link.

### P2.5. Medical / drug names in dictated messages must be corrected (2026-09-07, Pelacarsen)

Extends the HEPA rule (Stage 2) to prescriptions/trials: NDR dictated "Pascaleren" → **Pelacarsen** (Novartis Lp(a) trial); "LP little A" → **Lp(a)** (lipoprotein(a)); "CAG scores" measured before calcification/plaque → **CAC** (coronary artery calcium) score when the context is imaging. Write the correct term in the message AND tell NDR you corrected it in your reply. For a patient→specialist "thanks for the study" message, a structured "things I'm watching" bullet list (subgroup analysis, next trials, cohort characteristics) reads well — the user's own dictation was already organized that way and should be preserved as bullets, not flattened.

### P2.6. Calendar attendee history — the resolution source that gets missed (2026-09-19, "Harish Bhaksani" → Haresh Buxani)

When a voice-garbled **full name** fails across the People API (all three
vault accounts), the contacts sheet, AND Gmail, the last place to look before
giving up is **NDR's own calendar**. Past events he attended carry the
person's canonical display name and email as an attendee record —
independent of whether they're in Contacts at all.

Worked example: NDR dictated "meeting with **Harish Bhaksani**". Exhaustive
sweeps returned nothing:

- People API ×3 accounts, 12 phonetic variants (`Bhaksani`, `Bakhsani`,
  `Bhaxani`, `Baksani`, `Bhaghsani`, …) → zero
- Contacts sheet fuzzy scan (4,249 rows, 13 patterns) → zero
- Gmail search → one false positive from a 2010 Google Chat export
- Only "Harish" candidates that existed were unrelated (Harish K S / DRA,
  Harish Naik, Harish Sathya Lawyer)

What worked:

```python
cal = build_service('calendar','v3', service_name='google-draas')
r = cal.events().list(calendarId='primary', q='Harish', maxResults=25,
                      singleEvents=True, orderBy='startTime',
                      timeMin='2025-01-01T00:00:00Z').execute()
# then inspect attendees for each item
```

One hit: **"Investment Proposal Discussion with Harish"** — attendee
`haresh@buxani.com`, displayName **Haresh Buxani**, responseStatus `accepted`,
location *DRA Realty Office* (the same venue as the requested meeting), dated
14 Mar 2026. Confirmed via People API returning `Haresh Buxani` with the same
address plus `+852 9200 0520` (Hong Kong).

Rules:

1. **Search the calendar by first name only** — the surname is usually the
   corrupted part, and `q=` does a broad text match across summaries and
   attendees. Vary the first name too (`Harish`/`Haresh`, `Gitu`/`Jitu`).
2. **The `q=` filter matches the summary as well**, so events titled
   "…with <Name>" surface even when the attendee record is thin. Read the
   `attendees` array, not just the summary.
3. **Use venue + topic as corroboration.** The location matching the venue
   NDR named for the new meeting is strong evidence you have the right person.
4. **`responseStatus` tells you the relationship is live** (`accepted` beats
   `needsAction`).
5. If the calendar also draws a blank, **ask** — but only after this sweep.
   Report the candidates you did find so he can pick.
6. Recurring false-positive shapes to ignore: `[NNNN_Dra developers & projects
   pvt ltd.<Name>] Review & Approve` entries are the internal HR/attendance
   automation, not a person NDR is meeting.

This applies whenever you must resolve a person to an email/phone for a
calendar invite, a WhatsApp link, or an email draft — the three channels share
the same resolution problem.

### P3. The tool does not validate the phone number

`whatsapp_link` strips non-digit characters and builds the URL, but it
does not check whether the number is a real WhatsApp account. If the
contact record has a number that is no longer active, the link will
open WhatsApp with the recipient field pre-filled, but the user will
discover at send time. Prefer the **canonical form** from the contact
record (E.164 with the country code) over the raw display value.

**A number recalled from conversation/session history is NOT a resolved number (user-corrected 2026-09-18).** Stating "I have his number on file (+91 98841 27288)" and building the link from it is the same error as inventing one — the digit string had drifted in from elsewhere in context. Ayalur Vaidyanathan's link failed and his registry entry holds exactly ONE number, **+91 93425 57120** (contacts sheet row 502). Rules: (1) a phone argument is only ever taken from a live lookup made **in the same turn** (resolver → People API ×3 accounts → contacts sheet → Gmail signature → calendar); (2) when the user says a number is wrong, the first action is a fresh lookup for that person and you report the resolved digits — never defend the old number; (3) do not spend turns retrying a failed `memory` write while a link is owed. Full write-up: `contact-phone-lookup` → "NEVER QUOTE A PHONE NUMBER FROM CONVERSATION / SESSION HISTORY".

**Verify digit count BEFORE delivering — an extra digit silently builds a broken link (USER-CORRECTED 2026-08-31).** Indian mobile numbers are exactly **12 digits** after stripping (country code 91 + 10-digit mobile, e.g. `918150029900`). Observed failure: the agent passed `+9198150029900` — country code 91, then a stray `9`, then `8150029900` — 13 digits total. The tool strips the `+` and happily builds `phone=9198150029900`, which reverses to `+91 98150 029900` — not a real active number. The user noticed at tap time: "you have sent an extra digit 9 before it ... it is a broken link." Prevention: after composing the link, verify `phone=<digits>` in the URL is exactly 12 digits for Indian numbers; if it's 13+, one digit is wrong — re-check the source contact before delivering. The same check applies to the digit-stripped value the tool echoes back in its `url` field.

**The re-typing transposition variant (2026-09-04, Amit Jain Deloitte): NEVER re-type the resolved number into the `whatsapp_link` call — copy the exact digit string from the resolver/People API output.** The contact record +919704404000 (mobile 9704404000) was CORRECT; the agent re-typed it into the tool call as "+91 97040 44000", which strips to 919704044000 → mobile 9704044000 (the "44" block moved) — a silent transposition that only surfaced when NDR compared the link's number to the actual. The link "looked" like a valid 12-digit number so the P3 digit-count check never fired. Prevention: pass the phone argument verbatim (copy-paste the resolver's canonical value, no re-grouping/spacing by eye), and AFTER the tool echoes `phone=` in its `url`, diff it char-by-char against the canonical value — a mismatch means YOU corrupted it at the boundary, fix by re-calling the tool, not by editing source data. For batch/scripted flows, pipe the People API canonicalForm directly into `tools.whatsapp_link_tool` (sanctioned terminal fallback) so no LLM re-typing can occur between resolver and tool.

**The missing-digit variant (2026-09-02, Nishant Prakash Zara): the SAME 12-digit rule applies in the other direction — an 11-digit argument is equally broken.** The agent had the correct stored number (+91 99996 73483, verified People API ×2 accounts + contacts sheet row 2317 + his own email signatures all showing `9999673483`) but **typed the tool argument as `+91999673483` — dropped one `9` inside the number, 11 digits**. The tool silently built `phone=91999673483`. User: "you got the wrong number for Nishant Prakash." Re-searching every source returned the SAME correct number — the bug was the agent's own transcription into the tool call, not the stored data.

**When the user says a number is wrong after you used a verified source, the order of checks is:**
1. **Count the digits in the `phone` argument YOU passed** (and in the URL the tool echoed) — an 11-digit (dropped digit) or 13-digit (stray digit) Indian number is a self-inflicted encoding bug, fixed by re-calling the tool with the correct 12-digit E.164, no re-search needed.
2. Only if the passed argument was 12 digits AND the source itself is suspect, re-run the full source sweep (People API ×3 accounts → contacts sheet → email signatures → session history) per contact-phone-lookup's "When the User Says a Number is Wrong" protocol.
3. Report the exact digits found and let the user confirm before regenerating.

### P4. Group messages need an empty phone, not a fake one

When the user says "I'm putting the message in a group," pass `phone=""`
to the tool. Do NOT pass a single recipient's number hoping WhatsApp
will somehow route to the group — it won't, and you'll send a 1:1
message to the wrong person when the user taps the link.

### P5. Long messages still work, but watch the prefill — and the Telegram delivery limit (USER-CORRECTED)

WhatsApp's prefill is robust for messages up to a few thousand
characters. The **Telegram delivery** of the resulting wa.me link is
not. Telegram splits outgoing text at ~4096 characters per message
(both for caption bodies and for the body of a normal text reply). A
long voice-transcribed follow-up — 1–2 KB message + 1–2 KB of URL
encoding on top — easily exceeds 4 KB once you count the markdown
link wrapper. When that happens, Telegram delivers the result as **two
separate bubbles**, the wa.me URL is truncated at the split point, and
the user gets a broken link. The user has corrected this twice:
*"Can you give me Charan's WhatsApp message link again… it did not
come across properly because it got split into two messages here."*

**The tool now auto-splits** (2026-08-07): `whatsapp_link` detects when a
  URL would exceed Telegram's 4096-char single-message cap and returns a
  `parts` array — one complete, independently-tappable link per part, each
  sized so even the `[display text](url)` form fits one bubble. When the
  result has `"split": true`, deliver EACH part as its OWN separate
  Telegram message (e.g. via `send_message`); never paste multiple parts
  into one message or Telegram's splitter will cut a link in half.

**Optional compaction (fewer links = cleaner UX):** after `whatsapp_link`
returns, if the response would contain multiple parts, you may compact
the message first and call the tool again so the user gets one link.
Strategies that work, in order of preference:
  1. **Strip sub-bullets and the closing "Regards / phone" block.**
     Many voice messages end with a long sign-off; the user can add
     that themselves after tapping. Cutting "Regards, Nishant Ranka /
     +91 98800 55634" typically saves 60–80 chars.
  2. **Inline the numbered points as a single line per item** instead
     of multi-line paragraphs. "1. Reporting time — please confirm…
     2. Documentation — please confirm you have…" instead of
     "\n1. Reporting time\n\nPlease confirm…\n\n2. Documentation\n\nPlease confirm…".
     This is the single biggest win — long messages usually have
     paragraph-broken bullets that add ~2× the bytes.
  3. **Drop "Kindly confirm", "Awaiting your revert on this" and
     similar filler** that voice messages accumulate. Each cut saves
     20–40 chars and the user does not miss them.
- After rewriting, call `whatsapp_link` again. The link should fit in
  one Telegram bubble.
- Tell the user the link has the full message and they can scroll
  within the compose box.

**Symptom-recognition rule for an already-split message:** if the user
says the link "didn't come through properly" or "got split into two
messages" right after a `whatsapp_link` reply, this is the cause, not a
bug in the tool. The link is fine; the *delivery* chopped it. Re-send
using the compaction above and confirm in the re-sent message that it
is a single link.

### P14. One WhatsApp link cannot carry two matters for the same recipient (2026-09-18)

When a chase/broadcast task has **more than one item for the same person** (e.g. Bharat
owned both the Ranka Amber supplementary-sharing chase AND the Ranka NorthStar addendum
chase), a single WhatsApp message containing both bullet sets becomes long and muddy.

Failure observed: both items were drafted into one message; it was judged too long, the
Amber item was sent, and the NorthStar item was **silently dropped** — then mentioned only
as a trailing afterthought line. NDR reads the link label first and would have tapped send
without knowing a second message was still owed.

Rules:
1. **Count matters per recipient BEFORE drafting.** N > 1 ⇒ decide up front: one combined
   message (only if it stays short and legible) or N separate links.
2. If you send only one of them, **state the omission prominently** in the delivery message
   — a leading line, not a trailing asterisk. Prefer just generating both links; a chase is
   not a place to economise on a second link.
3. **The email layer does NOT have this constraint** — send N threaded emails to the same
   person freely; that's the natural shape and what NDR expects. The one-matter-per-message
   discipline applies only to the WhatsApp layer, because a link is one tap = one message.

### P5b. Very long messages fail on mobile WhatsApp — deliver an HTML file (USER-CORRECTED)

Distinct from the Telegram-split problem: even when the link arrives as
ONE bubble, a very long `?text=` parameter (~4 KB+ of encoded URL) can
fail to open or pre-fill on the WhatsApp mobile client. The tool's
auto-split (P5) already caps each part's URL at Telegram's 4096-char
budget, so this is now rare — the HTML fallback below remains for
extremely long messages the user wants in ONE tap. The user reported
the link "not working" and the follow-up request was explicit: *"If the
message is too long, let's generate an HTML file in which the link is
kept so I can click it and send the message to the group."*

### P5e. Interactive editable HTML builder — user regenerates the link himself (2026-08-29)

When the user asks for an HTML file he can EDIT and REGENERATE himself ("generate an HTML file with the entire message and allow me to edit the message and click a button that regenerates the WhatsApp link"), build the interactive builder — this is a distinct artifact from the static P5b page:

**Deliver:** `templates/whatsapp-message-builder.html` (copy from the skill, fill the phone + message, deliver with `MEDIA:`). Proven identical to this session's `20260829_NishantPrakash_WhatsAppMessageBuilder.html`.

**The template includes:**
- `contenteditable` editor with B/I/S/monospace toolbar (`document.execCommand` — deprecated but works everywhere)
- DOM-walker that converts rich text → WhatsApp markers (`<b>`→`*text*`, `<i>`→`_text_`, `<s>`→`~text~`, `<code>`→`` ```text``` ``)
- `⚡ Regenerate WhatsApp link` button — client-side `encodeURIComponent` into `api.whatsapp.com/send`, auto-splits long messages at a ~3700-char encoded budget into `parts` (mirrors the tool's P5 behaviour), one Copy + Open-in-WhatsApp button per part
- Editable phone field (10-digit numbers auto-get 91)
- Reset button restoring the original draft

**Pitfalls learned:**
1. **Verify the JS logic** before delivering — run the `buildLinks()`/`digitsOf()` functions under Node (`node -e`) with a sample text to confirm parts/splits behave. The template's logic was Node-verified before hand-off.
2. **Split-parts still apply** — the message in this session split into 2 parts (main body + closing/funding paragraph). Tell the user each part is its own WhatsApp message; same rule as tool-split parts.
3. **Phone number conflict check** — the contacts sheet said Nishant Prakash = +91 98300 24654 (row 617, Redsol Farmers Collective investor) but earlier 2026 threads used +91 99996 73483. When a name has divergent numbers across sources, bake in the sheet's number BUT flag the alternate explicitly and offer to regenerate — never silently pick.
4. **Bake the message into the HTML** exactly as the drafted/sanctioned text — don't paraphrase; the user edits from a correct baseline.

### P5c. Delivery format preference — code block for copy-paste (USER-CORRECTED 2026-08-21)

When the user asks for WhatsApp links and says they want them "presented
as a code section" / "with all the whatsapp markdowns so I can copy and
paste directly," deliver BOTH:

1. The tappable markdown link: `[Open WhatsApp — Name](url)`
2. A fenced code block containing the raw URL for copy-paste

The user uses the code block to copy the link and paste it into a
different app (group chat, another chat, their notes). Do NOT skip the
code block when this format is requested — the default tappable link
alone isn't sufficient for his workflow.

Example of the requested format:

```
🔗 Person Name — WhatsApp link (tap to open):
[Open WhatsApp — Person Name](<url>)

📋 Copy-paste version:

<url>
```

This format override applies when the user explicitly asks for code
blocks / markdowns / copy-paste ready delivery. It does not replace the
default single-link delivery; it augments it on request.

**Fix pattern (in order):**
1. **Condense the message first** — compact the draft (drop filler,
   inline bullets, shorten sign-off). Short URLs almost always work.
2. **If still long, build an HTML file** at `/opt/data/` (e.g.
   `kelsa_whatsapp_message.html`) containing:
   - A WhatsApp-style chat bubble showing the full message text (so the
     user sees what will be sent), using `white-space: pre-wrap`
   - A large green button (`background:#25d366`, border-radius:50px)
     whose `href` is the **exact URL returned by `whatsapp_link`** —
     never a hand-built URL
   - A smaller "Direct link (copy if needed)" anchor with the same URL
   - A note: "Tap the green button → WhatsApp opens with the message
     pre-filled → choose your group and send"
3. Deliver with `MEDIA:/path/to/file.html` in the response so it arrives
   as a downloadable file the user can open on their phone.
4. Mention that WhatsApp shows `＆` (fullwidth ampersand) in long
   encoded messages — that is correct encoding, don't edit it.

**Percent sign (`%`):** the tool encodes `%` exactly once (`%25`) and emits
`api.whatsapp.com/send` links (never `wa.me`). **BUT the `%` is STILL
corrupted by WhatsApp's own decoder — verified live 2026-08-10.** A short
test link (`api.whatsapp.com/send?phone=…&text=Test%3A%2012.5%25%20tax…`)
that survived Telegram's delivery intact still rendered `�` (U+FFFD)
instead of `%` in the WhatsApp compose box. This is NOT a wa.me redirect
problem and NOT a Telegram-truncation problem — it is WhatsApp's mobile
client re-interpreting the decoded `%` as the start of another escape
sequence. The `&`/`#` fixes in the tool sidestep this exact mechanism by
substituting **fullwidth characters** (`&`→U+FF06 `%EF%BC%86`, `#`→U+FF03
`%EF%BC%83`) whose UTF-8 escapes contain no `%26`/`%23` substrings. The
same fix is needed for percent but is NOT yet in the tool: add
`encoded.replace("%25", "%EF%BC%85")` in `_encode_wa_text()` (U+FF05
FULLWIDTH PERCENT SIGN ％; `%EF%BC%85` contains no `%25`, so no cascade —
a literal `%25` in source encodes to `%2525` and is untouched). Until the
tool is patched, the pragmatic fallback for user-facing messages is to
write "percent" in words (e.g. "12.5 percent") instead of `%`.

**Diagnostic for "broken character in WhatsApp link" reports:** first
generate a SHORT test link (a sentence with the suspect character, no
long message) and have the user tap it. If the short link is clean, the
problem was Telegram truncating the long URL — re-split/re-send. If the
short link is STILL corrupt, the problem is WhatsApp's decoder — apply
the fullwidth-substitution fix for that character (see above).

See `messaging-drafts` skill → `references/whatsapp-chunked-message-html.md`
for the multi-chunk variant when the message itself must be split across
multiple sends.

### P5d. Split-part delivery — the WORKING send_message call shape (2026-08-24)

When `split: true` and you deliver each part via `send_message`, the ONLY call
shape that works for the DM session is **person-to-person addressing**:

```
send_message(recipient_name='Nishant Ranka', platform='telegram', message=<part display_link>)
```

Tried-and-rejected variants (all fail on the Telegram gateway):
- `target='telegram'` → `Cross-user send blocked` (resolves to the home channel, not the session DM)
- `target='origin'` → `Unknown platform: origin`
- bare `message=` with no target → validation error (needs recipient or target)

Also capture NDR's explicit rule from the same session (user: *"I need you to just
generate the WhatsApp link using the WhatsApp link tool, not send the WhatsApp"*):
the deliverable is the **link(s) from `whatsapp_link`** — he taps, picks the group,
and pastes himself. Do NOT spend turns fighting delivery targets or attempting to
post to WhatsApp programmatically (Hermes cannot). Generate the link(s), deliver
them cleanly (one part per bubble), and stop. If delivery tooling is misbehaving,
put the links in the reply and tell him they're ready — don't loop on `send_message`.
(Re-validated 2026-08-29 with a 2-part split: `target='telegram'` → `Cross-user send blocked`,
`target='origin'` → `Unknown platform: origin`; embedding both `display_link` parts in the reply and
labeling them Part 1/2 worked — the reply-embedding fallback is fine for split parts too.)

**Reply-embedding two links in ONE bubble = the "links got merged" complaint (USER-CORRECTED 2026-09-02).** For a 2-part group-paste message, NDR reported "link 1 and link 2 have got merged somehow … it says part 1 then part 2 then YADG has come." Cause: both markdown links sat in a single Telegram reply. Fix: deliver EACH part as its OWN separate bubble via `send_message(recipient_name='Nishant Ranka', platform='telegram', message=<part link>)` — the same person-to-person shape as P5d, and it works for group-paste (no-phone) links too. One bubble per part, always, even when both parts are short.

- **Never hand-wrap the tool URL in your own markdown — short label + the tool's raw `parts[i].url`, one bubble per part (re-validated 2026-09-04).** On the Gunjur 2-part split the assistant delivered BOTH parts in ONE reply and hand-wrapped them as `[![text](url)]` (image-NESTED-in-link markdown). Telegram rendered it as one merged, garbled blob — user: "part one and part two have got mixed up in a single message … the LLM output is getting mixed up across messages." The tool's split was correct (each part had a complete independent URL); the hand-built wrapper + combining parts in one bubble caused the merge. Re-validated delivery shape that works: `send_message(recipient_name='Nishant Ranka', platform='telegram', message='[Open WhatsApp — Name (Part N)](<parts[i].url>)')` — `target='telegram'` → Cross-user send blocked, `target='origin'` → Unknown platform, both still true. Do not re-escape/restyle the label text; plain short label only.

**Voice-transcription names in a group draft are artifact-prone — drop them on command (2026-09-02).** In the DRA Creative group father-daughter story message, the dictated "Yadj — please push this message out to our existing leads" was a transcription error; NDR said "there's no YADG… it's Gowri and Bharat only we are sending it to." When NDR names a person in a group message you cannot resolve in ANY contact source, flag it in your reply (as done) and DO NOT treat the name as a recipient — on correction, regenerate the links WITHOUT the name and WITHOUT re-selling it. Recipient list in a group draft = only the people NDR confirmed. Also re-verify after regeneration that each part's URL contains the corrected content (drop the stale name from the text before re-calling `whatsapp_link`).

- **Group-paste split parts: send each part as its OWN bubble — never embed both in one reply (corrected 2026-09-02).** With a long group-paste message that splits into 2 parts, delivering both `display_link`s inside a SINGLE reply made the user read them as one merged message ("link 1 and link 2 have got merged somehow") and he asked for them regenerated as clean Part 1 / Part 2 links. Repair: call `whatsapp_link` again with the corrected text, then deliver each part via its own `send_message(recipient_name='Nishant Ranka', platform='telegram', message=[Open WhatsApp — <Group> (Part N)](<url>))` bubble. One bubble per part, short labels. This applies to group-paste links too (they already have no phone), not just person links.
- **Regenerate when the user says a dictated recipient name was a transcription error.** When a name can't be resolved in ANY contact source ("Yadj"/"YADG" — nothing anywhere), flag it in the reply and keep it as dictated only if the message is internal. If the user then says "it's a transcription error, ignore it, it's Gowri and Bharat only", DROP the phantom name from the body and recall `whatsapp_link` with the corrected text — do not keep shipping the unverified name through future regenerations.

**2026-09-02 correction — separate bubbles are the reliable path, NOT inline embedding:** with a 2-part split delivered as two `[text](url)` links inline in ONE reply, the user reported the links "got merged" (opening one showed content from both — "it says part 1 then part 2 then ... has come") and asked to "regenerate the WhatsApp links correctly, part 1, part 2, breaking it up into appropriate links". The fix that landed: re-call `whatsapp_link` with the corrected text, then send EACH part as its own Telegram bubble via `send_message(recipient_name='Nishant Ranka', platform='telegram', message='[Open WhatsApp — <Group> (Part N)](<url>)')` — short label, raw URL from the tool. When the user says split parts look merged / didn't come across correctly, regenerate and re-send as separate bubbles; do NOT just re-embed the old links in the reply.
- **Transcription-error names in dictation (same 2026-09-02 session):** user explicitly said the name "Yadj" in the drafted group message "is some kind of transcription error, ignore that, it's Gowri and Bharat only we are sending it to". Fix: regenerate the ENTIRE message without the phantom name and re-call `whatsapp_link` (which re-splits cleanly) — never hand-edit just the changed part of an already-split pair. State the corrected recipient set back to the user and re-verify the split boundaries after regeneration.

### P6. wa.me deep links cannot carry file attachments

The `whatsapp_link` tool only pre-fills text — it cannot pre-attach a PDF, image, or
document to the WhatsApp message. When the user dictates a message that says
"attaching the copy of mom's policy" or "see the attached report", the link will
open WhatsApp with text ready but the **user must tap the attachment (paperclip) icon
and pick the file themselves** before sending. Two patterns:

- **Best for one-off attachments**: draft the text via `whatsapp_link` and tell the
  user "the link has the text — open the chat, tap the paperclip to attach the PDF,
  then send." If the file is on Drive, give them the Drive link to download on
  their phone first.
- **Best when the attachment is the main payload**: skip WhatsApp and use a Gmail
  draft instead (see `email-drafter` skill). WhatsApp + PDF is awkward; email is
  the native channel for files.

Don't promise "the link will attach the file" — it can't, and the user will
discover the gap at send time.

### P7. User dictating a message to a third party is still a "send WhatsApp" task, not impersonation

A common voice-message pattern: the user dictates a message in
**second person** addressed to a named recipient, e.g. "Pratash, urgent.
Re: the Ranka Oasis email..." or "Amit, as per our last telephonic
discussion you were to check with your legal team...". The user is
**dictating instructions TO that person**, not asking Hermes to perform
that person's job.

Default action: this is a `whatsapp_link` task — the user is the sender,
Pratash / Amit / etc. is the recipient. Resolve the contact, draft the
message, generate the link. Do NOT act on the third party's behalf in
external systems (Gmail replies, Drive edits, Sheets updates) — that is
**impersonation**, not delegation.

**Pausing rule — when to clarify with the user**: if the dictated
message references actions that *only* the third party can take — e.g.
"open this Drive file owned by your legal team, attach the legal opinion
links, and reply-all to a thread involving external parties" — pause
and surface the role boundary. A "send a WhatsApp to Pratash with
these instructions" is the right shape; a "do this work as if I were
Pratash" is not.

The same rule applies to dictation that *names* the user in third
person ("Harish" for "Harsimran" in the user's voice, "Dharmesh" for
"DRA" etc.). Verify the intended recipient against the actual contact
record / Gmail thread, and produce a `whatsapp_link` for the
**real** person, not the STT-corrupted one.

### P8. Recipient identifiable but no WhatsApp number available — report the gap honestly

A common pattern: the user says "let's jointly call [Person]" or "send a
message to [Person]" where you can figure out who the person IS (from
Gmail threads, context, or memory) but have no WhatsApp number for them.
This happens with landlords, government officials, and external vendors
whose contact info lives only in email correspondence.

**What to do:**
1. Report what you found — name, email, role, project context.
2. Do NOT fabricate or guess a phone number. The user won't discover the
   broken link until send-time and will be frustrated.
3. For the task at hand: generate the link for the people whose numbers
   you have, and flag the gap for the person without a number. Ask the
   user to provide it, or note that you found their email if that helps.
4. If the person is someone the user will CALL (not WhatsApp), note that
   you don't have their number in your directory and ask the user to
   supply it.

**Session example:** Nishant said "make a joint call to Akbar as well"
for Aamir's message. You identified Akbar as Akber Hussain (AH Group,
akber@ahindia.com, Millers Road landlord) from Gmail threads, but no
phone number exists in Contacts, the Sheet, honcho memory, or email
signatures. Correct action: generate Aamir's link with the full context,
flag that Akbar's number isn't available, and ask the user for it.

### P9. "The URL I'm providing below" almost never arrives in the voice transcription (2026-08-29)

NDR frequently dictates: "Using the URL I am providing below, please… [draft the message with it]" or "I'm also sharing my model / the link". **The URL is NOT in the transcription** — voice STT drops bare URLs. Observed 2026-08-29: dictated "using the URL I am providing below… I'm also sharing my model"; no URL arrived; the assistant guessed a Drive spreadsheet (Palya_2A6G_LandCost_Realisation_Model_V2) and baked it into the WhatsApp link; the user corrected with the real URL (newlaunches-devanahalli.com row-houses page) — one wasted regenerate and a moment where the wrong link existed in the pipeline.

Rule:
1. When the dictated message references a URL/link/model "provided below" / "I'm sharing" / "the link is", check the message for an actual URL. Voice almost never carries it.
2. Do NOT substitute a guessed or related link (a Drive file you found, a model you think is right, an old link from a past session). A wrong link inside a pre-filled WhatsApp message is worse than a 30-second clarification — the recipient gets a broken or misleading reference, and NDR may send it without checking.
3. Draft the message text fully, embed a `[URL]` placeholder, then ask: "the URL didn't come through in the voice message — paste it and I'll regenerate the link in one step." If you have Drive candidates, list the top one as a suggestion but do NOT bake it in without confirmation.
4. Exception: if the user explicitly names a known file ("the Palya IRR model on Drive"), using that file IS authorized — but still confirm the exact one when several candidates match (Drive search for `name contains 'Palya' and (IRR or Model)` returns 5+ lookalikes).

### 6. Fallback: SMS via `sms:` URI

If WhatsApp is unavailable or the recipient is on a non-WhatsApp number,
fall back to a standard `sms:` URI. This is a hand-constructed URL —
there is no special tool, and ampersand handling is not an issue
because sms: URIs do not use `&` as a parameter separator (they use
`?body=` with a single body parameter).

```python
import urllib.parse
phone_digits = "919845070013"
body = cleaned_message
sms_url = f"sms:+{phone_digits}?body={urllib.parse.quote(body)}"
```

On Android the user gets a chooser; on iOS the URL opens Messages.app.
Report as a markdown link the same way as a wa.me link.

## 7. Fallback: mailto: for email-only follow-ups

When the user wants a "send a message" but the recipient is email-only
(landlords, government, regulators), use a `mailto:` URI. No special
tool needed; the same encoding rules apply (rewrite `&` in the body to
"and" first to be safe across email clients that may mis-split on `&`).

```python
import urllib.parse
mailto_url = (
    f"mailto:{recipient}?subject={urllib.parse.quote(subject)}"
    f"&body={urllib.parse.quote(body)}"
)
```

Note: `mailto:` legitimately uses `&` as a parameter separator (e.g.
`?cc=...&bcc=...`). The rewrite rule applies to `&` **inside the body
or subject**, not to the `&` between parameters — that one is correct
and required.

## 8. Verification

After presenting the link, do a quick sanity check:

1. The link starts with `https://api.whatsapp.com/send?` (or `sms:+` /
   `mailto:`). A `wa.me` URL is a sign of a stale/legacy path — regenerate.
2. The source text you passed had no raw `&` characters.
3. For individual contacts, the phone number is in the URL
   (`...send?phone=<digits>&text=...`).
4. For group messages, the phone number is NOT in the URL (only `?text=`).
5. The phone number has the country code (91 for India, no leading `+`).
6. **Indian numbers: the `phone=` digits are EXACTLY 12 long** (91 + 10-digit mobile). 13+ digits means a stray digit slipped in — re-check the source before delivering (see P3, user-corrected 2026-08-31).
6. If the result has `"split": true`, deliver each `parts[i]` as its own
   separate Telegram message.
7. The recipient is named in the message body or the surrounding prose
   so the user knows who they are sending to.

## 9. References

- `templates/whatsapp-message-builder.html` — interactive editable HTML builder: user edits the rich-text message in-browser and regenerates the WhatsApp deep link himself (P5e). Copy, fill phone + message, deliver with MEDIA:.
- `references/voice-transcribed-project-research-legacy-cataleya.md` — worked example of resolving a voice-garbled project name ("Century Katalya" → Legacy Cataleya) through phonetic search, landmark matching, clarify, and multi-source research (RERA, broker sites). Implements P2.3.
- `references/web-research-via-terminal.md` — Wikipedia REST API and
  MediaWiki API via curl: structured text extraction without web_search.
  Use this for the "research a topic, then WhatsApp" pattern the user
  frequently bundles in a single voice message.
- `references/landowner-group-milestone-update.md` — the recurring "give all the
  landowners an update" broadcast: ingest the WhatsApp group export the user
  uploads FIRST (dates/sequence are critical), ground private money context from
  the landowner's Drive folder without ever leaking it into the group text, the
  approved six-block structure (milestone → meeting request → two objectives →
  in-person-only decision → what happens next → one commitment), the psychology
  rules NDR dictates, WhatsApp-only formatting traps (no underline, never `~`),
  and part-splitting delivery for a 2-3 link broadcast. Verified 2026-09-12 on
  Ranka Northstar Landowners.
- `references/approve-then-link-draft-approval.md` — the approve-then-link workflow:
  present the full draft text for approval BEFORE generating any link; on a section
  revision show only that section, then regenerate links for the whole message; expect
  3–4 correction rounds on high-stakes broadcasts; offer full + concise variants; deliver
  split parts one bubble each. (NDR-corrected 2026-09-19.)
- `references/ndr-followup-patterns.md` — 10 worked patterns from real
  sessions: broker/chief single-recipient, institutional group-message,
  WhatsApp + email pair, apology/delay-with-reason, hospital insurance,
  landowner progress update, and single-ask review/confirmation. Includes
  a cheat sheet of `&`-to-`and` rewrites and a pitfall for resolving
  DRAAS colleagues not in Google Contacts. Use this for the
  follow-up-message patterns that NDR uses most often.
- `references/contacts-sheet-naming-edge-cases.md` — parenthetical name suffixes and employee-style contacts that block People API / contact_resolver. When resolver returns only same-first-letter false matches, fall back to direct sheet search.
- `references/voice-firm-name-gmail-discovery.md` — resolve a voice-garbled FIRM name
  ("Balan & Nambisar" → Balan + Nambisan Architects / BN Architects) when contacts sheets
  return nothing: search the raw voice spelling in Gmail, read NDR's own narration emails for
  the spelled-out name, search the email domain to enumerate the team, and extract
  phone/address from signatures. Includes the BN Architects contact card (Janice Rodrigues
  lead; HAL 2nd Stage; +91-80-25217543/44).
- `references/festival-greeting-broadcast-excel.md` — festival/forgiveness broadcast pattern
  (verified Sep 2026, Samvatsari): surname-scan the contacts sheet → Gmail last-touch
  recency → explicit-name expansion (voice→contact map incl. Mardia/Sisodia/Rakeshji
  Showoff/Kamal Bhavesh Surat) → relative exclusion → one script for all links → 3-tab
  clickable .xlsx with verification. Also carries the mailbox-identity pitfall
  (HERMES_SESSION_USER_ID=ndr + getProfile check when the request is NDR's but the session
  user is a colleague).
