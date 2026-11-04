---
name: whatsapp-drafter
description: |
  Drafts WhatsApp messages for any contact. Resolves contact from the Google Contacts
  Sheet (primary), presents all phone numbers, drafts message with correct tone/format
  (work vs personal), and generates wa.me deep-link URLs using the whatsapp_encode
  tool. NEVER encode URLs manually.
  If the contact is not in the sheet (e.g. personal relations: spouse, family), ask
  the user for the mobile number directly — do not search further.
  Trigger: "WhatsApp [name]", "WA [name]", "send [name] a WhatsApp", "message [name] on WhatsApp"
metadata:
  hermes:
    tags: [whatsapp, messaging, contacts, communication, draft, wa.me]
    linked_files:
      - references/google-contacts-sheet-access.md
      - references/whatsapp-url-encoding-research.md
      - references/google-sheets-api-read-write.md
      - references/bharat-hawaldar-contact.md
      - references/important-information-drive-doc.md
      - references/drive-contact-resolution.md
category: communication
version: 1.0.0
author: ndr@draas.com
---

# WhatsApp Message Drafter

> **Prakash (psingh@draas.com) specifics — see `references/prakash-draft-review-whatsapp-api.md`:**
> he insists on reviewing the full draft text before sending (fetch the real draft from Gmail,
> never reconstruct from memory); when he says he'll send it through his **own WhatsApp API**,
> give the raw message body only (no deep link); image captions cap at **1024 chars** — report the
> count and warn that edits overflow it.

## 1. Trigger Conditions

Activate when the user says anything like:
- "WhatsApp Manohar", "WA Raghu", "send a WhatsApp to Bhavesh"
- "message Nishant Prakash on WhatsApp about..."
- "draft a WhatsApp for the engineering group"

Noun resolver has already corrected the contact name before this skill sees it.

---

## 2. Stage 1 — Contact Resolution

> **🔴 Never invent an address or handle — including for a calendar invite or email you
> attach to the same task.** NDR's standing rule (2026-10-14): *"any email must be looked
> up through contact."* The Sheet is **not** authoritative on its own — it carries stale
> and phantom addresses, and a corrupted row can hold several addresses in one cell. For
> anything that carries an **email address** (calendar attendees, Drive shares, cc lists),
> also query online Google Contacts via People API and state which source each address
> came from. Rule, worked failure case (Roshini Ranka / `rivaanranka@gmail.com`) and the
> lookup recipe → `references/contact-email-resolution.md`.

**Primary: try the Google Contacts Sheet.** If the user provides a sheet URL (or you already know the sheet ID from references), use the Sheets API v4 with OAuth — see `references/google-contacts-sheet-access.md` for the working Python recipe.

**Secondary: ask the user for the mobile number.** If no sheet access is available, the contact is not in the sheet, or the user is asking about a personal relation (spouse, family) not in the contacts — ask directly. This is faster and more reliable than searching further.

**Sheet access workflow (when a URL is shared):**
1. Extract the sheet ID from the URL (`/d/{ID}/`)
2. Use `tools.gws_auth.build_service('sheets','v4',service_name='google-draas')` — get the key from `gws_resolve_account`. **Never** read a token JSON from disk: no `oauth-draas.json` exists any more; per policy OAuth tokens live only inside the `gws-vault` daemon and are reachable only through `build_service()` / the native `gws_*` tools
3. URL-encode the sheet tab name with `urllib.parse.quote()` — tabs often contain spaces/dots that break bare URLs. For the contacts tab the real title is `NDR DRAAS Google contacts.csv` (**not** `contacts`) — column map + fetch snippet: skill `contact-phone-lookup` → `references/contacts-sheet-tab-and-column-map.md`
4. Search columns A (First Name) and C (Last Name), then fetch full rows for phone numbers
5. Present matches with all available phones for the user to confirm
6. For writing/updating cells in this spreadsheet, see `references/google-sheets-api-read-write.md`

**If the contact is not found in the sheet**, or if the user says "my wife" / "my husband" / "my brother" / "brother-in-law" / "sister" / "mother-in-law" / any personal relation not in the business contacts: skip the sheet search entirely and ask the user for the mobile number. Do not search relation/spouse fields, do not cross-reference other rows, do not attempt to reverse-lookup via Drive, Gmail, or People API.

**Phone number from email signature — check before asking the user.** If the contact is a surveyor, vendor, or anyone who has emailed the user, their phone number is often in their email signature. Before falling back to asking the user:
1. Search Gmail for emails from this person (`from:contact@email.com`)
2. Get the full body of their latest email
3. Extract the phone number from the signature block (look for `Mob:`, `Mobile:`, `Phone:`, `Tel:` followed by digits)
4. Present the found number to the user for confirmation: "Found [name]'s number from their email signature: [number]. Send WhatsApp there?"

This was confirmed in Jun 2026: Venkatesh G's email signature contained `Mob: 9611501955` — extracted and used to generate the WhatsApp link without asking the user for the number.

**When the contact is not in the sheet and no email signature found — ask directly, no further search.** This is the expected path for: lawyers, personal physicians, family friends, and other professional relations not stored in the business contacts sheet. Do not search Drive, Gmail People API, or other sources beyond the email signature check. Just ask the user for the WhatsApp number. Example: "B.R. Krishna (lawyer)" — not in contacts sheet → asked user for the number directly. No search, no delay.

**Also update — group messages:** When the user explicitly says "it's a WhatsApp group message", skip contact resolution entirely. Encode the message and return the wa.me link directly.

**Critical pitfall — family relations and voice input:** When the user describes someone by relation ("my brother-in-law", "my sister", "my mother-in-law"), do NOT attempt to find them in the contacts sheet, Drive, or Gmail. These relations are often not in the business contacts. Ask the user directly for the mobile number. STT frequently renders Indian names phonetically ("Ranjit Rathur" was the STT form — the actual contact may not exist under that spelling in any system). If the user provides a name that yields no contacts match, confirm whether they meant a family relation before expanding the search.

**Specific hazard — "my wife" / personal relations:** Never search the sheet's relation fields (e.g. "Relation 1 - Label = Wife") to find a spouse. Those fields describe how a contact relates to OTHER people, not who is married to the user. A contact with "Relation 1 - Label: Wife" means that contact IS someone's wife (to their listed spouse), not that they are the user's wife. Searching this way will return the wrong person (e.g. "Roshini Singh" with spouse "Anshul Singh" is NOT the user's wife Roshini Ranka — she just has a husband listed). **Always search by the person's actual First Name + Last Name.** The user's spouse may share a first name with another contact in the sheet (two "Roshini" entries exist: Roshini Ranka = wife, Roshini Singh = unrelated Earth School mother). Name collision is a real risk.

**Pre-search before drafting:** Before generating a WhatsApp link for a personal contact ("my wife", "my husband", etc.), check if the contact name matches an existing user memory entry. If the user has already told you their spouse's name, cross-reference against the sheet to find the correct row. Do not rely on relation field searches — find the person by their actual name.

**Also:** before drafting, confirm the canonical name from the sheet matches the user's target. If the user says "my wife Roshini" and the sheet has two Roshinis, verify which one before proceeding. If uncertain, ask the user to confirm which entry or provide the mobile number directly.

**Do NOT:**
- Import `contact_resolver` as a Python module — it doesn't exist; you will get `ModuleNotFoundError`. Call the **tool** `contact_resolver` instead (it already returns the candidate rows with phones, emails and org); `entity_resolver` for projects/entities/land
- Use CSV export URLs (`/export?format=csv`) — returns 401 without credentials
- Pass raw (unencoded) sheet names in API URLs — returns 400 Bad Request
- Search relation fields first to find a person's entry — a contact may NOT have a relation label filled in. The contact's own name fields (First Name + Last Name) are the primary search key. Relation fields should only be searched to find *dependents* or *relatives* of a known contact (e.g., "find all contacts whose relation label is 'wife'"). If you search relation fields for "wife" to find someone's spouse, you may get the wrong person entirely if multiple contacts have that relation label.

**Contact resolution priority:**
1. **Primary**: Search by First Name (col A) + Last Name (col C) for exact or near-exact match
2. **Secondary**: Search by full name string across all text fields only if name fields don't yield results
3. **Last resort**: Ask user for the mobile number directly

Never return a phone number from a contact whose name doesn't match the user's target — even if the relation field looks relevant. A mismatch between the user's target name and the contact's actual name is a hard stop.

---

---

## 3. Stage 2 — Draft the Message

### CRITICAL: Always use the canonical contact name

After Stage 1 resolves the contact, you have two names:
- The **STT/user-provided name** (e.g., "Narsem Raju sir", "narsimha raju") — what was heard or typed
- The **canonical name** from the contact sheet (e.g., "Narasimha Raju" from Col A + Col C)

**Always use the canonical name from the sheet in ALL parts of the message — including any salutation, body reference, or closing.** Never use the STT or user-typed version in the drafted message.

Before calling `whatsapp_encode`, scan the draft for any instance of the user-provided / STT spelling and replace it with the canonical name. For salutations / addressed-as, use Col I (nickname) if present, otherwise Col A (first name only).

---

### MANDATORY: Bold Caption on Every Work Message

**Every work message — without exception — must have a bold caption as the very first line.** This is not optional formatting; it is the user's consistently stated preference. If the message relates to a project, land deal, entity, employee, vendor, legal matter, marketing task, or any business topic, it gets a bold caption.

**Format:** `*[Topic or Project Name]: [Concise Subject]*` — placed as the first line of the message, before any salutation.

**Examples of caption formats:**
- `*275 Acres — Lakshmi Pura, Bidadi — SC Verdict*`
- `*Structural Stability Certificate — Urgent — RERA Closure Blocked*`
- `*Bidadi Smart City — Follow Up*`
- `*TerraGreens Alipur 300: FW Agreement & MOU Uploaded*`

If the user has given you a specific caption in their instructions, use that exact wording. If not, derive it from the topic and subject matter.

### Work message (use when topic relates to projects, land, entities, accounting, legal, marketing, operations — to employees, vendors, partners)

**Format:**
```
*[Topic/Project]: [Subject]*

[Message body — direct, professional, no pleasantries]
```

### Advocate / legal counsel meeting request (new sub-pattern, June 2026)

Use when the user asks for a WhatsApp message to an **advocate, lawyer, or legal counsel** they have an existing email thread with, asking for a **meeting to share final documents** and get an opinion. Distinct from generic work messages:

- Contact is almost NEVER in the business contacts sheet, People API, or Drive — they are external professional counsel. **Skip contact resolution; ask the user for the number directly.**
- Tone is **deferential-professional**: use "Sir" suffix (Indian advocate convention), formal "Namaskar" or "Good morning" opener, no caption line (this is a peer-to-peer legal request, not a project task).
- Reference prior email context: "Hope you and the team at [Firm Name] are doing well" — confirms continuity without restating the email thread.
- Be specific about what is being shared: "final documents pertaining to [property/matter]" — don't dump document names in the WhatsApp message; the documents speak for themselves in the meeting.
- Make the ask explicit and time-bound: "I would like to meet you *today*" with a flexibility signal ("flexible and can come to your office or host a call").
- **Always ask the user for these 5 things before generating the link:**
  1. The phone number (raw 10-digit Indian mobile, no +)
  2. Advocate's name spelling — confirm what user said vs canonical (e.g. "P R Krishna" vs "PR Krishna" vs "P.R. Krishna")
  3. Firm name spelling — confirm ("Jain Patan Chetty" vs "Jain, Patan & Chetty")
  4. Preferred time today — or default to "flexible / your convenience"
  5. Location preference — his office, your office, or video call
- **Do NOT search Drive/Gmail/People API** for advocate contact details. They won't be there. The user has the number from a prior email thread — just ask for it.

**Template (copy and customise):**
```
*Adv. [Full Name]* Sir, Namaskar.

This is [Your Name]. Hope you and the team at *[Firm Name]* are doing well.

I have with me the *final documents pertaining to [property/matter name]* (including [1-2 key document types, e.g. "the executed Family Arrangement Deed and the connected loan / allocation letters"]). I would like to meet you *today* at a convenient time to go through them and get your opinion on the next steps.

Kindly let me know what time works for you — I'm flexible and can come to your office or host a call. I'll carry both the originals and a soft copy.

Thank you, Sir.

Regards,
[Your Name]
```

**Pitfall — don't apply the project-caption rule:** The bold-first-line caption rule (`*Project: Subject*`) is for internal work messages to DRA team / vendors / partners. Advocate requests are a different register — no caption, salutation comes first, deferential tone throughout.

**Pitfall — don't list document names in the WhatsApp message:** A long enumerated list of 8+ FSA documents would turn this into a checklist rather than a meeting request. The meeting IS the request; the documents are the in-meeting material. If the user wants to share a doc index, do that as a follow-up attachment or PDF, not the WhatsApp message body.

**Pitfall — Drive fullText search won't find the advocate's contact details:** When you search Drive for "PR Krishna" + "Jain Patan Chetty", you get zero hits even though the user has exchanged emails with them. The advocate's details live in the user's email thread (Gmail), not in their Drive documents. Don't waste tool calls looking in Drive — just ask the user for the number. Confirmed this session (June 2026): 0 hits across Drive fullText and Gmail (token scope issue blocked Gmail anyway — but Drive hit zero even with proper fullText search).

### Polite persistence follow-up (when contact has been unresponsive)

> **Variant — nudging on an email that went unanswered.** Trigger: *"check if he has responded
> to that email; if not, the same contents, referring to it — tell him maybe you are not
> checking the mail, so sending a message as well."* Full recipe (two-account Gmail search,
> proving "no reply" from thread `labelIds`, cross-thread check, the nudge shape, the
> don't-invent-new-asks rule) → `references/unanswered-email-to-whatsapp-nudge.md`.

Use when the user wants to follow up after silence — someone who hasn't replied to earlier
messages, and the user wants to be respectful but firm about needing a response. This is a
different register from the standard direct work message. The tone is warm-professional,
acknowledges the silence without blame, and makes a concrete request (call, update, next step).

**Format:**
```
*[Topic — Follow-Up]*

Good [morning/evening] [Title/Name] sir/madam. [Brief acknowledgment if appropriate.]

[One sentence: where things stand or what prompted this follow-up.]

[One sentence: what the user wants — a call, an update, direction on next steps.]

[One sentence: availability / openness to their convenience.]
```

**Rules for this register:**
- Acknowledge the other person's position ("I fully understand there are complexities") without
  being apologetic to the point of weakness
- State the specific ask clearly but without pressure — "I would very much like to speak with
  you today if at all possible" rather than "please respond ASAP"
- Close with an availability signal — puts the ball in their court without being passive
- No numbered tasks in this register — the goal is a conversation, not a checklist
- Drop the directness level slightly vs. standard work messages; persistence without frustration

**Format:**
```
*[Topic/Project]: [Subject]*

[Message body — direct, professional, no pleasantries]

[Numbered tasks if any:]
1. [Task description] — by *[deadline/time if given]*
2. ...

[Key data points as bullets if any:]
- [Point]
- [Point]
```

Rules:
- **First line MUST be bold caption** — `*Bold Caption Here*` — before the salutation
- No greeting ("Hope you're well", "Hi, how are you") UNLESS user explicitly asks
- Tasks → numbered list
- Deadlines/times → `*bold*`
- Keep it direct and functional

**Example:**
> *Ranka Oasis: Site visit confirmation needed*
>
> Please confirm availability for the site visit this week.
>
> 1. Confirm date — by *Wednesday 5pm*
> 2. Arrange access to the south plot
> 3. Send updated survey report before visit

---

### Personal / casual message (non-work: news, social, articles, non-project topics — to friends, family)

**Format:**
- No caption line
- Natural, warmer tone
- Still NO boilerplate opener ("Hope all is well", "Dear X") unless user explicitly asks
- Lists / bullets only if content naturally calls for it

**Special rule — Roshni Ranka (alias "RO"):**
Always use personal tone regardless of topic.

---

### WhatsApp formatting reference

WhatsApp (mobile app) supports limited markdown:

| Effect | Syntax | WhatsApp renders? |
|--------|--------|-------------------|
| Bold | `*text*` | Yes |
| Italic | `_text_` | Yes |
| Strikethrough | `~text~` | Yes |
| Monospace | `` `text` `` | Yes |
| Numbered list | `1.` prefix | Yes — plain numbered items render readably |
| Bulleted list | `-` prefix | Yes |

**Numbered lists in WhatsApp:** WhatsApp does not support true markdown ordered lists, but plain `1.` `2.` `3.` prefix syntax renders as readable numbered items on all WhatsApp platforms. This is the recommended approach when the message contains a list of documents or tasks. Do NOT use HTML or CSS workarounds.

**When a document list spans two sections (e.g., a master list + a sub-list):** Present each section with its own header/label, then the numbered items. Do not merge or deduplicate — if the same number appears in two sections with different items, keep both sections separate and label them clearly (e.g., "Complete mandatory documents list:" followed by items 1-19, then "List 9-19 — specific technical documents outstanding:" followed by items 9-19 starting at 9).

---

### Present the draft

Show the draft in a code block so the user can review it cleanly:

````
*Ranka Oasis: Site visit confirmation*

Please confirm your availability for a site visit this week.

1. Confirm date — by *Wednesday 5pm*
2. Arrange access to the south plot
3. Send updated survey report before the visit
````

Ask: "Looks good? If yes, which number — mobile, work, all numbers, or no number (group)?"

---

## 4. Stage 3 — Generate Link on Approval

Once the user approves the draft (or approves with minor edits), encode and generate the WhatsApp link.

**Platform availability check first:** If `send_message(action='send', target='whatsapp:...')` returns `Platform 'whatsapp' is not configured`, WhatsApp is not wired into this Hermes instance. Tell the user: WhatsApp isn't connected in Hermes right now — only Telegram is available. Ask which platform to use, or whether the user wants the deep link to open manually on their phone.

**Telegram limitations:** Telegram does not resolve phone numbers in `send_message` — `telegram:+91XXXXXXXXXX` will fail with "Could not resolve". Telegram requires a chat ID (numeric, e.g. `sales1.blr`) or a Telegram username. If the user provides a phone number for Telegram, ask for their Telegram ID or handle.

**For HTML cards:** When delivering the HTML card via Telegram, simply write the file to `/tmp/` and include `MEDIA:/tmp/filename.html` as a separate message. Do NOT pass it through `send_message` — Telegram's file/document mechanism preserves the clickable button inside the HTML. The user taps the button inside to open WhatsApp. This is the confirmed-working pattern (used for the Bajaj Allianz policy summary this session). The format must be `telegram:<chat_id>` not `telegram:+91...`.

### Long messages — ALWAYS use HTML card, not plain URL

**Confirmed problem (user-reported this session):** When the encoded WhatsApp URL is pasted as plain text in Telegram, Telegram splits it into multiple messages above ~4,096 characters. The URL arrives broken in WhatsApp — user reports "it keeps splitting" and "incomplete message."

**Proactive trigger — do NOT wait to be told:** If the message body is long (500+ words, 4+ substantive paragraphs, multi-section legal/financial content), generate the HTML card IMMEDIATELY. Do not generate a plain deep link first and wait for the user to report it got split — that wastes a turn and the user has to repeat themselves. Always estimate encoded length before choosing delivery format.

**Rule — no exceptions for long content:** If the encoded message is long (policy summaries, legal narratives, multi-section recaps, anything over ~2,000 chars encoded), always deliver via HTML card. The URL lives inside an `<a href>` in the HTML and never passes through Telegram's text-splitting pipeline.

**Delivery via Telegram:** Write the HTML card to a persistent path like `/data/hermes/cron/output/whatsapp-{contact}-{topic}.html` (e.g. `whatsapp-bharat-cancellation-deed.html`) and deliver as `MEDIA:/data/hermes/cron/output/whatsapp-...html`. Do NOT use `send_message` for the HTML file — send it as a Telegram file attachment.

**HTML card template (copy and modify):**
```html
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Message</title></head>
<body style="font-family:Arial,sans-serif;background:#f0f2f5;padding:20px;">
<div style="background:#fff;border-radius:12px;max-width:600px;margin:0 auto;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,0.1);">
<h2 style="color:#128C7E;margin-top:0;">[Subject]</h2>
<p>[Brief context line]</p>
<div style="text-align:center;margin-top:20px;">
<a href="https://api.whatsapp.com/send?phone=91PHONE&text=ENCODED_MSG"
   style="display:inline-block;background:#128C7E;color:#fff!important;text-decoration:none;
          padding:14px 28px;border-radius:8px;font-weight:bold;font-size:16px;">
   Open WhatsApp
</a>
</div></div></body></html>
```
Full recipe in `references/whatsapp-url-encoding-research.md`.

**Short messages (~2,000 chars encoded or less):** Plain deep link is fine — use wa.me or api.whatsapp.com URL directly in the chat.

### Delivery format preference — clickable hyperlink, not code block

**User preference (Nishant R, June 2026):** When presenting the final WhatsApp link, **do NOT wrap it in a code block**. Code blocks render as plain text on Telegram and the user cannot tap to open. Present the link as a **direct clickable hyperlink** in the chat — no backticks, no inline code formatting.

This applies to ALL WhatsApp link deliveries: wa.me links, api.whatsapp.com links, and any URL the user needs to tap to open. The only exception is when the user needs to copy-paste the message body itself (not the link) — in that case, show the message text in a code block, with the link separately as a clickable hyperlink.

### Message text draft (not deep link) — code block is correct

**User preference (Nishant R, June 2026, updated Jun 10):** When the user asks for a WhatsApp message draft to be presented as **plain text for copy-paste** (i.e. they will manually paste it into WhatsApp themselves, not use a deep link), deliver it as a markdown code block in Telegram. This lets them copy the entire message body cleanly with one tap. Do NOT generate an HTML card or deep link unless asked.

Trigger: user says "present as markdown code block so I can copy paste", "just give me the text", "I'll paste it myself", or similar.

Rule: short message drafts → code block. Long/complex messages (500+ words, multi-section) → still default to HTML card per the long-message rule above.

### WhatsApp URL format — api.whatsapp.com (RECOMMENDED DEFAULT) vs wa.me

**api.whatsapp.com (RECOMMENDED DEFAULT):** Does not have the double-decode problem. Use for all messages, especially complex/long ones. Format: `https://api.whatsapp.com/send?phone=91{PHONE}&text={ENCODED}`. No fullwidth hack needed.

**wa.me (fallback):** Has a double-decode problem — `%26` becomes `&`, `%3D` becomes `=`, breaking URL parameter parsing. Requires fullwidth character replacement hack. Use only if api.whatsapp.com produces garbled output.

**Fullwidth fix for wa.me (only when needed):** Replace `&` → `＆` (U+FF06) and `=` → `＝` (U+FF1D) before encoding. `urllib.parse.quote` encodes these to `%EF%BC%86` and `%EF%BC%BD`, which survive the double-decode.

**User preference (Roshini, 2026-05-27):** The user explicitly said `&` characters in the message body cause encoding breaks on mobile WhatsApp and must be replaced with a **fullwidth ampersand (＆)** — not double-encoding with `%2526`. When a message contains `&` in visible text (e.g., "BWSSB Water ＆ Sewage NOC"), replace `&` with `＆` before URL-encoding. This is the confirmed-working approach for this user and overrides the generic `%2526` approach.

**Recommendation:** Default to api.whatsapp.com for all messages. wa.me with fullwidth was NOT needed (used for Bajaj Allianz summary — clean). Default to api.whatsapp.com for all but the simplest messages.

**Link construction:**
```python
import urllib.parse
phone = '918050493448'  # no + prefix, country code prefix (91) included
message = """*Bold Caption*

Body text here."""
encoded = urllib.parse.quote(message, safe='')
link = f"https://api.whatsapp.com/send?phone={phone}&text={encoded}"
```

**Group message (no number):**
```python
import urllib.parse
message = "Hey Amir, Salman, any news from Devaya sir?"
message = message.replace('&', '＆').replace('=', '＝')
encoded = urllib.parse.quote(message, safe='')
link = f"https://wa.me/send?text={encoded}"
```

**Always use the Python recipe above** (fullwidth replacement + `urllib.parse.quote`) to construct wa.me links. Never hand-type or manually reconstruct a WhatsApp URL string.

---

## 5. WhatsApp URL Encoding

**⚠️ CRITICAL — Ampersand encoding (applies to ALL WhatsApp links, both wa.me AND api.whatsapp.com):**
Mobile WhatsApp WebViews (Android/iOS) incorrectly parse `%26` as a URL query separator and truncate the message. Fix: replace every `&` with `%2526` in the message body BEFORE URL encoding. This survives the WebView's incorrect single-decode pass.

**WRONG:**
```python
msg_encoded = quote(msg, safe="%0A%20")
# & → %26 — truncated on mobile WhatsApp WebView
**⚠️ CRITICAL — Ampersand encoding (applies to ALL WhatsApp links, both wa.me AND api.whatsapp.com):**
Mobile WhatsApp WebViews (Android/iOS) incorrectly parse `%26` as a URL query separator and truncate the message. Fix: replace every `&` with fullwidth ampersand `＆` (U+FF06) in the message body BEFORE URL-encoding. This survives the WebView's incorrect single-decode pass. The fullwidth character encodes to `%EF%BD%86`.

**WRONG — %2526 approach:**
```python
# ❌ BROKEN — %2526 double-encoding fails on WhatsApp mobile WebView
msg_fixed = msg.replace("&", "%2526")
```

**CORRECT — Fullwidth ampersand (verified working June 2026):**
```python
import urllib.parse

phone = '919900029200'  # no + prefix, prepend 91
message = """*Bold Caption*

Body text here & more."""

# Step 1: replace & with fullwidth ampersand (U+FF06) BEFORE URL-encoding
message = message.replace("&", "\uFF06")  # ＆

# Step 2: URL-encode
encoded = urllib.parse.quote(message, safe='')

# Step 3: build link — prefer api.whatsapp.com
link = f"https://api.whatsapp.com/send?phone={phone}&text={encoded}"
# & in URL params (phone=...&text=...) stays unencoded — it's not inside the message text
```

**Why this works:** `%EF%BD%86` (the URL-encoded form of `＆` U+FF06) survives WhatsApp WebView's incorrect single-decode. The `&` between URL parameters (phone=...&text=...) is NOT inside the message text, so it stays unencoded and the API parses it correctly.

**Confirmed working on:** api.whatsapp.com with messages containing company names like "O3 Infotech ＆ DRA Group", legal documents with "BWSSB Water ＆ Sewage NOC", and any `&` in visible text.

**Rule:** Every `&` in the message body — company names, legal references, any ampersand in visible text — must be replaced with `＆` (U+FF06) before URL-encoding. Never use `%26` or `%2526`.

**Phone number format:** `91XXXXXXXXXX` — no `+` prefix, no spaces, no dashes.

**api.whatsapp.com vs wa.me:** Default to `api.whatsapp.com`. wa.me has a double-decode problem that requires fullwidth character workaround (`＆` / `＝`) — use only as fallback.
| `phone` + `message` | Full deep link with number and pre-filled text |
| `phone` only | Deep link to number, no pre-filled text |
| `message` only | Opens contact picker with pre-filled text |

## 6. Pre-Draft Sanity Check (voice message inputs)

Voice transcription errors are common for names. Before generating a WhatsApp link, verbally (or in your head) read back the proposed message. If something sounds wrong — a name spelled oddly, a phrase that doesn't flow — it likely came from an STT mishearing. Common failures this session:

- "Roshni" vs "Roshini" — STT transcribed the user's phonetic input differently than the canonical contact name
- "Raghaz Nishant" → "Regards, Nishant" — garbled ending that required guesswork
- "Rada" → misheard fragment of a longer phrase

**Pre-draft readback rule:** After drafting, read the message aloud in your head. If a phrase sounds unnatural, awkward, or like a garbled name, it likely came from an STT mishearing. The user's approval is the final signal — but flagging confusion before generating the link is always better than sending the wrong thing.

**Group message — skip contact resolution entirely.** When the user explicitly says "it's a WhatsApp group message", do not ask for contact confirmation or try to resolve a contact name/number. Encode the message and return the wa.me link directly. Contact resolution is for individual 1:1 messages only.

**STT failure patterns observed (cumulative):**
- "Roshni" vs "Roshini" — STT transcribed the user's phonetic input differently than the canonical contact name
- "Raghaz Nishant" → "Regards, Nishant" — garbled ending of a WhatsApp sign-off
- "Rada" → misheard fragment (not a name to use in the message)
- Amar → Aamir (Amar Khan was actually Aamir Khan in contacts)
- Javad Benson Town → STT heard "Jawad Benzentown" — always confirm property/location names when the STT version sounds off
- Devaya → confirmed as Devaya (not a mishearing, but often confused with similar names in legal context)
- Bharat Hawaldar → "Havaldar" is the STT form of "Hawaldar" (DRA Pre-Sales Executive, Row 597); "Havaladr" / "Hawaldar" / "Havaldar" all resolve to the same contact. Do NOT confuse with "Babu Bharat Broker" (Row 522 — a different person entirely, broker, not a DRA employee).
- Aamir Khan → NOT the same as Bharat Hawaldar. When the user says "Bharat" in context of Allalsandra / North Star / DRA internal work, they mean Bharat Hawaldar (DRA employee), not Aamir Khan (external LLP contact). Treating them as interchangeable caused a misdirection in session 2026-05-04.
- Aamir Khan (Direct LLP) → user calls him "Ahmed Khan" in voice sometimes — the STT mishears Aamir as Ahmed. Resolve to Aamir Khan. Phone: +91 98458 81652. Known aliases: Ahmed Khan (STT artifact).
- Geography STT errors — location found but region wrong: STT correctly picks up the property/estate name but assigns it to the wrong state/region. E.g. "Serenity Estate" → STT heard "Chirchiganapalini" AND placed it in Bangalore, but actual location is "Chichuraganapalli, Tamil Nadu 635103". When the user corrects geography ("not Bangalore, it's Tamil Nadu"), this is a distinct failure mode from name misspelling. Save STT form → correct location mapping immediately.

**The workflow:** show draft in code block → ask "Looks good?" → wait for user approval → *then* generate link. Do not generate the link proactively.

**Pre-draft readback rule:** After drafting, read the message aloud in your head. If a phrase sounds unnatural, awkward, or like a garbled name, it likely came from an STT mishearing. The user's approval is the final signal — but flagging confusion before generating the link is always better than sending the wrong thing.

## 7. Contact Resolution Reference

**Date:** 2025-05-02

**Situation:** User shared a Google Contacts sheet URL (`docs.google.com/spreadsheets/d/1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`). Contact `Ashwin Pai` needed to be found.

**What failed:**
- `ModuleNotFoundError: contact_resolver` — module does not exist, do not attempt to import it
- CSV export URL (`/export?format=csv`) — returns 401 Unauthorized without credentials
- Bare sheet name in Sheets API v4 URL — returns 400 Bad Request because the tab name contains spaces and `.csv`

**What worked:**
- OAuth refresh (`/data/hermes/oauth-draas.json`) + Sheets API v4 with URL-encoded sheet name (`urllib.parse.quote("NDR DRAAS Google contacts.csv")`)
- Search columns A and C for name matches, fetch full row for phone numbers
- Full recipe in `references/google-contacts-sheet-access.md`

**Lesson:** The Google Contacts Sheet is accessible via OAuth + Sheets API v4. The sheet tab name must always be URL-encoded. People API is no longer the fallback path — use the sheet.

## 7. Rules Checklist

**DRA employees — which number to use:** For DRA employees (Bharat Hawaldar, Manjunath Licensing Engineer, etc.), prefer the `DRA` or `Work` labeled number. The user's instruction "his work number" is interpreted as the DRA Work number. Personal numbers are used only when the user explicitly asks for personal.

**Always:**
- Ask the user for the mobile number directly — this is the primary path in this environment
- Use `whatsapp_encode` from `/data/hermes/scripts/whatsapp_encode.py` — never encode manually
- Use `*bold*` for deadlines and key times in work messages
- Confirm the contact (name + number) before drafting
- Bold the caption line in work messages: `*Project: Subject*`
- Work vs personal: topic relates to a project, entity, land, or business relationship → work. Otherwise → personal
- **Roshni Ranka / "RO"**: always personal tone
- Group message: use `mode="text_only"` — no phone number in link

---

## 8. Sending through the bridge: draft → confirm

When the chat is on the user's auto-send list you can use `wa_send_message` directly. When it is not
(the normal case), the path is `wa_draft_reply` → show the text → `wa_send_confirmed`.

🔴 **The draft ID comes back SHORT and the send needs the FULL UUID** (confirmed 2026-10-17).
`wa_draft_reply` returned `345f1294`, but `wa_send_confirmed(draftId="345f1294")` failed with an
internal bridge error — while draft *listing* kept working, which makes it look like an outage rather
than a bad key. Sending with the full UUID
`345f1294-eeae-4c79-8492-dbf833f79079` went through immediately.

So: **before confirming, resolve the short id to the full UUID** via `wa_draft_status()` (no argument
lists pending drafts) or the draft list, then confirm with that. If a confirm fails, do NOT create a
replacement draft to "retry" — the original may still be live and you risk two messages. Check
`wa_draft_status(draftId)` for `state=sent` first, then decide.

## 9. Sharing documents with a contact over WhatsApp

NDR often wants files pushed to a vendor/agent ("send a WhatsApp to Jay with Mum's Aadhaar, PAN and
photo"). Sequence:

1. **Check whether this bridge can carry media** (`wa_send_media` present and working) before
   promising attachments.
2. **The path that always works: a narrow Drive folder + a link in the message text.** Create a
   small dedicated folder holding ONLY the files that person needs — never hand out the main case
   pack — set anyone-with-link → viewer, and put the URL in the message body.
3. **Place the folder inside the root TMP folder**, never at My Drive root (NDR's standing rule).
4. **Anyone-with-link cannot be given an expiry** through the Drive permissions API. So schedule the
   revocation explicitly: a `next-agent-action` row, or after the event has passed. Tell NDR the
   link stays open until that date — do not imply it self-expires.
5. **State what is NOT in the folder.** e.g. PAN carries no address — say so in the message so the
   recipient does not go looking for an address on it.

Say plainly in chat which mechanism you used ("the link is in the message; the files open in Drive")
so NDR knows he is not sending attachments from his phone.
