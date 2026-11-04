# Personal-contact WhatsApp drafting (absorbed)

> From the `messaging-drafts` → `references/personal-messaging-workflow.md` (archived 2026-09-20). Its references were re-homed into
> this skill's `references/`; its `find_contact.py` script is at `scripts/find_contact.py`.

name: personal-messaging
description: "When the user asks for a WhatsApp / SMS / message draft to a personal contact — colleague, partner, vendor, advisor — find the right person, honour user-vocabulary spellings from memory, compose the message, and deliver a wa.me / sms link via the `whatsapp_link` tool. Triggers: 'send a WhatsApp to [name]', 'message [name] on WhatsApp', 'draft a message to [name]', 'follow-up WhatsApp to [name]', 'text [name]', 'WhatsApp him/her'. Distinct from `email-drafter` (formal email, Gmail draft) and from `regulatory-complaint-escalation` (formal complaint to a regulated entity)."
version: 0.2.0
metadata:
  hermes:
    tags: [messaging, whatsapp, contacts, telegram, voice-memo, india, personal, dnd]
    category: productivity
---

# Personal Messaging

Class-level skill for the recurring pattern: user dictates a message to a personal contact (colleague, partner, vendor, advisor) and Hermes produces a wa.me link with the message pre-filled.

## When to load

- "Send a WhatsApp to [name]"
- "Message him/her on WhatsApp"
- "Draft a message to [name] about [topic]"
- "Follow-up WhatsApp to [name]"
- "Text [name]"
- User dictates a voice memo whose content is clearly a personal note to a named person
- **Medication procurement:** User asks to send someone (driver, family member) to buy medicine — load this skill AND the `references/medication-procurement.md` reference for the two-phase (verify then compose) workflow

This is **not** for:
- Formal email (use `email-drafter`)
- A regulatory complaint to an ombudsman (use `regulatory-complaint-escalation`)
- A message to the user themselves or a memory-only note

**Group messages:** The `whatsapp_link` tool does not support group delivery (no phone parameter targets a group), but you CAN draft a message for the user to paste into a group. Call `whatsapp_link(text=message_text)` without the phone parameter — this returns a wa.me link with the message properly URL-encoded and no pre-set recipient. The user taps the link, WhatsApp opens with the message pre-composed, and they paste it into the target group. Use this pattern when the user says "post this on the Creative Group", "send to the group", or addresses multiple people in one message.

**Collective recipients ("Accounts Guru", "accounts team", "accounts group") are a GROUP, not a person (corrected 2026-08-01):** When the user says "message for my Accounts Guru" / "share on the accounts group" / "send to the accounts team", the recipient is a **WhatsApp group** containing Eshwari and other accounts members — NOT a single contact. Do NOT resolve it to a person's phone number (the find_contact.py → Eshwari path is wrong here). Two consequences:
1. Generate the group-paste pattern link (`whatsapp_link(text=...)` with NO phone) so the user can paste into the group.
2. Address the message to the group collectively ("Accounts team,") not to one person.
If the user names a person AND says group ("for Eshwari and the accounts group"), the group-paste pattern still applies — address it to the group, mention Eshwari inside the body if needed. When unsure whether a collective noun names a person or a group, ask once rather than assuming a phone number.

## Workflow (5 steps, in order)

### Step 0 — Load this skill

Before touching any messaging tool, call `skill_view(name='personal-messaging')`. This skill encodes the user's preferred WhatsApp style, contact-finding workflow, and the single sanctioned tool for building wa.me URLs. Skipping Step 0 is the #1 source of user corrections in this domain.

### Step 1 — Identify the recipient

- If the user names the person, search memory for the contact card: name, phone, nickname, role, relationship. Memory frequently holds these (Aamir Khan, Sahabji/Manohar, Sahabji's phone 9845890316, etc.).
- **ALWAYS verify the phone number in Google Contacts before generating a wa.me link.** This is a hard user preference (corrected 2026-07-16). Even if memory has the person's name, Google Contacts may have a more complete or updated number. The user expects the link to include the phone number, not just the message text.
- **Contact lookup priority chain (HARD RULE — tool runs FIRST, always; corrected 2026-07-31):**
  1. **`scripts/find_contact.py` — ALWAYS FIRST, in the same session, immediately.** Run `/opt/hermes/.venv/bin/python /data/hermes/skills/productivity/personal-messaging/scripts/find_contact.py "<name or number>"` (or `cd /data/hermes/skills/productivity/personal-messaging && ...`) for ANY telephone or email lookup — WhatsApp recipients, calendar attendees, email drafts, vendor research, anything. **Use the full absolute path**: running `scripts/find_contact.py` from the default cwd fails with `can't open file '/opt/hermes/scripts/find_contact.py'` because the script lives in the skill directory, not `/opt/hermes/scripts/`. The script queries Google People API AND both DRA contacts sheets in one shot and auto-recovers the vault socket. Never answer from memory, never guess, never skip ahead. If it finds the person, use its labelled phones (prefer Work / Phone 1) and emails.
  2. **Memory** — after the tool run, cross-check for nickname/spelling/role context (Aamir, Sahabji, Bharat, Gowri, etc.). Memory supplies vocabulary and relationships, NOT ground-truth phone numbers.
  3. **Google Contacts (People API)** — direct `people().searchContacts(query=name, readMask='names,phoneNumbers,emailAddresses')` if the script's People section failed or you need a fresh query variant. If that returns nothing, also try `people().connections().list(personFields='names,phoneNumbers')` — searchContacts has limited field indexing and often misses contacts with comma-heavy or descriptive names.
  - **Email-prefix search is a powerful fallback.** searchContacts sometimes returns ZERO results for a display name that clearly exists (observed Jul 2026: "Roshni" and "Roshni Ranka" both → empty, but query `"rnr"` — her email login prefix — found Roshini Ranka with all phones/emails). If a name query fails, retry with the person's email-prefix/local-part (e.g. `rnr`, `psingh`, `bk`) — searchContacts indexes the email address even when the display-name match misses.
  4. **NDR DRAAS Contacts Sheet** (Sheet ID: `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`, sheet name `"NDR DRAAS Google contacts.csv"`) — search across all columns for name or phone when People API returns no match (the script already does this; run it directly if you need custom columns). This sheet has enriched metadata (role, company, department, notes) that the People API doesn't expose.
  5. **NDR CONTACTS sheet** (Sheet ID: `1fYa-t2RY1siy2qBgAH8uu_Jd2chjJ716BbcpxilpOK0`) — a smaller contacts spreadsheet with columns: SL.NO, NAME, COMPANY, DESIGNATION, ADDRESS, TELEPHONE, FAX, MOBILE, E-MAIL, WEBSITE, NOTES. Good fallback for traditional business contacts (lawyers, consultants, vendors). Has ~200+ entries. Search its NAME and MOBILE columns via Sheets API.
  6. **Employees sheet** — within the NDR DRAAS spreadsheet (id `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`), an `employees` tab stores DRA Group staff & family records (name, email, phone, role, Telegram ID). **`find_contact.py` does NOT search this tab** — if the script comes back empty for someone who should be on the org list, query it directly:
     ```python
     svc = build_service('sheets', 'v4', service_name='google-draas')
     res = svc.spreadsheets().values().get(spreadsheetId='1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g', range='employees').execute()
     # rows like: ['Roshini Ranka', 'rnr@draas.com', '+919845026390', 'admin', '7245204091']
     ```
     Verified 2026-09-10: `find_contact.py "Roshini Ranka"` → no People API match, sheet search garbled (false row), yet the `employees` tab had her in seconds (phone +91 98450 26390, role admin). Also check `references/dra-realty-group-roster.md` FIRST for org members — Roshini's row lives there too. Rule: for DRA staff/family, an empty find_contact.py is NOT "not found" until you've checked the roster reference AND the employees tab.
  7. **Gmail thread search** — when the person isn't found in any contact source but the user has exchanged emails with them before, search Gmail for the person's name. This is especially useful for professional contacts (advocates, consultants, vendors, government officials) whose email was shared in past conversations. Use the Gmail API to search for the person's full name (try multiple variants) and extract email addresses from From/To/Cc headers of matching threads. The Gmail API workspace is available via `/opt/hermes/.venv/bin/python` with `tools.gws_auth.build_service("gmail", "v1", service_name=...)`. For multi-step queries, write a standalone `.py` script to the temp directory and execute it with the venv Python — inline shell quoting with the `-c` flag tends to break on dict expressions (`m["id"]` → `m[id]` shadowing builtin `id`), and the `execute_code` sandbox doesn't have googleapiclient.
  8. **Manual People-API search in google-ahfl when find_contact.py returns nothing** (corrected 2026-08-14) — `scripts/find_contact.py` queries NDR's People API (google-draas) and the contacts sheets, but NOT the `google-ahfl` vault account. Government / phone-added contacts can exist ONLY there (observed: **Nagarajappa JDTP North was invisible to find_contact.py; found only via `build_service('people','v1',service_name='google-ahfl')` + searchContacts**). If a name search comes up empty, run a quick script searching google-draas AND google-ahfl (AND google-gmail) before concluding "not found" — see `messaging-drafts` → `references/messaging-links-workflow.md` for the multi-account rule.
  8a. **Recover a phone from a PAST session's generated WhatsApp link (2026-08-27)** — when the person is nowhere in contacts/sheets (dev vendors, agency leads, one-off collaborators), but Hermes has generated a WhatsApp link for them in a prior session, run `session_search(query="<name> WhatsApp")` — the old transcript contains the `api.whatsapp.com/send?phone=<digits>` URL; extract the digits. This is authoritative if it's the number NDR used recently, and the same search often also surfaces the recipient's email + email-subject used in parallel outreach. Example: **Devansh Goel** — not in Google Contacts or either contacts sheet (contact_resolver returned only same-first-letter false matches like "D Contat"); number +91 70657 03131 recovered from the Aug 21 pre-sales outreach session's generated link, email devanshgoel112233@gmail.com from the Aug 21 sent-mail thread. When reporting, note "let me know if the number has changed" since it came from session history, not a live contact source.
  9. **Ask the user** — if all sources fail, ask for the number/email. Do not invent or guess one.
- **Adding a NEW contact (NDR: "add X to my contact"):** dual-store write — (1) Google Contacts via People API `createContact` (names givenName/familyName, phoneNumbers mobile, emailAddresses work, organizations name+title, biographies → career notes); (2) append a row to the NDR DRAAS Contacts Sheet (id `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`, sheet `'NDR DRAAS Google contacts.csv'`) — columns observed: A=First Name, B=? C=Last Name, K=Organization Name, R='Work' label, S=email, AB='Mobile' label, AC=phone, CH=Nickname, CI=Notes. Verified 24-Aug-2026 on Munish Moudgil (People API `people/c5188874351235426364` + sheet row 4229).

- **Adding a NEW contact from a visiting card (business card) — full workflow (validated 2026-08-27, Ankush Musaddi):** When the user shares a visiting card image and says "update my contacts", the complete workflow is:
  1. **OCR the card** — `vision_analyze(image_url=...)` to extract name, org, title, phones, emails, address, website, RERA number.
  2. **Check if already in sheet** — fetch `'NDR DRAAS Google contacts.csv'` rows via raw Sheets API (`build_service("sheets","v4",service_name="google-draas")`), scan every row for the name — append only if new; update in place if existing.
  3. **Map to 93-column Google Contacts CSV schema** — the sheet has 93 columns (A..CM). Critical indices (0-indexed in a Python list):
     - `[0]` = First Name, `[2]` = Last Name
     - `[10]` = Organization Name, `[11]` = Organization Title
     - `[17]` = E-mail 1 Label ('Work'), `[18]` = E-mail 1 Value
     - `[19]` = E-mail 2 Label ('Personal'), `[20]` = E-mail 2 Value
     - `[27]` = Phone 1 Label ('Mobile'), `[28]` = Phone 1 Value
     - `[29]` = Phone 2 Label ('Mobile'), `[30]` = Phone 2 Value
     - `[39]` = Address 1 Label ('Work'), `[40]` = Address 1 Formatted
     - `[41]` = Address 1 Street, `[42]` = Address 1 City
     - `[45]` = Address 1 Postal Code
     - `[82]` = Address As (display name, e.g. 'Ankush')
     - Initialize a list of 93 empty strings (""), fill only populated indices, append via `values().append(range="'NDR DRAAS Google contacts.csv'!A:CM", valueInputOption="USER_ENTERED", insertDataOption="INSERT_ROWS")`.
  4. **People API** — search existing first (`people().searchContacts(query=name, readMask='names,emailAddresses')`). If found, `updateContact(updatePersonFields=...)`; if not, `createContact` with names, organizations (name+title), emailAddresses, phoneNumbers, addresses, urls (website). Report the `people/...` resource name back.
  5. **Email follow-up** — if the user also says "send the same inventory/pitch email I sent to X to this new contact": find the reference email via Gmail (`q='subject:... <domain>'`), extract the plain-text body (base64 or quoted-printable decode), and create a fresh draft to the new contact's email with the exact same body. Draft only — never auto-send. Recipient address must come from the new card/contact, and it must satisfy the email-recipients rule (in contacts once added).
- If the user describes by **role** rather than name (e.g. "operations coordinator for Dr. Haldipur"), first try searchContacts with the person's real name (ask the user if you don't know it). Then iterate through the priority chain above.
- **Enriched metadata from the contacts sheet:** Even when the People API search finds the contact directly (returning name + phone), check the NDR DRAAS Contacts Sheet for enriched context — job title, company name, and notes about the person. This helps you craft a more context-appropriate message (e.g. addressing Gowri as "Content & Marketing Head at DRA Realty" rather than just "Gowri Singh").
- **Multiple phone numbers on one contact:** People API search can return several numbers for the same person (work, mobile, legacy). Do NOT pick the first one. Disambiguate via the NDR DRAAS Contacts Sheet: find the exact-name row and prefer the **Work**-labelled phone (Phone 1) over Mobile/others. Real example: Bharat Hawaldar returned 3 numbers from People API; the sheet showed Work +91 99000 29200 as primary and Mobile +91 83173 20327 as secondary. Use `scripts/find_contact.py` for a one-shot lookup that prints both People API matches and labelled sheet rows.

- **WORK NUMBER ONLY for colleagues — NEVER personal/alternate (HARD RULE, NDR correction 2026-09-18).** For any colleague or reportee (Prakash, Anbu, Rahul/Vinod, Bharat, Sarthak, Sinchana, Gowri, Roshni, Manohar, Umesh, and any other DRA staff), ALWAYS use their **work / DRA mobile**. NEVER generate a link against their personal or alternate number — even when that number appears FIRST in the resolver output, even when it's the only one returned, and even when it's the number NDR has used before. Only use a personal/alternate number when NDR **explicitly instructs it for that specific message** ("send it to his personal number", "use the alternate"). When two numbers are on file and it's unclear which is the work line, **ASK — do not guess**. Applies to WhatsApp deep links, SMS, calls, and every other channel. NDR's rule verbatim: *"the only and only number we should be using for Prakash is his work number or any of my colleagues, which is a DRA mobile or work, unless I explicitly state to use their personal number or the alternate number."*
  - **Worked failure (2026-09-18, P-195 Ranka Udaya campaign):** `contact_resolver("Prakash Singh")` returned Phone 1 = +91 97399 32078 and Phone 2 = +91 99000 93816; the link was generated against **97399 32078** (wrong — personal) and NDR corrected it. The **work/DRA mobile is +91 99000 93816**. Note the resolver's "Phone 1" is simply label order, NOT a work/mobile signal — for Prakash the sheet's own labels are misleading too (his *work* number is listed second). Treat NDR's stated number as authoritative over the sheet's ordering.
  - **Known colleague work numbers** (use these; extend as NDR confirms): Prakash Singh = **+91 99000 93816**.
  - **When you must correct an already-issued link:** regenerate it, mark the old link explicitly **superseded — do not send** in your reply, and append a dated `CORRECTION` note to the tracker row (keeps the audit trail clean so the wrong link isn't used later).
- **When a contact has 2+ numbers, NDR wants a WhatsApp link for EVERY number — he decides which to use (confirmed Aug 2026).** For bulk employee/coworker messages (e.g. the RAI robo-call voice-recording drive), when `find_contact.py` returns multiple numbers for a person (Aravind Ahfl, Pawan, Ravi Office Boy, Sarthak), generate one link per number, label each link with the number (`Open WhatsApp — Aravind #1 (82481 51485)`), and send each as its own Telegram message. Do NOT silently pick one — NDR will tap whichever is the live number.
- If multiple people match, try **Gmail-context disambiguation BEFORE asking** (below), then ask if still ambiguous. The cost of a wrong number is high.
- **Same-name disambiguation via Gmail thread context (worked 2026-08-14):** when searchContacts/sheet return several candidates with the same first name but the conversation topic is known (fund manager, project, vendor), run a Gmail query pairing the name with topic keywords — `Kishan (algo OR AIF OR trading OR Nifty)` — and inspect the threads. Decisive evidence (a recurring "Performance Update" thread from one candidate's company domain) picks the right person with zero clarify round-trips. Real example: 4 "Kishan" candidates (Surana-jeweller, Nair, Thapar-driver, Shah) — only **Kishan Murjani Nair** (kishan@flamebackcapital.com, Flameback Capital, CFA) had performance-update threads matching NDR's "21+ percent returns" story, so he was the recipient. If no thread is decisive, fall back to `clarify` with the shortlist.
- If the contact is not in memory, not in Google Contacts, and not in the contacts sheet, ask for the number. Do not invent one.
- Confirm the country code. Indian mobile numbers starting `9845…` / `90080…` / `98801…` are all +91.
- **Designation is not part of the name.** If the user says "Nagrajappa JDTP", "JDTP" is his designation (Joint Director of Town Planning), not part of his name. Search contacts for the given name only. Same for "BBMP", "JD", "SE", "EE", "AEE" suffixes — strip them before searching.
- **"First Name, Full Name" apposition for employees (2026-08-28):** The user sometimes identifies an employee by both a first name AND a full name in apposition — e.g. "Rahul, Vinod Kumar Das, my employee." The full name after the comma is the canonical record name; the first name before the comma is how the user addresses them. Try searching the full name first (more specific), then the first name. Employees may NOT be in Google Contacts at all — fall back to asking the user for the phone number rather than burning the full contact lookup chain on a person who may only be in WhatsApp's local phonebook.

**"Assuming-done urgency" framing for internal design/architect requests (2026-08-27):** When the user asks for deliverables from an internal designer/architect (Sinchana, etc.), the standard framing is:
- State the need clearly: "I need X, Y, Z in one place — immediately."
- Express assumption: "I'm assuming this is already done."
- Set timeline: "Please share the folder link immediately."
- Name who's been informed: "Marked to [colleagues] for visibility."
This is warmer than a blunt demand but leaves no ambiguity about urgency. Use this framing for ANY internal deliverable request to design/architect staff. It signals confidence in the team while communicating pressure — the combination the user prefers.

**When searching for an email address (not just phone):** Gmail thread search often yields the most reliable result for professional contacts. After finding an email from Gmail, **confirm it with the user before using it** in a message or draft. Say something like "I found [person]'s email as [email] from your past thread about [topic] — is this the right one?" This prevents mistakes from:
- Multiple people with the same name
- Old/outdated email addresses from years-old threads
- Misspellings from voice transcription

### Step 1b — Project-context reconstruction from Gmail BEFORE composing (2026-09-19)

When the dictated message is **project operational content** — sequencing of site works, contractor mobilisation, inspection/energisation timelines, meter installation — and NDR/Anbu names people and roles the message assumes the recipient will recognise, **reconstruct the project's live context from Gmail first**. Do NOT compose from the dictation alone and do NOT ask the user who "Nandish" is before searching. The thread history names the counterparties, the work-order owners and the real technical vocabulary, which is what makes the draft read as credible to the recipient.

**Trigger:** a message that says "the DG person", "our building electrician", "Nandish team", "the transformer will be charged" — role references, not full names.

**Recipe (colleague session — search `google-draas` only, see `contact-phone-lookup`):**
```python
g = build_service('gmail', 'v1', service_name='google-draas')
for q in ['<Project> transformer', '<Project> electrical', '<Project> permanent connection',
          '<Project> meter', '<Project> LT panel', '"<Project> — OC Shared"']:
    r = g.users().messages().list(userId='me', q=q, maxResults=8).execute()
    # then format='full', decode body_text(payload), grep for role nouns
```
Then read the **most substantive thread** end-to-end (`subject:"<Project> — OC Shared"` style subjects carry the day-wise action items and every workstream owner).

**Worked example (Ranka Iris, 19 Sep 2026):** the dictation referenced "Nandish team … giving the meters for all the flats" and "our building electrician — the contractor who has done their flats' electrical work". Gmail reconstruction from `google-draas` produced, in order of value:
- `Shankar Electricals - HT/LT Work - Payment Details` (3 Sep) — the HT/LT work-order owner, completed vs pending list: *"Inspected approval — inspection from the Chief Electrical Technical Inspector is pending; Procurement of all the meters is pending; Charging of the transformer is pending; LT cable connection to the panel board and cubicle for the common area meters is pending."* This is exactly the work the dictated message schedules.
- `Ranka Iris — OC Shared: Action Items & Day-Wise Timeline Commitments Required` (8–11 Jun) — names **Nanjesh (BESCOM & Transformer)**, the BWSSB/water-meter owners, and the "15 working days from transformer order to final LT panel connection" timeline.
- `Regarding your services for the electrical inspection approval for Ranka Iris` (1 Sep) → **Deccan Electricals** (`deccanelectricals179rp@gmail.com`), engaged for Form 1/Form 3 + transformer inspection and charging.

**Outcome to report:** with the history in hand you can (a) confirm the recipient genuinely owns the topic, (b) tell the user in the delivery reply which *known* counterparties the dictated role-names most likely map to, and (c) keep unresolved tokens verbatim in the body while flagging them — here **"Nandish" had zero hits** across contacts, sheets and Gmail, so it was kept as dictated and flagged (candidate: the meter-supply team, or a garble of **Nanjesh**, BESCOM AEE +91 99002 49798). Never substitute a guessed name into the message body.

**Rule — role-names stay verbatim; only clearly-wrong terms get corrected.** "the DG person", "Nandish team", "the building electrician" are NDR's own vocabulary and the recipient knows them. Correcting them to the formal counterparty names is *inserting facts*. Flag the mapping in your reply to the user instead.

### Step 2 — Honour user-vocabulary

This is the most important step and the most commonly fumbled.

- **Voice transcripts of the user's own vocabulary are NOT ground truth.** When the user says a name out loud and the dictation software writes "Sabji" instead of "Sahabji", the user *means* "Sahabji" — that is the spelling the user has previously chosen and saved to memory. Use the memory spelling, not the transcript spelling. Same rule for: nickname variants, company name variants, place names, inside-joke words.
- Read the memory entry for the contact and use the spelling and nickname stored there. If memory says the nickname is "Sahabji" and the voice transcript says "Sabji", write "Sahabji" in the message and note the discrepancy briefly to the user so they know you corrected it. Don't lecture; just note it.
- If the user explicitly corrects a spelling in the same turn ("it's Sahabji, S-A-H-A-B-G-I"), do two things: (1) update memory with the correct spelling, (2) confirm the correction back so the user trusts it landed. Memory will hold; future sessions will not re-make the same mistake.
- If the user's correction triggers a memory-write and memory is at capacity, replace the oldest or most short-term entry (e.g. one-time TATs from a specific dispute) with the new durable fact. Do not silently drop the correction.

## User-specified bold formatting within message body (corrected 2026-08-28)

When the user explicitly says "highlight [X] in bold" or "put [Y] in bold" within a dictated message, honour that instruction literally in the WhatsApp markdown: wrap the target phrase in `*...*`. This is a deliberate emphasis request, not a suggestion — the user has a specific reason for wanting that phrase visually prominent in the recipient's chat (urgency, legal consequence, financial figure). 

**Rule: bold only what the user designated.** Do NOT add bold to other items unless the user also asked for that. When the user says "number them 1-6 and bold X", scope: numbers are plain, X is bold. When the user says "bold the portion about Y", scope: only Y gets `*...*`, everything else stays plain. Over-bolding dilutes the emphasis and costs a correction round.

Also applies when the user says "keep different formatting" / "use different formatting" in a multi-section message — that is a standing request for visual structure (bold headers, bullet sections, numbered items), not a one-off. Default to structured WhatsApp markdown for any message with 3+ distinct items.

---

### Step 3 — Categorise the message type

There are four distinct patterns. Identify which one applies before composing. (Pattern D added 2026-08-01 — internal performance follow-ups.)

---

**Pattern A — Direct dictation (simple)**

The user dictates a message they want sent verbatim or near-verbatim. Typical trigger: "Send a WhatsApp to [name] saying [message]". Compose close to their words.

**Pattern B — Response to a third-party message (iterative)**

The user describes *an incoming message they received* and *what they want to say back*. Typical triggers: "[Name] sent me a message saying X, help me draft a response", "In response to [name]'s message, I want to say Y". This pattern almost always requires **multiple refinement rounds**.

Step-by-step for Pattern B:

1. **Understand both sides.** Read the incoming message carefully (the user may describe it in a voice memo, paste it, or forward it). Understand what the sender said *between the lines* as well as explicitly. Note tensions, unasked questions, and any passive-aggressive framing.
2. **First draft** — compose a response that addresses:
   - What the user explicitly asked to say
   - Any points from the incoming message that the user didn't address but are implicitly expected (e.g. the sender asked about something — don't ignore it, redirect politely)
   - Don't overreach — stick close to the user's described intent
3. **Present the draft as a starting point.** Use language like "Here's a draft —" not "This is the final version." The user *will* refine.
4. **Expect 2-3 refinement rounds.** The user's first voice description often misses nuance. Each follow-up voice message will:
   - Correct a point you got slightly wrong
   - Add context they forgot to mention
   - Refine the wording on a specific paragraph
5. **Each round:** apply the correction, re-present the full revised draft in a clean form (not just showing the diff). Keep numbering/paragraph structure consistent so the user can track what changed.
6. **Final deliverable:** When the user is satisfied, generate the WhatsApp link. Do NOT generate it on the first draft — wait for the user to greenlight the message first.
7. **Explicit question in the incoming message the user's dictation did NOT answer (2026-09-15, Ranka Amber SSA → Raghu Iyer):** If the third party asked a direct question (e.g. "please confirm I don't need anything notarised or DFAT-attested before I leave") and the user's dictated reply didn't cover it, do NOT fabricate or guess the answer — insert a bracketed placeholder in the draft (`[your confirmation here]`), flag the open item to the user in the delivery reply, and HOLD OFF creating the Gmail draft / WhatsApp link until the user gives the answer. NDR's own stream-of-consciousness dictations routinely skip counterpart questions; guessing a legal/factual answer (notarisation, attestation, dates, amounts) produces a draft he has to unwind. Worked example: Raghu Iyer's 15-Sep email (24–25 Sep visit, DFAT question) → draft presented with placeholder, `draft_reply_create` deferred until NDR confirms the notarisation answer.

Common refinement signals in Pattern B for NDR:
- User refers to a sensitivity they want addressed (e.g. "the personal matters point — clarify that wasn't my intention")
- User re-frames who is a PoC ("Ashwin and I are the only PoCs, not just I")
- User sharpens the domain boundary ("Conrad was never in my domain, never interacted with you on it")
- Each refinement is a tight edit of one or two paragraphs, not a full rewrite

**Pattern C — Congratulation + request (government/authority figure)**

A sub-pattern where the message opens with sincere praise or congratulations on a public achievement, then segues into a gentle request. Typical situation: a government official (JDTP, BBMP engineer, BDA officer) did something noteworthy — the user wants to acknowledge it warmly before asking for a favour.

Key rules:
- Open with genuine, specific congratulations. Name the achievement explicitly (e.g. "the Karnataka Apartment Management Owners Bill amendment").
- Keep the praise warm but not obsequious. "Fantastic work, a step in the right direction" is right. Don't overdo.
- The segue is soft: "Gently following up..." / "Just checking in..."
- Acknowledge the official's busy schedule explicitly (\"I know you have a very busy schedule\").
- Frame the request as brief and low-friction (\"half an hour\", \"whatever works best for you\").
- Tie the request to a concrete forward outcome (\"so we can put in the sanction application and move things forward\"). See `references/government-official-messaging.md` for a worked example.
- **Draft-approval rule (NDR, 24-Aug-2026): *"lets agree on the draft then generate the link."*** For Pattern C (and any complex multi-paragraph message), the FIRST dictation is often an incomplete/garbled briefing — NDR will follow up with the corrected facts (project name, dates, officer names). Present the draft text for approval BEFORE calling `whatsapp_link`; when the correction arrives, treat it as authoritative and only generate the link once he greenlights ("please go ahead and make the WhatsApp message"). Worked example: khata-intervention message to Munish Moudgil (IAS, Special Commissioner Revenue & IT, GBA) — the first draft referenced a vague "old project"; the corrected brief had the full facts (Ranka Iris, Domlur; 3B+G+13; 50% TDR setback relaxation; started 2014, stalled, OC obtained; revenue software rejecting khata; technical team Santosh & Amar; RERA delay; customers occupying units; request to meet briefly just to say hello).
- **Gov officials are usually NOT in Google Contacts** — don't burn the whole lookup chain. Research them instead (see `references/government-official-messaging.md` → "Finding gov official contacts").
- No WhatsApp-style directness here — this pattern deliberately uses polite, deferential language (Sir, grateful, please let me know).

**Pattern D — Internal performance follow-up / escalation to a direct report (corrected 2026-08-01)**

When the follow-up is about a subordinate's pending work — "it's been 8 days and no response", "forwarding to Bharat is not an answer", "it's your job to follow up" — NDR wants **polite, cooperative framing with the SAME hard intent**, NOT blunt directness. The user's explicit correction after seeing a strict draft: *"Let's make the message very polite and cooperative in tone, but the intention is the same."*

Structure that landed (Prakash ICICI 8-day follow-up, Aug 2026):
1. **Open softly** — "I wanted to talk to you about..." instead of "You have not responded."
2. **State the fact plainly but not accusingly** — "they have now been pending for 8 days, and that is simply not acceptable." Keep the hard line ("is not an answer for me — it is absolutely not an answer") but in complete, measured sentences.
3. **Refer to prior instruction as context, not rebuke** — "I have shared with you before that Bharat is not to be used for this work."
4. **Offer an acceptable alternative** — "If you need Bharat's help, that is fine — but in that case I should have seen you engaging him properly: eight emails in one day, WhatsApp messages, calls, following up, and coming to me directly for any missing documents."
5. **State the standard as expectation, not threat** — "That is the follow-up I expect from you."
6. **Ask for their plan, and offer support** — "Please let me know if there is any issue from your side, and how you plan to resolve this — I am happy to support you in any way I can."

Key differences from the strict first draft: open with "I wanted to talk to you" not the accusation; "if you need Bharat's help, that is fine" instead of a flat ban; close offering support. The teeth are unchanged — 8 days unacceptable, forwarding ≠ answer, and the specific expected behaviour spelled out.

**Rule:** For direct reports / internal escalations, default to this cooperative-but-firm Pattern D. The blunt direct style (Pattern A) is for peers/vendors on neutral topics; Pattern C for authority figures. When in doubt between strict and polite for an internal follow-up, choose polite-cooperative — the user will sharpen it if needed, but starting blunt costs a correction round.

**Pattern F — Sensitive partner/advisor escalation (multi-year failed commitments with a shared intermediary)** (added 2026-08-28)

When the message is to a senior partner/advisor who introduced an intermediary (Devaaya, broker, agent) that has accumulated many failed commitments over multiple years, and the goal is to:
- Move to Plan B bypassing the unreliable intermediary
- Get the advisor to arrange a meeting where the intermediary transparently hands over contacts/communications
- Deploy a new resource from the advisor's network to work directly

**Distinctive tone structure that landed (Narsimha Raju, Aug 2026):**
1. **Warm opening** — `Hope you are well, sir.` Standard greeting, deferential.
2. **"We tried not to bother you" framing** — `We have tried very hard to avoid troubling you on this, but as you are well aware...` This defuses defensiveness; the advisor knows the history, no need to lecture them.
3. **Factual, not emotional, recitation of failures** — `this matter has now been running for multiple years with at least 50 failed commitments to us — and through you and commitments you yourself gave to us, at least another 10 failed commitments in getting the pending lands closed.` Numbers without adjectives. No "lies", "broken promises", "unreliable". Just the facts.
4. **Specific ask for meeting** — `We are therefore requesting you, sir, for an urgent in-person meeting with Devaaya present — a meeting that only and only focuses on Plan B (Dundi).`
5. **Explain WHY words aren't enough** — `Frankly, I don't think Devaaya can be part of Plan B anymore... Relying on his words alone is no longer wise. The sheer volume of failed commitments makes it clear that neither does he take his word seriously nor does he attach gravity or seriousness to the commitments he makes — and more importantly, he is not in control of things. I also believe he has not been transparent. That is my honest sense.` Frame as your honest assessment, not accusation. Use "I also believe" and "my honest sense" — personal, not objective.
6. **Concrete handover ask** — `under your instructions only, he transparently hands over all communications, all contacts, all his last message exchanges — and if needed, even makes calls to the farmers directly in front of you.` Specify the EXACT actions needed, not vague "transparency."
7. **Contingency offer** — `given the extensive contacts and goodwill you enjoy in that area, we would request — if you have another resource you can deploy with us — that we directly start engaging with the farmers ourselves to try and get closure.` Give the advisor a face-saving alternative: their other person, not a stranger.
8. **Close deferential** — `Kindly let us know when you are available for this meeting. We are ready at your earliest convenience.`

**Key rules for Pattern F:**
- NEVER accuse the intermediary directly (the advisor introduced them; attacking the intermediary attacks the advisor's judgment)
- Use "failed commitments" not "lies" or "broken promises" — it is factual and impersonal
- Frame the intermediary's shortcomings as "not in control" and "not transparent" rather than "dishonest" or "incompetent"
- Ask for the advisor's resource, not a new outsider
- The advisor's "goodwill in that area" should be acknowledged explicitly — you want to leverage it, not bypass them
- Close with availability matching the advisor's schedule, not your own

This pattern overlaps with Pattern D (internal escalation) in its polite-but-hard framing, but differs in the target: D is a subordinate, F is a peer/senior partner/advisor where the dynamic is respect-based, not authority-based.

**Pattern H — Milestone wishes to acquaintances / kitty-group members (restrained tone, corrected 2026-08-29)**\n\nWhen the user asks for anniversary / birthday / milestone wishes for people who are **acquaintances** — a kitty group, social circle, "people I hang out with" — NOT close friends or family, NDR's correction after seeing an over-warm first draft:\n\n- \"They are not that close to each other. I don't want to add too many emoticons. Maybe one emoticon in the entire message and no love and all that. And don't put the [event] tomorrow. I may not be going because my wife is not keeping well. So just wishing. That's it.\"\n\nRule for this pattern:\n1. **Max ONE emoticon in the entire message** (🥂 or 🎉 at the very end). Not per-line, not per-option — one total.\n2. **NO affection filler** — no \"lots of love\", no hearts, no \"cheers to you both!\" exclamations. Warmth comes from the wish itself, not emoticons.\n3. **Do NOT mention the associated event** (the Sai Bhajan, the party, \"see you tomorrow\") — NDR may not attend (wife unwell etc.), so the message must stand alone as a pure wish.\n4. **Deliver 2–3 short options** the user can pick from, each fully written.\n5. **Present each option as a ```code block```** containing the WhatsApp markdown verbatim (`*bold*` etc.) so the user can copy-paste into WhatsApp directly. This mirrors the code-fence copy-paste preference (2026-08-25) for deliverables the user will paste.\n6. Keep each option under ~3 lines: greeting/stated milestone + wish. No backstory, no filler.\n\nWorked examples that landed (Deepak & Gowri, 25th anniversary, Aug 2026):\n- Option A: `*Happy 25th Anniversary, Deepak & Gowri!*` + \"Wishing you both a wonderful day and many more years of togetherness and happiness ahead. 🥂\"\n- Option B: plain, no emoji — \"Deepak & Gowri, Congratulations on 25 years! Wishing you both a fabulous anniversary and many more happy years together.\"\n- Option C: `*25 years — congratulations!*` + \"Wishing you a lovely anniversary and all the best for the years ahead. 🎉\"\n\nFirst drafts that got corrected: emoji-heavy (🎉💍✨💕🥂), \"Lots of love from our gang!\", event invite / \"can't wait to celebrate with you at the Sai Bhajan tomorrow\". All removed per the correction.\n\n**Pattern I — Family/personal messages that open with a festive greeting (added 2026-09-16)**

When NDR asks a message to a FAMILY member (Dharmesh, Manish, Mamta) to OPEN with a
greeting — *"start the message a little politely wishing him Michhami Dukdam to him and
Bhabi... then get into this message"* — honour the requested structure exactly:
greeting first, personal bridge, then the ask.

1. **Greeting on top, standard spelling.** STT renders "Michhami Dukdam"; the standard
   Jain greeting is **Michhami Dukkadam** (the Paryushan/Das Lakshana forgiveness wish).
   Use the standard spelling in the body and flag the choice in the delivery reply (same
   rule as any name correction).
2. **Tone is a hard instruction, not a suggestion.** NDR may explicitly demand
   *"language, choice of words, tone, structure of sentence, everything very polite and
   cooperative always"* — open warmly (festival wish + "will catch you tomorrow at
   <place>"), then present the ask as a two-option choice in bullets ("Either you... or
   give me the go-ahead here on WhatsApp...") and close with a soft confirm. Polite
   framing does NOT soften the ask — the action items stay explicit.
3. **No caption on family messages.** The *Project — Subject* caption rule applies to DRA
   staff task messages; brothers/sisters get personal, caption-free notes even when the
   content is business-adjacent (succession, GPA, deeds).
4. **Correct voice-corrupted names in body + flag in reply** (extends the
   clearly-wrong-terms rule): "Manisha"→Maneesha Kongovi (CMS IndusLaw), "Rajan"→Rajendra
   (the separate-GPA recipient named in NDR's own emails), "Bhabhi" kept as dictated.
5. Structure greeting → personal bridge → bullets for options → closing line ("Look
   forward to seeing you tomorrow morning") has landed well.

**Pattern G — Vendor/contractor technical requirements clarification (added 2026-08-27)**

When the user is in an active build/consulting engagement (Devansh, Anuj, Parth on the DRAAS pre-sales AI system) and a vendor sends follow-up questions, NDR dictates a LONG dense voice memo of settled requirements — "what I need for Alpha v1 is very clear…" — and asks for a "detailed response with all WhatsApp markdown… keep different formatting… make it easy to read." This is a counterpart to the OwnerBrief PRD email (same content, WhatsApp medium): when he says "based on all of my content I've already put," reuse the PRD structure (Lead Gen → Nurture → Buckets → Calling → Human Interface → Objectives) rather than re-deriving from scratch.

**Pre-composition step — assess what the vendor's message actually needs:**
- The vendor will often send a detailed structured breakdown (Alpha scope questions, Beta scope questions, "what's already built vs needs building," data/inputs they need, phase boundaries). Read each section of their message carefully before composing — don't just summarise or acknowledge; address each point they raised within the response.
- NDR's voice dictation for the reply will be a stream-of-consciousness that touches on ALL the same topics the vendor raised. Map his dictation to the vendor's structure as you compose, not separately.

**Structure that landed (5-part series to Devansh, Aug 2026, across two rounds):**
1. **Confirm their understanding first** — open by acknowledging every point the vendor listed and confirming it's correct. Use a single bold header "ALPHA SCOPE — CONFIRMED ✅" followed by a bulleted list of each of their 11 points with ✓ marks. This establishes that the requirements are settled, not up for debate.
2. **"What's already built vs needs to be built"** — this is the critical clarification the vendor needs for scoping. Use a separate bold section with a clear bottom-line summary. Mark each system with ✅ (ready), 🟡 (needs check), or ❌ (needs building). This prevents the vendor from pricing work that already exists or assuming things are ready that aren't.
3. **"Data/inputs we will provide"** — another bold section separating what you CAN provide immediately (✅) vs what needs internal checking (🟡). Include a volume estimate for their sizing. Promise a timeline for the 🟡 items.
4. **Beta scope confirmation** — if the vendor asked about Beta, confirm their understanding and add any corrections or additions. Use another bold header and bulleted list.
5. **Scaling/Post-Beta boundary + human handoff rules** — define the boundary clearly so the vendor can separate phases. Provide specific handoff rules the vendor needs for Alpha scoping. Close with timeline estimate and a direct question.

**Within each part:** `*Bold section header*` + bullets — WhatsApp markdown sections (`*Core System Logic:*`, `*Cadence & Escalation:*`) each followed by `-`/`•` bullets. Never a wall of text.

**Extract structure from the ramble** — the dictation is stream-of-consciousness; the value is re-organising it into logical sections while keeping EVERY stated requirement (import leads, parallel enrichment, configurable outreach channel per lead, bucketing, cadence, call only when no WhatsApp engagement, full lead context, human intervention queue, in-system WhatsApp for humans, template/media tracking, multiple objectives site-visit → follower → tool adoption).

**Information gaps workflow — REQUIRED post-composition step (added 2026-08-27):**
After composing the full multi-part response, identify EVERYTHING you were unable to answer definitively and present them as numbered gaps to NDR. These typically include:
1. WhatsApp provider status (Exotel vs Ozontel — which is active?)
2. Vobiz relationship status and credentials
3. Call transcript storage location
4. Lead enrichment data availability
5. Volume estimates (leads/day, calls/day)
6. Human handoff rules confirmation
7. Kelsa API credentials sharing approval
8. Languages needed (Tamil for Chennai?)
9. Site visit process status
10. Any other data-source status questions

Present these as a clean numbered list under a "Gaps I need you to confirm" heading BEFORE asking him to send the messages. This prevents him from discovering missing information in the middle of the vendor conversation.

**Handle mid-task topic switches gracefully** — NDR will often ask an unrelated quick question (stock price, document lookup, calendar check) while you're in the middle of composing a multi-part response. Handle the quick question first, then resume the drafting. Do not complain about the interruption or lose your place — the message links are already generated and can be presented immediately when he returns to the topic.

**Numbered summary at the end** — "*In Summary — This Is What I Want Built:* 1… 2… 3…" recapping the full scope (8 items worked well).

**Close with a direct question** — "Do you need any clarification on the scope before we finalize the build plan?" keeps the ball in their court.

**Deliver each part as its own short-labelled inline link** in the reply (`Part 1 — …`, `Part 2 — …`, `Part 3 — …`, etc.) — the user taps each. If `send_message` delivery misbehaves, embedding the part links in your reply is acceptable (P5d in `messaging-drafts` → `references/messaging-links-workflow.md`); the user has never complained about that form.

---

### Step 4 — Compose the message

After categorising the pattern, compose accordingly:

**For Pattern A (direct dictation):**
- Follow NDR's WhatsApp style: direct, to-the-point, no pleasantries
- Keep the user's voice (run-on sentences, "wanna", Indian-English cadence)
- Do not insert facts the user did not say

**For Pattern B (response to third-party):**
- First draft covers the user's-described intent plus implicit expectations from the incoming message
- Expect iterative refinement — present each version cleanly
- Don't generate the WhatsApp link until the user confirms the message is final

**For Pattern C (congratulation + request):**
- Polite, respectful, formal but warm
- Lead with congratulations, then segue to the request
- Acknowledge their time constraints
- Tie request to a concrete outcome

**General rules (all patterns):**
- **NDR's preferred WhatsApp style (confirmed 2026-07-13):** Direct, to-the-point, no pleasantries. No "how are you" / "hope you are doing well" / formal greetings. Lead with the ask. Address people by first name only. This is a hard style preference, not optional. **Exception:** Pattern C (government officials) deliberately breaks this rule — use polite, deferential tone with "Sir" and formal framing.
- Keep the user's voice. If they say "wanna close out accounts and discuss a few Amber and Nosta matters", keep "wanna". Don't upgrade "wanna" to "would like to".
- Do not insert facts the user did not say. Don't add "Hope you're well" if the user didn't say it. WhatsApp pre-fills are not email — they are a copy-paste buffer, and padding feels wrong.
- Keep proper nouns exactly as the user said them. "Thaisildar" → check if the user actually meant "Tahsildar" (the correct Hindi/Urdu transliteration of the revenue officer). When in doubt, quote the user exactly and note the possible variant in the message body itself. Better: ask. Many users switch between spellings in different memos.
- If the message references people the recipient does not know (Amit Pujari, Anbu, Nishant Prakash), keep the names as the user said them. Don't add "(our land consultant)" annotations.
- **Proposal/deal WhatsApps to external parties: default to FULL commercial terms recap + honorific (NDR correction 2026-09-04, Anil Kumar of Avni Consultancy, Hoskote takeover).** A dictated first-pass like "this is the proposal we gave you, this is our final, any revert or update?" is a PREVIEW — NDR returns immediately asking for "the WhatsApp with the full details of the proposal" and "address him as X sir". For any final-position/deal message, put the COMPLETE terms in the first link: rate × area = gross consideration, deductions/budgets (e.g. arch/clubhouse works), withheld amounts, net to seller, and the full payment schedule (now / ~4 months / 12 months) — plus a total-transaction line. Open with the honorific NDR used ("X sir,") even though the style rules say first-name directness; external deal counterparties get the deferential form. Keep his closing line verbatim ("This is our final position... Please let me know if there's any revert or update on this."). Do not ship the short nudge first — regenerating costs a round.
- Run-on sentences are fine. WhatsApp tolerates them; tightening a 90-second voice memo into 3 bullet points usually loses information the user wanted conveyed.

**Special case — medication procurement:** When the message dispatches someone to buy medicines, follow the two-phase workflow in `references/medication-procurement.md`. Phase 1 (dosage verification) must complete and get user confirmation before Phase 2 (message composition). The message must include: molecule name, 2-3 common Indian brand names, exact tablet quantity, and dosing schedule instructions.

**Color-reference matching rule:** When the message references color-coded elements from an attached image (e.g. "the blue-outlined site", "the yellow-outlined building", "the green-outlined project"), use colored emojis (🔵🟡🟢🔴🟣🟠) that match the image colors — not generic symbols (🟦💛📗) or text-only labels. The user expects the emojis in the message to visually correspond to the outline colors they drew on the annotated image. This applies to any WhatsApp message that accompanies or describes an annotated screenshot, map, or diagram where the user has applied colored markings.

**Caption rule — colleagues always, vendors/third parties never (USER-CORRECTED 2026-08-31):** When generating follow-up/task messages, prefix the message to DRA colleagues (Anbu, Rahul, Sinchana, Prakash, Bharat — anyone internal) with a bold caption line `*<Project> — <Subject>*` (e.g. `*BuxRanka Hudson — 4.5 FAR Pre-DCR Quotation*`, `*Palya Land — Survey Clarification*`, `*Ranka Iris — Khata Issuance*`). Do NOT caption messages to vendors/third parties (Ketan Vyas, external lawyers, agency contacts) unless the user explicitly asks. NDR's rule verbatim: *"To my colleagues we should always caption the messages. To anybody who is not a colleague who is a vendor or third party, we are not going to caption messages unless explicitly asked to do so."* The caption is the bold first line of the WhatsApp text itself (a WhatsApp markdown header, not a Telegram label). Also confirm the RECIPIENT'S PURPOSE before composing: in this session Ketan Vyas's message was mistakenly about 18 & Oak when he is actually the khata-issuance contact for Ranka Iris (coordinating with Amar, the technical/software guy, via Rahul) — resolve which project/purpose the recipient belongs to from user context/memory before drafting.

**Structured multi-part messages — WhatsApp formatting is EXPECTED (confirmed 2026-08-01):** When the message covers multiple items (two land proposals, several properties, a list of requests), the user explicitly asks for "appropriate bullet points, highlighting/bolding, numbering as required" — and this is a standing preference for multi-part drafts, not a one-off. WhatsApp renders `*bold*` natively, and line breaks + `•` bullets survive the wa.me encoding, so build the message with real structure before calling `whatsapp_link`:
- **`*Header line*`** for the caption (e.g. `*RD URGENT — RD for two new properties*`)
- **Numbered bold items** per property/entry, each followed by un-bolded detail lines (`*1. 10 Acres — Rajabhats, behind Foxconn*` then `Villa Development · Outright ~₹60 Cr @ ₹6 Cr/acre` then `Kelsa: <url>`)
- **`•` bullets** for request lists
- **Inline URLs** (Kelsa lead links, maps links) go in the plain text of the message — the tool percent-encodes them; do NOT shorten or strip them
- Keep the structured draft visible in your reply AND pass the same structured text (with `*` markers) to `whatsapp_link` — don't flatten it to plain text in the link while showing structure in the reply; the two should match.
Worked example that landed cleanly: the 2-property RD request to Prakash Singh (Aug 2026) — bold caption, two numbered property blocks each with Kelsa lead URL, `•`-bulleted request list (Sunday visit, Tuesday financier meeting, cab advance offer), close with "Thanks so much, [Name]".

**Structured data for copy-paste into third-party WhatsApp chats — deliver as a ```code block``` containing WhatsApp markdown (NDR preference, 2026-08-25):** When the deliverable is *data* the user will paste into someone else's WhatsApp (pharmacies raising an invoice, vendor onboarding, KYC handoff), NDR wants it as a code section with `*field:* value` pairs, WhatsApp-markdown bold on every field label, NOT a wa.me link. Two tricks make this work:
- Put the whole block inside a Telegram ``` ```code fence``` ``` — this preserves the literal `*` markers on Telegram (a bare message would render them as bold and the copy would lose the markup). The user copies the fenced content and pastes it into WhatsApp, where the `*...*` renders as bold.
- Structure = `*Label:* value` on its own line, one pair per line, grouped by sections with a `*HEADER*` line; include a one-line closing request ("Kindly raise the invoice in the above name for subsequent health insurance claim").
- When assembling KYC/invoice details from Drive (PAN, Aadhaar), extract with `pdftotext -layout` FIRST — PAN/Aadhaar card PDFs in NDR's Drive are text-layer, not scans (PAN: INCOME TAX DEPARTMENT header; Aadhaar: UIDAI header). Cross-check the billing address between the Aadhaar print (often an older "registered" address) and the Google Contacts sheet Home address — they can differ (Charitra Murjani: Aadhaar lists A 004 Victory Harmony Apts, sheet lists A3-202 White House Apartments, both R.T. Nagar 560032). Present the sheet Home address as primary and flag the Aadhaar discrepancy; let NDR pick which a third party should invoice to. Also include PAN, Aadhaar no., DOB, gender, phone (from contacts sheet), email, and links to the source PDFs for verification.

### Step 5 — Call the `whatsapp_link` tool

This is the **only** sanctioned way to produce a wa.me URL. Manual encoding is banned — past failures (2026-07-09 %26 ampersand break, 2026-07-10 missing country code, 2026-07-11 hand-written wa.me URLs in markdown) are why this rule exists. The tool handles percent-encoding correctly and accepts phone in any format (E.164, local, with spaces/dashes).

```python
# Correct:
whatsapp_link(phone="9845890316", text="...")

# Also correct:
whatsapp_link(phone="+91 98458 90316", text="...")

# BANNED — never hand-encode:
f"https://wa.me/{phone}?text={urllib.parse.quote(text)}"  # NO
```

**Fallback if the `whatsapp_link` tool is not registered in your toolset:**  
The underlying Python function still exists. Call it directly via terminal:

```bash
cd /opt/hermes && /opt/hermes/.venv/bin/python3 -c "
import sys, json
sys.path.insert(0, '/opt/hermes')
from tools.whatsapp_link_tool import whatsapp_link_tool
result = whatsapp_link_tool({'phone': '+919841059898', 'text': 'message here', 'platform': 'telegram'})
data = json.loads(result)
print(data['url'])     # raw wa.me URL
print(data['display_link'])  # Telegram markdown link
"
```

Do NOT fall back to hand-encoding even when the tool is unavailable — always route through `whatsapp_link_tool()` directly. The percent-encoding quirks (fullwidth ampersand workaround, `#` handling) are baked into that function and bypassing them produces broken links (confirmed user correction 2026-07-20).

The tool returns a URL you should display verbatim. In Telegram output, use the markdown link form:

```
[Open WhatsApp to <Name>](<returned_url>)
```

The user opens it on their phone; the message is pre-filled; they tap send. **Never** send a message on the user's behalf — the tool does not send, it composes. Same safety boundary as `email-drafter`.

## Pitfalls

- **Kids/family emails often live in NDR's SENT mail, not the contacts sheet (2026-09-11):** Ruhaan (pebblyshark69@gmail.com) was in the contacts sheet, but Rivaan was in NO contact source — found via a topic search on google-draas SENT mail (`q='bambu'`, labelIds SENT), where NDR's earlier email to both boys carried their real addresses in the To: header. When you need a family member's email for a CC (e.g. "mark Ruhaan and Rivaan in every email"), search NDR's SENT folder for related topics first; the To: header of his own sent mail is ground truth.
- **"Kelsa Issues" / "Kelsa support team" is a GROUP, not a single person (2026-09-09).** When NDR dictates a Kelsa bug report/issue message ("WhatsApp message to my Kelsa Issues, support team… telling Ashwin…"), the recipient is the Kelsa dev/support WhatsApp group (Ashwin Hegde, Pavan Kumar O3, Guna Kelsa, etc.), NOT one person — use the group-paste pattern (`whatsapp_link(text=...)` with NO phone) and address the message by first name to whoever he names inside the body (worked example: "Ashwin, take a look at this video… So if Pawan you can confirm it…"). **Contact lookup trap:** `contact_resolver` returns false positives for Kelsa names — "Kelsa Issues" → single-char "K" at score 90; "Ashwin Kelsa" → Ashwin Bafna (unrelated). Do NOT use it for Kelsa staff. Run `find_contact.py "Kelsa"` instead — it returns the full O3/Kelsa cluster (Ashwin Hegde co-founder +91 99728 03083, Pavan Kumar O3 +91 98496 44710, Guna Kelsa +91 99723 12600, Ajay Customer Advocacy, Arnav, Apoorv, Shilpam, etc.). Full roster + who-does-what: `references/kelsa-support-contacts.md`.
- **"Saurabh" is a 9+ contact family — disambiguate by domain, and CONFIRM before generating the link (2026-09-09).** Candidates seen in the contacts sheet: Saurabh CA Ref VK (+91 98864 89873 — the chartered accountant; "Ref VK" = referred by Vinod Kumar Das, who handles Satvik/DRA KAAJ/Palya tax-legal matters), Nitin Osthwal Saurabh CA (+91 80889 79777), Kshitij Saurabh (kshitij.saurabh@icici.bank.in — ICICI bank pre-approvals: Ranka Udaya/Ranka Amber), Saurabh Vashishth (saurabh.vashishth@godrejfunds.com — Godrej/BRDPL), Saurabh Kotak (Saurabh.s.g@kotak.com — Kotak Wealth), Saurabh Jain (IRCA listing, met via DDR), Saurabh Gfm, Saurabh Kelsa, Saurabh Kumar (Chambers security). Domain map: tax/GST/income-tax/foreign-asset assistance → the CA (98864 89873); bank pre-approval → Kshitij Saurabh (ICICI); Godrej/BRDPL → Vashishth. When the matter is a tax case (e.g. DGGI GST writ), default to the CA but explicitly confirm the number in the delivery reply — a wrong pick in this family is easy and costly.
- **Bare-first-name families where the TOPIC doesn't disambiguate — deliver the draft + one link per plausible candidate, then ask (2026-09-14, "Santosh" electrical update).** Multiple contacts share bare "Santosh" (DRA office boy +91 90662 31897; blank-org +91 93410 55524; plus Mr. Santosh of the khata-software team with no number on file; plus Santosh Srinivas / Santosh Kumar BBMP RI as domain-specific hits). When NDR dictates a status update ("inform Santosh... Chief Electric Inspector visit done, BESCOM meters, energization timeline") and NO candidate clearly owns the topic, do NOT silently pick one. Compose the message from the dictation, generate the deep link per plausible number (label each: "Santosh (DRA)" vs "Santosh (93410 55524)"), and flag the ambiguity in the delivery reply — NDR taps the live one. Same for garbled site/term tokens that match no project ("Apache" for the CEI site): keep the dictated term verbatim, flag it, and regenerate on his correction. This extends the multi-number rule (link for EVERY number, NDR decides) to same-name-different-person cases.
- **Named person required for resolve flow — skip lookup for service persons / unnamed trade contacts (corrected Sep 2026).** When the user says "my AC technician" / "the plumber" / "the electrician" / "shower glass guy" without naming the person, do NOT run the full contact resolution chain (find_contact.py, People API, sheets, Gmail). These are WhatsApp-only contacts the user will forward the message to themselves — Hermes has no record of them. Jump directly to Stage 3 (compose) with a generic salutation like "Sir" and ask the user to paste/send it. The deliverable is the WhatsApp link with the message pre-filled for the user to forward. If the user offers a name later ("his name is Rajesh"), THEN resolve. The phone argument stays empty (the user picks the recipient in WhatsApp) unless the user explicitly provides a number. When the user says "I just added [Name] to my contacts" and the contact doesn't appear in the contacts sheet, People API (all three vault accounts), or any other source — stop searching. Google Contacts sync from phone/WhatsApp is not instantaneous (can take minutes to hours). Do NOT keep re-searching or burn time on Gmail extraction. Ask the user for the phone number directly. The sequence: full search → not found → "Contact not in any synced source yet — please share the number."
- **Unresolved recipient on a share/email/WhatsApp task — check Kelsa users, then ask; don't block the rest (2026-09-10, "India Reddras").** When the named recipient of a Drive share / email / WhatsApp isn't in any contact source, ALSO check the Kelsa pipeline user list (`kelsa_call_tool list_users` on the relevant pipeline — e.g. 519 DRA Land Proposal): DRA land-team members who have no Google Contacts row DO live there (Sinchana = sinchana@kelsa.io, id 31249). Include a `list_users` pass before concluding "not found" for internal colleagues. If still nothing (real case: "India Reddras" — zero hits across contacts sheet, People API searchContacts on google-draas/ahfl/gmail with phonetic variants Reddy/Indira/Indra/Indu, Gmail, and Kelsa users), ask NDR for the email directly — NEVER guess. And do NOT let the unresolved person stall the resolved-recipient half: complete the share/email/WhatsApp for the identified person, flag the unresolved one with the sources searched, and offer to add them once the email arrives. A Drive permission grant also needs a real Google address — an unresolved name blocks only that person's share, not the whole task.
- **Single-character contact_resolver matches are false positives (corrected 2026-09-07).** `contact_resolver` can return entries where only the first letter of the name matches (e.g. query "Kranti" → returns a contact whose full stored name is just "K"). These are almost never the right person. When the top match has a score < 80 and the match reason is a single character or initial, flag it as uncertain instead of presenting it as a candidate. Ask for more details or the phone number rather than making the user disqualify a wrong match.
- **WhatsApp-synced contact may already exist in People API under a full name — UPDATE, don't duplicate (2026-09-07, Kranthi).** When the user created a contact today via WhatsApp and asks you to add an email, the phone-synced People API record often exists already under a MORE complete name than the user's dictation (dictated "Kranthi" / "Kranthi Chaitanya" → actual record `Kranthi Chaitanya Pulluru` + phone). Before `createContact`, match by phone/name variants via `people().connections().list(personFields='names,emailAddresses,phoneNumbers,organizations')` + `searchContacts`, and if found use `updateContact(updatePersonFields='names,emailAddresses,phoneNumbers,organizations')` with the etag to add the email (people API get → merge emailAddresses+org → update). Creating a second "Kranthi Chaitanya" record with no phone is the classic duplicate mistake; if you already did, `people().deleteContact()` the duplicate and re-apply to the original. Find the email to attach by searching Gmail for the company domain (`q='flameback'` → `kranthi@flamebackcapital.com` from sender headers), and confirm the address with the user before use.
- **Dictated feedback/message IS the content — do NOT research the subject (NDR correction, 2026-08-24).** When NDR dictates a long feedback/message (e.g. "I am giving him a feedback on the AI agent Yashika") and says "I have already done all of the feedback, I don't need you to pull out her context from anywhere", do NOT run session_search / web_search / skill lookups about the subject. The dictation is complete and self-contained; compose the message from the dictation only. Researching the topic he already covered wastes a round and reads as not listening. Only research when he asks for it or when the message needs a fact (phone, email, address) he did not provide.
- **Multi-part delivery: use SHORT labels + raw URL, NOT the `display_link` (2026-09-03, Rahul task-list message).** When `whatsapp_link` returns `split=true`, the `display_link` embeds the ENTIRE part message as the markdown link TEXT (multi-line, with `*` and `\` escapes). Sent via `send_message`, that link does NOT render as a tappable link in Telegram — the user sees plain text / a broken link ("the link is not properly constructed"). Correct delivery per part: a SHORT label line (e.g. `🌀 WhatsApp — Rahul task list — Part 1/2 (intro + tasks 1–6)`) followed by a clean markdown link `[Tap to open in WhatsApp](<parts[i].url>)` using the part's RAW `url` field. Each part STILL goes as its own separate Telegram message. Do not inline the part's message text in the label.
- **Follow-up messages that cite emails already sent: fetch REAL dates + the exact address used from Gmail FIRST (2026-09-03, Dharmesh).** When NDR dictates a chaser like "you have not responded to the succession-documents emails I sent — mention the dates and his email address", do NOT guess dates/addresses from memory. Query Gmail across google-draas + google-ahfl + google-gmail (`succession to:ddr@draas.com`, `succession dharmesh`, etc.), list thread subjects + dates + the To: address, then embed the VERIFIABLE dates (e.g. "1 Sep: 'DR Family Succession Petition - Filing, Vakalat & GPA'" / "2 Sep" / "3 Sep") and the actual mailbox (dharmesh.ranka1@gmail.com — NOT the contact's other addresses ddr@draas.com / ddr@drahomes.in / dharmesh.ranka@gmail.com; the address that actually received the docs wins) in the message body. Voice fragments like "Dharmesh dot Rankavan" are corruptions of a real mailbox — NDR wants it named precisely so the follow-up reads as exact. Create the tracker row if the topic is a live commitment (succession commitment → P-066-style row), and keep the WhatsApp text structured (numbered items, bold headers) with a caption because the recipient is an internal colleague.
- **Multi-part WhatsApp link delivery — send each part to the user via `recipient_name` + `platform`, not `target` (verified 2026-08-24).** When `whatsapp_link` returns `parts` (split=true), the three links must each go as their own Telegram message. In this environment `send_message(target='telegram')` fails with "Cross-user send blocked" (even though the error text suggests bare 'telegram'), and `target='origin'` fails with "Unknown platform". The working form is person-to-person: `send_message(recipient_name='Nishant Ranka', platform='telegram', message=<part display_link>)`. All parts delivered to the requester's own DM are fine — the guard is about cross-user targeting, not self-delivery.
- **When send_message target= fails with "Cross-user send blocked"**, use `recipient_name='Nishant Ranka'` + `platform='telegram'` instead (verified 24-Aug-2026) — the requester's own DM is always an allowed person-to-person target; explicit `target='telegram'` / `target='origin'` / `telegram:<id>` forms are refused by the gateway.
- **Follow-up WhatsApp referencing unresponded emails: pull the ACTUAL address + dates from Gmail, never the contact record or memory (2026-09-03, Dharmesh succession).** When NDR dictates "he hasn't responded to my emails — mention his email address and give the dates of the emails," do NOT take the address from People API / contacts sheet (Dharmesh Ranka had 4 addresses: ddr@draas.com, ddr@drahomes.in, dharmesh.ranka@gmail.com, dharmesh.ranka1@gmail.com) and do NOT recall dates from memory. Search Gmail for recipient + subject keyword (e.g. `succession to:dharmesh.ranka1@gmail.com`, `succession dharmesh`) across accounts, extract each thread's Date + To: header, and compose with the REAL address (the Sep-1/2/3 succession thread went to **dharmesh.ranka1@gmail.com**) and exact dates. NDR's own dictation of the address is often garbled STT — "Dharmesh. Dott, Rankavan, whatever his email address is" is a lookup request, not a literal address; treat it as "find the address I used." Also verify the phone via find_contact.py as usual (Dharmesh Ranka = +91 98441 09010, DRA Projects Finance Director, brother).
- **Medical queries to healthcare providers: deliver the message, do NOT add your own medical commentary.** When the user drafts a message to a doctor, dentist, physio, or other medical professional asking about medication, treatment, or a health condition, your job is to deliver the user's message verbatim as they dictated it. Do NOT add your own medical advice, dosing opinions, drug-interaction notes, or disclaimers like "but I'm not a doctor" — the user is intentionally asking *their own doctor* and your unsolicited commentary is irrelevant at best, misleading at worst. If you feel compelled to say something, limit it to "message delivered — you're asking the right person" and nothing more. This applies to any health/medical message to any recipient who is a qualified medical professional.
- **Voice transcript ≠ user's intent.** Always cross-check the spoken name/nickname against memory. If memory says "Sahabji" and the transcript says "Sabji", use "Sahabji" and note the correction. The user is the source of truth for their own vocabulary.
- **"Mention his email address and the dates of the emails" — verify BOTH from Gmail thread metadata before composing (2026-09-03, Dharmesh succession follow-up).** When NDR asks the message to name the recipient's email address and the dates he sent emails (e.g. succession docs to Dharmesh: "Dharmesh dot Rankavan… whatever his email address is, please mention that and give him the dates of the email as well"), do NOT take the address from the contact record and do NOT guess dates. Search Gmail (`to:<name-email> succession` style queries across google-draas/ahfl/gmail) and read `Date`/`To` headers from the actual threads. The address actually used in the thread can differ from the contact card's primary (Dharmesh Ranka's record lists ddr@draas.com / ddr@drahomes.in, but the Sep-2026 succession thread went to **dharmesh.ranka1@gmail.com**). List the exact email subjects + send dates (verified 1 Sep, 2 Sep, 3 Sep) inside the message body; the email-address verification belongs in Gmail, not in the WA link tool.
- **Never hand-encode wa.me URLs.** Use the `whatsapp_link` tool. If the tool is not registered in your toolset, route through `tools.whatsapp_link_tool.whatsapp_link_tool()` via terminal (see Step 5 fallback). Do NOT improvise with `urllib.parse.quote` or hand-written markdown — the encoding quirks are baked into the Python function.
- **Don't insert facts.** No "Hope you're well", no "Please find below", no formal sign-offs. WhatsApp is informal by default; if the user wanted formal, they would have asked for an email.
- **Don't double-correct the user's other spelling choices.** If the user said "Thaisildar" and the user has not previously corrected it, leave it. The rule is: respect the user's exact words unless memory says otherwise. Multiple-spelling vocabulary is normal in India (Tahasildar / Thasildar / Thaisildar all appear in Indian English).
- **No hardcoded country code inference.** If the user gives a 10-digit number, default to +91 (India) since this is the DRAAS-Bangalore/Chennai user. If the user gives a number with a different country code, pass it through to the tool as-is — the tool handles E.164 conversion.
- **Avoid `#` in message text.** The `#` character (even URL-encoded as `%23`) can break wa.me URL parsing — WhatsApp or the OS may interpret it as a URL fragment identifier, cutting off everything after it. Use alternatives: remove the `#` entirely (write "PO/WO 750" instead of "PO/WO #750") or rephrase to avoid it. Same applies to any character with special URL meaning — if unsure, test the URL by inspecting the encoded `text=` parameter for unexpected truncation.
- **Memory-write race condition.** When the user asks you to remember a fact in the same turn as a message, the memory write can fail if memory is at capacity. In that case: still deliver the message (don't block on memory), but tell the user the memory write failed and offer to consolidate.
- **"RD" in DRA land vocabulary = Real Estate Due Diligence** (road access, competing projects/pricing, infrastructure, demand sources). When drafting an RD assignment message to Prakash Singh (land team), the standard brief covers those 4 items, notes the site-visit option with cab reimbursement, and flags ownership red flags (multiple owners, court stays) from the Kelsa lead. The Kelsa lead URL goes in the message so he opens the record directly.
- **Do not transcribe and translate.** The user said "Sahabji" — they mean "Sahabji". They did not ask for an English translation. The "translation" temptation comes from the LLM reflexively normalising Indian-English/Hindi/Urdu names to English. Resist it.
- **Designation suffixes are not part of the contact name.** When the user says "Nagrajappa JDTP" or "Mohan Sir JDTP GBA East", the suffix (JDTP, BBMP, GBA East, Sir) is a role/location descriptor, not the person's name in the contact book. Strip designations before searching Google Contacts. The contact is likely saved under just the given name. If the initial search with the full string returns nothing, retry with the bare name only.
- **Voice corruption of role acronyms (2026-08-21):** The user's voice may corrupt the role itself — "ADTP" for "JDTP" (Joint Director Town Planning), "GBA" for various town-planning divisions. This compounds the designation-as-name problem. When a search for "Mohan ADTP" returns nothing: (1) recognise the role acronym could be corrupted, (2) strip the entire suffix and search the bare first name via People API `searchContacts()` directly, (3) search with the correct role acronym as a separate fallback. **Worked this session:** contact_resolver("Mohan ADTP") → nothing; People API `searchContacts(query='Mohan')` on google-draas → "Mohan Sir JDTP GBA East" (+91 98868 85455). The stored display name includes the role in the field (e.g. "Mohan Sir JDTP GBA East") which the standard resolver's tokenisation misses.
- **Gov/official contacts may be stored under a DIFFERENT spelling than NDR dictates — search ALL phonetic variants AND the stored designation-as-name (2026-09-03, "Nanjesh").** NDR dictated the Ranka Iris electrical engineer repeatedly as "Nagesh" / "Nunjesh" / "Nanjesh"; the People API contact is stored as literally **"Nanjesh Bescom Aee"** (+91 99002 49798) — different first-letter spelling AND the designation (AEE) as part of the display name. `contact_resolver("Nagesh")` returns decoys (Nagesh Contractor 99001 43733, Nagesh U G Consultancy 98802 03100, H M Nagesh Electrical Consultants 98450 58642, Nagesh Y.N. ING VYSYA) — none is the BESCOM man. Recipe: iterate the spellings the user actually said (Nagesh→Nanjesh→Nunjesh), plus keyword variants ("bescom", "electrical inspector", "noc", "transformer"), via `people().searchContacts()`; when a hit has org=[] and bio=[] but the NAME carries the designation ("Bescom Aee"), that IS the person (a phone-added contact with no org enrichment). Cross-check against the tracker row that names the counterparty (P-066's H column says "Nagesh (NOC reqd)" / "Anbu (Nunjesh)") — the tracker keeps his role, the contact keeps his number.
- **"Accounts Guru" / "accounts group" is a GROUP chat, not Eshwari.** When the user says "my Accounts Guru" or "the accounts group" for an internal finance message, the recipient is the accounts team group chat (Eshwari + other members), NOT Eshwari alone. Confirming "Accounts Guru = Eshwari?" is a reasonable check, but expect the correction: it's a group. For group recipients, call `whatsapp_link(text=message_text)` WITHOUT the phone parameter (no pre-set recipient) so the user can paste into the group — see "Group messages" above.
- **Wait for promised attachments/amounts before composing.** When the user says "I'm going to share both the images and the exact amount right now, then you create the message," do NOT draft the final message yet — confirm the recipient and context, then hold until the images/amount arrive. A draft built on guessed figures wastes a round-trip.
- **WhatsApp-synced contacts may not appear in Google Contacts.** A contact that the user can see in their WhatsApp contact list may not be accessible via the People API or the NDR DRAAS Contacts Sheet. WhatsApp can sync contacts from the phone's local/SIM storage, which isn't exposed to the Google People API. If a search with both the full name (minus designations) AND the bare name yields nothing across all five contact sources, do NOT conclude the contact doesn't exist — tell the user you couldn't find it in indexed sources and ask them to share the phone number. "Not found in Google Contacts" and "not in the sheets" together still means "could not locate programmatically," not "doesn't exist." Prefer asking for the number over guessing which of several partial matches is correct.
- **Pattern B (response to a third-party message) will go through 2-3 refinement rounds.** Do not treat the first draft as final. Do not generate the WhatsApp link until the user confirms. Each round: apply the correction to one or two paragraphs, present the full revised draft cleanly. Keep paragraph structure stable so the user can track what changed.
- **Pattern C (congratulation + request) deliberately breaks the "no pleasantries" rule.** Government officials and authority figures expect formal, respectful language — "Sir", "grateful", "please let me know". Do not apply the standard WhatsApp direct-style rule to this pattern. Switch tone based on recipient type, not just medium.
- **Inline Python quoting breaks with Gmail API dict access.** When calling the Gmail API via inline shell (`python3 -c "..."`), dict key access like `m["id"]` gets mangled by the shell. The `execute_code` sandbox also lacks googleapiclient in its default path. Instead: write a standalone `.py` script to `/tmp/` with `write_file`, chmod +x it, and run it with `/opt/hermes/.venv/bin/python`. This avoids both the quoting issue and the dependency issue. The venv at `/opt/hermes/.venv/` has all Google client libraries installed.
- **Vault socket unreachable kills the whole contact chain.** If `gws_resolve_account` or `build_service` errors with "Vault socket unreachable at <path>", the env var `GWS_VAULT_SOCKET` may point at a stale path while the live socket lives elsewhere (observed 2026-07-31: configured `/opt/data/gws-vault/run/vault.sock`, live `/run/gws-vault/vault.sock`). Fix: locate the live socket (`find / -name vault.sock -maxdepth 6`), then in your script set `os.environ['GWS_VAULT_SOCKET']` and `os.environ['GWS_VAULT_TOKEN_DIR']` BEFORE importing `tools.gws_auth`. People API + Sheets is the backbone of Step 1 lookup — recover the socket instead of falling back to "ask the user". `scripts/find_contact.py` already does this recovery automatically.
- **A follow-up voice message that re-spells a name is authoritative.** Session example (2026-08-14): first dictation said "Nokesh Gandhi", the user's very next message said "Lokesh Gandhi" — and contacts confirmed Lokesh. When the user repeats a name with a different spelling, treat the NEWEST spelling as the correction, verify against contacts, and proceed (note the correction briefly). Do not re-ask which spelling is right.
- **"He hasn't responded to my emails" follow-ups — dig the REAL email dates + address from Gmail BEFORE composing (2026-09-03, Dharmesh).** When NDR dictates a follow-up referencing sent emails ("you have not responded to the succession docs I have been sending"), he expects the message to name the actual dates and the exact address used: search Gmail first (`succession to:<addr>` / `from:ndr@… subject:<topic>` across google-draas), pull the thread dates + subjects, and embed them in the message ("1 Sep: 'DR Family Succession Petition…' · 2 Sep: … · 3 Sep (today): … — emailed to dharmesh.ranka1@gmail.com"). Also resolve which of the contact's multiple emails was actually used (People API/sheets may list several — the Gmail To: header is ground truth). Do NOT say "the emails I sent" generically; NDR wants the audit trail visible to the recipient.
- **Dictated recipient names in group messages can be pure STT fiction (2026-09-02).** A voice-dictated DRA-group message named "Yadj" — no such person in Google Contacts, either contacts sheet, or the DRA roster. It was kept as dictated ("internal group, team knows each other"), and the user's corrective follow-up came back: "there's no YADG, it's some kind of a transcription error, you can ignore that — it's Gowri and Bharat only we are sending it to." Rules: (1) when a named recipient in a group message is NOT found in any contact source, flag it explicitly in the delivery reply ("X isn't in any contact source — kept as dictated; is the spelling right?") so the correction lands BEFORE he forwards it; (2) when the user says the name doesn't exist, DROP it entirely on regeneration — never preserve a transcription-fictional name for fidelity; (3) re-read the whole message when regenerating — instructions attached to the fictional person (here: "push this message out to our existing leads") die with the name unless the user re-assigns them.
- **Introduction messages that share a third party's contact details** ("I'm connecting X with Y"): look up the third party with the same lookup chain, include labelled contact lines in the message (`• Lokesh Gandhi` / `• Mobile: +91 …`), and if no email exists on file, state that plainly to the user rather than inventing one. Worked example: Lokesh Gandhi — phone only (+91 94488 45692); Gmail had only old WhatsApp-chat exports, no real address, so the message carried phone and NDR was told "no email on file anywhere".
- **Clearly-wrong technical terms in dictation: correct in the body, flag in the reply.** NDR said "HIPAA filters" meaning **HEPA** (the recipient — his wife — needs the right term to protect the device). When the dictated term is factually wrong (HIPAA/HEPA, ENY/EY, Roohan/Ruhaan), use the correct spelling in the message body and note the correction to the user in your reply. This is a correction, not "inserting facts" — the recipient must not act on a wrong term. **Project-internal vocabulary (2026-09-03):** also correct and flag rendered English from voice: "N-Transportal"→internal transport, "Mokap Vilas"→model villas, "GoDridge"→Godrej (commitment counterparty), "OSS"→Oasis (project), "Prism Grasston building"→Prizm & O3, "AW at the virtual side of Westcom"→AEE, BESCOM. Always state the correction in the reply AND put the corrected term in the body — the recipient (colleague or counterparty) must see the real term.
- **Real-estate project names + street names garble TOGETHER — verify via web search, correct both, regenerate on follow-up re-spelling (2026-09-12, Bharat Krishnamurthy chat).** One dictation contained TWO corrupted proper nouns in the same sentence: "Legacy Kataria on Kanigam Road" was really **Legacy Cataleya on Cunningham Road** (Legacy Global Projects, Vasanth Nagar — verified via web search; there is no Legacy Kataria in a Bangalore context matching the dictation, and no "Kanigam Road" for that project). Recovery recipe: (1) web-search the dictated project-name variants ("Legacy Kataria Bangalore") — the real developer's listing shows the actual address, which also exposes a garbled street; (2) put the VERIFIED names in the message body and flag every correction in the reply, so the recipient sees real terms and NDR can veto; (3) if NDR follows up with a voice re-spelling of the project ("Legacy cateleya"), treat it as authoritative and regenerate the link immediately — it confirms the correction, it doesn't reopen the draft; (4) when one proper noun in a sentence is garbled, actively re-check the other — STT usually mangles both ("Kataria"→"Cataleya" and "Cunningham"→"Kanigam" in the same pass). This extends the general "clearly-wrong terms" rule to real-estate deal chatter, where the recipient is a buyer deciding whether the property is worth a call — a wrong project/road kills the pitch.

## "Chief" is a SALUTATION, never a recipient name (verified 2026-09-10)

NDR's casual WhatsApp messages to brokers / associates / known contacts frequently
OPEN with the word **"Chief"** — "Hi Chief,", "Good morning chief!", "Hey Chief,",
"Chief, …".
- Aug 12 — *"Just here chief, a very casual message following up any updates on the various matters you are tracking"* → meant **"Hi Chief, …"** (message went to Sunny Sadhwani, the broker)
- Sep 7 — *"a whatsapp message for [X] telling him good morning chief, just wanted to give you a heads up — Architect has come back for the premium development…"*
- Aug 16 — delivered message body began *"Chief, if you don't mind…"*

**Rules:**
- When a dictation says "from the Chief" / "to the Chief" / "here chief", do NOT search the
  contacts sheet, People API or directories for a person called "Chief" (there is none —
  only job titles like "Chief Architect"). Read it as the salutation to open the message with.
- The recipient is whoever NDR named in the same sentence; if he named nobody, ASK — do not
  invent one. Never treat "Chief" as the addressee.
- Keep "Chief" verbatim in the drafted message body when NDR dictated it — it is his voice.

## Voice-memo dictation hygiene

When the message arrives as a voice memo (Telegram transcribes these automatically), expect:

- Long run-on sentences with "So" / "Basically" / "And also" / "Another matter" — leave them, they're natural WhatsApp cadence
- Names of people the user has not introduced (Amit Pujari, Anbu, Nishant Prakash) — keep them, do not ask "who is Amit Pujari?" — the recipient knows
- Indian-English spelling variance (Tahasildar / Thasildar / Thaisildar, CLU / Change of Land Use, conversion cancellation) — quote the user's choice, don't normalise
- Currency as "X cr" / "X lakh" / "Rs.X" / "INR X" — keep as user said

## Reference files

- **`references/contact-card-format.md`** — DRAAS contact cards, voice-memo dictation gotchas, memory-capacity replacement order.
- **`references/hospital-medical-contacts.md`** — NDR's recurring hospital contacts cluster (Dr. Haldipur, Trustwell, Shreedhar, insurance coordinators). Load when the message context is medical/surgical.
- **`references/medication-procurement.md`** — Medication procurement workflow: when the user asks to send someone to buy medicine (driver, family member). Covers two-phase flow: (1) dosage verification before sending, (2) message composition with molecule name, brand names, quantity, dosing schedule.
- **`references/dra-realty-group-roster.md`** — DRA Realty Group employee WhatsApp roster (14 people, verified Aug 2026): name → phone → email/role, org-broadcast pattern. Load for any org-wide message ("message everyone in the group") — resolve numbers from here, then follow `messaging-drafts` → `references/messaging-links-workflow.md` §4a for the batch one-link-per-person workflow.
- **`references/government-official-messaging.md`** — Government/authority-figure messages (Pattern C): congratulation + request structure, full worked example from this session (JDTP site visit after Karnataka Apartment Owners Bill amendment), tone rules, and common contexts. Load when the recipient is a JDTP, BBMP official, BDA officer, or any regulatory authority.

## Support scripts

- **`scripts/find_contact.py`** — one-shot contact lookup for Step 1: People API searchContacts + connections fallback, then both DRA contacts sheets, printing labelled phones (Work vs Mobile). Includes automatic vault-socket recovery. Run with full path: `/opt/hermes/.venv/bin/python /data/hermes/skills/productivity/personal-messaging/scripts/find_contact.py "Bharat Hawaldar"`

## Related skills

- `email-drafter` — formal email via Gmail draft. Different formality, different tool, same "draft, don't send" safety boundary.
- `regulatory-complaint-escalation` — formal complaint to a regulator/ombudsman. Different recipient type, but the underlying workflow (voice memo → research → multi-source) overlaps.

## Cross-skill references

- **Ranka Amber SSA registration** (Raghu Iyer landowner correspondence — thread facts, counterpart constraints, open DFAT/notarisation item): see `email-drafter` skill → `references/ranka-amber-ssa-registration.md`. Load when drafting ANY message/email around the Ranka Amber landowner group or Raghu Iyer.
