# Meeting scheduling — "1 hour this week" pattern

> Extracted from the messaging-drafts SKILL.md on 2026-09-20 (SKILL.md had hit the 100 KB limit).
> This is scheduling/calendar work rather than message drafting; kept for reference.

## Meeting — 1 Hour This Week
I want to hear your thoughts on:
• Specific question 1
• Specific question 2
• etc.

Nishant
```

**Natural case for project/topic descriptions:**
- Keep descriptions in natural sentence case: "Estate homes & Villa homes" not "Estate Homes & Villa Homes", "investor opportunities" not "Investor Opportunities"
- Do NOT artificially capitalize every word in headings or bullet descriptions
- Exception: Proper project names (Ranka Amber, Ranka NorthStar) keep their trademark capitalization

**Key distinction — Collaborative, not Commanding:**
- Ask for input: "Where do you feel comfortable starting?" rather than "This is what we're doing"
- Frame as discussion kickoff, not directive

### Pitfall — Document Bundle Email: Permissions Before Send (Jul 2026)

**Trigger:** You compile a multi-file research document (HTML timeline, report) and need to email a link to one or more recipients referencing a set of Drive files.

**Problem:** The natural sequence — create document → upload → send email with links — is WRONG for DRAAS. Doing it this way means the email goes out before the recipient has access to the underlying files, forcing a corrective follow-up.

**Correct workflow — Permissions FIRST, email SECOND:**

1. **Identify ALL files** referenced in the document/email — not just the main doc, but every supporting file (sale deeds, NOCs, title opinions, etc.)
2. **For each file**, set the recipient as **Viewer (Reader)** with **1-month expiry** on the permission
   ```python
   from datetime import datetime, timedelta, timezone
   expiry = (datetime.now(timezone.utc) + timedelta(days=31)).isoformat()
   drive.permissions().create(
       fileId=FILE_ID,
       body={'type': 'user', 'role': 'reader', 'emailAddress': 'recipient@example.com',
             'expirationTime': expiry},
       sendNotificationEmail=False
   ).execute()
   ```
3. **Remove unnecessary existing viewers** — if Nishant says "remove all other viewer access," remove user-type permissions for everyone except Nishant (owner) and the intended recipient. Do NOT remove 'domain' or 'anyone' permissions without explicit instruction. Note: you can only remove permissions on files you own (i.e., files where ndr@draas.com is the owner).
4. **Verify every file** — confirm the recipient shows in the permission list before sending
5. **Only THEN send the email** with clickable links to every referenced file
6. **Email must contain clickable links** — not just descriptive text. Each Drive link should be an `<a href>` in HTML or a raw URL in plain text that the recipient can tap to open.

**Why this matters (Jul 2026 evidence):** A Bin Mangala timeline email was sent with the document link but the referenced files (23 Drive files across Binnamangala and Arya Developers folders) had no permissions set for the recipient. The user had to explicitly correct: "you have not given viewer access to Rahul. Please give him complete viewer access on all files and folders for one month and remove all other viewer access."

**Checking permissions is cheaper than fixing after send:**

**Trigger:** Nishant says "create a detailed instruction sheet" or asks you to draft an email to a third party (Raghu, Manohar, etc.) explaining a multi-step process (DSC application, E-signing, registration steps, NOC process, etc.)

**Problem (corrected Jun 2026):** The default email style is brief and direct. But for instructional emails to third parties, "brief and direct" means "too vague to act on." The first draft in this session ended with "If you need any help or assistance with any step, let me know" — which sounds helpful but actually forces the recipient to figure out the process themselves. Nishant corrected: the email should be a **self-contained guide** the recipient can follow end-to-end without needing to ask for clarifications.

**Fix — write the full guide inline, not a summary:**

1. **Context section first** — 1-2 paragraphs explaining WHY this step is needed (e.g., "The sub-registrar only accepts E-signatures. Raghu cannot double E-sign as both himself and as a representative because the system doesn't support it.") This frames the motivation.

2. **Numbered step-by-step instructions** — Each step a clear heading with:
   - Exact portal URL (e.g., `https://esign.emudhra.com/`, not just "the eMudhra portal")
   - What to do at each stage
   - What documents to prepare (as a checklist)
   - Expected timeline / cost estimates
   - Expected outcome per step (e.g., "You will receive an email with a download link and password")

3. **Include the "why" for each major step** — Not just "upload documents" but "the video verification call is recorded — this becomes the legally valid audit trail." The recipient needs to understand the significance, not just the mechanics.

4. **Anticipated friction points** — Call these out explicitly:
   - "There may be a small video verification call that she will have to set up"
   - "Post that once she has uploaded all the documents, she will be issued a DSC which will arrive in an email with a link to the file as well as the password"
   - "Once that is installed on a computer, then when the document comes through for E-sign, she can use it"

5. **Summary table at the end** — Who does what, by which method. Quick reference the recipient can scan.

6. **Offer help as a closing sentence, not a crutch for missing detail** — One line: "If you need any help, let me know." But the steps should already be complete enough that the recipient doesn't need to ask.

**How this differs from the standard brief style:**

| Scenario | Approach |
|----------|----------|
| Internal coordination (Gowri, Anbu, Roshni) | Brief, direct — they have context |
| External party needing instructions (Raghu, Manohar, clients) | **Self-contained full guide** — assume zero prior knowledge of the process |
| Mixed (internal email that also instructs vendor) | Full instructions for the vendor part, brief for the internal coordination part |

7. **Verify portal/website names before finalizing** — Voice transcription easily mangles portal names (e.g., "eMudhra" → "MUTRA", "esign.emudhra.com" → "Mutra portal"). Before sending an email with a portal URL or platform name, **pause and verify the exact spelling with the user**. This is especially critical for:
   - URLs in step-by-step instructions — a wrong URL means the recipient can't follow Step 1
   - Official portal names (eMudhra, NSDL, MUTRA, Kelsa, etc.) — voice homophones are common
   - Company/project names that sound similar (e.g., "Ranka Amber" vs "Ranka Udaya")

   **Fix:** When the user dictates a portal name in voice, after drafting the email, quote the portal name back: "Just confirming — the portal is **esign.emudhra.com** (eMudhra), correct?" Do this as part of the pre-send confirmation, not in a separate round trip.

**Why this matters:** Writing a summary first and waiting for "redo this in detail" wastes a full round trip. The user's voice instruction "create a detailed instruction sheet" or "request him to please E-sign..." is already the signal to go deep. Default to detailed on first draft when the email content involves a technical/registration process for a third party.

### Creating a Google Meet invite (verified 2026-09-21)

NDR often says "create an event ... and Google Meet is sent out ... set it up for today".
Working pattern via NDR's own token (`service_name='google-draas'`,
`/opt/hermes/.venv/bin/python`, run from `terminal()`):

```python
cal = build_service('calendar','v3', service_name='google-draas')
ev = {
  'summary': '<name> (Vineet (Jiraaf), Vishwas (Vantage Point), Prakash, Nishant)',
  'description': <agenda + notes>,
  'start': {'dateTime':'2026-09-21T15:00:00+05:30','timeZone':'Asia/Kolkata'},
  'end':   {'dateTime':'2026-09-21T15:30:00+05:30','timeZone':'Asia/Kolkata'},
  'attendees': [{'email': e} for e in [...]],
  'conferenceData': {'createRequest': {
      'requestId': '<unique-string>',
      'conferenceSolutionKey': {'type': 'hangoutsMeet'}}},
}
r = cal.events().insert(calendarId='primary', body=ev,
        conferenceDataVersion=1, sendUpdates='all').execute()
r['hangoutLink']   # -> https://meet.google.com/xxx-xxxx-xxx
```

- **`conferenceDataVersion=1` is REQUIRED** — without it the `conferenceData` block is
  silently dropped and there is no Meet link.
- **`requestId` must be unique per event** — reuse across events collapses them onto the
  same conference; use a slug + date.
- **`sendUpdates='all'`** actually emails the invites; omit it and attendees get nothing.
- Report back BOTH `htmlLink` (calendar event) and `hangoutLink` (the Meet) — NDR asks for
  the Meet link specifically.
- **Check the slot is free first** (`events().list(timeMin=..., timeMax=...)` over the day)
  before creating, so you can flag a clash instead of double-booking.
- Emails come from the contacts roster — confirm each attendee address before insert
  (see the Calendar Event Confirmation Pitfall above). For external deal participants the
  addresses are in the thread headers, not the contacts sheet.

### Calendar Event Confirmation Pitfall (Jun 2026)

**Problem:** In a prior session, a calendar event was created with the wrong attendee email because the agent assumed the contact's email was correct without confirming. The user had to correct both the attendee name AND email address.

**Fix — always confirm ALL of these before creating a calendar event:**
- Event title
- Date and time (with timezone)
- Location
- Attendee(s) — BOTH display name AND email address(es)
- Whether to send update notifications

Show as a structured summary:
```
Title: Various Discussions
Date: 9 Jun 2026, 1:00-2:00 PM IST
Location: DRA Realty Office
Attendees: Nishant Ranka (ndr@draas.com), Nishant Prakash (nishantprakash@theyelloweye.com)
```
Wait for explicit confirmation before creating.

### Calendar Timing — Voice-Transcription Time Ambiguity (Jun 2026)

**Trigger:** User says a time range like "232-330" or "230 to 3" in a voice message to describe a meeting window.

**Problem:** Voice-transcribed time ranges are ambiguous. "232-330" could mean 2:30-3:30, but could also mean 2:30-3:00 with the next meeting at 3:30. The agent parsed it as one long slot; the user corrected it to a shorter slot + separate follow-up meeting.

**Fix — explicitly confirm the end time of each meeting slot, especially when meetings are back-to-back:**
1. Parse the voice-transcribed time range
2. When there are two meetings in sequence, confirm BOTH slots independently:
   - "Meeting A: 2:30-3:00, then Meeting B: 3:00-3:30?" vs "Meeting A: 2:30-3:30, Meeting B: 3:30-4:00?"
3. Don't assume intervals are clean 30/60-minute blocks — voice can mangle the boundary
4. If in doubt, spell out both options and ask the user to pick

**Pitfall example (Jun 2026):** User said "232-330 meeting with Anwar and Somaya" — agent created 2:30-3:30. User then clarified "Anwar Sumeya meeting from 230 to 3 pm, 3 to 330 itself" meaning Anwar/Sumaya was 2:30-3:00 and the Ravi meeting was 3:30-4:00. The correction cost an extra round trip.

### Pitfall — Multi-Stakeholder Task Assignment Emails (Voice Dictation)

**Trigger:** User dictates a complex email via voice assigning specific responsibilities to 3+ people (e.g., "Anbu does X, Bhavik handles Y, Nanjesh handles Z under Anbu's coordination").

**Problem:** Voice transcriptions parse cleanly but the agent's interpretation of who-does-what is often wrong. In one session, the responsibility breakdown required **4 rounds of correction** because the structure (primary responsibility vs coordination vs parallel workstream) wasn't made explicit before the draft.

**Fix — always present a responsibility breakdown table BEFORE writing the full draft:**

1. Parse the voice message into discrete tasks and assignees
2. Present a structured breakdown via Telegram:
   ```
   Responsibility breakdown I parsed:
   • Anbu: BWSSB road cutting, water meter installation (same team)
   • Bhavik: NOC/approval via consultant, meter release
   • Nanjesh (under Anbu's coordination): BESCOM road cutting, transformer delivery + installation, LT panel connection
   
   Confirm this breakdown is correct before I draft the email?
   ```
3. Wait for user confirmation before writing even a first draft
4. Only after the user confirms the structure, compose the email
5. If the user corrects the breakdown, update it and re-confirm before drafting

**Why this matters:** Writing a full draft with the wrong responsibility structure wastes time — the user has to read the whole email to spot the error, then correct it, then ask for a rewrite. A structured breakdown takes 2 lines of Telegram text and catches the error in seconds.

### Pitfall — Voice-Dictated Email Addresses Require Confirmation

**Trigger:** User says an email address in a voice message (e.g., "send to vkdas@draas.com").

**Problem:** Voice transcriptions easily mangle email addresses — similar-sounding names (Vinod/Vinood), domain confusion (drahomes.com vs draas.com), and letter repetition (vkdaas vs vkdas). In this session, the same address was corrected TWICE before getting right.

**Fix — before sending or sharing:**
1. **Pause and confirm** — If you don't have a pre-existing record of that person's email (in contacts sheet, memory, or past emails), ask the user: *"Confirm: vkdas@draas.com — is this correct?"* Spell it out letter by letter.
2. **Cross-check from a known source** — Before sending, search the contacts sheet, memory, or Gmail history for this person's verified email. If found, use that instead of the voice transcription.
3. **On first use, confirm** — The first time an email is used for a new recipient in a session, show the full address to the user and ask for confirmation before sending.

**Verification step:** Before calling `gmail.users().messages().send()`, print the To: address to stdout and verify against what the user said.

### Pitfall — Phone Number from Email Signature Is NOT Authoritative

**Trigger:** You extract a phone number from someone's email signature and use it for a WhatsApp link or contact record.

**Problem:** The number in someone's email signature may be their office/main line, not their personal/mobile number. The user's Google Contacts likely has the correct mobile number. Using the signature number without cross-referencing wastes a round-trip when the user corrects you (confirmed Jun 2026: Nilesh Prasar's email signature showed `+91-8095506021` but his Google Contacts had `+91 99059 54753` as mobile).

**Fix — always cross-reference before using:**
1. Extract the number from the email signature as a candidate
2. **Search Google Contacts (People API)** — `searchContacts(query="Last Name")`
3. **Search contact sheets** — NDR CONTACTS sheet, NDR DRAAS Google contacts
4. **Present ALL numbers found** with their sources (email signature, People API, sheet)
5. Let the user confirm which number to use
6. Only then generate the WhatsApp link

**Exception:** If the user explicitly says "use the number from their email signature" or "send it to this number," use it directly.
