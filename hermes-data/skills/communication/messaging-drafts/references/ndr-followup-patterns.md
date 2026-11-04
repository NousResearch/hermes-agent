# NDR Follow-up Patterns (worked examples)

Concrete patterns extracted from real Hermes sessions. Use these as templates
when the next voice-message follow-up comes in — they show the typical NDR
shape (multi-topic, named addressing, &-free text) and the working call to
`whatsapp_link`.

## Pattern 1: Single broker / chief follow-up (Sunny Sadhwani)

**Trigger**: user says "message [name], [name], also known as [name], do a
contact lookup" + voice message with multiple topics.

**Contact resolution**:
- One Google contact, two display names (alias). Trust the contact record.
- Email: `rsadhwani13@gmail.com`, mobile: `+91 98450 70013` → use
  `canonicalForm` `+919845070013`.

**Voice-to-text cleanup (the actual session)**:

| Voice transcript fragment           | Cleaned text in message                |
|-------------------------------------|----------------------------------------|
| "Premium FAR + TDR"                 | "Premium FAR plus TDR"                 |
| "FAR 4.5"                           | "4.5 FAR"                              |
| "multiple matters"                  | "the multiple matters"                 |
| "out of curiosity"                  | kept verbatim                          |
| "&" anywhere                        | "and" / "plus" / em-dash               |
| "₹27.5 Cr"                          | "Rs 27.5 Cr" (avoid `₹` in source)     |

**Working call**:
```python
whatsapp_link(
    phone="+919845070013",
    text="""Chief, checking in again — following up on the multiple matters
you are handling for me:

1) Ranka Iris — ...
2) Regalia units — ...
3) Hermitage rental (Mainan Dharmeshis, Floors 22 and 23) — ...
4) Legacy Catalina — ...

Thanks Sunny."""
)
```

**Tone markers**: "Chief" greeting (user used it), numbered items, "Awaiting
your update" close, "Thanks Sunny" sign-off (user used the first name).

## Pattern 2: Group message to institutional recipients (Godrej Venture)

**Trigger**: user says "use the WhatsApp tool, don't encode any number in
it. I'm putting the message in a group. This is for [name1], [name2], [name3]."

**Recipient resolution**: user picks the group in WhatsApp at compose time.
Pass empty phone to the tool. Do NOT pass a single recipient's number.

**Critical tool call shape**:
```python
url = whatsapp_link(phone="", text=GROUP_MESSAGE)
# → https://wa.me/?text=...  (no phone= segment)
```

**Voice-to-text cleanup (BuxRanka thread)**:

| Voice fragment                              | Cleaned text                  |
|---------------------------------------------|-------------------------------|
| "addressing Viraj, Saurabh and Amit"        | "Good morning Viraj, Saurabh, Amit." |
| "Amit, as per our last telephonic…"         | keep verbatim, formal opener  |
| "non appeal" / "non appealable"             | use "non-appealable" once     |
| "FEMA invested entity"                      | keep verbatim                 |
| "BDA has not designed a mechanism to split" | keep verbatim                 |
| "buy it and load up to that much"           | "buy it and load up to that extent" |

**Tone markers**: "Good morning [names]" opener, addressed paragraphs (one
per person), long multi-item ask, "Awaiting your update and feedback" close.

## Pattern 3: Pair WhatsApp link + Gmail threaded reply

This is the BuxRanka session's actual deliverable shape — voice message with
many topics, user asks for BOTH a WhatsApp group message AND a reply to an
existing email thread.

**Sequence**:
1. Voice → extract topics, identify named recipients.
2. Google People search to resolve contact identities (cross-check against
   Gmail thread participants — see `email-drafter` §P1).
3. Gmail thread search (use the BuxRanka example: query for the
   most-recent thread with all of `Amit OR Saurabh OR Viraj` + TDR/FAR
   keyword).
4. Fetch the latest message in that thread to get `Message-ID` and
   `References` headers (raw Gmail API — see `email-drafter` §P2).
5. Build Gmail draft via raw `users().drafts().create()` with
   `In-Reply-To` and `References` headers set, then verify per
   `email-drafter` §4 (draft exists, NOT in sent).
6. Build WhatsApp link via `whatsapp_link(phone="", text=...)` for the
   group version (terser, no subject line, addressed paragraphs).
7. Present both to user with clear "tap the link" / "review draft and
   send" instructions.

**Tonal split between the two**:
- WhatsApp: terser, conversational, addressed per person.
- Email: more formal, numbered (1) (2) (3), full context, signed
  "Nishant Ranka", includes the long-term context the email
  recipient may need to forward it.

## Pattern 4: Apology / delay-with-reason follow-up (Aamir Khan)

**Trigger**: user owes someone feedback, a review, documents, or a deliverable and is running late. Message includes:
- Sincere apology
- Specific deliverable (video feedback, PPT, document)
- Reason for delay that is specific and believable (mom's hospital discharge, needed to sit at a computer for the final 10%, can do 90% on phone via AI)
- Commitment with a specific time window ("today without fail", "before 3-3:30pm")
- No excuses or defensiveness — just facts + commitment

**Contact resolution**: standard Google Contacts lookup. NDR's Terra Greens partner Aamir Khan is at +91 98458 81652, email aamirkhan@me.com.

**Voice-to-text cleanup (the actual session)**:

| Voice fragment                                    | Cleaned text                                |
|---------------------------------------------------|---------------------------------------------|
| "I have this huge feedback put up"                | kept verbatim                               |
| "I wanted to clean up some aspects of it"         | kept verbatim                               |
| "90% of my work on the phone using my AI"         | kept verbatim ("my AI" is accurate context)  |
| "10% last bit and I didn't get a chance"          | "but that last 10% I didn't get a chance to do" |
| "I will do it today without fail"                 | kept verbatim                               |
| "getting mum's discharge sorted in the morning"   | kept verbatim                               |
| "most likely before 3, 3.30pm"                    | "most likely before 3-3:30pm"               |

**Tone markers**: open with "my sincere apologies" (stronger than "sorry"), then state the deliverable, then the reason (not as an excuse — as context), then a firm time-bound commitment. Close without "thanks" if the relationship is close enough — the commitment itself is the close.

**Working call**:
```python
whatsapp_link(
    phone="+919845881652",
    text="""Aamir, my sincere apologies — I have this huge feedback put up for the video and the PPT. I wanted to clean up some aspects of it, for which I need to sit on the computer. I'm able to do 90% of my work on the phone using my AI, but that last 10% I didn't get a chance to do. I will do it today without fail. Just getting mom's discharge sorted in the morning. As soon as I reach the office, the first thing I'll do this and post it — most likely before 3-3:30pm."""
)
```

## Pattern 5: Hospital insurance / discharge follow-up (Charan — Trustwell)

**Trigger**: user needs a status update on hospital insurance claim processing and discharge summary readiness. Message is to the insurance coordinator (not the operations coordinator).

**Key distinction**: there are TWO coordinator roles at the hospital:
- **Insurance coordinator** (Charan) — handles TPA pre-auth, claim filing, reimbursement
- **Operations coordinator** (Sridhar/Shridhar) — handles discharge summary signing, room logistics, doctor coordination

When NDR says "send it to Charan only" — they know the roles and expect you to distinguish them.

**Tone**: direct, numbered questions, no pleasantries. The recipient knows who they are and what's expected.

**Working call**:
```python
whatsapp_link(
    phone="+919845252011",
    text="""Charan — just following up. Has the insurance claim been filed? Is that process already underway? Also, has the discharge summary draft been given, or is the discharge summary already prepared and signed by the doctor?"""
)
```

## Pattern 6: Landowner / Partner Project Update (Ranka Northstar group)

**Trigger**: user needs to send a formal progress update to a landowner group, investor group, or JV partners about a real estate project. Message includes project milestones, regulatory updates, NOC status, government timelines, and next steps.

**Key differences from P2 (institutional group message):**
- Addressed to "everyone" / "hi everyone" — not named individuals (unless a PS is added)
- **Formal progress report**, not a request or follow-up
- Includes specific regulatory/approval language (gazette notifications, objection periods, departmental processing)
- Structured with section headings or bold labels
- Ends with "Regards, [Name]" not "Awaiting your update"
- May include a **PS addressed to a specific person** within the group (e.g. a senior partner who needs to handle a separate matter)

**Contact resolution**: these go to a WhatsApp group — pass empty phone to the tool.

**Voice-to-text cleanup (Ranka Northstar session)**:

| Voice fragment | Cleaned text |
|---|---|
| "the 30 day public objection period for the new update in modification of the KPTC and norms" | "The 30-day public objection period for the proposed modification (allowing buildings up to 21m to not require fire NOC)" |
| "by 15th of July, 15th to 21st of July, the notification will get gazetted" | "The target is for the notification to be gazetted between 15th–21st July" |
| "four airport NOCs, BESCOMB, BWSSB and PCB NOC and a BSNL NOC" | "4 Airport NOCs, BESCOM, BWSSB, PCB, and BSNL" |
| "apartment association owners and management bill, the SIR going on, the state SIR as well as the central SIR" | "the new Apartment Association Owners & Management Bill, state & central SIR" → rewrite `&` to "and" before calling the tool |
| "to Kanshika MAM's arrival in India" | "Kanshika Ma'am's arrival in India" |

**Preferred layout (worked example from session):**

```
Hi everyone — it's been a while since the last update, so wanted to keep
you all posted. I completely understand the anxiety to get this project
launched.

Update on KPTC Norm Modification:
The 30-day public objection period for the proposed modification
(allowing buildings up to 21m to not require fire NOC) expired on 4th
July. Officials are currently processing the few objections received.
The target is for the notification to be gazetted between 15th–21st
July. Post that, they will begin processing plan sanction applications
— including ours.

NOC Status:
All NOC applications have been submitted — 4 Airport NOCs, BESCOM,
BWSSB, PCB, and BSNL. Once taken up for processing, we expect
completion within roughly 3 months, given that department officials
are currently stretched with other government priorities (the new
Apartment Association Owners and Management Bill, state and central
SIR, etc.).

Next Steps:
Once our application clears Revenue department and moves to Town
Planning, we intend to start full site activity — setting up the
marketing office and other infrastructure — while we await the
sanction to begin construction.

Hope you find this update helpful.

Regards,
Nishant

P.S. — for Swaminathan Sir:
We are waiting for Kanshika Ma'am's arrival in India to get the
modification addendum to the JDA registered. This will be a
prerequisite for both banking and RERA applications going forward.
Request you to please advise.
```

**Tone markers**: "Hi everyone" opener (not "Good morning" — less formal for a long-term partnership), bold-capped section labels (UPDATE, STATUS, NEXT STEPS), numbered timelines, explanation of delays (government stretched = empathy, not excuse), Regards sign-off, and a distinct PS block for a senior addressee.

## Pattern 7: Single-ask review / confirmation request (colleague, content review)

**Trigger**: user needs a colleague to review a specific document, briefing, or deliverable and confirm its accuracy. Message includes:
- A specific artifact (HTML briefing, spreadsheet, PDF, proposal)
- A named recipient — the expert/subject-matter owner who needs to fact-check
- A clear ask: review every fact and figure, confirm correctness
- Context about who the deliverable is ultimately for (e.g. "for Gowri (content head) to work on")

**Key characteristic**: this is a **single-ask message**, not a multi-topic follow-up. There is one deliverable, one ask, one recipient. No numbered items. No closing pleasantries beyond a quick "thanks". Brevity signals focus and urgency.

**Contact resolution nuance**: the recipient is described by their role ("my colleague") and name — but the name may not exist in Google Contacts if the colleague is a new-ish joiner or the contact was never saved. Resolution sequence when the name doesn't appear in Google People search:
1. `session_search` for the person's name in past transcripts (most DRAAS colleagues appear in email CCs or past WhatsApp drafts from prior sessions)
2. Cross-reference the found phone/email against the NDR DRAAS Contacts Sheet (spreadsheet ID `1XbSRAXxPLY4cXMTm2rmvKh11Nx3x0aKUxxuWualoV9g`)
3. If still not found, ask the user for one of: phone, email, or confirmation of the person's role at DRAAS

**Voice-to-text cleanup (Prakash Singh session)**:

| Voice fragment | Cleaned text |
|---|---|
| "asking him if he has had a chance to review" | "Prakash, have you had a chance to review" (convert 3rd-person dictation to 2nd-person direct message) |
| "the Serenity Hill View complete content-related briefing and suggestions that I have prepared in HTML" | "the Serenity Hill View content-related briefing and suggestions I prepared in HTML" (drop "complete" and "that" — the artifact name doesn't need redundant modifiers) |
| "which I shared with him yesterday" | "yesterday? The one I shared with you" (keep the temporal anchor, convert to direct speech) |
| "that I prepared for the content head Gowri to work on" | "that I prepared for Gowri (content head) to work on" (parenthetical role clarification — the recipient may know Gowri by role or by name) |
| "give me confirmation that he is giving me his feedback and confirm that all the facts, figures, statements are all correct" | "please give me your feedback and confirm that all the facts, figures, statements are correct" (merge the redundant "confirm" + "confirmation" into one clear sentence, keep the triplet "facts, figures, statements") |

**Tone markers**: direct address with first name, no greeting, single paragraph or two short lines, "Appreciate it, thanks" close (not "Regards" — less formal for a colleague). No numbered items, no separate sections.

**Working call**:
```python
whatsapp_link(
    phone="+919739932078",  # Prakash Singh, DRAAS
    text="""Prakash, have you had a chance to review the Serenity Hill View content-related briefing and suggestions I prepared in HTML yesterday? The one I shared with you that I prepared for Gowri (content head) to work on.

I need you to review every fact and figure in there — please give me your feedback and confirm that all the facts, figures, statements are correct. Appreciate it, thanks."""
)
```

**Pitfall specific to this pattern**: the user's voice will mix **third-person dictation** ("asking him if he has had a chance...") with the **intended second-person message** ("Prakash, have you had a chance..."). Your job is to extract the intent and render it in second-person direct speech. Do NOT preserve the "asking him/her if..." framing verbatim — that is the user's instruction TO you, not the message content. Strip the framing, keep the substance, and address the recipient directly.

This is a corollary to §P7 (User dictating to a third party), but specific to single-ask review messages where the dictation framing is "asking him/her if..." rather than "tell him/her to...".

## Common ampersand rewrites (cheat sheet)

| Original                | Rewrite           |
|-------------------------|-------------------|
| `Premium FAR & TDR`     | `Premium FAR plus TDR` |
| `Amit & Saurabh`        | `Amit and Saurabh` |
| `FAR 4.0 & 4.5`         | `FAR 4.0 and 4.5` |
| `cost & timeline`       | `cost and timeline` |
| `terms & conditions`    | `terms and conditions` |
| `research & development`| `research and development` |
| `Q3 & Q4`               | `Q3 and Q4`       |
| `father & son`          | `father and son`  |

**When you can't rewrite**: if the user specifically wants the literal `&`
(e.g. quoting a brand name like "Smith & Co."), use the form
`Smith and Co.` in the message body and add a one-line note above the link:
"The user can manually change 'and' to '&' for Smith & Co. after opening
WhatsApp." This is rare — only when the brand is a known entity with the
ampersand in its actual legal name.

## Pattern 9: Mutual Introduction / Connect Two People via WhatsApp

**Trigger**: user wants to introduce Person A to Person B, and asks you to send a WhatsApp to each sharing the other's contact details. This is two reciprocal messages, not one.

**Key characteristics**:
- TWO `whatsapp_link` calls in the same response — one to each person
- Each message shares the other person's contact info (name, role, email, phone)
- Framing: "connecting you with [Name] regarding [topic]"
- Both links presented together in the same response

**Contact resolution**: standard Google Contacts lookup for both parties. If one has no phone number available, flag the gap — do not hold up the other message.

**Dialogue shape (from session)**:

| Utterance | What to do |
|---|---|
| "send Utsav Jain [his number], and send his contact as well to Manohar Singh, send a WhatsApp to each, sharing the contact of the other" | Generate TWO links: (1) Utsav ← receives Manohar's contact; (2) Manohar ← receives Utsav's contact |

**Preferred layout for each message**:

```
[Name] — connecting you with [Other Name], [Role] at [Company]. His details:

• [Other Name] — [Role], [Company]
• Email: [email1]
• Email: [email2] (if applicable)
• Phone: [phone]

[Context — 1–2 sentences about why they're being connected. E.g. "Feel free to connect on the Kelsa integration work."]
```

**Working call pair (from session)**:

Link 1 — to Utsav Jain (+91 98451 72180):
```python
whatsapp_link(
    phone="+919845172180",
    text="""Utsav — as discussed, connecting you with Manohar Singh, Managing Director of O3 Infotech. His details:

• Manohar Singh — MD, O3 Infotech
• Email (O3): msingh@o3infotech.com
• Email (Kelsa): msingh@kelsa.io
• Phone: +91 98458 90316

Feel free to connect on the Kelsa integration work."""
)
```

Link 2 — to Manohar Singh (+91 98458 90316):
```python
whatsapp_link(
    phone="+919845890316",
    text="""Manohar — connecting you with Utsav Jain regarding the Kelsa whitelabel/integration work we discussed.

• Utsav Jain
• Email: utsav108@gmail.com
• Phone: +91 98451 72180

Please connect and take it forward."""
)
```

**Pitfalls**:
- **Both numbers must be confirmed before generating.** Do not generate the first link and ask for the second number later — hold both until you have both numbers, then generate both links together in one response.
- **Use the same framing verb for both descriptions** ("connecting you with") so the introduction reads as a deliberate pairing, not two unrelated messages.
- **Keep the topic/context consistent** between both messages — the other party's description of "why" should match from both sides.

## Pattern 10: Forwarding Intermediary — "Please forward this to [Senior Contact]"

**Trigger**: user dictates a message intended for a senior person (govt official, partner, landlord) but sends it through an intermediary who will forward it. The intermediary is the WhatsApp recipient; the senior person is the ultimate audience.

**Key characteristics**:
- ONE `whatsapp_link` — to the intermediary, not the senior person
- Message opens with "please forward this to [Name/sir]" so the intermediary knows their role
- Message body is self-contained — the intermediary can copy-paste the WhatsApp text as-is to the senior person
- Content typically involves sensitive business info (MOU terms, land details, price per acre)

**Contact resolution**: the intermediary is usually a colleague, family member, or trusted contact found in Google Contacts or the NDR DRAAS Contact Sheet. The senior person may not be in the contact directory at all — the user expects the intermediary already has their contact.

**Voice-to-text cleanup (this session — Dharmesh + Irfan sir + PEP Realty MOU)**:

| Voice fragment | Cleaned text |
|---|---|
| "apparently he has visited a land in Bestamanahalli, next to TDS factory, Atibeli Aanikal" | "Apparently he visited a land in Besthamanahalli, next to TDS factory, Atibeli-Anekal road — he will know." |
| "if Mr. ML Srinivasa has taken him to the land" | "If Mr. ML Srinivasa has taken him to the land:" |
| "just tell sir that they have signed an MOU with me, a binding MOU, for sale of this entire land, so just keep him informed" | "Just tell sir that we have signed an MOU with PEP Realty — a binding MOU — for sale of this entire land. Just keeping him informed that it has been signed." |
| "next Sanjaya lands" | "next to Sanjaya lands" |
| "they did not have binding MOU with me for 3.8 crores per acre, for 44 acres and 7 gundas" | "they have signed a binding MOU with me at ₹3.80 Cr per acre, for 44 acres 7 guntas" |
| "behind that they said there is additional 100 acres which we will do in phase 2" | "Behind that, they said there is an additional ~100 acres which we will do in Phase 2." |
| "for the front lands, where RTC holder names are Ayappa, Yalappa, Muniraju, Ramayesh" | "This is for the front lands, where the RTC holder names are Ayappa, Yalappa, Muniraju, Ramayesh and various others." |

**Preferred layout**:

```
[Intermediary Name] — please forward this to [Senior Name] sir.

[Context about the senior's prior knowledge — "Apparently he visited a land in..., he will know. If [person] has taken him to the land:"]

Just tell sir that [key business fact]. Just keeping him informed that [summary].

[Specific details — price, acreage, phase split, names, etc.]
```

**Working call (from session)**:
```python
whatsapp_link(
    phone="+919844109010",  # Dharmesh Ranka
    text="""Dharmesh — please forward this to Irfan sir.

Apparently he visited a land in Besthamanahalli, next to TDS factory, Atibeli-Anekal road — he will know. If Mr. ML Srinivasa has taken him to the land:

Just tell sir that we have signed an MOU with PEP Realty — a binding MOU — for sale of this entire land. Just keeping him informed that it has been signed.

If it is the same land — next to Sanjaya lands — they have signed a binding MOU with me at ₹3.80 Cr per acre, for 44 acres 7 guntas. Behind that, they said there is an additional ~100 acres which we will do in Phase 2. This is for the front lands, where the RTC holder names are Ayappa, Yalappa, Muniraju, Ramayesh and various others."""
)
```

**Pitfalls**:
- **The message IS for the senior person, addressed via the intermediary.** Do NOT rewrite it as "please tell Irfan sir that..." (that turns it into a request to relay, which changes the tone). Instead, write the message AS IF it were directly addressing the senior, and frame it as "please forward this to X" at the top. The intermediary simply copies and forwards the text.
- **Verify you have the right intermediary contact.** The DRA/group has multiple people with the same first name (e.g. Dharmesh Ranka at DRA Projects vs Dharmesh Samanth at Young Designs). Confirm by role/family context — the user will say the name in a specific project/land context that disambiguates.
- **Do NOT extract the senior person's phone number from the intermediary's contact.** There is no implication that you should look up the senior's number. The user is relying on the intermediary's existing relationship to handle the forward.
- **No conversion to second-person for the intermediary.** Unlike Pattern 7 (single-ask review) where you strip the "asking him/her to..." framing and address the recipient directly, here you keep the "please forward this to [X]" framing because the intermediary is a channel, not the audience.

## Pattern 11: Research + Share Supporting Docs + Assign Queries via WhatsApp

**Trigger**: user asks you to research a real estate project (address, RERA number, survey number, BBMP sanction plan details), compile findings, upload a brochure/document to Drive, share with a colleague, and send a WhatsApp message with the details plus specific queries the colleague needs to investigate.

**This is a compound multi-step workflow, not a single WhatsApp message.** The steps must execute in order before the link is generated.

**Full sequence:**

1. **Research the project** — web search for name, location, builder, RERA number, BBMP approvals, specifications. If the project name is voice-garbled, use phonetic variants and landmark matching (see P2.3) then clarify with user before proceeding.

2. **Download supporting document** — if a brochure PDF or other document exists online, download it.

3. **Upload to Drive TMP folder** — use the TMP folder (`18p74II2uL32sNDzDDwXzmlOUdJJOTmE-`) for new files.

4. **Share with colleague** — grant viewer access using the colleague's official DRA email (from Google Contacts or memory).

5. **Draft the WhatsApp message** with three distinct sections:
   - **Project summary** — bullet-point key data (address, builder, unit details, RERA #, status)
   - **Supporting document link** — the Drive link to the brochure
   - **Numbered queries** — specific questions the colleague needs to answer (e.g. BBMP OC status, extra floor regularization, as-built justification)

**Working example from session (2026-07-29 — Legacy Cataleya for Rahul Vinod Kumar Das):**

```python
whatsapp_link(
    phone="+919900093813",  # Rahul VK Das
    text="""Rahul — here are the details on Legacy Cataleya (Cunningham Road, next to Ranka Chambers).

Brochure uploaded to TMP folder and shared with you:
https://drive.google.com/file/d/FILE_ID/view

Project summary:
• Builder: Legacy Global Projects Pvt Ltd
• Address: No. 30, Cunningham Road, Vasanth Nagar (next to Ranka Chambers)
• Type: 4 BHK — 24 units, G+14 floors
• Unit size: 4,800-5,100 sq.ft.
• Land area: 0.76 acres | FAR: 3.53
• RERA: PRM/KA/RERA/1251/309/PR/171016/000867
• Status: Ongoing (54% complete), extended completion 30 Sep 2026

Need you to check with BBMP on the following:

1. OC Application — what is the current status of the Occupancy Certificate application for this project?

2. Extra Floor Regularization — there is one additional floor built beyond the sanctioned plan. Has this been regularized? Have they spoken to BBMP about regularizing it as part of an As-Built drawing submission?

3. As-Built Justification — under the As-Built drawings, how is that extra floor being justified? What is the mechanism being used (FAR exemption, density premium, or otherwise)?

Please find out and let me know.

Thanks"""
)
```

**Key structural rules:**
- **Three sections, not one wall.** Project summary → Document link → Queries. Each separated by a blank line.
- **Bullet the project summary** (builder, address, type, size, RERA). This gives the colleague everything they need without opening the brochure.
- **Number the queries** (1, 2, 3) — the colleague can report back by number.
- **Drive link goes in the message body** (not as a separate WhatsApp attachment — wa.me cannot carry files). The file must already be shared with the colleague before the message goes out so they can open it immediately.
- **Verify the colleague's email** for Drive sharing — use `vkdas@draas.com` or whatever the primary DRA email is from Google Contacts.

**Common research sources for Bangalore projects:**
| Source | What it provides | URL pattern |
|---|---|---|
| Official project site | Brochure, specs, amenities | variable |
| AurumProptech (RERA aggregator) | Full RERA record (approvals, FAR, financials, architect, milestones) | aurumproptech.in/pulse/rera/... |
| MagicBricks / 99acres / Housing.com | Address, pricing, possession status | per project |
| Legacy official site | Unit sizes, floor plans, brochure download | legacy.in/project-name |

**Pitfall — voice-garbled project name**: "Century Katalya" → actual was **Legacy Cataleya**. Both the builder name ("Century" → "Legacy") and project name ("Katalya" → "Cataleya") were wrong. Don't assume only one word is garbled. Use landmarks (e.g. "next to Ranka Chambers") as the most reliable anchor, search real estate portals by location, and present top candidates via clarify for confirmation.

**Pitfall — Drive sharing before WhatsApp**: Always share the file with the colleague's email BEFORE generating the link. If the link goes out before the share permission propagates, the colleague gets a "request access" screen instead of the document. Grant via `drive.permissions().create(fileId=..., body={'type': 'user', 'role': 'reader', 'emailAddress': '...'})`.

**Pitfall — multiple emails for one colleague**: Some colleagues have multiple emails (vkdas@draas.com, c2vdas@gmail.com, vkdas@drahomes.in). Share with the primary DRA email first, and also copy secondary emails if the colleague might check those. Google Drive sharing accepts multiple permissions on the same file.


## Common ampersand rewrites (cheat sheet)

**Trigger**: user needs to apologize to a single landowner (not a group) for a delay in the exit/final settlement payment for their undivided land share. Message includes:
- Apology for the delay
- Regulatory/approval-related explanation (not a personal reason)
- Specific authorities named (sub-division authority, town planning authority, BBMP)
- Reference to GPA (General Power of Attorney) from landowners
- Personal visit to departmental contacts as evidence of action
- Assurance of clarity within 1-2 days
- Commitment to confirm exact timeline for documentation closure -> final payment
- Formal closing with "Regards"

**Key differences from P4 (personal deliverable delay):**
- P4: Personal reason (mom's discharge), personal deliverable (video feedback), time-bound commitment ("before 3-3:30pm")
- P8: **Regulatory/bureaucratic reason**, landowner exit payment (Rs Cr level), timeline is "a day or two" for clarity followed by payment, uses "Sir" formality, mentions specific government departments

**Key differences from P6 (group landowner update):**
- P6: Group WhatsApp with milestone updates, section headers, PS block for senior partner
- P8: **One-on-one** to a single landowner, apology-first framing, about exit payment specifically (not project milestones)

**Contact resolution nuance**: Landowners for Ranka Northstar may be in the NDR DRAAS Contact Sheet but without a phone number (only email). If so, the whatsapp_link tool cannot be used -- present the message draft as plain text and ask the user for the phone number to generate the link, or tell them they can copy-paste the message manually.

**Voice-to-text cleanup (this session's actual transcript):**

| Voice fragment | Cleaned text |
|---|---|
| "separation star" | "sub-division authority" (STT corruption of real-estate term) |
| "town planning authority" | kept verbatim (correct transcription) |
| "GPA that I have from the landowners" | kept verbatim |
| "Exiting You" | "existing documentation" or "exit documentation" -- context: documentation of existing approvals does not create extra cost |
| "the LN" | keep as "the lender" or "the LN" verbatim (if the user uses LN as known abbreviation) |
| "departmental contact points yesterday directly personally" | "I personally met the departmental contacts yesterday" (reorder for readability, keep "personally" for weight) |
| "in our day or two" | "in a day or two" |
| "you get your full and final payment" | kept verbatim |

**Tone markers**: Opens with "Sir, apologies" or first-name-only apology per NDR style for landowners with established relationship, states the nature of the regulatory issue clearly (not vague), references specific authorities by name (credibility signal), mentions personal visit to department (shows effort, not passive waiting), commits to "a day or two" (specific window), closes with "Hope you find this in order" + "Regards, Nishant". The message should sound anticipatory -- the delay is almost over, not open-ended.

**Preferred layout (worked example from session):**

```
Sunder, apologies -- just bear with me a day or two. I will revert on the
timeline shortly -- I'm getting exact clarity on this.

There's been a regulatory-related matter we're working through. We are
seeking confirmation from both the sub-division authority as well as the
town planning authority -- which ultimately has to approve our project
plan based on the GPA I hold from the landowners -- that our existing
documentation does not create any additional cost implication or
documentary burden.

Alongside this, there has been a query/development from the lender which
we are busy sorting out, just to ensure we execute proper documentation
with you.

I personally met the departmental contacts yesterday and they've assured
me they will give absolute clarity in a day or two. Following that, I
will be able to confirm the exact timelines to close our documentation
-- subsequent to which you receive your full and final payment.

Hope you find this in order.

Regards,
Nishant
```

**Pitfall specific to this pattern**: The voice dictation will use the phrase "something like this please" at the end -- meaning the user wants you to take ownership of the phrasing. Do not repeat word-for-word. Reorder for readability, fix STT corruption of domain terms ("separation star" -> "sub-division authority"), and add paragraph breaks between distinct ideas (apology -> regulatory explanation -> lender development -> personal visit -> commitment -> close). The user's raw voice is a stream of consciousness; the message should be structured.

## Pattern 12: Long-silence reconnection apology after a business meeting (Pankaj Kumar / Blue Hat)

**Trigger**: NDR met someone (promoter/investor counterparty) weeks/months ago, the counterparty made a serious proposal (equity/land-for-equity), and NDR never reverted — a sincere apology + catch-up request must now go out. **CHANNEL: WhatsApp, NOT email — user-corrected 2026-09-05.** (NDR dictated the apology and Hermes drafted a Gmail draft; user: "Not email I need a WhatsApp message"; the draft had to be deleted. See SKILL.md P11.)

**Message structure that worked** (Pankaj Kumar, Blue Hat Solutions — defense-tech, IPO-bound, Peenya manufacturing; meeting had been Sat 27-Jun-2026, 3–5 PM, DRA Realty Office Queens Corner; silence ≈ 10 weeks):
1. Sincere apology for the long silence since the meeting (name the time span — it proves the apology is real)
2. Reason, concrete and non-evasive: medical situation on the wife's side, took a complete backseat, follow-up slipped through — explicitly framed as "not how I operate"
3. Gratitude: he took the time, made the presentation, offered an opportunity "not offered easily" (the company is his baby) — this honors the counterparty's effort
4. Status: situation now under control, "we've done everything we can, the rest is beyond our control"
5. Reconnect ask: coffee at HIS place of work (the factory) in the next couple of weeks; "I will give you a buzz and come through" — low-pressure, user-initiated next step

**Context recovery when NDR forgets the exact meeting/company name** ("I forget the name of his blue something… a manufacturing facility in Peenya, when did I last meet him"):
- The company was **Blue Hat Solutions** (aerospace/defense, BH-ERU pylon) — session history + Drive research folder `Personal/Research/Blue Hat Solutions` held the deep-dive.
- The meeting date came from the **Google Calendar** event titled "Discussion about Partnership Opportunity - Pankaj Kumar / Blue Hat Solutions", Sat 27-Jun-2026 — the calendar event title itself resolves both the date AND the nameless company.
- Contact came from People API `searchContacts('Pankaj')` → `pankaj@bluehatsolutions.com`, `+91 93790 95455`. Verify every email against NDR's contacts before drafting (email-drafter HARD RULE).

**Working call** (phone verified 12 digits `919379095455`):
```python
whatsapp_link(
    phone="+919379095455",
    text="""Dear Pankaj,

I owe you a sincere apology for the long silence since our meeting in June. I have been meaning to write for a while.

Shortly after we met, there was a serious medical situation in the family - on my wife's side - that needed a lot of attention. ... Unfortunately your follow-up simply slipped through ... That is not how I operate ...

You took the time, you made the presentation, and you gave me an opportunity that I know you would not have offered easily - the company is your baby ...

The situation at home is now under control - we have done everything we can ...

I would genuinely love to catch up over a coffee at the factory sometime in the next couple of weeks, if that works. I will give you a buzz and come through.

Warm regards,
Nishant"""
)
```

**Pitfalls**:
- **Never default to Gmail draft for "write back to [Name]"** — NDR corrected this exact case. WhatsApp link first; email only if he says email/address or the content needs attachments.
- **Resolve the calendar event before drafting** — the meeter may not remember the date; the event title carries it.
- **Do NOT invent the company name from a vague hint.** "blue something" was resolved by session history, not guessing.
- If the email draft was created by mistake, delete it in the same turn the user corrects the channel (delete via `drafts().delete(draft_id)` with the draft resource id, then verify gone via `drafts().list()`).

## Pattern 13: Site-visit reschedule after last-minute cancellation (polite, flexible window)

**Trigger**: a site visit (property due-diligence / decision visit) was cancelled at the last minute by the seller/builder team; they offer videos/photos as substitute; NDR wants to press for an in-person reschedule within a narrow window because it's a decision-critical visit.

**Message structure that worked** (Ashwin Pai, Century Real Estate, +91 99720 42131 — Century Regalia 4th-floor visit, 5-Sep-2026 cancelled "because of some heavy material coming in to the site"):
1. Thank him for organizing the visit (credit the effort even though it fell through)
2. State the cancellation neutrally + the substitute offer (videos/photos)
3. Explain why in-person matters: "this is a critical decision for me" — no whining, just decision-weight
4. Give a precise alternative: Monday/Tuesday/Wednesday next week, "any time that suits the site is absolutely fine" (NDR explicitly wanted flexibility given ongoing construction — do not propose a fixed slot unless he asks)
5. Context that explains urgency: closing this chapter, considering an alternative property (Legacy on Cunningham Road) — so the seller understands the window is real
6. Warm close ("Thanks again for your help. Regards, Nishant")

**Pitfalls**:
- **Disambiguation of common first names via session context**: 20+ "Ashwin" contacts exist; the right one was **Ashwin Pai (Century)** because the 26-Aug session had already arranged THIS visit via "Anthony Mathew (Century Real Estate, +91 96069 13114) and Ashwin Pai (Century, +91 99720 42131)". When a visit/reschedule message names a person, search sessions for the arranging conversation before picking a contact — the visit context (project name, organizer) disambiguates faster than any contact-score.
- **Keep the alternative-project mention factual but present** — it is leverage for urgency, not a threat; NDR's dictation put it in ("considering another alternative at Legacy on Cunningham Road") and it belongs in the message.
- Do NOT invent "they were happy to reschedule" — NDR's dictation said they offered to reschedule; if the user doesn't say it, don't assert it.
