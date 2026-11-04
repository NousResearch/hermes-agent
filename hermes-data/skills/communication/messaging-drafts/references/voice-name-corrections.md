# Voice-name corrections (confirmed mappings)

NDR dictates names/companies phonetically; these mappings are confirmed against contacts/threads.

| Voice said | Actual | Source |
|---|---|---|
| "Jiraffe Capital" / "Jiraffe" | **Jiraaf Capital** — Vineet Agrawal, vineet@jiraaf.com, +91 97390 99290; team: Prakash (prakash.m@jiraaf.com, +91 73765 50564), Kanishk Sharma (kanishk.s@jiraaf.com) | Gmail thread "DRA - Jiraaf"; contacts sheet (index key 'vineet') |
| "Lakshmi Nesia" / "Lakshmi Kant" / "Lakshmi Broker" | **Lakshmikanth Broker Necia** — +91 89040 33887 (work) | contacts sheet row 1731, index key 'lakshmikanth broker necia' |
| "Sevkinapalli" | **Sevaganapalli** (Sevaganapalli Land Partners, Ranka Oasis) | Gmail thread "DRA - Jiraaf" |
| "Nishant Prakash's team" | **Nishant Prakash** — nishantprakash@theyelloweye.com (Yellow Eye), on Jiraaf thread | thread participants |
| "Vinit" (in Jiraaf context) | **Vineet** Agrawal (same person) | self-correction in dictation |
| "Endeslaw" / "Industlog" (counsel context) | **IndusLaw** / CMS INDUSLAW — G Vivekanand, vivek.g@cms-induslaw.com | legal-matter-coordination voice-correction table |
| "Divesh" (Flameback context) | **Devesh Sati** — devesh@flamebackcapital.com, Flameback Capital (contacts sheet row 4248) | contact_resolver @ 'devesh' |
| "Nitin Oswal" / "Ostwal" / "Osthwal" | **Nitin Kalpesh** — Kalpeshnitin@gmail.com (the CA/tax advisor NDR calls "Nitin"; the row itself carries these as `voice_misspellings`) | contacts sheet, Ranka Oasis tax threads |
| "Biaadarhalli" | **Byadarahalli** (Ashok Kumar's Schedule B lands, Sec-281 application) | DRA KAAJ 281 record |
| "Paliya" (281 context) | **Palya** (Ashok Kumar's land, Sy 8/1) | DRA KAAJ 281 record |
| "Sartak" | **Sarthak** Sharma — admin3.blr@draas.com, +91 98937 91110 | org roster, Kelsa creator records |
| "Fortunately and Sartak" | **UNCONFIRMED — flagged, likely "Roshini and Sarthak"** | do NOT silently resolve; he copies Roshini on accounts matters |
| "Aamir, Aasad" (attendance / Kelsa late-sign-in context) | **Anbarasan (Anbu)** — a DRA PM, pm2.blr@draas.com. The ENTIRE name pair was garbage; it resolved cleanly to a different real contact (Aamir Khan) | NDR corrected it himself in the next voice note, 2026-10-05 |

## Mis-parse pitfalls
- "NSTL" → NSDL (National Securities Depository; tax/bond dues).
- "Bucks Ranka" → Bux-Ranka Developers Private Limited (BRDPL).
- "Dhamid" (Gunjur/Saturday context) → Amit Jain (Deloitte), meeting at Ranka Chambers.
- "Rajan" (GPA context) → Rajendra (NDR's own 3-Sep email spells "GPA to rajendra is ok").
- "Harish Bhaksadi" → Haresh Buxani (seller-group rep, haresh@buxani.com).
- "Sharaaf" → Amit Saraf (Godrej side) — confirm before use.
- "pre-TCR" → **pre-DCR** (BBMP Development Control Regulations; the pre-submission work on the
  Hudson Circle / 4.5 FAR drawings). Always write **pre-DCR** in the email even when the dictation
  says "pre-TCR", and flag the substitution — the thread uses pre-DCR throughout.

## Workflow reminders for voice drafts
1. Resolve proper nouns via contact_resolver / entity_resolver first; for thread replies, cross-check against actual thread participants.
2. Strip mid-sentence self-corrections ("sorry, to Vinit" → Vineet).
3. If a financial figure is dictated, CLARIFY before generating — never guess amounts.
   Example: "you have 1-3 crore… If I can…" → real message was **8.13 crores GBA TDR for Godrej, waiting on Lakshmikanth's confirmation to commit**.
4. Reply-all: primary addressee in To, rest of thread participants in Cc, matching the sender's own earlier Cc pattern on that thread.
5. **Read the `voice_misspellings` field on `contact_resolver` results — it is a built-in garble dictionary.**
   The contacts sheet stores previously-learned mis-hearings against the real row, so a query for the
   mangled form can land on the correct contact with a high score and a `voice_misspellings` string
   listing every variant. Example: querying "Nitin Oswal" resolves to **Nitin Kalpesh** with
   `voice_misspellings: "Nitin Ostwal, Ostwal, Osthwal…"`. When a garbled name returns a candidates list
   carrying a `voice_misspellings` field that matches what you heard, that is the strongest available
   signal — it beats phonetics.
6. **Never silently resolve a garbled proper noun. State the reading you inferred, and ask.**
   Put the correction in the delivery, not just in the artifact: *"Corrected: 'Biaadarhalli' →
   Byadarahalli; 'Nitin Oswal' → Nitin Kalpesh (CA) — the contact on record."* Where the garble
   cannot be resolved confidently (a name that could be a person or a filler word), name your best
   guess **and** say it is unconfirmed, then offer to regenerate. NDR corrects fast when given a
   specific claim to react to; he has rejected drafts built on a silently-chosen reading.
   Real case: *"Fortunately and Sartak"* → delivered as "Roshini and Sarthak" with the garble
   flagged explicitly, because he copies Roshini on every accounts matter but the phonetics did
   not support a silent call.
7. **A garbled name can resolve cleanly to the WRONG real person — confirm before sending.**
   On 2026-10-05 *"send a telegram … to Aamir, Aasad"* resolved to a genuine contact (Aamir Khan),
   but the intended recipient was **Anbu (Anbarasan)**; NDR corrected it on the next note. A clean
   `contact_resolver` hit is therefore NOT sufficient evidence that you found the right person.
   Two extra checks before any peer-to-peer delivery:
   - **Reachability check when he names a platform.** He said "send a telegram" — run
     `send_message(action='list')` first. The tool resolves recipients from the identity registry
     only; if the named person is **not** in that list, do NOT substitute a similarly-named contact
     and do NOT let it fall through to the home channel. Surface the gap and offer the standard NDR
     fallback: **Telegram if reachable, else a WhatsApp deep link** (`whatsapp_link`).
   - **Say the resolved recipient back in the reply** ("Sending to Anbu / Anbarasan M on Telegram")
     so he can catch a wrong resolution in the same breath as the delivery.
   - Never use an **unlabelled** phone from the contacts sheet for a send — per the hard phone rule,
     ask if no Primary/DRA/Work label exists.