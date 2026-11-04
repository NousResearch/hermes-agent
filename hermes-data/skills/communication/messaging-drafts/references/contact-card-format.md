# Personal contacts — quick reference

When the user asks for a WhatsApp / message to a named person, search the conversation and memory for the contact card. Common fields:

| Field | What it captures | Example |
|---|---|---|
| Full name | Legal name as the user uses it | Manohar Singh |
| Nickname / address | How the user addresses them in messages (Sahabji, Bhai, Boss, Sir) | Sahabji (S-A-H-A-B-G-I) |
| Phone | E.164 or local Indian 10-digit | 9845890316 |
| Role | Colleague, partner, vendor, advisor, family | Colleague & partner at DRAAS |
| Relationship context | One sentence so the LLM picks the right tone | "colleague and partner" — first-name + nickname is fine |

## DRAAS contact cards (snapshot 2026-07-13, will age)

- **Manohar Singh** — phone 9845890316. Address as **Sahabji** (NOT "Sabji" — voice transcription gets this wrong). Role: colleague and partner. Tone: casual, first-name, Hindi-English mix.
- **Aamir (Hussain) Khan** — phone +91 98458 81652, email khan.hussain.aamir@gmail.com. Address as Aamir (or "Aamir bhai"). Role: trusted reviewer/advisor on real-estate documents. Tone: respectful, first-name, formal-casual.
- **Kishan Murjani Nair** — phone +91 98450 20921, email kishan@flamebackcapital.com (Flameback Capital, CFA). Runs the algo-trading platform NDR calls "Algotradie"; NDR has 21+% returns with him; Roshni's cousin (sheet note). Address as Kishan. Context: AIF plans, ₹11 Cr raise from 15-30 people, Marwadi/Jain network tapping.
- **Lokesh Gandhi** — phone +91 94488 45692, no email on file anywhere (Gmail holds only 2013-14 WhatsApp-chat exports). NDR's school friend, senior in business development at EY, deeply connected in the Jain community. Introduced to Kishan (2026-08-14) for community-network raises.

## Voice-memo dictation gotchas (India)

Names that voice transcription routinely mangles — always cross-check with memory before composing the message:

| User's actual | STT often writes | Notes |
|---|---|---|
| Sahabji | Sabji, Saabji, Saabjee | The user has explicitly corrected this. Memory says Sahabji. |
| Tahasildar | Thasildar, Thaisildar, Tasildar | The revenue officer. Three spellings are all in circulation in Indian English; do not auto-correct without user input. |
| Kelsa | Kelsey, Kalsa | Kelsa is a vendor/product name. Don't normalise. |
| DRAAS | Dras, DRAAS, Draaz | Acronym, keep as DRAAS. |
| Amit Pujari | Amit Pujari (stable) | Some names STT gets right. Don't pre-emptively "correct" names the user has not corrected. |
| Lokesh Gandhi | Nokesh, Lokesh | STT wrote "Nokesh" on first dictation; the user's own follow-up message said "Lokesh" — contacts confirm Lokesh. Follow-up spelling wins. |
| EY (Ernst & Young) | ENY, E&Y | Dictated "ENY" — means the firm EY. Use EY in the message body. |
| HEPA (filter) | HIPAA | "HIPAA filter" is a medical acronym misused for HEPA. Correct in body (recipient must not water a HEPA filter) and flag to user. |
| Ruhaan (son) | Roohan, Ruhan | Memory spelling is Ruhaan (see user profile). |

## When memory does not have the contact

1. Ask the user for the phone number and the preferred addressing form.
2. Do NOT search the web for the person's number.
3. If the user says "you should know this", check session_search for previous contact mentions before asking again.

## When memory is at capacity and the user asks to remember a fact

Order of replacement (drop in this priority, but always confirm with the user first):

1. **One-time TATs from a specific past dispute** (e.g. "Bajaj Life dispute GRO email from 2026-07-11") — most likely to be stale
2. **Tool quirks the agent has internalised in code** — these belong in skills, not memory
3. **Old project conventions from a completed phase**
4. **Never drop:** user-vocabulary facts (Sahabji spelling), user-enforced security rules, contact cards for active recurring contacts

The "Sahabji" lesson is the canonical example: this is a personal-vocabulary correction that must NOT be auto-dropped, no matter how full memory is, because it directly affects the user's trust in the messaging flow. If you have to choose between keeping a one-time dispute TAT and keeping "Sahabji spelling", keep Sahabji.
