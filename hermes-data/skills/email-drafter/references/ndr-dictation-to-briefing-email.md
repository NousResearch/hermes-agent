# NDR multi-point voice dictation → structured briefing email

NDR often sends ONE long voice message that is a stream-of-consciousness reply covering
5–10 separate topics (answers to a colleague's request, decisions on several proposals,
instructions, and a thing he will do himself), ending with:

> "please draft a reply all email and keep it ready."

This file is the recipe for turning that dictation into a clean, sendable draft.

## 1. "Reply all" + "keep marking it only to <Person>" — NOT a contradiction

When NDR says **"draft a reply all email"** and in the same breath **"keep marking it only to
<Prakash>"**, he means:

- **Reply ON THE EXISTING THREAD** — preserve `In-Reply-To` + `References` + `threadId` so the
  reply nests in the master conversation. That is the "reply all" part.
- **To = that one person only. Add NO Cc.** Do not include the other thread participants.
  He wants thread continuity without looping everyone in.

If the source thread is single-recipient anyway (only Prakash → NDR), reply-all and "only to
Prakash" coincide — **say that plainly** ("his original had no other recipients, so reply-all
would have been him alone anyway") rather than treating it as an unresolvable conflict or
asking him to choose.

Never add a Cc "for information". He decides recipients, and he says so.

## 2. Structure the dictation, keep his ordering

He dictates in the order he wants the points read. Do not reorder or merge his topics.

- **Numbered sections**, one topic each, each with a bold mini-heading.
- **Answer/clarify first.** When he opens with "I'm not sure what that means… please clarify",
  section 1 is a question back to the counterparty, not a decision.
- **Approval conditions as bullets** (his checklists: RERA number, verified phone/QR/locations,
  AI-enhancement limits). Name who owns the subjective call ("that is a subjective call, and
  Gowri will take it").
- **Each proposal → decision + what he wants back.** He rarely just says yes; he says
  "happy to look at it and approve — give me a proposal", or "send me the budget and your
  expectation on calling, lead generation and volume".
- **The big item gets a bulleted sub-list of its dimensions.** The RG model came out as four
  bullets: price floor (Rs. 1.2 cr all-in) / how to quantify it "working" / which new business /
  proposal + vertical + profit share.
- **Closing line answers any timeline the counterparty raised.** Prakash wrote "I want to clear
  and complete within December 2026" → closing line: "let us use this to get to a decision well
  before December 2026." Do not invent a harder commitment than he gave.
- **His words are the decision.** "We do not want to go below Rs. 1.2 crores all-in" goes in
  verbatim; do not soften or round it.

## 3. Formatting

- **HTML** for anything with 3+ distinct points (this is NDR's standing rule for multi-point
  internal briefings) — numbered/bold headings, `<p>` and `<ul>`, **no forced `<br>`**.
- Always ship a **plain-text alternative** part alongside the HTML.
- **ASCII-only in the body**: `Rs. 1.2 crores`, `Rs. 70-80 lakh` — avoid `₹` and en-dashes.
  Currency symbols and typographic dashes are a mojibake risk in the generated part; there is no
  upside to them.
- Sign off `Regards,\nNishant`.

## 4. Source the facts BEFORE writing

- **The checklist he refers to verbally may not exist as a standalone email.** He said "the
  checklist that we have prepared last time and I've shared with Gowri" — a search for it
  (`checklist gsingh@draas.com`, `"RERA number"`, `subject:(collateral OR creatives)`) turned up
  no standalone checklist mail; the guidance actually lived in his 2-Oct reply on the
  *Ranka Udaya RG scheme_ newspaper insert* thread. **Write "the checklist I shared with Gowri
  earlier" WITHOUT citing a mail, and tell him you could not find a standalone copy.** Never
  invent an email reference.
- **Colleague addresses come from Gmail headers, not the contacts sheet** (already the skill's
  headline rule — re-confirmed here: Gowri's sheet row carries only personal yahoo/gmail
  addresses, wrong for work).
- Resolve garbled names before using them: voice **"Sartak" → Sarthak Sharma**.

## 4b. "I have done an earlier session on this" — search, then ASK, never cite a guess

NDR often opens a dictation with a hedge like:

> "I have done an earlier session on cash flow planning. Whether it is a document on my drive,
> whether it is a part of some session memory, I am not sure."

Treat that as **a research instruction, not background colour.** Handle it in this order:

1. **Search all three stores before writing**: `gws_drive_list(query="name contains '<topic>'")`,
   `gws_gmail_search` on the topic plus the counterparty, and `session_search`. Then report what you
   actually found — including confidently reporting *"no earlier email or session on this exists"*
   when that is the truth. "Not found" is a valid and useful answer; do not pad it.
2. **Do NOT put a guessed prior-art reference into the draft.** Naming the wrong document sends the
   colleague hunting for something that does not exist. Ask NDR to confirm the artefact, and offer a
   one-line pointer once he does.
3. **Offer the closest match you did find**, with enough detail for him to recognise it (exact title,
   tab names, date range, who owns it) — he is usually remembering *something*, and he will confirm
   or correct in one line. Also list any adjacent prior assignments on the same topic so he can pick.
4. **Flag garbled dictation instead of papering over it.** Write the generic phrase in the draft, and
   quote the unresolvable word back to NDR verbatim for him to supply. Never invent a vendor,
   expense head, or document name.

Worked example (2026-10-19, DRAAS company cash flow planning → Sarthak, Cc Roshni): searches found
no earlier company-level cash-flow session or mail; the only on-Drive candidate was
`NDR RNR DRA Realty - Master Cashflow Planning Sheet` (tabs *Outflows* / *Inflows* / *Available
Balances* / *Projects Related* / *RNR NDR Personal*, monthly grid Oct-26 forward, last modified
Jul-2025), plus two project-level Sarthak assignments (Ranka Oasis two-model brief 13-Sep-2026,
leasehold built-to-rent 8-Oct-2026). **The draft cited none of them**; all of it went back to NDR as
a question, together with the one unresolvable dictated word.

## 4c. New-thread briefing vs threaded reply

The briefing pattern is not always a reply. When the assignment is genuinely NEW (NDR → Sarthak
`admin3.blr@draas.com`, Cc `rnr@draas.com`, 2026-10-19), build it as a **new** draft: stdlib
`EmailMessage` + `set_content` + `add_alternative`, then `drafts().create` with **no `threadId`** —
not `draft_reply_create`. Everything else in this file applies unchanged: ASCII subject, HTML +
plain-text pair, numbered bold sections, no forced `<br>`, sign-off `Regards,` /
`Nishant Ranka` / `ndr@draas.com`.

Subject convention for an assignment: `Task: <the work> - <the deliverable> (<phase>)`, e.g.
`Task: Cash Flow Planning - Complete Company Cash Flow Plan and Live Dashboard (Phase 1: Outflows)`.

Keep NDR's own phase/numbering vocabulary in the draft (here: literally "Phase 1 / 2 / 3") — it is
how he will refer to the work in every later follow-up, and renumbering breaks the thread.

## 5. DRA colleague address map (verified from Gmail headers)

| Who | Address | Notes |
|---|---|---|
| Prakash Singh | `psingh@draas.com` | sales & marketing; sends as this |
| Gowri Singh | `gsingh@draas.com` | marketing / content approvals. Sheet has ONLY `gowrisingh72@yahoo.com`, `gowrisingh1341@gmail.com` — do not use for work |
| Sinchana Gowda | `sgowda@draas.com` | architect (Oasis) |
| Bharat Hawaldar | `sales1.blr@draas.com` | sales; sends as this |
| Sarthak Sharma | `admin3.blr@draas.com` | cash-flow / financial models |

## 6. Always verify the draft (don't trust create-time success)

1. `format="raw"` → parse with `email.message_from_string` for the real **Message-ID**
   (`format="metadata"` can return it empty).
2. `EmailMessage` with `In-Reply-To` + `References` + `Subject` + To/Cc.
3. `drafts().create(body={"message": {"raw": ..., "threadId": <master>}})` — **threadId is
   mandatory** or Gmail mints a fresh thread.
4. `drafts().get(format="full")` → assert `threadId == master thread`, To/Cc correct, and the
   parts are `('text/plain','text/html')` (a demoted HTML part shows literal markup to the
   recipient — the failure NDR actually reports).
5. `drafts().list()` → confirm the draft id is present.

`scripts/create_threaded_reply_draft.py` does all of this in one run.

## 7. Worked example (2026-10-16)

Prakash Singh — *"Request for Approval and authorization for Sales and Marketing activities for
DRA Realty (ongoing and upcoming residential developements)"*, thread `1a143287bd9ac2d9`.
Draft `r-6890670267315170977`. Five sections: authorization clarify / content approval through
Gowri / sales coordinator approved to hire / bandwidth-discounted pricing / Rental Guarantee
model. To Prakash only, no Cc. **Draft only — NDR sends.**

## 8. The NEW-email variant — briefing a reportee (2026-10-19)

Not every dictation is a reply. When NDR describes an assignment with no existing thread
(*"I want to give a briefing note to Sarthak on email, marking Roshni in CC"*), there is no
`In-Reply-To` and no `threadId` — `draft_reply_create` is the wrong tool. Build a fresh draft with
`EmailMessage` (`new-email-html-attachment-nonascii.md` §4b): To = assignee, Cc = the person he
named, no `threadId`.

Worked example — the DRAAS company cash-flow plan brief: To `admin3.blr@draas.com`, Cc
`rnr@draas.com`, draft `r3833345223300570304`, 10 numbered HTML sections + plain-text part, no
attachment. Tracker row P-511.

### 8.1 The closing "let me know" belongs to the ASSIGNEE, not to NDR

He dictates long briefs *about* a third party, in the second person to that third party: *"Give me
an estimate of when you can finish the outflow side"* / *"read this requirement and come back to me
if everything is clear or not clear to you."* Those "you"s are the **assignee**. They go into the
email as his closing ask — do not bounce them back to NDR as questions, and do not drop them. The
ask for a **dated estimate** plus a **clarity confirmation before starting** is load-bearing; it is
what makes the brief answerable. If a "let me know" could plausibly be either, put it in the email
(harmless there) AND report your own open questions to NDR separately.

### 8.2 "I did an earlier session on this — a doc or session memory, not sure" = PRIOR-ART SEARCH

That phrasing is an instruction, not colour. Search all three stores (Drive, Gmail, session
history), then report what you found and state explicitly that you **cited none of it** pending his
confirmation. Never name a prior artefact in the draft on your own initiative — a wrong skeleton
reference derails the assignee. Offer it instead: *"tell me if you want X named as the starting
point."*

### 8.3 An unresolvable dictated word is OMITTED and quoted back — never guessed

The overheads list contained *"...telephone, milk, supplies. Rekarri."* "Rekarri" is not an expense
head. Write the generic parent (*"and other miscellaneous items"*), quote the word back verbatim, and
log it as an explicit unresolved ⚠ flag on the tracker row. Same rule for garbled entity names:
normalise the ones with an established mapping ("Sivaganapalli"/"SLV" → Sevaganapalli) and list the
corrections back to him.

## 8. A dictation that ASSIGNS work is TWO deliverables — the draft AND a tracker row

Verified 2026-10-19 (NDR → Sarthak, company cash flow plan). The draft was built, verified and
reported as "draft only, nothing sent" — and NDR's very next message was:

> "Can you confirm if this is added to his Master Task Tracker as well for Sartak?"

**Drafting the email is HALF the job.** Whenever the dictation *assigns an exercise or task to a
named reportee* (a fresh assignment — not a reply on a live thread), create the
`NDR_Master_Task_Tracker` row in the SAME turn and report the P-id next to the draft id. The row is
the durable artifact; the draft is only how the reportee first hears about it. See the
`personal-task-tracker` skill for the write recipe (fresh `All Tasks!A1:R400` read → id2idx with
`+1` → append → verify via the returned `updates.updatedRange`).

Put in the row's Notes: NDR's own label vocabulary **verbatim** (e.g. "Phase 1 / 2 / 3"), the draft
id, and `NOT sent — NDR sends`. Keep his labels as the shared vocabulary across every follow-up,
row and message — renumbering confuses the thread.

## 9. Never invent an item in a dictated list — omit it and quote it back

Same session: NDR dictated a corporate-overheads list ending `"...telephone, milk, supplies.
Rekarri."` — the last word was unintelligible. Correct handling: write the list WITHOUT it (here
"and other miscellaneous items"), quote the word back to NDR **verbatim** in the reply, and log it
as an explicit ⚠ unresolved warning on the tracker row. Never guess an expense head, vendor name or
figure from a garbled token — same rule as §4's "never invent an email reference". Any unresolved
item must be visible on the row, not silently dropped, or the next session will treat the list as
settled.
