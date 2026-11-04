# Dispute / escalation reply — stay polite, put the facts on record

Class: NDR is accused (in writing, to the whole family/board) of overriding a control or
safeguard, and must reply **without heat** while establishing the facts. Real case 2026-10-03:
Bhavik Ranka (first cousin, co-promoter of Ranka Iris) sent a "Note of Gratitude" to
Dharmesh/Piyush/Sanjeev/NDR/Manish announcing he was stepping off Iris effective 2 Oct —
alleging a payment (₹8,000 reimbursement to Anbarasan) was released "without my approval,
following instructions from NDR", "not the first time", safeguards not valued, "no real role
for me to play here anymore". NDR wanted a firm, factual reply, explicitly NOT a hurt one.

## The shape of the reply NDR wants

1. Grace first, briefly. "I am sorry to read that you feel this way, and it was never my
   intention that anyone should feel their inputs were not valued." One or two lines. Do not
   argue with the feeling; argue with the facts, later.
2. Then: **"it is important to get the facts on record for everyone's reference"** — and
   **"please correct me if any fact or figure below is wrong."** This framing is what lets a
   very hard message be sent politely to a family group. Use it verbatim-ish; NDR asked for it.
3. Facts, numbered, each one checkable — every claim anchored to a date, an amount, a Kelsa
   record, or an email. No adjectives, no "you always/never".
4. **Reframe the hurt, don't deny it.** NDR's line: *"I think the hurt should have been the
   other way around."* The reframe is structural, not emotional: if I had overridden a decision
   of yours without coming to you and explaining, that would be unfair — an already-approved
   payment being held back without one word to me is the more serious of the two.
5. Context that deserves to be on record — the counter-ledger of unpaid/voluntary
   contribution (see below).
6. Close on the resignation without forcing anyone: it is your right; but professionally it
   would not be accepted with last-stage engineering work, customer handovers, licences and
   RERA exposure pending → **call for a board meeting** to decide forward, and record that NDR
   cannot give further time/inputs, and ask the board to resource the critical closing matters.

## Family-vs-business register (this is the whole craft)

- He is a first cousin AND a co-promoter. Nod to the family relationship once, then speak in the
  register of a company record. Never "you are family so you should have…" (that reads as guilt).
- Highest regard for the elders is stated where a salary/authority decision touched them
  ("with the highest respect for Sanju uncle, we permitted you and Piyush to draw a salary").
- No sarcasm, no "as I already explained", no rhetorical questions. Direct statements only.
- Do not name a vendor/employee as suspect anywhere in the email — put the allegation's
  *verification* on record instead ("we have the vendor's number; anyone may call him").

## Fact-forcing BEFORE you draft (non-negotiable order)

Never draft from NDR's dictation alone — the dictated dates are approximate and the record is
authoritative. For this class:

1. **Gmail ops token of the session user** (`service_name='google-draas'`, own-account path) —
   `gws_gmail_search(from:<name>)` → `gws_gmail_thread_get` on the accuser's message to get the
   verbatim text, the addressee list (it tells you the audience: family + board), and the date.
   Duplicate message_ids matter: search returns thread hits, the actual allegation email is often
   *not* the newest message in the thread.
2. **Kelsa is the ledger.** The disputed amount, its narration, the requester, the payment proof
   attachment, the approver's name + approval date, and the current stage all come from
   `get_lead(pipeline_id=555, lead_id=…)`. Recipe and the "stuck backlog" sweep (stage
   `Expense Approved` = approved but unpaid) live in `kelsa-mcp` → §"Audit: which approved
   reimbursements are stuck unpaid". This is what turns "I approved it" into
   "raised 31 Aug, approved 5 Sep, still unpaid 28 days later".
3. **Confirm the "not the first time" claim with a sweep, not a memory.** One
   `search_leads(pipeline_id=555, query="stage:Expense Approved", per_page=100)` produces the
   pattern evidence (three of Anbu's Iris reimbursements approved and unprocessed). Only put
   "this is not the first time" in the email if the sweep supports it. When he later asks for the
   whole ledger (*"give me a list of all the petty cash we paid to X from project Y"*), do the full
   by-person sweep — recipe in `kelsa-mcp` →
   `references/petty-cash-enumeration-by-person-project-company.md` — and read
   `list_lead_notes` on the older stuck records too: there the note trail reversed the allegation
   entirely (the other side had written "Please release" in 2022 on a record still open four years
   later). That became the strongest paragraph of the email.
4. Where a fact cannot be verified, **say so in the same breath** ("I am not sure at whose end
   that fault lies"), and never fabricate a date, amount or ID.

## Iteration 2+: NDR escalates the ask — HARDEN, don't rewrite

These emails arrive in waves. Expect *"can you please update the email once more?"* followed by a
long dictated addition, and sometimes a second, then a third wave. Rules learned 2026-10-03:

- **Add, never replace.** Every prior section stays, new ones are inserted and the whole thing is
  renumbered (A…G). NDR is building a record to be read by the board — losing an earlier point is
  the failure mode, not length.
- **The emotional register is a moving target — re-read the latest brief, don't inherit the last
  one.** The first brief for this email said explicitly *"I want to reply back strongly"* (firm,
  not hurt); the second wave said *"my email should show hurt and pain for my end"* while still
  *"polite without hurting"*. Both coexist: firmness in the facts, hurt in the closing register.
  State the register you used back to him each pass so he can correct it.
- **Preserve his rhetorical devices verbatim-ish, as questions aimed at the other side** — they are
  the sharpest part and he constructed them deliberately:
  - *"you are setting aside either my judgement, or my credibility, or my ethics. Please tell me
    which of the three, and how you would see it differently"*
  - *"give me one proven communication, after that day, where you told me you were still keeping my
    approved payment on hold, and why"*
  - the **assume-the-worst concession**: *"let us assume we caught him red-handed … my answer would
    be exactly the same, and I want the board to decide whether I am right or wrong"* — this device
    makes the big-picture argument unanswerable; keep it intact when he supplies it.
  - *"were you expecting that I would make good ₹8,000 from my own pocket?"* — keep, but flag it as
    the line most likely to read as a taunt in a family email and offer to cut it.
- **"Free of cost, not on salary" must be repeated every time it appears.** NDR: *"I want to keep
  highlighting free of cost, not on salary. It's important for us."* Do not merge two mentions into
  one for elegance — the repetition is deliberate, and it is the counter-ledger to "your safeguards
  were not honoured".
- **The counter-ledger items he wants enumerated** (take his list, don't invent): liaisoning through
  Rahul / his office; sub-registrar matters through Bharat; Anbu on engineering and site; the DRAAS
  domain he still pays Google for; the Kelsa software he still pays O3 Infotech for; his own time and
  repeated customer meetings; and — specific and loud — *open the Iris customer group and see that
  the only person responding is me*, with silence from the other side on engineering items.
- **Flag risk items and let HIM decide — never silently soften, never silently keep.** Two classes
  showed up: (a) language that reads as our own admission (grey-area work / pre-OC road-cutting / a
  ramp on a public road) — recommend keeping the business-risk logic while dropping the incriminating
  specifics; (b) lines that may cost him more than they win (taunts, open rhetorical questions).
  Present both and ask which way he wants it. He wants the trade-off on the table, not pre-decided.
- Draft iterations stay **in chat** until he says finalize.

## What to name publicly vs hold back

- State the disputed amount, its purpose, the approval date, and that proof of payment exists —
  with the attachment available to the group.
- Name the "not the first time" pattern as a fact, but consider holding the *specific other
  records* back as board-meeting material rather than putting three more sums in an email that
  is already long. Ask NDR which he wants.
- Do NOT editorialise against the other side's competence; "I am not getting into a blame game"
  is itself a line NDR wants.
- Ask for confirmation of any fact only NDR can know (e.g. "the cheque was signed in front of
  me — please confirm") as a **direct question in the email**, and flag to NDR that you have
  made it public, because that changes its character.

## Mandatory: hand back a confirm-list with the first draft

NDR ideates before sending ("we'll ideate, debate, discuss and then finalize"). Every draft must
end with an explicit list of items only he can settle — at minimum:
- voice-garble terms ("chipper" vs **chamber** — the Kelsa narration says chamber and wins);
- dates that are vague in dictation ("the very next day" — next day after what?);
- whether a confirmation question should be public in the email or kept quiet;
- whether the extra stuck records are named now or held for the board meeting.

Present the draft **in chat**, not as a draft in his mailbox, until he says finalize. He asked for
"just show me here". When he does finalize: `gws_gmail_reply_draft` threaded to the source
message — never send.

---

# Phase 3: the rebuttal wave — answering THEIR counter-email

The other side replies with their own chronology, their own attachments and their own list of
"safeguards". The job changes from *building NDR's record* to *dismantling theirs*, and NDR's brief
flips hard: from "put everything on record" to **"keep it short, don't escalate"**. Real case
2026-10-09: Bhavik's reply was a full counter-timeline (10/12/18/19/30 Sep, 1 Oct, four attachments)
naming two safeguards as "overruled", and asserting he never signed the cheque NDR said he signed.

## The strongest move: find where their REASON MOVED

The first email said the ₹8,000 was held on **suspicion about the payment itself** (amount not
justified, "further checks and balances"). The reply said it was held because **the Cauvery inlet
water was not flowing** and the monoblock pump was unresolved. Those are different matters — and the
second one has no connection at all to the man whose money was withheld.

Open on that, in one line, then ask the direct yes/no:

> *"Your email today says the ₹8,000 was withheld for the inlet water issue. So may we take it that
> no further verification was pending on the ₹8,000 itself? Please confirm."*

This collapses a long argument into a single question they cannot answer without conceding. **Always
diff the reason across waves before drafting** — a moved reason is the whole case, and it is usually
sitting in plain sight between the two emails.

## Rules for the rebuttal reply

- **Answer their claims point by point — one numbered point per claim.** Do not restate NDR's earlier
  email; respond only to what the reply actually alleged. Points that carry: the reason moved; what
  the work order does and does not say; whose design the failure was; the money that was withheld
  elsewhere; the neutral cheque line; the closing principle.
- **"We / our", not "you / your", for anything jointly owned** — the chamber, the sizing error, the
  design failure, the decision. NDR: *"wherever we say you it should not be you… it was designed at
  our end, executed by us."* The deliberate effect is to blunt the accusation without softening the
  substance. **But he also said do not blanket-replace "I"** — keep "I approved", "I instructed him",
  "my position", "my resource". "We" owns shared failure; "I" carries his own authority and principle.
- **Do NOT turn it into an ego tussle over the third party.** NDR pulled the draft back when it read
  as him defending Anbu: *"we don't want it to seem like Bhavik is trying to prove Anbu is not sincere
  and Nishant is trying to prove that he is."* Compress the defence of the employee to a bare
  paragraph and rest the case on **the process**, not on anyone's sincerity. The user's goal is a
  short, non-escalating reply that closes the issue, not a win on the merits.
- **Only claim what the documents carry — go and check.** Where they assert a contract term (30%
  released on commissioning; meter installation; chamber-to-manhole), pull the actual work order from
  Kelsa pipeline 537 and say plainly what it does and does not contain: *"how are we to enforce what
  is not in it?"* Where they make an unevidenced *"this has happened several times"*, demand the
  instance: *"give the instance, the date, the document or the chat, and I will examine it."*
- **Verify every dictated quote before it goes in.** NDR dictated February Kelsa notes from memory
  ("12th of February, I have posted a note…"). The actual notes read: 7 Feb 2026 —
  *"@Bhavik Ranka - please confirm the air / pressure is there and it will need mono block. We need to
  clear this for vendor as he claims his job is done. Please confirm and approve."*; 12 Feb 2026 —
  *"Bhavik as discussed we are clearing this."* Pull them with `list_lead_notes` and quote **verbatim**.
  Close-enough dictation loses the entire point — and he will ask you to confirm you checked.
- **Park contradictions you cannot win; flag them to NDR in chat.** He asserted a cheque was signed in
  front of him; the reply insisted no cheque was signed that day and attached a cancelled cheque copy.
  Neutralise, don't fight: *"I am not going to argue about when it was signed — whether then or later,
  a cheque was ultimately signed and issued."* Then tell NDR you did this and offer to call it out
  directly if he wants. Same for the accounting gap he raised (₹27,500 claimed against a ₹24,600
  challan): answer it narrowly and put the two matters in separate points so they stop being conflated.
- **Corrections to NDR's own earlier email.** When the reply shows a figure NDR previously stated was
  wrong (he had said the ₹10,100 was released ~1 Oct; the reply shows it went in a cheque deposited
  23 Sep), simply drop or fix it rather than defend it. NDR's instruction was to ignore it — do not
  re-litigate his own arithmetic.
- **End on the principle, not the paperwork.** NDR's closing rule, stated plainly: *"when a man spends
  his own money on my word, in faith and trust, I will not withhold his payment — whether the job is
  complete or not"*, followed by *"I am happy to have a face-to-face discussion."* Drop any
  "I am not looking to escalate" preamble when he asks for it removed.

## Finalising a rebuttal (this one WAS sent)

- **Mirror the counterparty's formatting.** NDR: *"Just like Bhavik has highlighted specific portions,
  bolded specific portions, please make sure you reply in a similar format."* Bold the key sentences,
  send `html=true`.
- **Reply-all on the same chain**, using the recipient set the counterparty's original used — here To
  Bhavik, Cc Dharmesh, Piyush, Sanjeev, Manish. Thread it with
  `gws_gmail_reply_draft(message_id=<their latest message>, cc=…, html=true)`.
- **⚠️ Verify the draft by reading it back — `gws_gmail_get` returns an EMPTY body and no To/Cc for a
  draft.** Read it with `users().drafts().get(userId='me', id=<draft_id>, format='full')` via
  `tools.gws_auth.build_service('gmail','v1', service_name='google-draas')`. That returns the real
  `to` / `cc` / `subject` / `In-Reply-To` / `References` headers. Confirm the thread ID matches the
  original chain, and count the `<b>` runs in the HTML part, before telling NDR it is ready.

---

# Phase 4: replying IN-LINE, in colour, to their in-line reply

Some waves come back as the counterparty's *own* annotations inside NDR's quoted email — their
comments dropped in beside each of NDR's numbered points, usually in red. Real case 2026-10-09
(Ranka Iris drainage thread): Bhavik answered all 13 of NDR's points in-line in red. NDR's brief:
*"like he has done, he has replied in line to my comments in red colour… I want to reply in line to
his comments in blue colour."*

## Build the HTML body yourself — Gmail's quote cannot do this

Write one HTML body (a single MIMEText html part), shaped per point as: point header in bold →
their comment muted → your reply in blue.

- Your reply: `<span style="color:#1155cc">…</span>` (Google blue).
- Their comment being answered: `<p style="color:#555555">Your comment: …</p>` — so the two voices
  are colour-coded the same way the counterparty did it.
- **Keep every original number**, including the points they did not comment on. Answer those with a
  one-liner ("Noted and closed.") so the reply has no dangling headings.
- Reply only to what this wave actually said; do not restate the earlier emails.

### Three colour layers — verified 2026-10-10 ("Note of Gratitude" wave 3)

When the counterparty has already annotated NDR's email in colour, **mirror the structure exactly
rather than paraphrasing.** The body stacks three colour layers per point:

| Layer | Markup | Content |
|---|---|---|
| original point | `<p style='color:#444444'>` | NDR's own earlier point, muted grey, verbatim |
| their comment | `<p style='color:#CC0000'>` | the counterparty's words **reproduced verbatim in their own red** — never paraphrased, never tidied, never shortened |
| NDR's reply | `<p style='color:#0000CC'>` | the reply, opened with `<b>My reply:</b>` |

- Open the body with the key: *"Bhavik — my replies are in blue below, against your responses in
  red."* NDR asks for this framing explicitly.
- Point headers (`<p><b>On Point 1</b></p>`) sit above each stack, and **every original number is
  kept** — including points they did not comment on (answer those with a one-liner so nothing
  dangles).
- Reproducing their exact words is the point: it puts their text on the record and lets the reply
  stay short. Resist improving their grammar.
- **Show the paired replies in CHAT for approval before building the draft.** For in-line waves he
  asks for *"the exact comments replies… and then once I approve it please draft the reply email."*
  Present the quote→reply pairing as plain text, take his corrections, and only then build the
  HTML. Same rule as Phase 3: iterations stay in chat until he says finalize.
- **A later wave that asks to insert a point into an already-built draft is an IN-PLACE DRAFT
  EDIT — do not rebuild it.** NDR's phrasing is *"we have to insert this point back in that
  draft."* Fetch the live draft with `format="raw"`, patch the HTML as text, and
  `drafts().update()` preserving `threadId` + headers so the same draft id and thread survive.
  Full recipe and the MIME-parsing pitfalls: `references/draft-in-place-edit-preserve-thread.md`.

## Tool: thread it — and know the attachment limit

- `tools.gws_skill_bridge.call("draft_reply_create", service_name="google-draas",
  message_id=<their latest message>, to=…, cc=…, body=<html>, html=True)`. Threads correctly via
  In-Reply-To/References/threadId. `gws_gmail_reply_draft` (native) also threads.
- **⚠️ `draft_reply_create` / `draft_create` themselves are MIMEText-only — those two pass no
  attachments.** When NDR says "the attached image is self-explanatory" or "please see the image
  attached", the images will NOT be on the draft built that way. Either (a) tell him plainly to
  attach them in Gmail before sending, or (b) build the draft as raw MIME with attachment parts — see
  `references/gmail-draft-with-attachment-mime.md` in this umbrella. Do not let him find out after
  sending, and do not report the draft as complete while it is missing images he asked for.
- **Reply-all using the Cc set the counterparty's own email carried.** Read it off the thread with
  `gws_gmail_thread_get` — the forwarded copy sitting in Sent is the most reliable place to recover
  the full Cc list.
- **Verify by reading back.** The bridge's `draft_get` returns the body but NOT Cc/subject. For
  headers use `tools.gws_auth.build_service('gmail','v1',service_name='google-draas')` →
  `users().drafts().get(userId='me', id=<draft_id>, format='metadata')` and print
  To/Cc/Subject/In-Reply-To/References. Confirm the threadId matches the chain, and count the blue
  `<span>` runs in the body before telling NDR it is ready.
- House Python for these scripts: `/opt/hermes/.venv/bin/python3`, with
  `sys.path.insert(0, "/opt/hermes")` before importing the bridge.

## Brevity is the register now — a late wave is a timebox

By wave 3+ the instruction becomes *"keep it very, very brief… running out of time on this, no more
time allocation for it."* That is a real constraint, not a mood:

- Open with the colour key in one line: *"Bhavik, my replies in blue below. Running out of time on
  this — there is no further time allocation for it."*
- Short sentences, one idea per blue block. Treat every extra paragraph as a failure.
- **Still verify before inlining.** Even mid-timebox, pull the record and quote it verbatim (here the
  7 & 12 Feb Kelsa notes on the KYG invoice — see Phase 3). A document quoted back at them accurately
  is the highest-value line in a short reply.
- **Names and documents get dictated loosely — flag, never silently reconcile.** He said "Jay Ram
  sir" in one point and "Manju" in another for what reads like the same chamber work order. Write
  both as given, and put the ambiguity in the hand-back list alongside the other items only he can
  settle. Same rule for any figure or date you were given by voice.