# Cross-channel follow-up: anchoring a new ask on a prior commitment

Use when NDR says a variant of: *"we already have an email and a WhatsApp on this, use
that to build the context"*, *"follow up with X on the pending R&D / pending task"*, or
*"I need him to give me a plan"*. The deliverable is usually BOTH a WhatsApp message and
an email draft, and they must read as one continuous programme with the earlier mail —
not as two fresh, disconnected requests.

Worked example this reference is drawn from: Bharat Hawaldar, Ranka Oasis competitor
R&D (Oct 2026).

---

## 1. Recover the existing thread before writing a word

Do not draft from the request alone. The prior mail carries the ask, the project list,
the standing instructions, and the deadlines that the follow-up has to honour.

```
1. contact_resolver(query="<first name>", context="<project>")   # canonical + phone/email
2. gws_gmail_search(query="<project> competitor")                # find the brief
   gws_gmail_search(query="from:<firstname>", max_results=25)    # profile their traffic
3. gws_gmail_thread_get(thread_id=...)                           # FULL body, not snippet
4. wa_search_messages(chatId=..., query="<topic word>")
   wa_search_messages(chatId=..., query="R&D", fromMe=true)      # what NDR already said
5. wa_read_messages(chatId=..., limit=30)                        # LAST commitment comes here
```

Gmail search is AND-of-all-terms — a four-word query silently returns `count: 0`.
Search two or three words, then widen with `from:`/`to:` filters. A zero result is a
query problem far more often than a missing email.

## 2. Find the anchor: the date the counterparty already committed to

`wa_read_messages` on the specific chat is where it lives — e.g. NDR asking for a
completion date on 27 Oct and Bharat replying *"Sir i will provide you this by Sat
(31st oct)"*.

**That date is the anchor. Do not overwrite it.** NDR's own commitment to a date is a
contract; a follow-up that quietly resets it reads as an extension. Instead:

- open with *"your commitment of <date> stands"*, then
- **insert an earlier checkpoint** (the plan, one day out), and
- instruct them to flag slippage **at the checkpoint, not at the deadline** —
  and to say what they are dropping to hold the other dates.

Result: outer date preserved, 48 hours of warning bought, no renegotiation invited.

## 3. Ask for the plan before the report

The recurring shape when NDR wants visibility on delegated work — encode it as three
lists, one page, with a hard time:

| Ask | What must be in it |
|---|---|
| **Whom you have identified to meet** | person, designation, date and time, per project. "If a project is not being visited, say so and say why." |
| **How you are visiting** | day-by-day schedule — project, product type, contact, morning/afternoon; a minimum cadence or an explanation |
| **Which counterparties you are talking to** | the list with firm, catchment, years active, deals closed in last 12 months and at what ticket size; each row marked **"booked with date"** or **"not yet booked"** |

The "booked with date versus not yet booked" marking is the load-bearing part — it makes
*identified* mean *calendarised* and removes the space for a list of names to pass as a plan.

## 4. Reserve the last meeting for NDR

When NDR says *"get me in on the last meeting so I can also do my Q&A"*:

- "Keep the **final** meeting of the round for me" — with a named date and a fallback date.
- Specify a partner "of real size", not the smallest one who will take the slot.
- **30 minutes of his own Q&A**, after the counterparty work has already been done — so
  the ask is his questions, not a repeat of the homework.
- **Pre-read must arrive before he walks in**: their catchment, deal history and
  underwriting position. Write it as *"I do not want to hear the numbers for the first
  time in the room."* Book the invite the moment it is fixed.

## 5. Convert a rambling dictation into numbered deliverables

NDR's voice note will carry the strategic reasoning ("price is the only lever we have",
"Grade A product", "50 to 60 villas, or even 70", "balance in years two and three is
residual only"). Keep that reasoning — it is what makes the email persuasive — but
attach every ask to a **date, a named document, or a number**. A follow-up email that
ends without dates is a failed follow-up.

Order the email: what stands → what I need first (the plan) → what is mine (the meeting)
→ the decision I am waiting on (the numbers) → the date table → the standing instruction.

## 6. Two registers for two channels, same asks

- **WhatsApp** = action and timeline only. Numbered asks, bold on dates and document
  names, no justification, no restating of the commercial logic. NDR's own median to a
  direct report is ~100 characters; a 1,500-character structured message is acceptable
  and normal when it carries a plan, but every line must be an instruction.
- **Email** = the same asks *plus the why*: why the plan precedes the report, what
  "identified" is defined as, why the number must come off counterparty underwriting
  rather than a brochure. NDR signs `Regards,\n*Nishant Ranka*`.

Carry the earlier mail's standing instruction forward verbatim so the two messages are
clearly one programme — e.g. *"record what is claimed versus what is shown; if it is not
on paper, write 'claimed verbally'"*.

## Pitfalls

- **Project-name drift between channels.** WhatsApp said *"Riverstone Oasis"*; the email
  thread and every document said *"Ranka Oasis"*. Voice dictation mangles project names.
  Use the name from the email thread / `entity_resolver`, never the dictated one, and
  never propagate a mangled name into a written email. `Riverstone Farms` is a genuinely
  different project — do not blend them.
- **Don't re-ask what the prior mail already asked.** The 23 Oct brief carried the ten
  projects, the five data points and the 20-partner list. The follow-up's job is the plan
  and the decision, not a second copy of the brief — reference it by date instead.
- **Verify the draft, don't trust the create.** `gws_gmail_get(message_id=...)` and check
  `labelIds` contains `DRAFT`, plus the to/subject/body. Never call send.
- **`wa_send_message` may really send.** If the contact is on the auto-send list it
  returns `SENT (this contact is on the auto-send list)` — only then may it be reported as
  sent, with the outbox id. A `drafted=true` response is the normal outcome for everyone
  else and is not an error; say the message is in the WhatsApp app for approval.
- **Bharat Hawaldar (+91 99000 29200, sales1.blr@draas.com) is on NDR's auto-send list** —
  a WhatsApp to him goes out immediately without NDR seeing it first. Get the wording right
  before calling the tool.
