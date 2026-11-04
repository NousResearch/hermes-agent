# Landowner Group Messaging (multi-party, high-stakes)

For the **Ranka Northstar Landowners** group and comparable landowner/partner
groups where NDR is the developer and the counterparties are a collective of
landowners. These are the messages with the longest life and the highest stakes —
they carry commercial commitments, so precision beats speed.

Group created 15 Jul 2025, renamed "Ranka Northstar Landowners" 28 Nov 2025.
Members include the Swaminathan family (V. Swaminathan), the Vaidyanathan family
(A. S. Vaidyanathan, Sreeraman, Lakshmi), and Sunder Padmanabhan (Mumbai-based,
the only non-Bangalore member).

---

## THE APPROVAL GATE — do not skip this

For anything significant on a landowner group, the sequence is **strict**:

1. **Draft as plain text in chat.** Not links. Not a file. Not an attachment.
2. **He edits.** Expect **two to four rounds** of refinement. He dictates the
   change verbally and usually wants **only the changed section re-shown**, not
   the whole message re-pasted — see "Show the delta" below.
3. **Only after he approves**, generate the WhatsApp links.

He states this explicitly: *"just show me the two objective portion of the draft.
Once I approve it, we can go ahead and make WhatsApp links for the entire
message."* And earlier: *"show me draft 1 and draft 2, the entire text content?
So I can see if I decide which draft to use."*

**Do not pre-emptively generate links.** Producing links alongside an unapproved
draft makes him re-request and wastes a round.

### Show the delta, not the whole document

When he asks to see a revision, **re-present the changed section only**, with a
one-line note of what else changed. He asked for exactly this ("redo the two
objectives, part two… just show me the two objective portion"). Re-pasting a long
message for a one-line change is a legibility failure.

### Offer options when the message is long or tonal

He asks for *"one or two drafts if required"*. For long landowner messages, give a
full version and a concise version, each split into parts, and say which you
would send. Do not collapse them into one "compromise" draft.

---

## Link generation

- **Always** generate through the `whatsapp_link` tool / sanctioned path. Never
  hand-encode a URL and never use `wa.me` (its redirect corrupts `%` on mobile).
- **Group links: omit the phone number.** With no phone, the link opens WhatsApp
  and he picks the group himself. Do not guess a group JID.
- **Long messages split into `parts`.** Deliver each part as its own link and tell
  him to send them **in order, one at a time**. 2 parts is common; 3 for the
  longest messages.
- After generating, sanity-check for grammar/typo slips that survive the split
  (e.g. "everyone consideration" for "everyone's consideration" — an apostrophe
  breaks some mobile clients, so prefer the apostrophe-free read where it still
  parses). Regenerate rather than shipping a visible typo.

---

## Ingesting the chat export

He exports the group chat from WhatsApp (**Export chat → Without media**) and
attaches the `.txt`. **Ingest it before drafting** — he says the dates and
sequence are critical, and they are: the group's history is the context that makes
a message land.

What to mine from the export:
- the **exact chronology** of past commitments ("we will share plans once
  in-principle approval is received") so the new message can point back to them;
- **who raised what**, and who stayed silent;
- **tone precedent** — the register used in previous updates;
- **open promises** that the new message either fulfils or must acknowledge.

Do not treat the export as read-only background. Cite it: "as I had committed on
5 Aug" is materially more persuasive than a fresh assertion.

---

## Voice-transcription hazards — verify every proper noun

He dictates by voice and the transcript **will** mangle names. Check every name,
number and term against the **document record** (Drive, the index sheets, the
executed instruments) before it goes out. Known examples:

| Heard | Correct |
|---|---|
| Kanshika | **Kankshita** Swaminathan |
| Alal Sandra / Alala Sundara | **Allalasandra** (also Allalsandra in older files) |
| "legal ears" | **legal heirs** |
| Vedyanathan | **Vaidyanathan** |
| Kata | **khata** |

**Flag discrepancies to him rather than silently normalising** — and when you
correct a name, say which spelling you used and why in one line. He has never
objected to being corrected on a name; he has objected to a wrong name going out.

---

## Message architecture for landowner updates

He wants these to **look** like a considered circular: bold headings, bullets,
numbered objectives, blank lines between paragraphs, "titles to each paragraph
which are bold and underlined if required… whatever is required to make it look
cozy and readable."

That is a request for **WhatsApp markdown** — `*bold*` (single asterisks; double
asterisks do not render on WhatsApp), bullets as `•`, numbered points as `1)`,
section headings in bold on their own line.

Standard shape of a major landowner update:

1. **Bold heading** naming the subject and the news.
2. **Greeting** — a festival greeting or a warm opening where the occasion calls
   for it (he specifically wanted Ganesha Chaturthi greetings plus the note that
   the good news arriving on that day was "serendipity"). Do not invent
   religiosity; use it when he asks for it.
3. **The milestone / news**, in bullets.
4. **The ask** — meeting, dates, confirmations, action per person, named.
5. **The objectives** — numbered, one per decision required.
6. **The hard part** — the sensitive constraint, stated plainly but with the
   solution advertised ("we do have a solution, it needs everyone's buy-in").
7. **What follows** — the timeline and commitments, in bullets.
8. **Close** — gratitude for their patience, signed with his name.

### Tone calibration

- **Warm and respectful by default**, and *demonstrably* respectful of the years
  these families have waited. He asks for "a very, very nice cooperative message
  using all psychology of human psychology elements… make the landowners feel
  positive… that now there's light at the end of the tunnel."
- **Informal when he says so.** He explicitly asked to make one message "more
  informal" — the tell is phrases like "why don't you…", "Simple.", "my treat",
  "Just let me know." Drop the corporate register on those.
- **Firm on the commercial position, soft on the delivery.** The content of a
  hard line ("these matters are between you as landowners, it has nothing to do
  with me") stays hard; the packaging stays polite. Never soften the position to
  make the message nicer.
- **Named asks beat broadcast asks.** "Sundar sir — please propose a date.
  Everyone else — keep it free" outperforms an unattributed request.

---

## Pitfalls

- **Do not conflate refundable and non-refundable.** He corrected this directly:
  *"There is only refundable. There is no non-refundable at all."* When he says
  "refundable", write only "refundable" — do not helpfully pair it with
  "non-refundable" from an earlier draft. Read back what he actually said, not
  what the last version said.
- **Do not carry wording forward from a rejected draft.** Each round is a
  rewrite of the specific text he named, not a superset of every prior version.
- **Do not assume you have the group chat.** Hermes cannot read WhatsApp. If he
  asks "do you have the full conversation?", the answer is no unless he has
  attached an export — say so plainly and ask for the export rather than
  improvising from the emails.
- **Do not put a wrong reference number in a landowner/Bharat message.** Sheet row
  numbers and document numbers must be read from the live sheet on the day. He
  said "row 224" when the index had it at **row 244** — the correct behaviour is
  to check, send the correct number, and flag the discrepancy, not to silently
  use either one.
- **Small touch: if a dictated message trails off into thinking aloud** ("as they
  just sent it to me on WhatsApp… that's the message you"), **leave that out of
  the sent text** and say in one line that you did. He confirmed this was the
  right call.
