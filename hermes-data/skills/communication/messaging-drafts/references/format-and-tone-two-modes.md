# Email format & tone: the PRINTED-vs-EMAIL split

NDR has corrected format on the same class of work repeatedly, and the correction
has always been **in the same direction**: he wants MORE formatting in the email,
not less. Getting the axis wrong costs a full redraft, so read this before
drafting anything that leaves the building.

## The real axis is the SURFACE, not the recipient

| Surface | Format |
|---|---|
| **Email** (to a colleague, an authority, a grievance cell, a vendor) | **Rich HTML, inline CSS.** All facts and figures in the body. Bold on every identifier, date and figure. Bullets and numbered lists. Tables for anything tabular. Attach the actual evidence files. |
| **Printed letter** for hand-delivery, signing, or enclosure in a physical bundle | **Clean one-page letter.** Prose, letterhead, signature block, no decor. |

Same facts, two different formats. An email is read on a phone and has to be
scannable; a signed representation is read as a document and has to look like one.

🔴 **Do NOT read this as "authority letters = plain text."** That inference has
been made once and was wrong. NDR's actual words, twice, on grievance emails to
government offices:

> "It's all text. There is no bold. There is no bulleting. Please redo that so
> that the email becomes very, very easily readable."

> "the emails again don't contain any of the facts and figures properly
> represented. The email body itself should have all of the facts, figures, maybe
> even all of the necessary attachments and a strong case made out plus
> highlighted with bullets, bold, maybe use HTML CSS to make it so it's easily
> readable. This email is pure text and it's not easily readable."

The first of those was answered with plain text (a misread) and he sent it
straight back. So: **plain-text email = FAIL**, whoever the recipient is.

## What "rich HTML" means concretely

- Inline styles only — no `<link rel=stylesheet>`, no `@page`. External CSS is for
  printable artifacts, not mail.
- Serif body (Georgia / Times), ~14–15px, line-height ~1.6, max-width ~860px.
- A **summary callout box at the top** carrying the request or the headline fact in
  one or two lines — this is the first thing read.
- **Bold section headers**, each with a rule under it.
- **Bold on every figure, date, reference number, address and name.**
- Bulleted lists for what is done / what is held; numbered lists for what is asked.
- A **table for any enumeration** — appointments, document lists with what each
  document proves, contact cards, status of open items with colour-coded cells.
- Colour-coded status cells: green HAVE/Delivered, amber PARTIAL, red MISSING/Bounced.
- **Attach the real files**, not just a Drive link, when the evidence matters.

Mechanics: build the HTML as a file in `/tmp`, create the draft through
`tools.gws_skill_bridge.call("draft_create", ..., html=True)` — `html=True` is
mandatory or the tags ship as literal characters — then verify by decoding the
draft's HTML through `build_service("gmail","v1")`. `gws_gmail_get` returns an
empty body for an HTML-only draft; that is not a failure. Full recipe:
`html-email-draft-large-body.md`.

## Tone mandate (government / authority correspondence)

Formatting is rich; **tone** is the opposite of aggressive. NDR:

> "the entire action pack, the content is cooperative, polite tone, seeking help,
> not prescriptive, not trying to use legal arguments, not expressing frustration,
> not holding anyone accountable, but seeking help and just showing that it's...
> it should elicit sympathy on the person. So it should create a desire in the
> person reading to want to help."

- Open with who is writing and for whom, and ask for **guidance**, not action.
- "We may have misunderstood — please guide us" rather than "the rule is wrong".
- Human facts early and plainly: age, widowhood, date and place of birth, why the
  document was never issued. No adjectives.
- State everything already done voluntarily (notices, affidavits, fees) as good faith.
- Say explicitly that **no relaxation of verification is being sought** — this
  defuses the fear that the office is being asked to cut a corner.
- Cite the law only as quiet background, never as a threat or a deadline aimed at
  the reader.
- Close with gratitude and an offer to supply anything further.

So the rule is: **rich formatting, humble voice.** The formatting is for
legibility; the voice is for sympathy.

## The alternatives trap

NDR's internal position is often "either name is fine, we don't mind which one
they print." That is correct internally and **wrong in writing**. In a letter to
an authority, offering alternatives reads as a request to *change* a name — which
triggers a different and much harder process.

In writing there is exactly ONE stated ask: the name already printed on the
existing passport. Variants are described only as *aliases of the same person*,
evidenced by official records. Keep the "either is fine" position in the internal
strategy document only.

## Tone when the email ASKS a counterparty to do work

The government mandate above is one direction of humility. There is a second, and getting it
wrong reads as **rude**. NDR, on a draft asking a designer to prepare 13 hoarding posters
(2026-10-17):

> "you cannot keep the same thing in the draft, it looks rude. ... we need to get the 13 posters
> prepared from Gowri ma'am, so we are requesting her, can't be so rude."

What he objected to was the scope-boundary paragraph:

> "preparing these posters was **not our scope**, but we have prepared them anyway… Please treat
> the attached posters as a **reference only**."

Two failures in one line: it **draws a boundary / claims credit** at the exact moment you are
asking a favour, and "reference only" devalues the work you are handing over.

**Rule: in a request email, never state what is not your scope, never justify why you did the
groundwork, and never hedge your own deliverable with "for reference only".** Convert the same
fact into a service to the counterparty:

> "We have prepared a direction poster for each of the 13 locations ourselves, purely so that the
> distance, the drive time and the arrow direction are settled before design begins. Please treat
> these as a working reference – **the design itself is entirely in your hands**."

- Say what you did **for them** ("so that X is settled before you begin"), not what you were or
  weren't obliged to do.
- Explicitly hand over ownership of the deliverable ("the design is entirely in your hands") —
  it removes any sense that you are pre-approving their work.
- Add a next step before the sign-off ("happy to walk you through any of the locations before you
  begin").
- Keep the actual asks as numbered points. A request email should read as cooperative, never as
  a scope ruling.

This is the same failure class as the alternatives trap below: an internal position (who owns
what, either option is fine) that is correct internally and wrong the moment it is written down.

## Tone when the email QUESTIONS a professional's own record or output

Third direction, and the easiest one to get badly wrong: you are writing to a doctor, lawyer,
architect or consultant about something **they produced** that looks wrong. Correcting a
professional reads as an attack on their competence unless it is deliberately built not to.
NDR on a clinic query, having rejected a firmer first draft (2026-10-22):

> "make sure the message is very polite, cooperative, not trying to in any way sound, point
> out that the doctor was wrong."

The shape that worked:

- **Own the possibility of being wrong first.** *"I want to be careful here, because this may
  simply be me misreading the note."*
- **State the gap as YOUR assumption, not their error.** *"So I assume no examination was done
  for me … but I am not certain, and I would much rather ask than assume."* Same fact, no
  accusation.
- **Explain the mechanism neutrally** where you have a hypothesis — *"this may have carried
  over while one prescription was closed and the next opened"* describes how the system fails,
  not how the person failed.
- **Absolve explicitly.** *"There is no complaint intended here at all — Dr. <name> was
  extremely helpful, and I am only trying to keep my records correct."*
- **Give the real motive**, which is usually administrative, not adversarial (he scans and
  retains records; he wants the filed version accurate).
- **Ask for the corrected artefact**, not for an apology or an explanation of fault.
- **Urgency marker ≠ sharper language.** A `🔴 CRITICAL — URGENT` headline at the top is fine
  and he asks for it; the body underneath still has to be the softest version.

Banned in this mode: *"please do not leave it ambiguous"*, *"I was genuinely surprised to
read…"*, and any itemised list of what the professional failed to mention or do.

Domain-specific version with a full template:
`family-health-operations` → `references/diagnosis-provenance-challenge.md`.

## Length — rich format is NOT licence to be long

Formatting and length are separate axes and NDR has now corrected the length axis
explicitly. On a task-instruction email to Sarthak (2026-10-23):

> "Detailed email … but don't make it too verbose. Nobody reads very verbose emails.
> But all the key points and basic instruction to Sarthak, please do the needful to get
> the address updated."

So: **rich HTML, but short.** The failure mode is a beautifully formatted four-page email
that nobody finishes.

- **One purpose per email.** If two unrelated asks arrive in the same voice note, send two
  emails (he asks for this himself: *"a separate email to X marking Y and Z"*).
- **Lead with the ask in one line**, then the minimum background needed to act on it, then
  numbered steps. If a section is only there to show how hard you worked, cut it.
- **Cut the reasoning out of the body.** The trace, the computation and the "here is how I
  found this" belong in chat to NDR, not in the email to the person doing the task.
- **State facts the executor needs, not the story of the facts.** "Address on record is X"
  is needed; three sentences explaining the depository chain are not — point to the source
  instead.
- **Keep tables only where they replace prose.** A 6-row facts table is compact; a table of
  things already said in the paragraph above is length.
- **End with the single instruction and a deadline**, not a summary.

Length by artifact — get this right or the email is wrong in one direction or the other:

| Artifact | Length |
|---|---|
| Task instruction to a colleague / admin | **Short** — one screen, numbered steps |
| Field / site-visit checklist that someone physically walks around with | **Long by design** — the nine-section brief is a walk-around document, do not compress it |
| Review memo to a family member / board colleague on a document | **Medium** — the analysis, not the appendix |
| Draft to an authority / regulator | Long is acceptable; the case has to be made — but every paragraph must carry a fact |

## Follow-up on an ask that is already live on email AND WhatsApp

When NDR says *"we already have an email and a WhatsApp on this, use that to build the
context"*, read `cross-channel-followup-anchoring.md` before drafting anything. In short:

- Recover the prior mail **full body** (not the snippet) and the counterparty's **last
  committed date** from `wa_read_messages` on that chat.
- That commitment **stands** — never quietly reset it. Insert an earlier checkpoint and ask
  them to flag slippage there, not at the deadline.
- Ask for the **plan before the report**: whom / how / which counterparties, each marked
  "booked with date" or "not yet booked".
- Reserve the **final meeting for NDR** with a pre-read sent before he walks in.
- WhatsApp = action and timeline only; the email = the same asks plus the why.
- Project names drift between voice dictation and written records — use the email-thread
  spelling, never the dictated one.

## Quick decision rule

| Document | Mode |
|---|---|
| Email to staff / travel agent / accountant / vendor, any facts | Rich HTML |
| Email to a government office, grievance cell, regulator | Rich HTML, humble voice |
| Email asking a vendor / partner / consultant to produce work | Rich HTML, request framing — no scope or credit language |
| Email / WhatsApp querying a professional's own record or output (doctor, lawyer, architect) | Rich HTML (deferential register; own the misreading first, never imply they erred) |
| Printed one-page representation, affidavit cover, covering letter in a bundle | Clean plain letter |
| WhatsApp message | Plain text always — no HTML in WhatsApp |
