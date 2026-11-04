# Sensitive feedback to a colleague — "maintain the intent, make the language collaborative"

Verified 2026-09-20 (NDR dictation: *"in as polite and cooperative a language as
possible. The intent is what I want to tell him, maintain the intent, but the
language and choice of words make it as collaborative, saying it's a joint
problem we need to solve, but it's a critical, critical one."*)

## The pattern

NDR regularly dictates hard feedback for a reportee and then explicitly asks for
a **tone transformation** in the same breath: keep the substance, soften the
delivery, frame it as shared rather than top-down. The dictation sounds harsh —
it *is* harsh, that's the point — and the request is to translate, not sanitise.

The failure mode is softening so far that the issue stops being visible. NDR
wants both: unmistakably serious, unmistakably collaborative.

## Rules

1. **The intent is fixed. Only the register changes.** If he says the Iris
   basement concreting and the block work were "a big disappointment", the
   message must still convey a real quality failure — not "some opportunities to
   improve". Deleting the substance to sound kind is a failed transformation.

2. **Before drafting, separate the three layers of the dictation:**
   - **The trigger** — the specific artifact under discussion (a video, a site
     visit, a report). Usually *not* the actual problem.
   - **The issue** — the thing that genuinely worries him. Here: a divergence in
     what counts as acceptable finishing quality, which will affect *all* future
     finishing work if unresolved.
   - **The examples** — concrete instances he cites (Iris basement, block work).
     These carry the weight and must survive into the draft, but they are
     evidence *for* the issue, not the issue itself.

   NDR is often explicit that the trigger is not the point — "I am not about this
   specific job", "it's not about X specifically". Honour that distinction; if
   you frame the message around the trigger, you've missed the ask.

3. **Language that reads collaborative without diluting:**
   - "I'd like your view" / "I want to understand how this looks from your side"
     — genuine invitation, not a rhetorical setup.
   - "This is something we need to solve together" — joint ownership.
   - "I may be seeing it differently" — leaves room without withdrawing the
     concern.
   - Name the gap as a **shared** problem: *"if what you consider acceptable and
     what I consider acceptable are far apart, that's a gap for us to close"* —
     not "your standard is wrong".

4. **Keep the escalate-or-fix fork in.** When NDR dictates a remedy — here, "if
   the gap is big we need someone whose quality perception aligns with mine" —
   carry it as a **last-resort** option, framed conditionally (if the gap turns
   out to be that wide). Stating it flatly reads as a threat; dropping it
   entirely loses a real instruction. Put it after the joint-framing, not before.

5. **The ask is a call.** Sensitive feedback should not be resolved over
   WhatsApp text. End with a request for a conversation, and make watching /
   reading the referenced artifact the precondition:
   *"Please take a look at the video, then let's have a call."*

## Formatting

Follow the standard NDR WhatsApp convention (`messaging-drafts` → `references/messaging-links-workflow.md` SKILL.md):
• bullets for lists, short sections, each thought on its own line, **bold** key
items, single clear call-to-action at the end. No wall of text. This holds for
difficult messages too — a dense paragraph reads as a lecture.

## Colleague-contact rules that apply

- Address him by his preferred name — `addressed_as` from `contact_resolver`
  (Anbarasan → **"Anbu"**).
- Use the **WORK / DRA mobile**, never a personal number, unless NDR says
  otherwise for that specific message. Full rule + verified numbers in the
  `contact-phone-lookup` skill's Verified Phone Numbers table.
- Resolve the number from that table *before* building the link — it is
  authoritative over anything the contacts sheet or People API returns first.

## Handling a missing artifact

If the referenced video/document link cannot be produced (wrong channel, no
access, not found), do **not** stall the draft. Deliver the message with an
explicit `[LINK]` placeholder, state the blocker, and ask for the URL in the
same reply. Wave the URL as the only outstanding item — it keeps the handoff a
one-line paste for NDR instead of a fresh drafting round.

See `references/youtube-metadata-extraction.md` in the `research-web-tools`
skill for the channel-uploads resolution ladder and its pitfalls.
