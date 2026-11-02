# Mirrored messages, identity resolution, and dictation fidelity

Lessons from a Century Regalia follow-up-message thread (Oct 2026) where two
drafted WhatsApp messages had to be corrected twice.

## 1. A "tell X what I told Y" request creates TWO drafts that move together

NDR asked for a message to Vineet, then a message to Nishant Prakash saying
"here is the message I sent to Vineet, please follow up and get me closure".

The second message **quotes the first verbatim**. So the pair is coupled:

- When he corrects the primary text, you MUST regenerate the second link too,
  with the corrected quote embedded. Never leave the quoted copy stale.
- Say explicitly in your reply that both were regenerated and why. A stale
  embedded quote in a follow-up message is worse than no message — it asks a
  third party to chase on wording that was already rejected.
- Keep one source string for the primary body and interpolate it into both the
  standalone message and the quoted block. Hand-editing two copies is how they
  drift.

## 2. Do not invent questions the user did not dictate

NDR dictated: "just checking Vineet hopefully if you are an architect if you got
all the relevant structural details".

Rendered literally, that became **"are you an architect?"** — which is wrong and
slightly insulting to a counterparty who is an investor, not an architect. What
he meant was "if you've engaged an architect / if your architect has reviewed".

Rule: when dictation is ambiguous about a person's ROLE, render the intent the
request depends on (has your architect reviewed it), not a literal question
about who they are. If identity or role genuinely matters to the ask, resolve it
from contacts first instead of asking the counterparty.

Signals that you are about to over-literalism a dictation: the sentence contains
"if you are a...", "hopefully you have...", "just checking whether you are...".
Those are hedges about a state of affairs, not interview questions.

## 3. Resolve first-name-only counterparties before drafting

"Vinit" and "Dilav" were unresolvable from the message alone. Resolution path
that worked: contact_resolver across all three Google identities, then Gmail
sender search, then the contact sheet, then session history. "Vinit" resolved to
**Vineet Agrawal of Jiraaf Capital** (vineet@jiraaf.com / +91 97390 99290).

- Never guess a first name. Two different "Vinit"s in a real-estate file is a
  live risk, and a wrong recipient is unrecoverable.
- "Dilav" matched nothing anywhere. The correct move was to say so and ask,
  which is what happened — do not silently substitute the nearest name.
- Note the near-collision in this account: **Nishant Prakash** (Yellow Eye,
  +91 99996 73483) is a different person from NDR himself (Nishant Ranka).
  Don't collapse them.

## 4. Drive links inside a WhatsApp message

When the message is "here are the files", the message is worthless without the
links — NDR corrected exactly this. Every such message must carry the full
`https://drive.google.com/file/d/<id>/view` URL per file, plus a line stating
the access granted and its expiry.

Grant the permission FIRST, verify it (reader role + expiry date), then generate
the link. Stating "30-day viewer access" in a message for a share that silently
failed is a factual misstatement to a third party.

If the counterparty's domain blocks inbound external shares (jiraaf.com did),
Drive refuses the named permission. Do not silently drop the claim — either
fall back to the anyone-with-link permission that is already set and say so
plainly to NDR, or escalate. Never tell the recipient they have been shared
when they have not.

## 5. Corrections are cheap; send-then-correct is not

Nothing in these drafts was sent autonomously — links were handed over for NDR
to tap. Keep it that way for counterparty messages with numbers or commitments
in them. A wrong "are you an architect" in a live WhatsApp thread costs
credibility that a regenerated link never does.
