# Group broadcasts to a NAMED WhatsApp group — what a link can and cannot do

Verified 2026-09-20 (NDR: *"give me a WhatsApp link for the group where Jay and
Chanakya both are part of the group"*).

## The hard limit

`whatsapp_link` group mode is **empty-phone + pre-filled text**
(`args = {"phone": "", "text": msg}`). The user still picks the recipient/group
at compose time. There is:

- **no deep-link format that targets a named WhatsApp group**, and
- **no way to address the message to a group in the URL**.

Do NOT invent `chat.whatsapp.com/<invite>?text=...` — the invite link accepts no
`text=` parameter and the text is silently dropped. Group *invite* links join a
group; they do not compose a message into it.

## What to deliver instead (NDR-accepted shape)

Two artifacts, in this order:

1. **The full message as a copy-paste code block** (fenced ``` ``` in the chat
   reply). This is what actually gets pasted into the group — one tap to copy on
   mobile, no link tap-through, no recipient picking. For a group broadcast this
   is the primary deliverable, not the fallback.
2. **The group link** (`{"phone": "", "text": msg}`) as a convenience for the
   same message, labelled so it's clear the user picks the group.

State the limitation plainly in the reply — *"there's no way to pre-fill a
message into a named group from a link; here is the message as a copy-paste
block"* — rather than staying silent or promising a link you cannot produce.

## Prerequisite gates before drafting the broadcast

Group broadcasts that announce "I have sent you the document" fail soft if the
underlying facts aren't verified first. Confirm all of these BEFORE writing
message text (2026-09-20 attempt stalled on every one):

- **The email being referenced actually exists** — search the sending account's
  sent mail (`in:sent`) for the recipient + recency. A voice note saying "the
  email I sent to X" is not evidence the email exists; if the newest thread to
  that person is years old, say so instead of summarising a phantom email.
- **The document is identified** — get the filename/Drive link. Without it the
  "here's what we changed" section is fabrication.
- **What changed and why** — especially for a "poor-card" / weak-document
  instruction. If the change history isn't in this session, ask; do not
  reconstruct it from the voice note alone.
- **Every recipient's email address** — group members named only by first name
  ("Jay") must be resolved (full name + email) before any Drive sharing is
  promised. Check `contact_resolver` → People API → contacts sheet; if nothing
  matches, ask. Do not promise access to an unresolvable address.
- **Access-grant list** — name exactly which addresses get the document, and
  flag any you cannot resolve.

## Resolved case (2026-09-20) — the shape that was accepted

The stalled broadcast above completed once the blockers cleared. What worked:

- **Message length landed at ~1,390 chars / ~200 words** for a five-section
  brief (title, what it is, 3 numbered changes, link, open items). This is the
  practical ceiling — do NOT pad to hit a word count the user says out loud
  ("less than 200 words"); a numbered change list plus a link and asks is
  naturally ~200 words.
- **Sections that earned their place:** `*The document*` (one sentence, what it
  is), `*What changed in v3*` (numbered), `*Document*` (bare link on its own
  line), `*Open items*` (• bullets, each an ask with an owner named).
- **Bold the section headers and the change headlines only.** Body text plain.
  WhatsApp renders `*bold*` and `_italic_`; there is no header syntax.
- **Address the group by name in the first line** ("Emailed to you, Chanakya —
  and to Jay"), then speak to both.
- **End with a single call-to-action**, not three.
- Delivery = the fenced code block (primary) + the phone-less link (secondary),
  exactly as above.

### Do not silently drop caveats to look cleaner

The document carried its own internal caveat — the headline figures were *"to be
reconciled line by line against the site register … before this section is
relied on."* The user asked to summarise "whatever we changed and why."

Including an unverified number in a group broadcast because the user asked for a
summary is a trust problem, not a formatting one. Two honest options, and you
should **offer both** rather than picking silently:

1. Keep the figures but state the condition ("must tie back to the actual site
   register").
2. Cut the figures from the group message entirely and keep the ask
   ("the §7 figures must be reconciled before anything goes out").

Flag this as a judgement call in the reply to the user — surface it, don't bury
it. Verified 2026-09-20: the user was told the figures were unverified and given
the option to cut that bullet.

## The Gmail-account blocker

When the referenced email was sent from `nishantranka@gmail.com` or
`ndr@ahfl.in` and those vault tokens are dead (`invalid_grant: Token has been
expired or revoked`), no amount of searching finds the email. Send the re-auth
button (`references/oauth-reauth-terminal-pattern.md`) and ask for the missing
facts in the same reply — batch the blockers into one message rather than a
multi-turn interrogation.
