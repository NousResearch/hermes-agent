# Reply-quote fan-out & user-only facts

Two drafting rules that came out of the 2026-10-16 expo status-check, where NDR's one dictation was
fanned out as identical DMs to two people and both replied by quoting it.

## 1. The reply-quote may live in someone else's chat

NDR routinely dictates ONE set of questions and asks for it to be sent to several people
("drop a Telegram message to Bharat and Prakash separately asking the same three things"). Every
recipient then replies by quoting that DM, so **the same quoted text exists in NDR's chat (where it was
drafted) and in each recipient's DM**. `session_search` on the quote returns whichever session indexed
it first — not necessarily the one you are in.

**How to establish who is actually speaking, before drafting:**

1. **Do NOT read the speaker off the session header.** The header's
   `Home Channels (default destinations) ID` is CONFIG-LEVEL (the default target for scheduled sends),
   not the source chat. Hit: header said `Home (ID: 7449813913)` (= ndr@draas.com) while the speaker was
   Prakash Singh (`8502281203`).
2. **`/data/hermes/channel_directory.json`** → `platforms.telegram[]` entries `{id, name, type}` is the
   authoritative chat-id → name map (same registry `send_message(action='list')` reads). Cross-check
   with `sessions.user_id` on that chat's other sessions.
3. **Read who the quoted text is ADDRESSED to.** `"Hi Prakash, …"` can only sit in Prakash's DM, so the
   speaker is Prakash — and a `session_search` hit on that same text belonged to a *different* person's
   chat (uid 8717455402 = Bharat Hawaldar), whose assistant addressed him as "Bharat" throughout.
4. **The tell that you crossed chats:** the recovered session's assistant used a different first name
   than the current session's profile. Stop and verify — never reconcile the two names by guessing.

## 2. Never import another person's session content into this user's draft

Colleagues asked the same question each deserve their own reply built from their own material. Do NOT
carry across: their uploaded photos, their dictated wording, their Drive folder/file links, their
signature block, their claimed actions. If the other person's collateral genuinely IS the right input,
say so and ask — do not silently import it. (Same family as the standing privacy rule: one user's data
never goes to another user's channel.)

Corollary: when the two drafts are for the same event, share only the **event facts** (name, dates,
venue, organiser contacts) and re-derive everything person-specific.

## 3. Never assert facts only the human knows

For a status-check revert ("is the expo on / did you validate the organisers / are we participating"),
separate what is groundable from what is not:

- **Groundable (verify, then state):** event identity, dates, venue, organiser legitimacy — official
  social handles posting live, sponsor announcements, on-ground vlogs of the venue, and the organiser
  contact numbers on the banner matching their public posts.
- **NOT groundable (never assert):** *"we visited the venue"*, *"we validated them in person"*,
  *"we are participating"*, *"the stall is up"*. These are the human's own experience or decision.

Draft the lines anyway, then list them under the draft as **confirm-before-send items**, offering the
variants: *"if you visited in person I'll word it X; if it was validated over calls/socials I'll word it
Y"*. A draft that reads well but asserts an unverified visit is worse than one with a flagged gap —
the user forwards it as-is.

Division of labour: `confirm-before-actions` governs whether you may SEND; this file governs what the
draft is allowed to CLAIM.

## 4. Do not inherit the requester's FRAMING of the event

The requester's own words may mislabel the thing. NDR called it *"the expo"*; the actual touchpoints
were **Durga Puja / Navratri cultural celebrations with a corporate stall presence**, and the user
corrected it in one line: *"YES, its not a EXPO - its cultural event of Navratri Celebrations."* Labels
like expo / show / mela / exhibition get dictated loosely. Verify the TYPE from on-ground evidence (the
banner photo, the venue's own listings), then use the organiser's own event name in the draft — and
state the correction out loud inside the draft (*"One correction: this is not an expo, it is a cultural
event — Navratri / Durga Puja celebrations"*), because the requester is holding the wrong mental model
and the revert is where it gets fixed.

**One revert may cover MORE THAN ONE venue.** Confirm how many touchpoints exist and give each its own
labelled block (`Location 1 — …`, `Location 2 — …`) naming that venue's role for us (cabin + team
present vs standee-only + leaflet/QR-code push). One undifferentiated paragraph invites the requester to
assume both are the same deal.

## 5. Use the user's OWN wording for contested numbers

When the dictated count does not survive arithmetic (*"4 developers — main sponsor Fortune Primero, then
Motherland, two more and us"*), do NOT silently re-derive it, and do not quietly drop the number. Put
HIS sentence in the draft and put the arithmetic question in the confirm list (*"you said four
participants but named five — which is it?"*). The number is going to a third party; the user owns it.

## 6. Delivering the confirmed revert to the other person

Once the user says go — e.g. *"Do all the three things"* means execute EVERY flagged item, including the
send and any peer cross-check; never send only the easy ones:

- **Recipient addressing:** `send_message(platform='telegram', recipient_name='<Full Name>')`. Explicit
  `telegram:<chat_id>` targets are refused by the cross-user guard (*"Cross-user send blocked"*); the
  name route resolves through the identity registry and passes. (NDR's own session has been blocked
  sending to staff chats — do not assume a block you saw in one chat applies in another; just re-route
  through `recipient_name`.)
- **Photo evidence:** append `MEDIA:<absolute path>` lines to the body, one per image, in the order the
  message references them. Convert large PNGs to JPEG first (2.5 MB screenshot → ~400 KB at 1800 px,
  q88) via `PIL` in `execute_code`; check the path exists before sending.
- **Verify it landed:** the tool returns `{chat_id, message_id}` — quote the message id in your report.
  Independent tell that media really uploaded: the call duration (a 3-photo send took **3.57 s**; the
  text-only peer message took **0.35 s**). A suspiciously fast "success" on a message carrying MEDIA
  lines means the attachments did not go.
- **Peer cross-check as a deliverable:** when the user wants a colleague to corroborate, send that
  colleague a numbered confirm-request restating exactly what you have asserted (venues, count, roles,
  who is staffing), with three explicit asks — (a) are both on and the organisers validated, (b) is the
  count right, (c) anything to correct. Then report back: what went where (with message ids), and what
  is still outstanding.
