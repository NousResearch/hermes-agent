# Matter-update briefing drafts (legal/tax updates to counsel + advisors)

Pattern from NDR's voice-dictated revisions (P-141, Ashok Kumar GST matter, WP 7542/2025,
30-Sep-2026). Use when NDR says "prepare an update for <names>" about a court matter, or
when he re-dictates a revision ("personalize all three messages").

## NDR's second-pass rules (the correction he actually wanted)
1. **Personalize the greeting by name** — "Hi Darshan, good morning." / "Hi Saurabh, good
   morning." Never a generic "Good morning." shared across recipients.
2. **Drop the numbers when the recipient knows the matter.** Explicit NDR: "We don't need to
   get into too much detail because he knows the IA, he knows the matter. So we don't need to
   refer to the numbers and all." → no case/WP numbers, no IA numbers, no filing dates, no
   judge name for people already in the matter. Refer to "the Ashok Kumar GST matter".
3. **Role-split the ask per recipient — lawyers ≠ non-lawyer advisors:**
   - Counsel who APPEARED in the matter (e.g. Darshan/Akshay of Trust Law): update + "since
     your team appeared in the matter, just wanted to know what you're doing on it and how
     you'd suggest we take it forward." Do NOT re-ask "what exactly happened in court" —
     NDR already knows from them; the message is about next steps.
   - Non-lawyer advisor being informed (e.g. Saurabh, the CA): update in plain terms + ask
     for advice: "no order was passed, it's posted to the 13th, seems to be going from date to
     date — what do you think, how do you think we should handle this?"
4. Keep the factual update identical in shape: hearing took place → adjourned to <next date>
   → still at preliminary stage → no order published → both IAs pending. One paragraph.
5. Deliver as WhatsApp deep links via `whatsapp_link` (platform='telegram'), display_link in
   the reply, "links only — nothing sent". Numbered per recipient with name + firm + number.

## Building from NDR's dictation
NDR describes the messages as flowing prose, not templates; parse per-recipient intent from
his words ("For Saurabh, this is what happened in court... So just wanted to know what is
your advice"). Grep the context (tracker notes, previous session) for who is counsel vs
advisor before drafting — the previous pass had sent identical lawyer texts; the revision
demand was personalization + trimming.

## Verification
- Show each numbered link with its pre-filled text so NDR can read before tapping.
- Recipient phone numbers come from the contacts sheet / prior chat log — confirm before linking.