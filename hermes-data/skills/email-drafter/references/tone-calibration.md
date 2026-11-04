# Tone calibration — draft in NDR's voice, don't invent it (Sep 2026)

NDR's explicit instruction (Flameback/Kishan advisory round, Sep 2026): before drafting any
person-facing email for him, study his ACTUAL prior correspondence with that counterparty —
email and WhatsApp — and mirror the register. Never guess his voice.

## Procedure that worked

1. Pull NDR's sent mail to the target counterparty from google-draas:
   `q='(to:<addr> OR to:<domain>) after:2026/06/01 -in:chats'`
   - List metadata (Subject / From / To / Date) for the full sweep (~26 hits over 3 months).
   - Fetch full bodies of the 5-10 most recent substantive replies, NOT just snippets —
     the register lives in the opening line, the push paragraphs, and the closing.
2. Read for register and reproduce it:
   - Warm but direct: credit the good first, then the hard number.
   - Signature idioms: "My view is that...", "I'll be honest with you", "threadbare
     discussion", "let's align", numbered point-by-point asks, "let's block time".
   - Short paragraphs, businessperson not consultant, no AI-sounding bullet soup.
   - Always closes with a concrete next step (meeting, deadline, "come prepared with X").
3. When NDR says future drafts should match tone, offer the source samples he liked most
   (a strong dated sample: NDR's "Re: Friday" to Kishan, 29 Aug 2026).

## Verified Gmail mechanics for this sweep

- WHOAMI check first: `build_service('gmail','v1',service_name='google-draas')` +
  `getProfile()` must print `ndr@draas.com`; else prefix
  `HERMES_SESSION_USER_ID=7449813913` and rerun.
- Bodies: `format='full'`, walk `payload['body']['data']` (NOT `part['data']`).
- Dedupe by message id; sort by Date; the sent sweep overlaps the inbox sweep.
- Full worked example: `private-investment-due-diligence` →
  `references/flameback-survival-2026-round2.md` (incl. tone section + per-email draft pattern).