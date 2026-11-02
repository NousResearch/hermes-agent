# Voice-garbled firm name → contact discovery via Gmail (worked 25-Aug-2026)

Scenario: NDR says "architect Balan and Nambisar" / "Balan & Nambisar" and wants past email
contact info for scope-of-work outreach. No hits in contacts sheets or People API for either
name — the firm lives only in email threads.

## Resolution chain (in order of probe)

1. **Search all 3 vault accounts for name + context words.** `Balan Oasis`, `Nambiar Oasis`,
   `architect Balan`, and the raw voice spelling `Nambisar` all hit only in `google-draas`
   (ndr@draas.com). Zero in google-ahfl / google-gmail. The voice spelling `Nambisar` itself
   was the highest-signal query — it appeared only in NDR's own narration emails.
2. **Read the hits — NDR often SPELLS the name in his own dictation narration.**
   A 19-Mar-2024 email "Email for Balan and Amritsar as well as my total narration" contained
   the literal text: *"detailed email to the architecture firm Balan and Nambisar"* plus the
   full Sevaganapalli villa project brief (10 ac, next to Clover Golf, mix of 1200/1800-sqft
   plots). This confirmed: voice "Balan and Nambisar" = **Balan + Nambisan Architects (BN Architects)**.
3. **Search the firm's email domain** (`bnarchitects`) to enumerate all people + thread history.
   The "Expected scope of work - Sevaganapalli Villa Project" thread (Mar-2024 → Feb-2025) had
   the full team in From/To/Cc:
   - Janice Rodrigues (Associate Architect) — janice@bnarchitects.co.in — authored the scope
     statement/proposal, main POC on the thread
   - Arjun Nambisan (principal) — arjun@bnarchitects.co.in ["Nambisar"]
   - Arun Balan (principal) — arun@bnarchitects.co.in ["Balan", older Ranka South City thread 2014]
   - Jeevan Somaiah — jeevan@bnarchitects.co.in; Suresh BNA — suresh@bnarchitects.co.in
4. **Pull full bodies for signatures** — People API / sheet had NO phones for these people, but
   every Janice email signature carried the firm card:
   - BN Architects, #226, 1st 'C' Cross, 18th Main, HAL 2nd Stage, Bangalore-560008
   - Ph: +91-80-25217543 / 44 · www.bnarchitects.co.in · IG @balan_nambisan_architects
   Address appears in `text/plain` bodies of `format='full'` — regex for `\+?\d[\d\s\-]{8,}\d`.
5. **Report with role labels + CC pattern.** NDR's loop on Ranka Oasis was NDR + Roshini + Bharat;
   BNA side Janice (lead) + Arjun + Jeevan + Suresh. Recommend the same loop for a new
   scope-of-work outreach.

## Pitfalls
- Never assume contacts sheets/People API hold professional firms — architects, consultants, and
  government people often exist ONLY in Gmail signatures. The Gmail-signature extraction path is
  the authoritative source for these.
- "Nambisar" in NDR's voice ≈ "Nambisan" (firm principal); "the below address" in a voice memo
  often means the actual address got lost in transcription — ask for it rather than guessing.
- Worked example of the wider rule (memory): searching the user's OWN narration emails can
  reveal the true name behind a garbled voice noun, because NDR dictates instructions containing
  the name spelled correctly.