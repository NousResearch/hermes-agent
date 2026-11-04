# Personalized update messages — "personalize all three" (NDR, 2026-09-30)

Canonical case: court-hearing update WhatsApp messages from NDR to counsel (that appeared in the
matter) and a non-lawyer advisor (CA). NDR dictated the same event update and asked to "personalize
all three messages" — the pattern below is what he approved.

## Pattern

- **Open each message with "Hi <FirstName>, good morning."** The name personalizes even when the
  body is near-identical. Use first names only.
- **Trim the detail — the recipient knows the matter.** Drop case numbers, IA numbers, judge
  names, filing dates unless explicitly asked. Keep the update to: hearing happened → next date →
  no order published → IAs still pending.
- **Tailor the ask by role:**
  - Counsel who appeared in court → *"since your team appeared in the matter, just wanted to know
    what you're doing on it and how you'd suggest we take it forward"* — next step / plan, NOT
    "what happened in court" (NDR already knows that from them).
  - Non-lawyer advisor (e.g. CA) → inform + invite advice: *"no order was passed, it's posted to
    the 13th, going from date to date … just wanted to know your advice — how do you think we
    should handle this?"*
- Same-body recipients (e.g. two counsel at one firm) differ only in the greeting name.
- One message per recipient, generated via `whatsapp_link` (platform='telegram', display_link in
  the reply) — a link NDR taps, never a claim of a send. Nothing is "delivered" until he taps.

## Why it matters

The first drafts (previous session) were technically complete but heavy — full writ number, both
IA numbers, judge name, filed dates, and an identical "what exactly happened in court" ask for all
three. NDR corrected: drop the numbers ("we don't need to refer to the numbers and all"), and
split the ask by role. The lighter message is what he actually wants to send.

## Verbatim guidance he gave in that correction

> "We don't need to get into too much detail because he knows the IA, he knows the matter. So we
> don't need to refer to the numbers and all. Then just since your team appeared in the matter,
> wanted to know what you're doing. … For Saurabh, it is more about … this is what happened in
> court. Saurabh was not the lawyer. … So just wanted to know what is your advice. I don't think an
> order got passed and now it's posted to the 13th. Obviously, it's going from date to date to date
> as a trend. What do you think? How do you think we should handle it?"