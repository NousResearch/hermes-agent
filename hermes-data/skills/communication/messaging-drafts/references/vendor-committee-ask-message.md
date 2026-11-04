# Vendor / committee "ask" message — rates, permission, booking

Verified 2026-10-05 (Durga Puja stall + gate branding, sent to Sumanta Basu of Sukriti
Cultural Association; drafted from SCA's own sponsorship deck).

## WhatsApp, not a call script

When the next step after logging an activity is "get the rate / permission / booking from
them", Bharat wants a **WhatsApp message**. Offered "call script / message", he answered
*"Better whats app message."* **Default to the tappable message; do not offer a call script
as the primary option.**

## One message per committee — not one per ask

When two tracked activities involve the same committee at the same venue (e.g. a stall space
and entry-gate branding, both at the same pandal), **fold them into a single message** as
numbered asks. One conversation beats two, and the same person answers both.

## Shape

- **WhatsApp markdown**: `*single asterisk*` bold, `•` bullets. Never `**`, never Telegram
  escaping. Pass the text to `whatsapp_link` unmodified.
- Open with `*Namaskar <Firstname> ji,*` for an Indian vendor/committee contact, then one line
  of who you are, plus a warm acknowledgement of whatever they sent you (their deck, proposal,
  quotation) — it is the reason for the contact.
- Then **numbered ALL-CAPS headings with `•` bullets underneath**, in the order a site team
  actually needs the answers:
  - `*1. PERMISSION*` — is what we want to do even allowed here?
  - `*2. SPACE AND RATE*` — size available, rate for the whole period, flat vs daily
  - `*3. WHAT IS INCLUDED*` — covered/shaded space, power point, table and chairs, may we
    bring our own standees and a screen
  - `*4. DATES AND PAYMENT*` — deposit, terms, and the deadline to block the space
  - `*5. <second ask>*` — e.g. gate branding: who to speak to and the rate
- **Turn unresolved facts into QUESTIONS.** A date contradiction found between the tracker and
  their deck goes in as *"Please confirm the dates — 16 to 21 October 2026?"* so their reply
  settles it, rather than being asserted either way.
- Close with availability (*"Happy to visit the site or connect on a call"*) and a signature
  block: name, org, email.
- Budget ~150–200 words. It has to be readable and **forwardable** on a phone — assume the
  recipient will forward it onward to their own people.

## Delivery

- Pull the number from the **source document** — for a deck, the last slide is the
  authoritative booking-contact list. Prefer the first/admin contact; don't invent a number.
- Generate with `whatsapp_link` and deliver the returned **tap link**. Say plainly it is a
  GENERATED link and **nothing has been sent** — he must tap and press send.
- **Render the message text in the reply as well**, so he can review wording before tapping.
- **Offer — do not auto-generate — links for the other listed contacts.** He will ask if he
  wants them.
- Note anything you deliberately did not send: a phone number from a dictated briefing that
  was left out, a name that couldn't be verified, etc.
