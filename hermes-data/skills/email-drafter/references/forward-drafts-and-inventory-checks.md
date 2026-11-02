# Forwarding mail to colleagues — and the "bunch of emails" inventory check

## Rule 1 — never accept the recollection as the inventory

NDR describes mail from memory ("there are a bunch of emails yesterday from Kotak … Nilesh has
sent it"). Almost always the count or the sender is approximate. Verify the ACTUAL inventory
before drafting, and report the real number even when it contradicts him.

Exhaustive sweep, in this order:

1. `gws_resolve_account` — list every vault account; search all of them, not just `google-draas`.
   (ndr@drahomes.in mail also lands in the draas mailbox, so draas coverage is usually enough —
   but check rather than assume.)
2. Keyword + date: `(<bank>) (<phrase>) newer_than:Nd -in:sent`
3. `in:anywhere "<phrase>"` — catches archived
4. `in:spam newer_than:Nd`
5. `<date range> has:attachment -in:sent -in:spam` — catches a covering mail whose SUBJECT hides
   the content
6. `from:<bank domain>` over 7–30 days, plus the officer's own domain (e.g. `from:kotak.com`,
   not just `kotak.bank.in`)

2026-10-23 worked example: "a bunch of Kotak balance-confirmation emails from Nilesh" resolved to
ONE thread of THREE messages, no attachments at all — Nilesh Prasar (DVP & Branch Manager) had
replied "Sent on registered mail ID with request ID 50536257", i.e. the certificate went to the
registered email, not into NDR's inbox. The honest report was: one bank email, the confirmation
itself is not here. Drafting several forwards to match his recollection would have been
fabrication.

## Rule 2 — forwarding is draft-only, and there is no forward tool

There is no `gws_gmail_forward_*`. Compose a NEW draft via `gws_gmail_draft_create` with:

- `subject`: `Fwd: <original subject>`
- body: 2–3 line covering note (why it matters, what to do with it), then
  `---------- Forwarded message ---------` with From / Date / Subject / To and the original body
- `to` = the colleague who needs to act, `cc` = whoever should be on record

Never `messages().send`. Drafts only; NDR sends.

## Rule 3 — say so when the recipient already has it

If the addressee of your forward is the original recipient of the bank mail (Eshwari was), the
forward is for the FILE, not for information — say that in the covering note ("forwarding for the
record … please keep the confirmation and the request ID in the audit file") and flag it to NDR
in the approval message. He then either confirms or names a different recipient.

## Colleague addresses used for cc (verified)

- Roshni / Roshini Ranka — rnr@draas.com
- Sarthak Sharma (NDR's admin) — admin3.blr@draas.com
- Eshwari Chamundeshwari (accounts & finance) — echamundeshwari@draas.com
  (distinct from Eshwari Anand, eshwariand2024@gmail.com)
