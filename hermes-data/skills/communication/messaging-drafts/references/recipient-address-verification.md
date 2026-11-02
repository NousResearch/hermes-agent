# Verifying which address a colleague actually uses

The contacts sheet is a directory, not an authority. It carries **stale addresses, an
obsolete-address column, and other-domain addresses for the same person** — and a draft to
the wrong one either bounces or, worse, goes to a live mailbox nobody reads.

Worked case (2026-10-23, Bharat Hawaldar):

- `contact_resolver("Bharat")` returned the labelled **Work** email as
  `sales1.blr@drahomes.in` — wrong company domain.
- His actual address is **`sales1.blr@draas.com`**, confirmed two independent ways from his
  own mail: messages with `from: BHARAT H <sales1.blr@draas.com>`, and a `To:` line on an
  internal circular listing `Bharat Hawaldar <sales1.blr@draas.com>`.
- The same contact row also carried an "Obsolete" field holding several other staff
  addresses (`blr.sales1@draas.com`, `mmallick@draas.com`, …) — that column is junk for
  addressing purposes; never send to anything you found there.

## The rule

1. Run `contact_resolver` for the name and the context.
2. **Then confirm from a recent message header** the person themselves authored:
   `gws_gmail_search("from:<name> newer_than:60d")` — the `from` field is the truth.
   If they have never written, search for their name in a `To:` line of an internal
   circular, a meeting invite, or a Drive share notification
   (`"Spreadsheet shared with you"` mails name the sharing account).
3. If the resolver's answer and the header disagree, **the header wins** — say which you
   are using and why in the chat reply, so NDR can correct it in one word.
4. Domain tells you the entity: `@draas.com` = DRA Realty / DRAAS Bangalore staff,
   `@drahomes.in` = DRA Homes Chennai, `@redsoul.co.in` / vendor domains = external.
   NDR's own rule is `ndr@draas.com` for company/vendor/KYC forms, `ndr@drahomes.in` for
   personal-share matters — the recipient's domain follows the same corporate logic.

## Same discipline for the Cc line

- When NDR says "marking Roshni and Eshwari", resolve each: Roshni Ranka =
  `rnr@draas.com`; Eshwari Chamundeshwari = `echamundeshwari@draas.com`. Spell the name as
  the person uses it — voice notes give "Ishwari"/"Ishwari" for Eshwari, and "Sartak" for
  Sarthak (`admin3.blr@draas.com`).
- Never Cc a *vendor* on an internal instruction mail unless NDR asked for it; when he does
  ask (e.g. the counterparty must coordinate directly), put them in Cc rather than To, so
  the action stays visibly with your own person.

## Access hand-off in the same mail

When the email points a colleague at a deliverable, grant the access and prove it in the
same turn — do not just paste a link:

- Sheet/Drive doc → `drive_permissions_add(role="writer")` for a colleague who has to add
  data back (writer, not reader), `role="reader"` for review-only.
- KML / PDF artifacts → check `drive_permissions_list` first; if `anyoneWithLink: reader`
  is already there, the link works as-is and no grant is needed — say so.
- State in the email *which address* the share was made to, so a failed share is obvious
  on the first reply rather than at the deadline.
