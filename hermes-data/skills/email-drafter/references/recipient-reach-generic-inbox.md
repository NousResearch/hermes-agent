# Recipient reach — a generic firm inbox is not the man

A draft that "went out" to `info@`, `office@`, `sales@` or `admin@` has not necessarily reached
the person NDR is chasing. This bites later, on the phone: NDR follows up with *"I have sent you a
detailed email yesterday"* while the mail sits unread in a company catch-all.

## Before the draft goes out

- `contact_resolver(<name>)` and read **every** email on the row. Jayaram Gowda (row 1409, *Jayram* /
  *Vardhan Ventures*) carries both `info@vardhanventures.in` and `mejayaramgowda@yahoo.co.in`.
- If the only usable address is generic, say so in the draft report: *"this went to the firm's general
  inbox, not to him personally — his personal id is X, want it re-issued there?"*
- **Do not silently add a second address.** A duplicate outbound mail to a third party is NDR's call.

## When a follow-up claims the mail was sent

- Prove it left: `gws_gmail_search(query='to:<addr> newer_than:5d')` → `labelIds` contains `SENT`;
  `gws_gmail_get` → the exact `Date:` header and the attachment list. A draft in Drafts is not sent.
- Check the date against what NDR says out loud. In the 2026-10-29 Jayaram case the mail left at
  **06:58 IST that morning**, not "yesterday" as dictated — the WhatsApp text was corrected to
  "this morning" and the change disclosed.
- Note whether a reply already exists (`from:<domain> newer_than:3d`); "neither has replied yet" is
  what justifies the chase.

Full WhatsApp-side handling: skill `whatsapp-reply`, `references/verifying-dictated-premises.md`.
