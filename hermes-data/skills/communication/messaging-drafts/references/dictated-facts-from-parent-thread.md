# Recovering dictated facts from the parent mail thread (before drafting)

Verified 2026-10-13 (Bharat's "creatives for the upcoming events towards
Sajjapur" email request).

## The rule

When a **voice request names facts you cannot trust verbatim** — a venue, event
dates, a deadline, a project list, colleague names — do **not** guess the
garbled proper noun, and do **not** ship a bracketed placeholder if the fact is
recoverable. **Search the requester's own mailbox for the parent thread first
and lift the facts from it.**

This does two things at once: it de-garbles the proper noun, and it makes the
new draft *thread-consistent* with what the requester already told the team.

## Worked example

Request (voice): *"draft an email to Nishansar, Roshni ma'am, Prakash and Gauri
ma'am … these are the creatives which we have worked for the upcoming events
towards **Sajjapur** … by tomorrow is a deadline … printing at least by 12 pm …
on 15th in the afternoon everything is ready and the stall can be set up."*

Unresolved on its own: the venue, the event window, and what "the creatives"
actually were.

Two Gmail searches settled all of it:

```
gws_gmail_search(query="Sarjapur")                    # -> the Sarjapur-belt thread
gws_gmail_search(query="standee OR stall OR leaflet") # -> the stall thread
```

Both surfaced Bharat's own **12-Oct email, "Plan of Action – Property Stall at
Cultural Events (15–21 Oct) – Designs for Printing"**, which carried:

- the venue — *"cultural activity venue opposite TISB, Dommasandra Road"*
- the event window — **15–21 October**
- the four offerings — Ranka Oasis, Ranka Amber, Ranka Udaya, Rental Guarantee
- the plan — design by 13th, printing 14th, stall set up by 15th

The draft was then written against those facts, needed no placeholder, and read
consistently inside the same context the recipients already had. (Note
"**Sajjapur**" ≈ **Sarjapur** — and the actual venue sat on the
Sarjapur–Dommasandra belt, which is why the search term worked.)

## Resolving DRA STAFF addresses — mail threads, not the contacts sheet

`contact_resolver` holds **external** contacts; DRA staff are not usefully in it
(it returned no match for Sinchana or Gauri). Get staff addresses from the
mailbox instead — `gws_gmail_search(query="<first name>")` returns the exact
address **and** the spelling:

| Person | Address |
|---|---|
| Nishant Ranka (NDR) | `ndr@draas.com` |
| Roshini Ranka — "Roshni ma'am" | `rnr@draas.com` |
| Gowri Singh — "Gauri ma'am" | `gsingh@draas.com` |
| Prakash Singh | `psingh@draas.com` |
| Sinchana Gowda (design coordinator) | `sgowda@draas.com` |
| Sarthak Sharma (admin/finance) | `admin3.blr@draas.com` |
| Eshwari Chamundeshwari (accounts) | `echamundeshwari@draas.com` |

If `contact_resolver` errors with *"has no gws_service configured in their
profile"*, pass the account explicitly — `account_email="sales1.blr@draas.com"`
for Bharat's session. The tool then works normally.

## Voice-name decoding for DRA staff

| Dictated | Actual |
|---|---|
| "Nishan sir" / "Nishansar" | Nishant Ranka |
| "Gauri mam" / "Gaurimiam" | Gowri Singh |
| "Roshni ma'am" | Roshini Ranka |
| "Shinjana" | Sinchana (Gowda) |
| "Sajjapur" | Sarjapur |
| "Prestige Homitage" | Prestige Hermitage |
| "Sarchapur Bagalur Road" | Sarjapur–Bagalur Road |

## Defaults for these drafts

- **To:** NDR; **Cc:** the other named colleagues — NDR is the approver and the
  body addresses the others directly. State the split in the reply so the
  requester can flip it.
- **Draft only.** Create a Gmail draft, never send (SOUL hard rule). Report it
  as sitting in Drafts for the requester to send.
- Include the Drive folder link for the referenced assets, and flag it if you
  could not verify the folder's sharing settings.
