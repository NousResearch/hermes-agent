# Updating an already-created Gmail draft: delete + recreate

Verified 2026-10-12 (Prakash Singh event email, corrected twice by voice dictation).

The native Gmail tools have **no draft-update endpoint** — `gws_gmail_draft_create` only
creates. To change the body/subject of a draft that already exists:

1. `gws_gmail_draft_delete(draft_id=...)` — keep every draft_id returned by the create call;
   you need it to remove the superseded draft.
2. `gws_gmail_draft_create(...)` with the full corrected content — a NEW draft_id comes back.

Worked example: event email to psingh@draas.com was created (r611903204354015894), then
corrected ("Dr. Sandara Road, Dommasandra") → delete old, create new
(r-2280287219243335666), then corrected again ("Rental Guarantee Model" not "Rental
Yield (RJ Model)") → delete, recreate (r6096243470355297220). Never stack duplicates —
each stale draft left in Drafts becomes a wrong-version send risk for the human.

## Voice-dictation correction loop
Bharat/NDR dictate event/email content and correct proper nouns in follow-up messages
("It's Rental Gurentee model", "Dr. Sandara Road- Dommasandra"). Treat every correction as
a full recreate, not a patch. Also re-confirm the ambiguity in your reply
("Item 4 now reads Rental Guarantee Model — confirm") and re-run gdoc/draft verification
after each version.