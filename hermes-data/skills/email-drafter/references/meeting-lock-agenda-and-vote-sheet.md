# Locking a multi-party meeting → vote-tracked agenda sheet

Verified 2026-10-24, the family governance thread "Call for a meeting"
(Nishant / Dharmesh / Manish — DRA Projects wind-up & liquidation).
Thread id `1a1178935be29c75`; meeting locked Mon 26-Oct-2026 16:00–18:00 IST.

## Trigger shape

NDR proposes slots in a thread → a participant replies "Monday works, add to the agenda the
following: - X - Y" → NDR: *"reply back now… guys just confirming this is our Monday, this is the
timing, and this is the agenda. Update the agenda items, share it, and create a calendar event
adding \<names\> for that Monday… blocking a 2 hour slot at \<entity\> at \<office\>."*

Deliverables in one turn, in this order:

## 1. Lock the slot explicitly, in the first line

> **Locked: Monday, 26 October, 4 to 6 pm, at DRA Projects Private Limited, Ranka Chambers.** Two
> hours, in office. I have sent the calendar invitation to both of you.

Take the slot from **NDR's own offer list**, cross-checked against the participant's confirmation —
not from the participant's message alone ("Monday works" does not itself name 4–6 pm).

## 2. Consolidated agenda — merge every participant's additions, marked

Restate NDR's earlier agenda **in full**, then fold each participant's items into the section where
they belong and tag them `(Manish)` so it is visible what came from whom:

- "Golfshire / Hermitage sale" → the **Commercial & assets** section.
- "Hermitage DOD/Parking/EGM" → **not** a new row; appended to the existing
  "Hermitage — cases, vakalats, EGM" line.

Sections that work for a governance/wind-up meeting: **A. Decisions** · **B. Matter-wise updates** ·
**C. Commercial and assets** · **D. Litigation and tax**.

Close with the method statement and a deadline for changes:

> Every item closes on Monday as one of three things: decided, assigned with a date, or consciously
> dropped. Nothing gets carried forward without an owner and a date.
> If either of you want to add or drop anything, reply before Monday morning so the list is final
> before we sit down.

## 3. Calendar event — 2 hours, office address, agenda in the description

```
start/end : <date>T16:00:00+05:30 … T18:00:00+05:30, timeZone Asia/Kolkata
summary   : <Entity> — Wind-up & Liquidation Review (<initials>)
location  : DRA Projects Private Limited, Ranka Chambers, 4A/4B, 4th Floor,
            No. 31 Cunningham Road, Bengaluru 560052
attendees : mdr@draas.com, ddr@draas.com, dharmesh.ranka1@gmail.com
reminders : popup 60 min, email 1440 min
sendUpdates: "all"
```

- Put the **full numbered agenda in `description`** — that is what people read on their phone.
- Add **both** of a person's addresses when they mail from two (Dharmesh: `ddr@draas.com` corporate
  + `dharmesh.ranka1@gmail.com` personal, which is what he actually replies from).
- **Check for a clash before creating.** This class routinely collides with a family/social event —
  here the JITO Diwali evening sat on the Thursday alternative slot, which is part of why Monday won.
- No native calendar tool exists: `build_service("calendar","v3", service_name="google-draas")`,
  `events().insert(calendarId="primary", body=..., sendUpdates="all")`.

## 4. The vote-tracked agenda spreadsheet (NDR spec, 2026-10-24)

> "Each agenda item we say what exactly is the discussion point description and we say NDR vote,
> DDR vote, MDR vote. We create three columns… Where we don't know, leave it blank."

Columns, in this order:

`# | Section | Agenda Item | Discussion Point / Description | NDR Vote | DDR Vote | MDR Vote | Decision / Action | Owner | Date`

- **Vote columns stay BLANK.** They are filled in the room; pre-filling them defeats the purpose.
- **Descriptions: write them where the scope is known; do not invent matters.** For a decision item
  ("Parking — decide whether to allot, sell or hold, and to whom") the description is easy. For a
  matter-wise update on a file you have no current facts on (Mantri, Kothanur, the two Goa matters,
  Nelamangala) write only *"Present position of the X matter and the next step"* and **tell NDR
  those rows are deliberately thin and offer to fill them** — never manufacture status text.
- Formatting: frozen header row, header fill `#2a6496` with white bold text, wrap + top-align body,
  column widths ~36 / 150 / 260 / 420 / 90×3 / 300.
- `spreadsheets().create` with `properties.locale="en_GB"`,
  `timeZone="Asia/Kolkata"` (NDR's DD/MM/YYYY convention), then `values().update` in one call, then
  `batchUpdate` for the formatting.

## 5. Sequencing NDR actually wants

He does **not** want the sheet attached to this mail. His words: *"Just prepare an agenda
spreadsheet for me, send this email out, prepare the spreadsheet, subsequently I'll send an email
saying guys just for tracking the agenda items."* So:

1. This mail goes out with the agenda **inline** and no attachment.
2. The sheet is created in his Drive root, **unshared**.
3. A **later** short mail introduces it for tracking — draft that only when he asks.

Do not pre-empt steps 2–3 into step 1 because it seems tidier.

## Pitfalls

- Keep the calendar event and the mail in the **same script**, calendar first — the mail body
  promises "I have sent the calendar invitation", so it must already exist when the draft is made.
- Verify the reply is threaded on the **participant's latest message**, not on NDR's own last sent
  mail, or the draft nests as a sibling mid-thread.
- A person's address in the Cc list and their address in the calendar invite are not always the
  same one; resolve from the thread headers, and add both when they use two.
