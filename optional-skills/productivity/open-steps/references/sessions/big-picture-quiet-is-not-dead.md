# Example 1 - the map that recommended deleting a working product

Synthetic. A small parcel-tracking product, two years old, one founder and two
contractors. No confidential or third-party data.

The user asked: *"What have we actually built here? Half of it feels dead."*

## What a first pass produced

It listed the directories it could think of, sorted them by date, and called
everything old a candidate for deletion.

> **Stale, consider removing:** `notifier/` (7 months), `geocode/` (9 months),
> `pdf/` (11 months), `retry/` (7 months), `csvimport/` (10 months),
> `webhooks/` (8 months).
>
> Six of eleven modules are unmaintained. Recommend a cleanup sprint.

Confident, tidy, and wrong in a way that costs a weekend. Every one of those
six is imported by the running product. `geocode/` had not changed in nine
months because it was finished - it turns an address into coordinates, and
addresses have not changed. "Unmaintained" was never measured; it was inferred
from a date.

## What the skill produces

Two signals instead of one, and the parts enumerated from `git ls-files`
rather than from memory.

> _Measured 14 Aug. Stages are dated where they stand; anything undated in this
> file is not measured._
>
> ## What this is
>
> A parcel-tracking service for small shops. Sellers paste a tracking number,
> customers get a page that updates itself and an email when the parcel moves.
> It runs as one web service with a background worker.
>
> ## What is in it
>
> | Feature | What it does | Stage | Last worked on | Signal |
> |---|---|---|---|---|
> | Tracking page | The page customers watch | live (12 Aug) | 12 Aug | active |
> | Carrier polling | Asks carriers where a parcel is | live (9 Aug) | 9 Aug | active |
> | Emails | Tells the customer it moved | live (6 Jan) | 6 Jan | stable |
> | Address lookup | Turns an address into a map point | live (3 Nov) | 3 Nov | stable |
> | Receipts | Makes the PDF receipt | live (21 Sep) | 21 Sep | stable |
> | Spreadsheet import | Bulk-adds parcels from a file | not checked | 4 Dec | stable |
> | Partner callbacks | Tells a shop's own system | live (2 Feb) | 2 Feb | stable |
> | **Old SMS sender** | Sent texts before we moved to email | not checked | 18 Mar | **unused - 11 months** |
>
> ## Worth retiring
>
> - **Old SMS sender.** Nothing has called it since the switch to email in
>   March, and no build or deploy script mentions it. It still holds the
>   Twilio credentials. Removing it is about an hour.
>
> ## What is next
>
> - Finish the spreadsheet import - never confirmed working (report, 4 Dec)
> - Decide on the old SMS sender - retire candidate above
>
> _Age and wiring measured 14 Aug, fresh this pass. Stage comes from session
> reports and is only as current as the date beside it._

## What changed, and why each change is in the rules

| Change | Rule |
|---|---|
| Six "stale" became `stable` | Each is imported by the live product. Quiet plus wired in is finished, not dead - Step 3 |
| The real dead one appeared | `sms/` was never in the hand-written list. `git ls-files` found it - rule 2 |
| "Recommend a cleanup sprint" became one named item | The skill recommends, it does not act - rule 6 |
| Two rows say `not checked` | Nobody could prove those shipped. A guess would have read the same as a fact - rule 1 |
| Both next items name a source | Neither was the agent's idea - rule 4 |

The lesson that cost the most: **the retire list got shorter and became
useful.** Six confident candidates were noise a person has to disprove one by
one. One candidate, with the reason it is dead and the credential it still
holds, is a decision they can make in a minute.
