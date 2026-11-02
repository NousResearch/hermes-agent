# DRA–Jiraaf Thread Follow-ups — corrected land numbers, framing, contacts (2026-09-21)

Second round of the Ranka Oasis structuring thread: NDR's pending follow-up to
Vishwas on the DRA–Jiraaf thread and the RERA-villa analysis email to Anbu.
Carries the CORRECTED land-holding numbers (the first draft had them wrong), the
no-percentage framing, and the wrong-Vinay correction.

## Thread anchors (DRA - Jiraaf)
- Thread messages list under `q='subject:(DRA - Jiraaf)'` in google-draas.
- Original: Nishant prakash <nishantprakash@theyelloweye.com> → ndr@draas.com,
  Vineet <vineet@jiraaf.com>, Prakash <prakash.m@jiraaf.com>,
  vishwas@vantagepointadvisors.in (2026-09-01).
- NDR's own last message (2026-09-14) 1a09e429abe72448: TO Vishwas Rao
  <vishwas@vantagepointadvisors.in>; CC nishantprakash@theyelloweye.com,
  ndr@drahomes.in, vineet@jiraaf.com, prakash.m@jiraaf.com,
  kanishk.s@jiraaf.com, psingh@draas.com.
- Latest inbound: Prakash <prakash.m@jiraaf.com> 2026-09-21 06:32, Message-ID
  `<PN0P287MB2595C285FF5EAFD1CD6F4A8B93842@...>` — use THIS as In-Reply-To and
  append to References for the corrected follow-up draft.
- CRITICAL: a draft claimed created on a previous turn may NOT exist — always
  `drafts().list()` first (see SKILL.md pitfall); here only the Anbu RERA draft
  existed, so the Vishwas follow-up was rebuilt FRESH (threaded via headers).

## Land-holding map numbers — source of truth = Sinchana's PDF

The map is 'Oasis Sevaganapalli Land Area Plan.pdf' from Sinchana's email,
filed as `20260917_Ranka_Oasis_Sevaganapalli_Land_Holding_Map_Owner_Wise_DRA_Realty_SPL_JDA.pdf`
(Drive id 1SF2b-Dfvit0w6SVETCkO4d5XwsRzNH_a, under Sevaganapalli Land Partners >
01_Legal_and_Title_Docs). Extract text with pymupdf `page.get_text()` — the
tables come out as label/owner/deed columns; render to PNG at 180+ dpi if OCR
of the vector map is needed.

### Approved layout (Phase 1 — DTCP sanction 13-01-2026), survey-wise:
| Sy No | cents | acres | Owner / holder | Type | Registered instrument |
|---|---|---|---|---|---|
| 158/1C9A | 25 | 0.25 | DRA Realty | Owned | SD 21785/2024 14.11.2024 |
| 158/1C9B | 69 | 0.69 | SPL | Owned | SD 21201/2023 16.10.2023 |
| 166/2B2 | 10 | 0.10 | SPL | Owned | SD 4350/2024 02.03.2024 |
| 166/3A | 138 | 1.38 | SPL | Owned | SD 21201/2023 |
| 166/3B | 4 | 0.04 | SPL | Owned | SD 21201/2023 |
| 166/3C | 119 | 1.19 | SPL | Owned | SD 21201/2023 |
| 166/3D | 54 | 0.54 | SPL | Owned | SD 21201/2023 |
| 166/3E1 | 21 | 0.21 | SPL | Owned | SD 21201/2023 |
| 166/3E2 | 40 | 0.40 | SPL | Owned | SD 21201/2023 |
| 166/3F | 70 | 0.70 | SPL | Owned | SD 21201/2023 |
| 167/2C | 36 | 0.36 | SPL | Owned | Exchange Deed 3906/2025 03.07.2025 |
| 167/2D | 6 | 0.06 | SPL | Owned | SD 21201/2023 |
| 176/1B2D | 16 | 0.16 | DRA Realty | Owned | ED 4292/2024 16.02.2024 |
| 176/2B4A | 22.5 | 0.225 | DRA Realty | Owned | ED 4292/2024 |
| 177/1A1A | 2.5 | 0.025 | DRA Realty | Owned | ED 4292/2024 |
| 177/1A1B | 23.5 | 0.235 | DRA Realty | Owned | ED 4292/2024 |
| 166/1 | 35 | 0.35 | Ramesh Reddy & DRA | JV | JDA 7963/2025 31.10.2025 |
| 167/1G | 7 | 0.07 | Ramesh Reddy & DRA | JV | JDA 7963/2025 |
| 168/1B | 53 | 0.53 | Ramesh Reddy & DRA | JV | JDA 7963/2025 |
| **Total** | **751.5** | **7.52** | | | |

- **SPL own land = 5.67 ac** (567 cents) per row-sum; NDR reads the map's own
  "OWN LAND" summary as **~5.46 ac** — flag the delta (5.46 vs 5.67) to NDR;
  use HIS figure in the draft when he dictates it (user's number wins in the
  email, flag the difference in the reply).
- DRA Realty own land = **0.90 ac** (89.5 cents) — and "substantial amount of
  DRA Realty has gone into relinquishment of road" inside the layout: this is
  why the email focuses on SPL alone.
- JV parcels = **0.95 ac** (95 cents: 166/1, 167/1G, 168/1B).
- Additional held (not yet approved): SPL 1.54 ac (158/1A1B, 158/1C1, 158/1C2,
  167/1I), DRA 0.73 (158/1C3, 158/1C4, 158/1C6), K Harish JV 0.50 (177/1B,
  177/2A1); total 2.77 ac.
- 167/1I = 0.07 ac (part of SD 21201/2023, retained by SPL — never exchanged).

## Corrected follow-up draft (r-2410090405069030372, Re: RE: DRA - Jiraaf)

NDR's dictated framing (embeds the landed corrections):
1. DRA Realty left with only very small parcels — substantial part went into
   **road relinquishment** within the layout (this justifies focusing on SPL).
2. SPL's own land in approved layout ≈ **5.46 ac** (use NDR's figure; map rows
   sum to 5.67 — flag), DRA Realty ≈ 0.90 ac, balance = JV; survey-wise table
   + registered-deed refs live in the attached map.
3. **NO PERCENTAGES** — "we talk about the plot area and built-up area to be
   given to Giraffe, let's not talk about percentage at all". Offer Giraffe his
   total land area + built-up area entirely from within SPL's own holding per
   the agreed commercials.
4. This **avoids the contribution deed** required from DRA Realty → SPL and the
   stamp-duty/registration on it; move straight to private-limited conversion.
5. Asks Vishwas to confirm ("please let me know if this approach works").
6. 6:00 PM today call: Giraffe wants to discuss a **temporary interim loan
   arrangement to SPL** to get the ball rolling; request Vishwas to join even
   half an hour, assess against NDR's red flags whether requirements for BOTH
   Giraffe and NDR are met; if he green-lights, he drives the documentation
   end-to-end himself.
- Built fresh with MIME multipart/mixed (text/plain + attached
  'Ranka Oasis - Sevaganapalli Land Holding Map (Owner-wise).pdf' =
  local /opt/data/oasis_land_area_plan.pdf), In-Reply-To/References to
  Prakash's 2026-09-21 message, drafts().create().
- FINAL draft id **r-3028275399535956241** (after one delete+recreate —
  see consistency lesson below; an earlier intermediate
  r-5845431757749387224 carried the 5.46 lead and was deleted).

## Internal-consistency lesson (2026-09-21, corrected in the FINAL draft)

NEVER write a lead total that contradicts the survey-wise line items below it in
the SAME email — a lawyer counterparty WILL add the rows. The first build said
"SPL own land ≈ 5.46 ac" while listing items that sum to 5.67; caught before
delivery and rebuilt with **5.67 as both the stated total AND the row-sum**,
flagging the 5.46-vs-5.67 delta separately to NDR in the reply. Rule: when the
user dictates a round figure that differs from the schedule sum, either use the
sum and tell him (preferred for anything he attaches), or if you use HIS figure
then state it as "≈ per map summary" and keep line items visibly consistent or
omit the row list.

## Map has TWO table variants — always state which one you used

The Sinchana holding-map PDF carries (a) the REGISTERED schedule (survey-wise
extents, total 751.5 c = 7.52 ac — SPL 567 c, DRA 89.5 c, JV 95 c) and
(b) an "ACCORDING TO DWG" drawing table (rounded extents, total 7.49 ac,
2.258 additional) plus a colour-code legend that prints OWNER NAMES ONLY —
no per-owner acre totals are physically printed next to the legend swatches.
When quoting "per the map", pin the figure to the registered schedule; the
drawing column and the legend are NOT per-owner authoritative totals (this is
why NDR's "5.46" and the row-sum "5.67" can both trace to the same file).

## Wrong-Vinay correction (HARD contact lesson)

- The RERA-villa analysis email to Anbu originally carried **Cc: Vinay T
  <vinay@vnv.ca>** (VNV Consultants) — **WRONG.** NDR: "of vnv consultant
  that's wrong this is [Vinay] liaisoning Tamil Nadu... so it is not Vinay T,
  please keep that rectification in mind". NEVER assume a first name maps to a
  specific consultant: for TN RERA liaisoning work the Vinay is the **TN
  liaisoning** contact, not Vinay T of VNV. When NDR says he'll send the
  address, don't guess — remove the wrong Cc via raw-edit (see SKILL.md) and
  leave the slot open.
- Mark: no verifiable "Mark" in contacts/Gmail — per NDR's standing rule never
  put unverifiable recipients on an email. Keep his name only in the dictated
  body ("coordinate with Vinay and Mark"), never in To/Cc until an address is
  provided.

## RERA-villa comparable analysis background (same session)

- Oasis registration = **TNRERA/30/LO/3130/2026** layout of 130 plots (07.08.2026,
  promoter DRA Realty + Ramesh Reddy + Sevaganapalli LP, Hosur/Krishnagiri).
- Comparables that DID market villas off a plot/layout approval: Sobha Gardenia
  (TN/01/Building/0043/2018, 79 villas; + layout 0018/2017, 0132/2020, building
  annexes 0256/2020, 0403/2023 — the "Share of Plots and Villas among Promoters"
  annexure is the mechanism), RHYTHM BY DAC (TNRERA/35/BLG/0237/2026), THE
  LORD'S BY DAC (TNRERA/35/BLG/0146/2026), SREE DAKSHA'S LEIA
  (TNRERA/11/BLG/0021/2026), Palm Beach Phase II, Aurum Villa Phase 3, House of
  Palisade by DAC (Building 0050/2024 + Layout 4130/2023). Pattern: layout +
  **phase-wise Building registration** ("Group Development building of N Villas")
  → recommend 3 phases of ~20 villas each from a base villa sanction.
- Gardenia reference folder: https://drive.google.com/drive/folders/11veB0fb4iQ6aIUj2Dw-5S5jhRGSl5YFm