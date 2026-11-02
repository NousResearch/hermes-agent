# NDR Email HTML Format Preferences (2026-09-15, Ranka Udaya IRR review email)

NDR's standing rules for rich/HTML email drafts addressed to him or sent on his behalf.
Triggered whenever a draft needs "rich text format or HTML" and for any structured review /
model / report email.

## Core preferences

1. **FULL-WIDTH HTML layout.** The draft must occupy the entire width of the available
   screen. Do NOT wrap content in a narrow centered container (e.g. `max-width:600px;
   margin:0 auto`) — NDR explicitly rejected that ("it sits in the center with lots of
   white space to the left and right... that's not how I want the emails to be drafted").
   Use `width:100%` on the outer `<table role="presentation">` and body, no max-width
   squeeze, and let section banners span edge to edge.

2. **Key sections highlighted.** Use colored banner header rows (e.g. dark blue `#1f3a5f`
   for scenario headers), label-value tables with bottom borders, green `#0b6e4f` for the
   key number (package price / IRR), light-gray `#eef2f7` callout strips for notes and
   housekeeping. Goal: "all the key elements, key sections are properly highlighted and it
   is easy to read."

3. **Terminology: "weblink", NEVER "webhook".** When pointing to an online sheet/doc,
   call it the "weblink to the online workbook/sheet" (the workbook carries summary +
   all tabs). NDR corrected "webhook" explicitly — he does not know/use that term.

4. **Plain text vs HTML both fine, but if HTML requested -> rich formatted.** NDR asked
   "use rich text format or HTML to make the email so all the key elements key sections
   are properly highlighted". Default for review/structured emails: rich HTML.

## Working draft pattern for this kind of email (verified 2026-09-15)

- Correction loop: NDR reviews the draft and dictates changes; each round = regenerate the
  draft (delete old draft ID, create new) rather than editing in place. Always verify the
  NEW draft after creation.
- Draft API quirk (critical, see gws-automation `references/gws-skill-bridge-draft-create.md`):
  `draft_create` takes `body=<HTML STRING>` and `html=True` (boolean flag). Passing
  `html=<html string>` silently drops the HTML and stores the plain body as text/html.
- Always pull the draft back and probe payload markers (rupee figures, `width="100%"`,
  recipient emails) — a 4 KB payload when 11+ KB of HTML was sent means the HTML was lost.
- Structure for a "review my model" email:
  1. Hello + weblink to the workbook (summary tab + detail tabs live in the sheet)
  2. Housekeeping strip (rename, folder, ownership, editor/viewer roles)
  3. Scenario 1 block (unchanged baseline) with label-value table
  4. Scenario 2 block with the corrected deal terms (package price, payment schedule,
     guarantee, escalation, occupancy efficiency, deposits, construction cost, velocities)
  5. Comparison block (what output to produce: IRR + profit + payback, win/loss by how much)
  6. Requested action + reply channel (email only) + variations to run later

## Rundown of the Ranka Udaya IRR review email (worked example)

- To: Sarthak Sharma (admin3.blr@draas.com), Cc: Prakash Singh (psingh@draas.com)
- Subject: "Ranka Udaya - IRR Model v2 (Rental Guarantee & Pure Plot Scenarios) - Review Request"
- Sheet: `20260914_Ranka_Udaya_IRR_v2` (id `1iyI8Atwx0Vx5UdhTddRWQidtuxGYXGMsX00Tzn5C2Qs`),
  renamed from `Ranka_Udaya_IRR_v2`, ownership transferred to ndr, moved into the Ranka
  Udaya folder; Sarthak editor, Prakash viewer.
- Package price: **Rs 1.2 crore** (Rs 60L plot at registration + Rs 60L construction:
  Rs 20L at construction start month 0 + Rs 10L at end of each quarter, months 4/8/12/16,
  final at handover → 20 + 40 = 60). G+3 (~2,000 sqft, 8 rental keys), rent guarantee
  Rs 50k/month from registration day, 3-yr and 5-yr models, 5% escalation both ways,
  occupancy Y1 0 / Y2 50 / Y3+ 90 (90 cap), 10-month tenant deposit inflow + return
  outflow, construction cost Rs 2,500/sqft × 2,000 = Rs 50L, NO FF&E, RG velocity 5/3/2
  - RG velocity 5/3/2 switchable, balance ~19 plots at 1/2/3 per month at Rs 4,250–4,500/sqft.

  ## Same structure for multi-recipient strategy/leadership emails (worked 2026-09-15)

  The identical full-width HTML banner + label-value-table pattern carries over to
  NDR's internal strategy emails (e.g. "Ranka Udaya - Lead Gen vs Sales: Strategy
  Reflection & the Rental Guarantee Pivot" to Prakash/Bharat/Gauri). Same rules:

  - Section banners numbered 1..N with the dark blue `#1f3a5f` underlines, key numbers
    green `#0b6e4f`, reference callout (`#eef2f7` strip) right after Hello with the
    workbook weblink.
  - The body mirrors NDR's dictation order: where we are → the pivot → package table →
    positioning/messaging → switch discipline → why-15 rationale → what I need from you.
  - Recipient verification is a HARD gate before drafting: resolve every name via
    find_contact.py / employees tab / Gmail sweep (see contact-lookup-fallbacks.md,
    "Resolving a voice-dictated recipient"). Unresolvable names are flagged, never
    guessed; draft goes to verified addresses only.
  - Verify the draft's HTML part by decoding `parts[].body.data` (drafts().get has no
    `raw` — see draft-create-verify-api.md "no raw envelope" pitfall).