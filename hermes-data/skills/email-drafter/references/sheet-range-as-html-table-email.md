# Putting a model/sheet table into an email as styled HTML

## Trigger
"Take the table from column X to column Y, rows A to B, put it in the email, present it in HTML
so it's easily readable, and caption it <name>."

Typical use: a scenario / sensitivity table from a real-estate financial model sent out as a
**final offer** to a counterparty or partner.

## Rule 1 — the numbers come from the sheet, never from the dictation
Dictated figures arrive mangled: "23, 17 and 12 crore profitability" was really
₹23.17 Cr / ₹17.92 Cr / ₹12.67 Cr; "4750 costing" was the ₹4,750/sqft construction rate.
Always read the range live:

```python
svc = build_service("sheets", "v4", service_name="google-draas")
r = svc.spreadsheets().values().get(
        spreadsheetId=SID, range="'Sensitivity Analysis'!A1:P30",
        valueRenderOption="FORMATTED_VALUE").execute()["values"]
```
Read a margin wider than the quoted block so you get the header row, the caption row above it and
the footnote assumptions — those become the table's header and its footnote.

## Rule 2 — table shape
- 8–9 columns maximum. Merge compound dimensions into one label column
  ("GW ₹3 Cr | Base" rather than two columns of repeated text).
- `border-collapse:collapse; max-width:760px; width:100%` — 760px is what survives Gmail,
  Outlook and phone previews.
- `<thead>` with a 2px accent rule and small uppercase letter-spaced labels; `<tbody>` with 1px
  hairlines; numeric cells right-aligned; label cells left-aligned and bold.
- Colour only the decision-relevant cell (net profit): green ≥ ₹8 Cr, amber ₹0–8 Cr, red negative.
  Shade alternate scenario groups (e.g. every "Upside" row) at ~#faf8f1.
- Footnote under the table: the model's own drivers — BUA, developer share, lot sizes and prices,
  and whether profit is stated before or after goodwill/stamp duty/marketing.
- Use `&#8722;` for a true minus sign in negatives, `&#8377;` for ₹.

## Rule 3 — caption
An `<h2>` in a serif face for the deal name ("Salman Airport Road Property — Final Offer") plus a
small uppercase letter-spaced kicker line under it naming the table
("Consolidated twelve-scenario comparison & ranking"). **Repeat the caption verbatim as the
Subject line** — NDR names the mail from the caption, so guessing a subject gets it rejected.

## Rule 4 — always keep a plain-text part
`set_content(...)` with the argument in prose, then `add_alternative(html, subtype="html")`.
A table-only HTML mail is unreadable in preview panes and in the terminal.

## Rule 5 — offer structure when the mail is a negotiating position
1. One paragraph saying what was modelled (the dimensions: cost, pricing, goodwill options).
2. **"What the two scenarios actually assume" — BEFORE the table, not folded into the intro.**
   NDR added this by correction: *"we need to add where we explain right in the beginning what is
   a base scenario and what is an upside scenario… and then we have to explain to him what is he
   reading in the table below."* Define each scenario purely in **funding terms**, per lot: who
   buys, at what price, and — the part that carries the commercial argument — whether the money is
   **fully upfront and funding the entire cash-flow requirement** until that payment is exhausted,
   or **post-RERA and milestone-linked**, arriving as milestones fall due. Say explicitly what is
   **not** construction-linked. Then close the section with the one-line summary
   ("base is 15/18/22, upside is 18/22/25 — the only difference is X and Y").
   Write this section first, unprompted, even if the dictation buries it mid-note.
3. **"What you are reading in the table below"** — a short column-by-column legend in plain words:
   what net revenue is net *of* (GST, marketing), what total cost carries (goodwill, stamp duty,
   approvals, construction), that profit is what is left to the developer, what peak equity means,
   and that IRR is annualised on that equity. Never hand an external counterparty a table
   unexplained — the recipient has not seen the model.
4. The table.
5. "What the numbers say" — walk the reader down the cost ladder for each goodwill option and
   name the exact point where the deal stops working (state the loss figure plainly).
6. The two pricing structures in words.
7. **"Our offer"** — Recommended option FIRST, labelled "Recommended", then "Baseline". Each one
   has to be executable by the recipient without another call: who pays what, when, against what
   commitment, and what the contingent leg is triggered by. State conditional money as a
   **trigger**, not a hope: "the balance ₹2 Cr is paid only when he delivers the upside — Lot 1 at
   ₹18,000 fully upfront and Lot 2 at ₹22,000 post-RERA."
8. Close with what is acceptable, what is not, and the single next action — and **keep NDR's
   walk-away sentence intact**: "This is the only offer we can make him. If it works, take it to
   Salman and come back. If it does not, we drop it — the margins are too tight to stretch any
   further." Do not soften or paraphrase it into a hedge.

### Reconcile the arithmetic before sending
Lots × area × price must tie to the sheet's gross revenue (three 15,000 sqft lots at
15k/18k/22k = ₹82.5 Cr), and profit must tie to net revenue minus total cost. If the dictation
disagrees with the model, **the model wins** — but say so explicitly in the reply rather than
silently correcting NDR.

### When the structure changes, rebuild — don't patch
Adding a section above the table means a new draft: create it, verify the HTML renders and the
row/column counts are intact, **then delete the superseded draft**, so exactly one live draft
exists and NDR cannot send the wrong version.

Do not soften a loss figure. "At ₹5,250 construction we lose ₹1.19 Cr" is the sentence that makes
the position land; hedged phrasings get negotiated away.

## Recipient address selection
Pick the address NDR actually uses for that *class* of mail — read it off Gmail headers, do not
use the contacts sheet. Example: Aamir Khan — DRA project/brief mail goes to
`khan.hussain.aamir@gmail.com`; investor / Terra Green / LLP mail goes to `aamirkhan@icloud.com`.
Flag the choice in the reply so it can be repointed in one word.
