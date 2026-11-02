# Vendor / integration brief emails — one canonical link per role, and the single-clean-draft rule

Verified 2026-10-26 (Bharat → Sinchana A Raj, Kelsa / O3 Infotech, Ranka Amber Unit 203).

## 1. "Keep only one cost sheet, we don't have to do unnecessary things"

When a brief hands over reference documents, each DOCUMENT ROLE gets exactly ONE link:

- Allotment letter → one doc
- Cost sheet **with** payment schedule → one workbook (prefer the one whose tab already
  carries the tentative cost *and* payment schedule — it plays both roles, so do not send a
  second "payment plan" file)
- Agreement for Sale → one draft proforma

Do not ship alternate versions, dated duplicates, exported PDFs *and* the Sheet, or the
"with logos" copy. Any extra link is a question the vendor will ask you instead of working.
If the user names three documents, send three links — not four because you found a nicer one.

## 2. Revising a draft: delete the superseded one, always

The Drafts folder must hold ONE version of a pending email. When the user corrects the brief:

```python
gws_gmail_draft_delete(draft_id=<old>, service_name="google-draas")
gws_gmail_draft_create(...)   # fresh clean draft
```

Two live drafts with slightly different content is how the wrong version gets sent days later.
Present the full draft text in the chat for review, then wait — the sanctioned outcome is a
draft, never a send. Add no cc, no attachment, no extra addressee that was not asked for; offer
them in one line instead ("say the word and I'll cc Nishant sir / attach the file").

## 3. Structure that a non-technical vendor contact can act on

1. **Reference line** — name the principal's earlier email by date AND full subject; say that
   mail carries the scope and questions, and this one carries the inputs to start.
2. **Objective in one paragraph** — the mechanism, in plain business language: all unit data
   lives in one master sheet; at conversion we enter only unit number + face price; rates and
   statutory charges are predetermined; the documents generate themselves; dates compute from
   the agreement date and re-date on slip.
3. **The documents** — numbered a) b) c), one link each, one line on what each is for.
4. **Attachments the user handles himself** — say "attached with this mail" and let him attach;
   never claim it is attached if he has not attached it yet.
5. **Pilot case block** — unit, level, facing, SBA, RERA carpet, UDS, consideration, RERA no.
6. **What we need back** — the numbered questions, restated so they can be answered without
   re-opening the earlier mail.
7. **What is coming separately** — name the remaining scope so nothing reads as missing.

## 4. Ranka Amber Unit 203 — the current pilot pack (Bharat's Drive)

- Allotment letter v4 (RERA model form): `1AyMP_icSjQjDPQDb3LrI1WQJtDdSfzaKuGa7M5kRJIE`
- Cost sheet + payment schedule, ₹11,499/sft all-inclusive, tabs *Cost Sheet (Simple)* /
  *Unit 203 — Rs 11,499* / *Build-up and notes*: `1ZDHEJa-kQzXwsZvM_3oCll0OfyXkm6aRe3uXrfSAFdM`
- Agreement for Sale draft: `1IdKmbUUTie1R-Aux3LkIBq_j9FEz80HyxCOoeOfM72s`
- Execution area statement — Bharat attaches it himself.

Unit 203: Level 2, East, faces driveway · SBA 1,572.30 sft · RERA carpet 1,146.41 sft
(106.50 sq.m) · UDS 691.06 sft · consideration **₹1,80,79,877** (1,572.30 × 11,499, correct).
NDR's 23-Oct mail quotes ₹1,80,79,749 — a typo, ₹128 short; his own back-solved basic rate in
the cost sheet is ₹10,443.80/sft.

## 5. Traps seen in this class of brief

- **Allottee name left blank is intentional** on an allotment letter (returned filled by the
  allottee). Say so explicitly or the vendor files it as a data defect.
- **Floor wording drifts between documents** ("Level 2 (first Floor)" in the cost sheet vs
  "Second Floor (Level 2)" in the letter). Check and align before circulating — the vendor
  maps floor from whichever file you sent.
- The two area bases stay live: 1,572.30 sft marketed vs 1,380 sft SSA-Sch-B. Name which basis
  the Agreement will carry before anyone prices or registers.
- Scope of the Kelsa conversion-to-collection pack is FIVE documents (allotment letter,
  Agreement for Sale, legal set, lender-wise APF pack, payment plan); this brief covers three
  and defers the other two by name.
