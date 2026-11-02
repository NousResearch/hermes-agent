# Ranka Oasis — Structuring & Legal Email Matrix (Sep 2026)

NDR's Ranka Oasis (Sevaganapalli, Hosur TN) launch work spawns a recurring
batch of advisor emails. When he dictates "the Ranka Oasis structuring email"
or similar, this is the recipient matrix and the structure snapshot to embed.

## Recipient matrix (verified Sep 2026)

| Role | Recipient | Address | Notes |
|------|-----------|---------|-------|
| Structuring/tax advisor | Vishwas Rao (Vantage Point Advisors) | vishwas@vantagepointadvisors.in | sheet row 4121, +91 77608 02361 |
| Tax/CA team | Saurabh T Jain (TSJ & Associates LLP, CA) | stjain@gmail.com (work; from RERA fund certificate 2026-09-15); firm info@tsjca.com | ICAI Membership 221686, UDIN 21221686AAAAMA1354, #40 4th Floor Lakshmi Complex KR Road BLR-560002, Ph 080-26705995. People API draas people/c1406711652723992519 + gmail people/c4649091425659277737; sheet row 3248. From CA certificate for Svamitva Emerald Square (Shankeshwar Landmark LLP) |
| Tax/CA team | Nitin Osthwal Kalpesh ("Nitin Osthwal", official DRA CA) | Kalpeshnitin@gmail.com | Full name Nitin Osthwal Kalpesh (given by NDR 2026-09-15). Phone +91 80889 79777. People API draas people/c9155704659781544135; sheet row 2333. Saurabh's bio: "Partner / Cousin of Nitin Osthwal — our official CA for all company & personal matters" |
| Legal (IndusLaw) | Vivek G | vivek.g@induslaw.com | sheet row 1048 "G Vivekanand / Induslaw", +91 93810 71075 |
| DRA internal (tax emails Cc) | Sarthak Sharma | admin3.blr@draas.com | People API, org "DRA Realty Private Limited", +91 98937 91110 |
| DRA internal (both email sets) | Prakash Singh | psingh@draas.com | row 2500, +91 99000 93816 |
| DRA internal (legal email Cc) | Bharat Hawaldar | sales1.blr@draas.com (CONFIRMED 2026-09-15 — all employees use draas.com, never drahomes.in; sheet's drahomes row is stale) | see contact-phone-lookup skill |

## Email set shape (as dictated 2026-09-15)

1. **Tax email — TWO completely disconnected copies** (fresh threads, do NOT
   cross-reference/reply into each other):
   - Copy 1: **To Vishwas Rao**; Cc Sarthak Sharma + Prakash Singh.
   - Copy 2: **To Saurabh T Jain (stjain@gmail.com) + Nitin Osthwal (Kalpeshnitin@gmail.com)** (draft r8527481941715893055 created 2026-09-15); Cc Vishwas Rao + Sarthak Sharma + Prakash Singh. |
   - Content: entity structure, approvals status, end objective (mandatory
     plot+villa sale, milestone-funded construction), the two routes, 4
     numbered questions (Route B feasibility / GST direct billing vs 5%
     composition without ITC / income-tax timing on plot sale / deferral
     structuring + stamp duty), ask box requesting written opinion.
2. **Legal email — To Vivek G (IndusLaw)**, Cc Prakash Singh + Bharat
   Havaldar; state that Prakash will drive it to conclusion.
   - Content: prior title-opinion work reference, project status (HNDTA pattern
     approval, DTCP plan sanction, RERA plot registration, construction
     started), piecewise land additions under one "Ranka Oasis" brand, the two
     routes, agreement-set review keeping TN RERA in mind, TN RERA plot→villa
     marketing question (addendum/update), clubhouse-membership society model
     for common-area maintenance (30–35 yr memberships + annual membership &
     usage charges; enforce monthly maintenance via club access), urgency
     (early investors need sale deeds/agreements).

## Structure snapshot (embed in both emails)

- **Sevaganapalli Land Partners** (partnership firm): DRA Realty Private
  Limited 90% / Nishant Ranka 10% — holds core land.
- **DRA Realty Private Limited** — additional registered land.
- Two JDAs: Purshottam Reddy; Harish Reddy (JDA land added piecewise).
- Approvals: HNDTA pattern approval → DTCP plan sanction (sub-division/layout
  to sell plots) → TN RERA registration (as **plot** project). More pieces
  (1/2/4 ac) get their own pattern + RERA numbers, all marketed as **Ranka
  Oasis**.

## Two routes being weighed (both emails)

- **Route A:** Agreement of Sale now; construction plan approved & built in
  company's name; customer funds construction progressively; **composite sale
  deed** (land+villa) at the end → heavier stamp duty, GST fully paid by us to
  contractor.
- **Route B:** **Sale deed for plot today** + parallel construction agreement
  (with us or a contracting sister concern); 10–20% advance + slab/milestone
  payments; GST paid directly by customer to contractor (ITC benefit passed)
  → lower stamp duty base, cleaner GST flow.

## Format

Both sets are NDR's dictated HTML/CSS briefing emails: color-coded section
boxes, entity/structure graphic via styled divs + tables, numbered question
list, highlighted yellow "Request/Urgency" box. Build HTML with inline CSS
(no external assets), create with `draft_create(..., html=True)` after
verifying mailbox = ndr@draas.com.