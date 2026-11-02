# Millers Road Office — Structural Test Reports & Terrace Structure Review (2026-09-21)

Draft-only email to the structural consultant for the 70/3 Millers Road office refurbishment. Load this when working on any Millers Road / India Chai office building email, NDT report analysis, or Tushar/Anbu/Aamir recipient resolution.

## Property identity (from lease v7 thread)
- 70/3, 70/4, 70/5, 70/6, 70/8 Miller's Road, Vasanth Nagar, Bangalore. Commercial building: Ground (partly tenanted), Floors 1–3 (~2,600 sft each), Terrace (~4000 sft land). ~20-year-old RCC frame. Known as "India Chai" office building (ground tenant Chai Lounge / Indira Coffee House).
- Lessors: M. Akber Hussain 30%, M. Akther Hussain 30%, Mrs M. Asma Banu 20%, Mrs M. Sara Banu 20% (all 829/1 Periyar EVR High Road, Kilpauk, Chennai), GPA holder Avoor Mohammed Muzzammil.
- Lessee: M/s DRA Realty Private Limited (CIN U70200KA2022PTC163215), represented by Nishant Dinesh Ranka.
- Lease thread: "Millers Road Property — Final Lease Agreement & Next Steps" (Aug 2026), v7 lease deed. AH India contacts: Atheeq <padirector@ahindia.com>, Akber <akber@ahindia.com>.

## Recipient canonical mapping (NDR voice → real identity)
- "Tushar Giri sir" → **Tushar Patil <structure.pune@gmail.com>**, Designage Consultants Pvt Ltd, Warje, Pune. Chartered Engineer M-145727-0, +91 90110 76607. He signs the Ranka Amber structural stability certificates; the "Giri" dictation is a persistent garble (also seen June 2026). No actual "Tushar Giri" exists in NDR's contacts/mail.
- "marking Amir" → **Aamir Khan <khan.hussain.aamir@gmail.com>** (also aamirkhan@icloud.com) — broker/conduit on the lease + Nova Spaces façade proposal.
- "Anwar Asan" → **voice error for Anbu Asan = Anbarasan** (pm2.blr@draas.com). NDR later confirmed explicitly: "tushar, anbu and aamir".

## NDT report (drives this email)
- Lab: Academy of Concrete Technology (ACT), Mallathahalli, NABL-accredited. Tests 16–17 Sep 2026, commissioned via Nordstar Novaspaces / Excel Projects & MEP (Zakir Kalami <zaak@novaspaces.in>, Nova Spaces Ulsoor Road).
- Rebound Hammer: 40 readings, 31.7–36.8 → ~22–38 N/mm² indicative (IS 516 cl 8.1: ±25% accuracy — indicative, not definitive).
- UPV: 30 on columns/beams (3012–3830 m/s), 10 on slabs (3594–4059 m/s). Mostly "Good" per IS 516 grading; a few borderline just under "Good" threshold; none "Doubtful".
- Filing: Drive → DRA Realty → Millers Road Office Building (created with 7-folder template) → "Millers Road Office Building Architectural and Engineering" / `20260917 Nordstar Novaspace - ACT NDT Report - Millers Road India Chai Building.pdf`, file id 1vUlNQBGOTmmDTuZ5XHl-g35dNO8lKFMV.

## Email draft (created in parallel session, verified DRAFT)
- Draft id: r3553559989664943956. To: Tushar Patil <structure.pune@gmail.com>. Cc: Anbarasan <pm2.blr@draas.com>, Aamir Khan <khan.hussain.aamir@gmail.com>.
- Subject: "Millers Road Office — Structural Test Reports & Terrace Structure Review"
- HTML body: context box (property/plan) → why tests done → results summary → highlighted ask box (terrace steel structure ~1,300 sft, up to 10 ft ceiling, fully roofed: grand boardroom + pantry + 2 cabins + seating; do current results give confidence for (1) terrace support & (2) interior mods, or recommend additional investigation — targeted cores, visual survey, deflection check, load assessment) → dead-load-reduces note.
- Attachments (3, verified present): ACT NDT Report PDF; `Millers Road Building - Existing - Chai Lounge with Chajjas.png` (real photo, Indira House with curved white chajjas on 2nd/3rd floors — from proposal PDF page 4, img-007); `Millers Road Building - Proposed Glass & Aluminium Facade Render.png` (DRA "Experience Nice" twilight render, all-glass + perforated aluminium — proposal PDF img-010).

## Image extraction trick (proposal PDF → attachments)
- Original proposal: `Ranka_Asset Upgrade_Proposal.pdf` from Zakir Kalami (10 Aug 2026), forwarded by Aamir (19 Aug).
- Extract images with `pdfimages -j /tmp/proposal.pdf /tmp/pdfimg/img`, then inspect the JPEGs (img-007 = existing building photo, img-010 = glass facade render; page 8 interiors are bathrooms/lift renders — not useful for this email).
- Rename meaningfully before attaching (existing vs proposed names above).

## Structural reasoning (for the email + any follow-up)
- Removing chajjas/fins/canopies = dead load OFF the frame → frame strength test was always overkill for this decision; frame is not the risk.
- Real risks: (1) cut line where chajja steel anchors into beams/slab edge — demolition must hand-break near frame (no heavy breaker within ~300mm) and immediately re-render/protect the exposed edge; (2) facade wind load on existing frame — cladding anchors must be engineer-designed, not site-decided.
- Report caveat: rebound ±25% means indicative only; borderline UPV members (slabs 3594/3727/3727) warrant targeted re-test + visual close-up; cores remain conditional.