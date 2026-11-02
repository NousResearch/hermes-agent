# Millers Road Office Building — Structural Testing & Consultant Briefing (2026-09)

## Property identity (canonical)
- **70/3 Millers Road, Vasanth Nagar, Bangalore** — "India Chai building" / "Indira House" / "Chai Lounge" (ground floor cafe).
- ~20-year-old RCC-framed commercial block, G+4, ~2,600 sq ft per floor plate.
- DRA Realty is the **lessee** from Akber Hussain (AH India) — it is NOT a Ranka project; no project folder existed before 2026-09.
- Nova Spaces / Excel Projects & MEP call it "DRA Asset Upgradation" — refurbishment proposal: strip chajjas/precast fins/canopies → floor-to-ceiling glass curtain wall + aluminium cladding, interiors redone. Frame untouched.

## Structural consultant (the "Tushar Giri" identity)
- NDR calls him "Tushar Giri sir" but his real identity is **tushar patil <structure.pune@gmail.com>**, Warje, Pune (+91 9011076607). Gmail From is always "tushar patil"; no "Giri" anywhere in headers/signature. Do the same Amber/Oasis structure work.
- "Tushar" in contacts sheet (timebroadbandindia.com) is a different person.

## NDT report (the file NDR uploaded 2026-09-19)
- **Nordstar Novaspace LLP.pdf** → renamed to `20260917 Nordstar Novaspace - ACT NDT Report - Millers Road India Chai Building.pdf`.
- Commissioned by Nordstar Novaspaces (Kiran Kumar) → executed by **Academy of Concrete Technology (ACT)**, Mallathahalli — NABL accredited lab. Dates 16–17 Sep 2026. Both certs say "Existing Building, Millers Road, Vasanth Nagar".
- **Rebound hammer**: 40 readings, RN 31.7–36.8 → est. 22–38 N/mm². Consistent, normal for 20-yr-old concrete. IS 516 cl 8.1 caveat: ±25% accuracy → indicative, not definitive.
- **UPV**: 30 readings on columns+beams (3012–3830 m/s), 10 on slabs (3594–4059 m/s). Per IS 516 grading most "Good"; a few columns/beams just under the Good line (borderline); none "Doubtful".
- NOT in the report: visual survey, laser/deflection survey, photos, engineer's signed verdict. Only two lab certs — gap vs the 31-Aug scope letter.

## Drive filing (done 2026-09)
- No Millers Road folder existed → created `DRA Realty > Millers Road Office Building` with standard 7-subfolder template.
- Report filed at `DRA Realty > Millers Road Office Building > Millers Road Office Building Architectural and Engineering`.
- File: `20260917 Nordstar Novaspace - ACT NDT Report - Millers Road India Chai Building.pdf`, Drive file id `1vUlNQBGOTmmDTuZ5XHl-g35dNO8lKFMV`.
- Drive download recipe: `drive.files().get_media(fileId=...)` works fine (387,252 bytes).

## Briefing email to consultant (draft r3553559989664943956, 2026-09-21)
- **To**: tushar patil <structure.pune@gmail.com> — **Cc**: Anbarasan <pm2.blr@draas.com>, Aamir Khan <khan.hussain.aamir@gmail.com>
- Subject: `Millers Road Office — Structural Test Reports & Terrace Structure Review`
- Structure that worked (HTML): context box → plan bullets (strip/remove/facade/no frame change) → why tests done → results table → **highlighted yellow ask box** (terrace structure: 1,300+ sq ft steel frame, ~10 ft ceiling, grand boardroom + pantry + 2 cabins + seating; ask = confidence check + does he recommend more investigation: cores/visual/deflection/load assessment) → context note (dead load reduces, frame untouched).
- Aamir's active address is khan.hussain.aamir@gmail.com (Sep-2026 Cc on NDR's Riverstone/Millers mail); sheet also lists aamirkhan@icloud.com / aamirkhan@me.com — headers win.

## Attachments — photo extraction from proposal PDF (reusable technique)
- Source: `Ranka_Asset Upgrade_Proposal.pdf` (2.4 MB, from zakir@novaspaces.in 10-Aug, attachment id from Gmail messages().attachments().get()).
- Gmail attachment download: fix base64 padding `4 - (len(data)%4)` before urlsafe_b64decode.
- Extract page images with pymupdf:
  ```python
  import pymupdf
  doc = pymupdf.open(path)
  for page in doc:
      for img in page.get_images(full=True):
          pix = pymupdf.Pixmap(doc, img[0])
          if pix.n > 4: pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
          pix.save(f'/tmp/proposal_imgs/p{page.number+1}_img{j+1}.png')
  ```
  Note: `import fitz` is deprecated → use `import pymupdf`. Don't forget `os.makedirs`.
- Page 4 contains BOTH needed images: `p4_img2.png` = existing building photo (Indira House, chajjas visible) and `p4_img3.png` = proposed render (all-glass DRA "Experience Nice" facade, perforated aluminium side screen, stilt parking).
- Verify image content with vision_analyze BEFORE attaching (photo vs render classification worked well).

## Scaffold decision logic (structural analysis — layman explanation)
- Removing chajjas/extensions = DEAD LOAD OFF the beams/columns → never makes frame worse. Frame strength test was overkill for the decision.
- Real risks: (1) cut line where chajja steel anchors into beam/slab edge — careless demolition shears cover + exposes top steel → corrosion; needs method statement (hand-break near frame, no heavy breaker within ~300 mm, immediate re-render). (2) Cladding wind load hung off existing frame needs engineered fixing detail, not site-decided.
- Borderline UPV members → targeted re-check (UPV + visual close-up at those grids) is cheap and closes the file.
- Recommended scope for the building: visual survey + rebound is what's actually needed; UPV/cores/rebar conditional; soil SBC not needed (structure stood 20 years); STAAD/ETABS only if findings warrant.