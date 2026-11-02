# Charitra Murjani Medical Communication Context

## ⚠️ Two Separate Account Contexts

Charitra's care involves **two independent communication channels** on different accounts:

| Context | Account | Vault | What it's for |
|---|---|---|---|
| **Medical/Doctor comms** | ndr@draas.com | google-draas | Treatment discussion, PET scans, drug queries with Dr. Sameer/Dr. Annie |
| **KIRAN PAP logistics** | ndr@ahfl.in | google-ahfl | Infusion confirmations, prescriptions, Medybiz pharmacy coordination |

Treat them as separate — do NOT cross accounts. A drug-alternative query belongs on draas.com; a cycle-12 infusion confirmation belongs on ahfl.in.

---

## Context A: Medical/Doctor Communications (ndr@draas.com)

### Account
- **Account**: ndr@draas.com (google-draas vault)
- DO NOT use ahfl for doctor queries

### Doctors
| Person | Email |
|---|---|
| Dr. Sameer Rastogi | samdoc_mamc@yahoo.com (NOT dr.sameer.rastogi@gmail.com) |
| Dr. Annie (Annie Baa) | anniekbaa@gmail.com (NOT anniejames555@gmail.com) |

### Patient / Family
| Person | Email | Notes |
|---|---|---|
| Charitra Murjani (Chinky) | **charitrakamath@gmail.com** ← ACTIVE | `murjanicharitra@gmail.com` is OBSOLETE — NDR confirmed wrong Aug 2026. Thread consistently uses charitrakamath@gmail.com. |
| Roshni Ranka | **rnr@draas.com** | `rmurjani@gmail.com` is older/obsolete |

### Active Thread
- **Subject**: "Re: Post-Chemo PET-CT Scan (Ms. Charitra Murjani) and Guidance on Maintenance"
- **Thread ID**: 19b21cfeb1c6b4a7 (on google-draas)
- **Started**: Dec 2025, ongoing Aug 2026

### Current Treatment (Aug 2026)
- Pembrolizumab (Keytruda) immunotherapy + Axitinib 10 mg daily
- Known Axitinib brands: Axinix 5mg (Cipla), **Glenmark AXZYB 5 mg** (sourced Aug 2026 when Axinix was out of stock)
- PET shows stable disease; mass near airway/food pipe

### Her genomic / tumour profile — pull this FIRST for any therapy question
From Murjani Medical Drive folder (folder id `1erVpDFXh9tdJuhsN5N36Zt69yjX4QG-V`).
Download with `gws_auth.build_service("drive","v3",service_name="google-draas")` +
`MediaIoBaseDownload`, then extract with **`pdftotext -layout`** (`pymupdf`/`fitz` is NOT installed).

| File | Carries |
|---|---|
| `20251223 Charitra R NGS Panel Medgenome.pdf` | **TMB 2.17 mut/Mb (LOW)**; PIK3CA p.Met1043Ile **VAF 5.2%** (subclonal); **NR1D1:NCOA2 fusion, 632× read depth**; HRD negative, GAS 30 low; MSI field not clearly legible — confirm with Dr Annie (almost certainly MSS) |
| `20251206 Charitra R PD-L1 NGS Medgenome.pdf` | PD-L1 (22C3) **60% tumour cells**, 10% immune cells |
| `20251215` / `20251217` Charitra R Pathology + IHC Amended TMH | Dr Bharat Rekhi: undifferentiated sarcoma with epithelioid cytomorphology, **FNCLCC grade 3**; "resembles ASPS in focal areas"; recommends high-throughput NGS incl. RNA fusions. Original 2020 right great-toe tumour: TFE3 patchy positive |
| `20260102 Charitra R NGS RNA Fusion Panel TMH.pdf` | **Scanned, no text layer** — fusion-expression confirmation must be read off by Dr Annie |
| `20260921 Charitra PET CT Scan Report St Johns.pdf` | SUVmax **11.9**; mass 6.4 × 6.8 × 9.6 cm; increased internal necrosis |
| `20260317 charitra Murjani PET scan report.pdf`, `20260629 …` | earlier scans for the SUV trajectory |

**SUVmax trajectory to always quote: 21.6 (Dec) → 18 (Mar) → 14.8 (Jun) → 11.9 (Sep).** The current
regimen is working — never propose interrupting it without saying so.

**HLA typing has never been done.** It gates every neoantigen / CAR-T / TCR option and costs ~₹5–10k.
Flag it as the cheapest high-value missing test.

### Therapy-option questions → use the `clinical-dossier` skill
When NDR asks "should we consider X for Chinky" (vaccine, ADC, CAR-T, Chinese trial, off-label
combo, named-patient import), do **not** answer generically — load
`research/clinical-dossier`. It carries the method (ground in her NGS/PET first; the
adjuvant-vs-bulky-disease test; clonal-vs-subclonal targetability; India CDSCO pathway; honest
verdict shape) and the knowledge bank
`references/personalised-cancer-vaccines.md`. Deliver as HTML/CSS to the Drive TMP folder, then
draft the doctor email from this file's patterns.

### Drug-Out-of-Stock Email Pattern
When Axitinib (or similar) is unavailable at Nishant's 3 pharmacies:
1. Find thread on ndr@draas.com (google-draas vault, NOT ahfl)
2. Confirm drug name + dose from thread body
3. Verify recipient emails from thread headers (never guess)
4. Reply to latest message with proper threading (In-Reply-To + References)
5. Body: state drug + dose + brand, stock running out, ask for alternative generic
6. CC charitra (**charitrakamath@gmail.com** — NOT murjanicharitra@gmail.com) + Roshni (rnr@draas.com)
7. Create as DRAFT only; verify in Drafts folder after creation

### PET-Cycle Follow-Up Email Pattern (proven 2026-09)
When a new Charitra PET-CT has been done and NDR wants to update Sameer/Annie in
the same chain ("same chain of mails I have been sending him"), the anchor thread
may have BRANCHED. There are TWO similar threads on google-draas:
- **Old/historical**: `19b21cfeb1c6b4a7` "Re: Post-Chemo PET-CT Scan (Ms. Charitra Murjani) and Guidance on Maintenance" — last msg Aug-2026 from Roshni.
- **New branch (12-Sep-2026)**: `1a09498067899ab8` — single NDR sent message (msg id 1a094a096c8458bb, Message-ID `<CAPDkZRYjuMReSOVTZ=x38x2BODn5OptaXG_0bNRE8U4U9RzukA@mail.gmail.com>`) asking Sameer about the necrosis-vs-active contrast question ahead of the 21-Sep scan. To: SAMEER RASTOGI samdoc_mamc@yahoo.com; Cc: annie baa anniekbaa@gmail.com, Charitra Kamath charitrakamath@gmail.com, Roshni Ranka rnr@draas.com.

**Pitfall — "continue the chain" means the LATEST SENT message branch, not the recorded thread ID.** When NDR says "same chain / same continuance", search SENT mail first (`q='from:me in:sent subject:"Post-Chemo PET-CT" newer_than:2m'`), take the most recent sent message's thread + Message-ID as the anchor, and check whether an EARLIER botched draft went only to Annie (known failure mode: 13:16 12-Sep draft went To annie only; the corrected 13:29 one went To Sameer). Recipients for PET follow-ups are always: To SAMEER RASTOGI samdoc_mamc@yahoo.com; Cc annie baa anniekbaa@gmail.com, Charitra Kamath charitrakamath@gmail.com, Roshni Ranka rnr@draas.com.

Post-PET results email shape (21-Sep-2026 follow-up draft r-7886512501656619298):
1. Greet both doctors; state scan done on <date> at St John's.
2. Summary: overall impression (STABLE DISEASE), SUV trend (21.6→18→14.8→11.9), no new visceral/bone/brain disease.
3. Two new watch signals, explicitly numbered: left supraclavicular node increase (0.9x0.5@1.9 → 1.1x1.0@2.9); new right thoracic/paraspinal subcutaneous lesion 1.1x1.6 cm SUV 6.1 (?infective vs ?early met). Optional: mild pleural increase.
4. Numbered questions: characterise the subcutaneous lesion (exam/US/biopsy threshold); supraclavicular node (US now vs reassess); continue Keytruda+axitinib unchanged; next PET interval (3-month vs earlier).
5. One-line note on the necrosis-vs-active point: Sep scan was standard FDG PET — if he wants contrast / extra sequence, we'll coordinate with Dr Annie.
6. Draft only; verify To/Cc/thread after creation.

---

## Context B: KIRAN PAP Logistics (ndr@ahfl.in)

### Account
- **Account**: ndr@ahfl.in (google-ahfl vault)
- DO NOT use draas.com for PAP logistics — the thread does not exist there

### What This Channel Is For
Every 2-3 weeks, NDR submits **Infusion Confirmation Form** (signed by Dr. Annie) + **Prescription** (next scheduled infusion date) to the KIRAN Patient Access Program (MSD/Medybiz) so they release the next cycle's free vials of Keytruda.

### Key Participants
| Person | Role | Email |
|---|---|---|
| **Kiran PAP Team** | Program coordinators (Medybiz) | **kiranpapv3@medybizpharma.com** (To) |
| **Charitra Murjani** | Patient | **charitrakamath@gmail.com** (Cc) — verify from thread headers each time |
| **Roshni Ranka** | Family | **rnr@draas.com** (Cc) |
| **Dr. Annie K. Baa** | Prescribing oncologist | **anniekbaa@gmail.com** (NOT on email CC — signs the infusion form) |

### Active Thread — KIRAN PAP
- **Subject**: `RE : [Ticket No:7027] Infusion Confirmation Form and Prescription`
- **Thread ID**: 19f657773f75c067 (on google-ahfl)
- **Started**: Jul 2026, ongoing with every cycle
- **Submission email**: kiranpapv3@medybizpharma.com
- **Patient SPS ID**: 1868282 (must be on the infusion confirmation form)

### Document Types & Naming Convention
Files go into the **Murjani Medical** folder on Drive (folder id: 1erVpDFXh9tdJuhsN5N36Zt69yjX4QG-V).

**Naming pattern observed from existing files:**
```
YYYYMMDD Charitra [Document Type] [Details].pdf
```
Examples:
- `20260722 Charitra Keytruda Infusion Confirmation MSD.pdf` — the signed infusion confirmation form
- `20260722 Charitra Pembrolizumab Prescription St Johns.pdf` — the signed prescription
- `20260803_Charitra_ChinaClinicalTrials_Research_v1.0.html` (research, not submitted to PAP)
- `20260722 Charitra Cough Prescription St Johns DrAnnie.pdf` (ancillary prescriptions)

**Each cycle creates 2 documents to submit:**
1. **Infusion Confirmation Form** — signed by patient rep + Dr. Annie, showing cycle number, vials infused, infusion date
2. **Prescription** — showing Inj. Pembrolizumab 200 mg, next scheduled infusion date

### Submission Workflow (per cycle)
1. **Receive documents** — NDR sends scanned PDFs of the signed infusion form + prescription (may be scanned images)
2. **OCR/extract** — if scanned (0 text layer), use pymupdf to render at 150 DPI then `vision_analyze` to read all text
3. **Name the files** per convention: `YYYYMMDD Charitra Keytruda Infusion Confirmation MSD.pdf` and `YYYYMMDD Charitra Pembrolizumab Prescription St Johns.pdf`
4. **Upload to Drive** — into `Murjani Medical` folder
5. **Find the thread** — on google-ahfl, search `q='kiranpapv3 subject:Infusion Confirmation'`
6. **Compose email reply** — Reply-all in the existing thread:
   - **To**: kiranpapv3@medybizpharma.com
   - **Cc**: charitrakamath@gmail.com, rnr@draas.com, anniekbaa@gmail.com
7. **Draft body pattern** (from Aug 3 email):
   - State: sharing Infusion Confirmation Form (signed) for Ms. Charitra Murjani (SPS ID: 1868282) and the prescription for the next scheduled infusion
   - Ask them to ensure Keytruda under KIRAN PAP is available on the scheduled infusion date (from the prescription)
   - Ask them to confirm the prescription's doctor/hospital details are in order — because a prior cycle had issues
   - Request confirmation by the next day to avoid last-minute surprises
8. **Attach** the two PDFs from Drive
9. **Create as DRAFT only** — verify in Drafts folder, report to user

### Pitfalls
- **Email address verification**: Charitra has multiple emails on record (`charitrakamath@gmail.com`, `charitra_murjani@yahoo.com`, `Charitramurjani77@yahoo.com`, `murjanicharitra@gmail.com`). The ACTIVE one used on ALL thread cycles is **charitrakamath@gmail.com**. Always verify from the LATEST thread message's Cc/To headers before drafting — do NOT use any other address.
- **Dr. Annie is NOT on the email CC** — she signs the documents but is not a thread participant. NDR may explicitly ask to add her (anniekbaa@gmail.com).
- **Roshni Murjani (rmurjani@gmail.com, Charitra's sister)** is distinct from Roshni Ranka (rnr@draas.com, NDR's wife). Roshni Murjani was CC'd on earlier threads (Feb 2026) but recent cycles use rnr@draas.com instead. Verify from the latest thread to avoid dead addresses.
- **Scanned PDFs have 0 text layer** — use pymupdf render at 150 DPI → `/tmp/...png` → `vision_analyze`. Do NOT expect pdftotext to work. 300 DPI can OOM the VPS; 150 DPI is safe.
- **OTP-based delivery** — Medybiz releases vials by OTP; the thread shows this is the normal process. The email should ask them to confirm availability before the infusion date rather than assume auto-delivery.
- **The thread lives on ndr@ahfl.in** — build_service('gmail','v1',service_name='google-ahfl'). Default gmail tools route to google-draas and will not find these messages.

---

## Pitfall — Multiple email addresses per person (cross-context)
All known addresses for a person across BOTH contexts are listed, but NOT all are active. Check the LATEST thread message's headers for the correct address — not the full list. When NDR sent to `murjanicharitra@gmail.com` (one of Charitra's old addresses), he confirmed it was wrong. The active address for Charitra on both threads is `charitrakamath@gmail.com`.

---

## Context C: Review-day / post-consult communication (proven 2026-09-23)

### Post-review THANK-YOU reply-all pattern
After a review meeting with Dr Annie, both doctors typically reply on the thread: Sameer replies briefly (e.g. "Same treatment. I am not in the country currently. Dr Annie can discuss further."), Annie replies with a reassuring summary (stable disease, continue same regimen, the paraspinal lesion is infected → good news). When NDR says "reply all and thank both, short and nice":
- Anchor `draft_reply_create` on the **LATEST INBOUND message** (e.g. Annie's reply message id 1a0cd45d420544ab, not NDR's own sent message) so the thread nests correctly.
- To: SAMEER RASTOGI <samdoc_mamc@yahoo.com>, annie baa <anniekbaa@gmail.com>
- Cc: Charitra Kamath <charitrakamath@gmail.com>, Roshni Ranka <rnr@draas.com>
- Body shape (short only — he explicitly said "short and nice"): thank Dr Rastogi for the prompt reply even while travelling; thank Dr Annie for the calm positive outlook and continued care of Chinky; confirm "we will continue with the same regimen as advised"; warm sign-off "Nishant (for the family)".
- Draft only; verify To/Cc/thread via `draft_get` after creation.

### Consult-prep Q&A — plain-English explanation preference (user style)
When NDR asks for Q&A to read before/after a consult, he wants LAYMAN plain-English answers first, then technical framing. Define every term in one line before discussing:
- **SUV** = metabolic activity number on the PET; normal tissue ~1-2, tumour 10-20+. High-ish ≠ cancer; inflammation routinely hits 4-8.
- **Node size pairs** = dimensions in cm (e.g. 0.9x0.5) followed by @SUV (e.g. @1.9). SUV ~2.5-3.5 is the gray zone.
- **Pleural effusion** = fluid in the sac around the lung (not inside lung tissue); visible/measurable on the CT component of the PET-CT; "mild increase" = fluid layer slightly thicker than prior scan. Thoracentesis (needle drain) triggers: symptomatic breathlessness, large volume (>1/3-1/2 chest), or diagnostic doubt — small asymptomatic effusions are monitor-only.
- **Paraspinal subcutaneous lesion** = small lump in the fatty layer just under the skin beside the spine, outside the ribcage. New small lesion while the main tumour is shrinking usually means reactive/inflammatory, not resistant spread; action ladder = physical exam → US-guided core biopsy (fast, 100% answer) → rescan in 4-6 wks and biopsy only if it grows.
- **Delivery form**: for reading with the family NDR asked for **HTML/CSS, not a Google Doc** — build a styled self-contained HTML page with clickable citation links.

### PET: necrotic vs active tumour differentiation (Dr Sameer's point, AIIMS)
Standard FDG PET cannot cleanly separate necrotic core from viable tumour. Techniques to differentiate (research-backed):
1. **Contrast-enhanced PET-CT** (IV contrast on the CT part) — necrotic tissue does not enhance, viable tumour does. Cheapest; MOST LIKELY what Sameer meant ("was it done?"). Ask the scan centre: did the PET include IV contrast?
2. **Dual-time-point FDG PET** (delayed imaging ~2-3 h post-injection) — viable tumour keeps accumulating FDG, necrosis/inflammation does not. PMID 28583277.
3. **68Ga-FAPI PET-CT** — targets fibroblast activation protein in tumour stroma; superior for sarcoma response assessment vs FDG; theranostic bonus (Lu-177 FAPI therapy). PMIDs 38365545, 40456674, 37117120. AIIMS does FAPI; confirm with St John's nuclear medicine in Bangalore.
4. **18F-FLT PET** — proliferation tracer correlating with Ki-67; research-grade.
Practical read for Chinky's case: SUVmax still ~11.9-14.8 means a large rim is still metabolically alive (necrotic core alone would read SUV ~2-3) — the technique quantifies how much viable rim remains at the airway/oesophageal edge, which drives clinical risk.

### Future-therapy shortlist carried into consults (Sep-2026 research, verified)
- **B7-H3 (CD276)**: surface protein common on sarcomas, rare on normal cells → good target. ADC = antibody-drug conjugate (antibody binds B7-H3, delivers chemo inside tumour cell); CAR-T = patient T-cells re-engineered to recognise B7-H3.
- **B7-H3 ADC HS-20093**: NCT05276609 (ph1 solid tumours), NCT06935409 (ph3 osteosarcoma — sarcoma-specific late-stage), NCT07464327 (ADC+PD-L1 vs chemo), NCT06332170 (combinations).
- **B7-H3 CAR-T**: NCT07751380 (sarcoma, children/young adults), NCT04864821 (CD276 CAR-T solid tumours), NCT07222735 (radiation + B7-H3 CAR-T). PubMed: 42641597, 42517272, 42674817. ChiCTR2500108722 = Chinese registry (chictr.org.cn) — not cross-indexed on ClinicalTrials.gov, verify there.
- **Utidelone + anlotinib** (salvage axis, China): utidelone = next-gen taxane-like microtubule stabiliser (new chemo mechanism), anlotinib = multi-target TKI approved in China for STS. UTISARC phase II (anthracycline + anti-angiogenic refractory STS) PubMed 42286614; anlotinib+ICI meta 42762479; anlotinib sarcoma meta 42496766; sintilimab+anlotinib in UPS case 42344169.
- **Actionable ask to put to the doctors**: B7-H3 IHC on the ORIGINAL biopsy wax block is cheap, one-time, and keeps the ADC/CAR-T door open if progression comes. Frame: NOT changing treatment today — current plan is working; just want options on the record + consider the B7-H3 stain.

### Deliver Q&A / research docs: HTML → Drive TMP folder (public link)
NDR's TMP folder on Drive: id **18p74II2uL32sNDzDDwXzmlOUdJJOTmE-**. Upload HTML via `drive.files().create` with `MediaFileUpload` (mimeType text/html), set permission `{'type':'anyone','role':'reader'}` so the link is public, then give the view link (`drive.google.com/file/d/<id>/view`) plus the raw download link (`drive.google.com/uc?id=<id>&export=download`). Verify the permission grant returns role reader before handing over the link.