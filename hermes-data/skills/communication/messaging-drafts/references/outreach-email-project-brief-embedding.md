# Outreach emails MUST embed the real project brief (corrected 21-Sep-2026)

Scenario: NDR asks for an outreach email to an architect/consultancy ("reach out to Janice,
mark Arjun, brief her about Riverstone and our new Miller's Road office"). The first draft was
a generic 3-paragraph note that *said* "we'd like to give you a briefing on Riverstone" without
actually containing the brief. NDR corrected: **"the draft does not contain any briefing about
what Riverstone is … there's a briefing note somewhere which we have given earlier to Karthik
… Just look for that and add the briefing note here … And then the Millers Road office briefing
note is also not added … add all of that and redo the draft."**

Rule: when the user says "give them a briefing on X" in an email, the email body must carry
**actual substance about X** — project model, acreage, locations, scope of work — not a promise
to brief them verbally. Search for existing brief material BEFORE drafting.

## Proven sourcing path (worked 21-Sep-2026)

1. **Find the prior briefing by searching Gmail for the project + a likely recipient name.**
   `TerraGreens Riverstone` and `Riverstone Karthik` missed; the winner was
   `Riverstone briefing` and plain queries like `Karthik Riverstone` / `Kuteer` / `Kurg` /
   `"estate living"`. The brief NDR meant = **`Terra_Greens_Project_Brief_Kartik.docx`**,
   attached to the 7-Jul-2026 email "Scope of work and detailed portfolio" from Aamir Khan
   (aamirkhan@icloud.com) → Karthik Krishna (karthik@evervantage.in / karthik.krishna@equa.in),
   Cc NDR. Doc = Terra Greens luxury estate & hospitality portfolio (the "estate living"
   story: each location = resort zone + investor-leaseback villas + 1-acre estate plots).
2. **Pull the attached docx from Gmail via the attachments API** and extract text
   (zipfile + regex over `word/document.xml`; strip tags, restore `</w:p>` → newline).
   The brief contained the exact portfolio table: Chikkaballapur 110 ac, Coorg/Madikeri 70,
   Ooty 70, Bandipur 280, Bidadi 100, Alipur 230 (golf estate, no leaseback).
3. **For a building/property brief (Miller's Road / Akber office): search Drive fullText.**
   `fullText contains 'Millers Road'` / `'India Chai'` / `'Akber'` surfaced the lease deed
   (20260709_LeaseDeed_MillersRoad_DRA_vs_Akber_v5/v7), the Nova Spaces facade proposal
   (glass + aluminium cladding, remove concrete projections), and the structural-test email
   (20-yr-old RCC building). Email threads named `DRA Asset Upgradation Proposal — Facade
   Costing Required` and `Proposed Structural Tests for India Chai Office Building` gave the
   renovation scope verbatim (facade cladding + interiors).
4. **Realize "Aqbar/Agbar" = Akber Hussain** (landlord of the Millers Road India Chai
   building, ahindia.com) — voice garbles of that building name resolve via the
   `millers-road-akber-lease` reference under the same umbrella.
5. **Redo the draft by deleting the old draft and creating a new one** (bridge/draft create)
   with the brief content inline: Riverstone/TerraGreens story in 2 paragraphs + Miller's Road
   facade/interiors scope in 1 paragraph + the meeting ask. Recipients unchanged
   (To Janice BNA, Cc Arjun Nambisan + Aamir Khan khan.hussain.aamir@gmail.com).

## Pitfalls
- Do NOT draft credibility emails from general knowledge of the project. NDR keeps written
  briefs (docx attachments, Drive docs, old email threads) and expects them reused verbatim-in-
  substance. A draft that asserts "we will brief you" with no content will be rejected.
- The Aamir on this email = **khan.hussain.aamir@gmail.com** (Riverstone partner, his Google
  identity). Do NOT use aamirkhan@me.com / aamirkhan@icloud.com for the Cc — me.com has no
  Google Account (Drive 400) and iCloud is his non-Google writer identity.
- First draft created as a fresh draft; when redoing, delete the old draft id so NDR's Drafts
  folder doesn't accumulate duplicates.