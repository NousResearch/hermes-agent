# Medication Procurement Workflow

When the user wants to dispatch someone (driver, family member, colleague) to buy medicines, the workflow has two phases: **verify the dosage first**, then **compose the message**.

## Phase 1 — Dosage verification

**Always verify before sending.** The user explicitly asked for this ("do that analysis before we send him a message asking him to buy the medicine").

### Key data to collect

| Field | Example |
|---|---|
| Age | 48 |
| Weight | 72-75 kg |
| Symptoms | Throat discomfort, phlegm, feeling unwell |
| Proposed drug + dose | Azithromycin 500mg 3-day course |
| User's rationale | "I can feel something in my throat, phlegm — thought it's better I take a 3-day course to settle" |

### Standard reference for common antibiotics

| Drug | Standard adult dose | Standard duration | Notes |
|---|---|---|---|
| Azithromycin 500mg | 500mg OD (once daily) | 3-5 days (total 1.5g) | Fixed adult dose, no weight adjustment needed for 50-100kg range. Z-Pak equivalent. |
| Amoxicillin 500mg | 500mg TDS (3x daily) | 5-7 days | Weight-based: ~20-40mg/kg/day in divided doses |
| Amoxicillin-Clavulanate 625 | 625mg BD (2x daily) | 5-7 days | Standard for respiratory infections |
| Doxycycline 100mg | 100mg BD Day 1, then OD | 7-14 days | Alternative for respiratory infections |

### Present the verification clearly

Format: verdict first (✅/⚠️/❌), then supporting detail in a compact table or bullets.

**Include these caveats for any antibiotic recommendation:**
- Antibiotics treat bacterial infections, not viral ones — if the infection is viral, the drug won't help
- Common side effects (nausea, loose stools for azithromycin)
- Drug-specific warnings (QT prolongation with macrolides, photosensitivity with doxycycline)
- If user has a known medical condition or takes other medication, flag a doctor consult

## Phase 2 — WhatsApp message to the procurer

### Required details to include in the message

The procurer (driver, assistant) needs exact instructions to buy the right product. Always include:

1. **Molecule name** — the generic scientific name (e.g. "Azithromycin 500mg as Azithromycin Dihydrate")
2. **Common brand names in India** — give 2-3 recognizable brands (e.g. "Azithral 500 (Alembic), Azax 500 (Sun Pharma), Zithrin 500 (Shreya) — any of these is fine")
3. **Exact quantity** — number of tablets and dosing schedule (e.g. "3 tablets only — one tablet after food, once daily for 3 days")
4. **Any special instructions** — "after food", "with water", "not on empty stomach"

### Message structure template

```
[Name], please buy this medicine for me:

Medicine: Azithromycin 500mg (molecule name — Azithromycin Dihydrate)
Brand: Azithral 500 / Azax 500 / Zithrin 500 — any of these works
Quantity: 3 tablets only (one 500mg tablet after food, once daily for 3 days — three-day course total)

[Optional additional context/name of person who needs it]
```

Keep it concise — the procurer just needs the product name, brand options, and quantity.

## Pitfalls

- **Don't skip Phase 1.** The user's exact request was "do the analysis before sending". Present the analysis, let the user confirm, then send.
- **Don't assume the user knows the molecule name.** They may only know the brand name (e.g. "Azithral") or just "AZ500". Provide the full generic name.
- **Indian pharmacy reality:** Most Indian pharmacists will dispense the generic or whatever brand they stock. Giving 2-3 common brand names ensures the procurer can buy from any pharmacy without being told "we don't have that brand, come back with a prescription".
- **Prescription requirement:** Azithromycin is a Schedule H antibiotic in India — technically requires a prescription. Many pharmacies dispense it without one for a 3-day course, but flag this to the user if appropriate.
- **Quantity confusion:** Don't say "3-day course" without specifying "3 tablets total" (one per day). Some people interpret "3-day course" as 3 boxes. Be explicit: "ONLY 3 tablets".
