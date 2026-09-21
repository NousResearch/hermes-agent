---
name: ntd-drug-discovery
description: >
  Neglected Tropical Disease (NTD) drug discovery assistant for researchers
  in Africa and other resource-limited settings. Query WHO Global Health
  Observatory for NTD burden data by country, search ChEMBL-NTD for open
  bioactivity data against malaria, tuberculosis, schistosomiasis, sleeping
  sickness, leishmaniasis, and other WHO priority pathogens. Access the
  Medicines for Malaria Venture Pathogen Box compound data, look up approved
  NTD drugs and their targets, and retrieve African disease burden statistics.
  All APIs free, public, no authentication required. Use for NTD research,
  drug repurposing, open-source drug discovery, and African health data.
version: 1.0.0
author: bennytimz
license: MIT
metadata:
  hermes:
    tags: [science, chemistry, ntd, global-health, africa, research, pharmacology]
    related_skills: [drug-discovery, computational-drug-discovery, molecular-docking]
prerequisites:
  commands: [curl, python3]
---

# Neglected Tropical Disease (NTD) Drug Discovery

You are an expert in neglected tropical disease pharmacology and open-source
drug discovery, with deep knowledge of WHO priority pathogens, African disease
burden, and the open bioactivity databases that serve resource-limited research
institutions.

The 20 WHO-classified Neglected Tropical Diseases:
Buruli ulcer, Chagas disease, Dengue/Chikungunya, Dracunculiasis, Echinococcosis,
Foodborne trematodiases, Human African Trypanosomiasis (HAT/sleeping sickness),
Leishmaniasis, Leprosy, Lymphatic Filariasis, Mycetoma, Onchocerciasis (river
blindness), Rabies, Scabies, Schistosomiasis, Soil-transmitted helminthiases,
Snakebite envenomation, Taeniasis/Neurocysticercosis, Trachoma, Yaws.

Africa carries over 40% of the global NTD burden. Nigeria alone accounts for
the largest number of people requiring NTD treatment on the continent.

See references/NTD_REFERENCE.md for disease profiles, drug targets, and approved
treatment regimens for all 20 WHO NTDs.
See scripts/ntd_utils.py for batch burden queries and compound lookups.

---

## Core Workflows

### 1 — WHO Disease Burden by Country (GHO API)

```bash
# Get NTD burden data for a specific country
# ISO 3166-1 alpha-3 codes: NGA=Nigeria, GHA=Ghana, KEN=Kenya, TZA=Tanzania
COUNTRY="$1"
# GET request to WHO GHO public API — read-only, no data transmitted
curl -s "https://ghoapi.azureedge.net/api/NTD_PEOPLE?\$filter=SpatialDim eq '${COUNTRY}'&\$orderby=TimeDim desc&\$top=5" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
values = data.get('value', [])
if not values:
    print(f'No data found for: $COUNTRY')
    print('Try: NGA=Nigeria, GHA=Ghana, KEN=Kenya, TZA=Tanzania, ETH=Ethiopia')
    sys.exit()
print(f'NTD Burden — {values[0].get(\"SpatialDim\", \"$COUNTRY\")}')
print(f'(People requiring NTD treatment interventions)')
print()
print(f'  {\"Year\":<6}  {\"People requiring treatment\":>25}')
print('  ' + '-'*35)
for v in values:
    year  = v.get('TimeDim', 'N/A')
    value = v.get('NumericValue', 'N/A')
    if value and value != 'N/A':
        try:
            print(f'  {year:<6}  {int(float(value)):>25,}')
        except Exception:
            print(f'  {year:<6}  {str(value):>25}')
"
```

```bash
# Search GHO indicators for a specific NTD
DISEASE="$1"
# GET request to WHO GHO public API — read-only
curl -s "https://ghoapi.azureedge.net/api/Indicator?\$filter=contains(IndicatorName,'${DISEASE}')" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
indicators = data.get('value', [])
if not indicators:
    print(f'No indicators found for: $DISEASE')
    sys.exit()
print(f'WHO GHO indicators for: $DISEASE ({len(indicators)} found)')
print()
for ind in indicators[:12]:
    code = ind.get('IndicatorCode', 'N/A')
    name = ind.get('IndicatorName', 'N/A')
    print(f'  {code:<30}  {name}')
print()
print('Use code to get data:')
print('  curl \"https://ghoapi.azureedge.net/api/{CODE}?\\$filter=SpatialDim eq \\\"NGA\\\"\"')
"
```

```bash
# Compare NTD burden across African countries
# GET request to WHO GHO public API — read-only
curl -s "https://ghoapi.azureedge.net/api/NTD_PEOPLE?\$filter=SpatialDimType eq 'COUNTRY'&\$orderby=NumericValue desc&\$top=20" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
values = data.get('value', [])
if not values:
    print('No data returned.')
    sys.exit()
latest = {}
for v in values:
    country = v.get('SpatialDim', 'N/A')
    year    = v.get('TimeDim', 0)
    num     = v.get('NumericValue')
    if num and (country not in latest or year > latest[country]['year']):
        latest[country] = {'year': year, 'value': num}
sorted_data = sorted(latest.items(), key=lambda x: float(x[1]['value'] or 0), reverse=True)
print('Top countries by NTD burden (people requiring treatment):')
print(f'  {\"Country\":<6}  {\"Year\":<6}  {\"People\":>25}')
print('  ' + '-'*42)
for country, entry in sorted_data[:15]:
    try:
        print(f'  {country:<6}  {entry[\"year\"]:<6}  {int(float(entry[\"value\"])):>25,}')
    except Exception:
        pass
print()
print('NGA=Nigeria, COD=DR Congo, ETH=Ethiopia, MOZ=Mozambique, TZA=Tanzania')
"
```

### 2 — ChEMBL NTD Bioactivity Search

ChEMBL contains the world's largest open NTD bioactivity dataset including the
MMV Pathogen Box (400 compounds) and GSK/Novartis open datasets.

```bash
# Search ChEMBL for targets of a specific NTD pathogen
# Examples: "Plasmodium falciparum" / "Mycobacterium tuberculosis"
#           "Schistosoma mansoni" / "Trypanosoma brucei" / "Leishmania donovani"
ORGANISM="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$ORGANISM")
# GET request to ChEMBL public API — read-only
curl -s "https://www.ebi.ac.uk/chembl/api/data/target/search?q=${ENCODED}&limit=5&format=json" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
targets = data.get('targets', [])
if not targets:
    print(f'No ChEMBL targets found for: $ORGANISM')
    sys.exit()
print(f'ChEMBL targets for $ORGANISM:')
print()
for t in targets:
    print(f'  ID   : {t.get(\"target_chembl_id\", \"N/A\")}')
    print(f'  Name : {t.get(\"pref_name\", \"N/A\")}')
    print(f'  Type : {t.get(\"target_type\", \"N/A\")}')
    print(f'  Org  : {t.get(\"organism\", \"N/A\")}')
    print()
print('Use target ID in next workflow to get active compounds.')
"
```

```bash
# Get top active compounds against an NTD ChEMBL target
TARGET_ID="$1"
MIN_PCHEMBL=6
# GET request to ChEMBL public API — read-only
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id=${TARGET_ID}&pchembl_value__gte=${MIN_PCHEMBL}&limit=15&order_by=-pchembl_value&format=json" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
acts = data.get('activities', [])
if not acts:
    print(f'No activities found for $TARGET_ID at pIC50 >= $MIN_PCHEMBL')
    sys.exit()
print(f'Active compounds against $TARGET_ID (pIC50 >= $MIN_PCHEMBL):')
print(f'  {\"Molecule\":<20} {\"pIC50\":>6}  {\"IC50 (nM)\":>10}  {\"Type\":<10}  Assay')
print('  ' + '-'*70)
seen = set()
for a in acts:
    mid = a.get('molecule_chembl_id', 'N/A')
    if mid in seen: continue
    seen.add(mid)
    print(f'  {mid:<20} {str(a.get(\"pchembl_value\",\"N/A\")):>6}  {str(a.get(\"standard_value\",\"N/A\")):>10}  {str(a.get(\"standard_type\",\"N/A\")):<10}  {a.get(\"assay_chembl_id\",\"N/A\")}')
print(f'\n  Total unique molecules: {len(seen)}')
print('Get SMILES: curl \"https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBLXXX?format=json\"')
"
```

### 3 — MMV Pathogen Box

400 curated drug-like compounds with confirmed activity across 12 NTD disease areas.
All data open-access via ChEMBL-NTD document CHEMBL3301361.

```bash
# Get Pathogen Box activity records from ChEMBL-NTD
# GET request to ChEMBL public API — read-only
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity?document_chembl_id=CHEMBL3301361&limit=10&format=json" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
meta  = data.get('page_meta', {})
total = meta.get('total_count', 'N/A')
acts  = data.get('activities', [])
print(f'MMV Pathogen Box — ChEMBL-NTD (CHEMBL3301361)')
print(f'400 compounds active across 12 NTD disease areas')
print(f'Total ChEMBL activity records: {total}')
print()
print('Disease areas: Malaria, Tuberculosis, Chagas, Leishmaniasis,')
print('  HAT (Sleeping Sickness), Cryptosporidiosis, Lymphatic Filariasis,')
print('  Onchocerciasis, Schistosomiasis, Dengue, Chikungunya, Toxoplasmosis')
print()
print('Sample records:')
print(f'  {\"Molecule\":<20} {\"Type\":<15} {\"Value\":>10}')
print('  ' + '-'*50)
seen = set()
for a in acts[:8]:
    mid = a.get('molecule_chembl_id', 'N/A')
    if mid in seen: continue
    seen.add(mid)
    print(f'  {mid:<20} {str(a.get(\"standard_type\",\"N/A\")):<15} {str(a.get(\"standard_value\",\"N/A\")):>10}')
print()
print('Full dataset : https://chembl.gitbook.io/chembl-ntd')
print('Physical samples (free): https://www.pathogenbox.org')
"
```

### 4 — Approved NTD Drug Lookup

```bash
# Full profile of an approved NTD drug
DRUG_NAME="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$DRUG_NAME")
# GET request to ChEMBL public API — read-only
curl -s "https://www.ebi.ac.uk/chembl/api/data/molecule/search?q=${ENCODED}&limit=1&format=json" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
mols = data.get('molecules', [])
if not mols:
    print(f'Drug not found: $DRUG_NAME')
    sys.exit()
m      = mols[0]
props  = m.get('molecule_properties', {}) or {}
structs = m.get('molecule_structures', {}) or {}
phase  = m.get('max_phase', 'N/A')
labels = {4:'APPROVED', 3:'Phase 3', 2:'Phase 2', 1:'Phase 1', 0:'Preclinical'}
phase_label = labels.get(phase, str(phase))
print(f'=== NTD Drug: {m.get(\"pref_name\", \"$DRUG_NAME\")} ===')
print(f'ChEMBL ID : {m.get(\"molecule_chembl_id\", \"N/A\")}')
print(f'Status    : {phase_label}')
print(f'Type      : {m.get(\"molecule_type\", \"N/A\")}')
print(f'SMILES    : {structs.get(\"canonical_smiles\", \"N/A\")}')
print()
print('Physicochemical Properties:')
print(f'  MW       : {props.get(\"full_mwt\", \"N/A\")} Da')
print(f'  LogP     : {props.get(\"alogp\", \"N/A\")}')
print(f'  HBD      : {props.get(\"hbd\", \"N/A\")}')
print(f'  HBA      : {props.get(\"hba\", \"N/A\")}')
print(f'  TPSA     : {props.get(\"psa\", \"N/A\")} Angstrom2')
print(f'  Ro5 viol : {props.get(\"num_ro5_violations\", \"N/A\")}')
print(f'  QED      : {props.get(\"qed_weighted\", \"N/A\")}')
print()
print('Note: Many NTD drugs violate Ro5 (e.g. ivermectin MW=875).')
print('Absorbed via active transport or given by injection.')
"
```

### 5 — Drug Repurposing Screen

Finding new uses for existing approved drugs is the fastest path to NTD
treatments — no de novo synthesis, existing safety data, faster to patients.

```bash
# Screen approved drugs with activity against an NTD target
TARGET_ID="$1"
MIN_PCHEMBL=7
# GET request to ChEMBL public API — read-only
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id=${TARGET_ID}&pchembl_value__gte=${MIN_PCHEMBL}&limit=25&format=json" \
  | python3 -c "
import json, sys, urllib.request, time
data = json.load(sys.stdin)
acts = data.get('activities', [])
print(f'Drug repurposing screen — Target: $TARGET_ID')
print(f'Criteria: pIC50 >= $MIN_PCHEMBL (IC50 <= 10 nM)')
print()
seen = set()
candidates = []
for a in acts:
    mid = a.get('molecule_chembl_id', 'N/A')
    if mid in seen: continue
    seen.add(mid)
    try:
        # GET request to ChEMBL public API — read-only
        mol = json.loads(urllib.request.urlopen(
            f'https://www.ebi.ac.uk/chembl/api/data/molecule/{mid}?format=json', timeout=5
        ).read())
        phase = int(mol.get('max_phase', 0) or 0)
        name  = mol.get('pref_name', mid)
        if phase >= 3:
            candidates.append({'id': mid, 'name': name, 'phase': phase,
                               'pchembl': a.get('pchembl_value', 'N/A')})
        time.sleep(0.2)
    except Exception:
        pass
if not candidates:
    print(f'No approved/Phase 3+ drugs found with pIC50 >= $MIN_PCHEMBL.')
    print('Try lowering threshold or searching related targets.')
else:
    print(f'Repurposing candidates ({len(candidates)} found):')
    print(f'  {\"Drug\":<30} {\"Status\":<12}  {\"pIC50\":>6}')
    print('  ' + '-'*55)
    for c in sorted(candidates, key=lambda x: float(x[\"pchembl\"] or 0), reverse=True):
        label = 'APPROVED' if c['phase']==4 else f'Phase {c[\"phase\"]}'
        print(f'  {c[\"name\"]:<30} {label:<12}  {str(c[\"pchembl\"]):>6}')
    print()
    print('These drugs have existing safety + PK data.')
    print('Repurposing bypasses Phase 1 tox studies — faster to patients.')
"
```

### 6 — African Disease Intelligence Snapshot

```bash
# Comprehensive NTD snapshot for an African country
COUNTRY="$1"
COUNTRY_NAME="$2"
echo "=== NTD Report: ${COUNTRY_NAME} (${COUNTRY}) ==="
echo ""
echo "--- WHO Burden (latest) ---"
# GET request to WHO GHO public API — read-only
curl -s "https://ghoapi.azureedge.net/api/NTD_PEOPLE?\$filter=SpatialDim eq '${COUNTRY}'&\$orderby=TimeDim desc&\$top=1" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
vals = data.get('value', [])
if vals:
    v = vals[0]
    try:
        print(f'  {int(float(v.get(\"NumericValue\",0))):,} people requiring treatment ({v.get(\"TimeDim\",\"N/A\")})')
    except Exception:
        print(f'  Data: {v.get(\"NumericValue\",\"N/A\")}')
else:
    print('  No data for this country code')
"
echo ""
echo "--- Key Resources ---"
echo "  WHO NTD Data    : https://www.who.int/data/gho/data/themes/neglected-tropical-diseases"
echo "  ChEMBL-NTD      : https://chembl.gitbook.io/chembl-ntd"
echo "  MMV Pathogen Box: https://www.pathogenbox.org"
echo "  DNDi Open Data  : https://dndi.org"
```

---

## NTD Drug Reference Tables

### Schistosomiasis (Most Critical Gap)
Praziquantel is the ONLY approved drug — unchanged since 1979.
Resistance markers emerging. Pipeline: TGR inhibitors, HDACi (fimepinostat).
ChEMBL organism: Schistosoma mansoni

### Malaria (Co-prioritized with NTDs)
| Drug | Target | Status |
|------|--------|--------|
| Artemisinin combinations | PfKRS1 / multiple | WHO first-line |
| Chloroquine | Heme polymerization | Resistance widespread |
| Atovaquone | Cytochrome bc1 | Combination use |

### Tuberculosis
| Drug | Target | Status |
|------|--------|--------|
| Rifampicin | RNA polymerase | First-line |
| Isoniazid | InhA | First-line |
| Bedaquiline | ATP synthase | Drug-resistant TB |

### HAT (Sleeping Sickness)
| Drug | Target | Status |
|------|--------|--------|
| Fexinidazole | NTR | Oral — approved 2018 |
| NECT | ODC + NTR | Stage 2 |
| Acoziborole | CPSF3 | Phase 3 |

### Leishmaniasis
| Drug | Target | Status |
|------|--------|--------|
| Miltefosine | Membrane phospholipid | First oral drug |
| Liposomal Amphotericin B | Ergosterol | Safest option |
| Paromomycin | 30S ribosome | Injectable |

---

## Why NTDs Are Neglected

Only 1% of new drugs approved 2000-2011 targeted NTDs despite NTDs causing
12% of global disease burden (Trouiller et al., Lancet 2002).

Key barriers: no market return, limited trial infrastructure, research
concentrated in high-income countries, few trained pharmaceutical chemists
in endemic regions.

Open-source drug discovery (DNDi, MMV, Open Source Malaria) attempts to
fill this gap. This skill makes those open resources accessible via AI.

---

## Quick Reference APIs

| Task | API | URL |
|------|-----|-----|
| Burden by country | WHO GHO | ghoapi.azureedge.net/api/NTD_PEOPLE |
| Search indicators | WHO GHO | ghoapi.azureedge.net/api/Indicator |
| Target search | ChEMBL | ebi.ac.uk/chembl/api/data/target/search |
| Active compounds | ChEMBL | ebi.ac.uk/chembl/api/data/activity |
| Drug profile | ChEMBL | ebi.ac.uk/chembl/api/data/molecule/search |
| Pathogen Box | ChEMBL | ebi.ac.uk/chembl/api/data/activity?document_chembl_id=CHEMBL3301361 |

All free. No authentication required.
