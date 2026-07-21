---
name: computational-drug-discovery
description: >
  Computer-Aided Drug Design (CADD) assistant covering structure-based and
  ligand-based drug discovery workflows. Fetch protein structures from AlphaFold
  and RCSB PDB, search compound libraries in ZINC and PubChem, predict ADMET
  properties via pkCSM, run QSAR and similarity searches, and reason through
  molecular docking, virtual screening, homology modelling, and pharmacophore
  concepts. Use for computational chemistry, structural biology, in silico
  screening, and target-based drug design questions.
version: 1.0.0
authors:
  - bennytimz
license: MIT
metadata:
  hermes:
    tags: [science, chemistry, cadd, structural-biology, research]
prerequisites:
  commands: [curl, python3, jq]
---

# Computational Drug Discovery (CADD)

You are an expert computational medicinal chemist fluent in both
**structure-based (SB)** and **ligand-based (LB)** drug design approaches.

**SB methods** — used when the 3D structure of the target protein is known
(X-ray crystallography, cryo-EM, NMR, or AlphaFold prediction).

**LB methods** — used when protein structure is NOT known but active ligands
are available (QSAR, pharmacophore modelling, similarity searching).

See references/CADD_METHODS.md for the full 8-method CADD framework.

---

## Core Workflows

### 1 — Protein Structure Retrieval (AlphaFold)

Fetch AI-predicted 3D protein structures for any UniProt ID. Free, no auth.

```bash
# Get AlphaFold structure metadata for a protein
UNIPROT_ID="$1"
curl -s "https://alphafold.ebi.ac.uk/api/prediction/${UNIPROT_ID}" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
if not data:
    print('No AlphaFold entry found for this UniProt ID.')
    sys.exit()
entry = data[0]
print(f\"Protein     : {entry.get('uniprotDescription', 'N/A')}\")
print(f\"Gene        : {entry.get('gene', 'N/A')}\")
print(f\"Organism    : {entry.get('organismScientificName', 'N/A')}\")
print(f\"UniProt ID  : {entry.get('uniprotAccession', 'N/A')}\")
print(f\"pLDDT (avg) : {entry.get('confidenceAvgLocalScore', 'N/A')} (>90=very high, 70-90=high, 50-70=low, <50=very low)\")
print(f\"Length      : {entry.get('uniprotSequenceLength', 'N/A')} aa\")
print(f\"PDB file    : {entry.get('pdbUrl', 'N/A')}\")
print(f\"View        : https://alphafold.ebi.ac.uk/entry/{entry.get('uniprotAccession', '')}\")
"
```

```bash
# Search UniProt for a gene name to get its UniProt ID
GENE="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$GENE")
curl -s "https://rest.uniprot.org/uniprotkb/search?query=${ENCODED}+AND+organism_id:9606+AND+reviewed:true&fields=accession,gene_names,protein_name,length&format=json&size=5" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
results = data.get('results', [])
if not results:
    print('No UniProt entries found.')
    sys.exit()
print('Top UniProt matches for human proteins:')
for r in results:
    acc   = r.get('primaryAccession', 'N/A')
    genes = r.get('genes', [{}])[0].get('geneName', {}).get('value', 'N/A') if r.get('genes') else 'N/A'
    name  = r.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')
    length = r.get('sequence', {}).get('length', 'N/A')
    print(f'  {acc}  |  {genes}  |  {name}  |  {length} aa')
"
```

### 2 — Experimental Structure Search (RCSB PDB)

```bash
# Search RCSB PDB for experimental structures of a target
TARGET="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$TARGET")
curl -s "https://search.rcsb.org/rcsbsearch/v2/query?json=%7B%22query%22%3A%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22full_text%22%2C%22parameters%22%3A%7B%22value%22%3A%22${ENCODED}%22%7D%7D%2C%22return_type%22%3A%22entry%22%2C%22request_options%22%3A%7B%22paginate%22%3A%7B%22start%22%3A0%2C%22rows%22%3A5%7D%7D%7D" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
ids = [r['identifier'] for r in data.get('result_set', [])]
if not ids:
    print('No PDB structures found.')
    sys.exit()
print(f'PDB IDs found: {ids}')
import urllib.request
detail = json.loads(urllib.request.urlopen(f'https://data.rcsb.org/rest/v1/core/entry/{ids[0]}').read())
struct = detail.get('struct', {})
expt   = detail.get('exptl', [{}])[0]
print(f'Top result: {ids[0]}')
print(f'  Title     : {struct.get(\"title\", \"N/A\")}')
print(f'  Method    : {expt.get(\"method\", \"N/A\")}')
print(f'  Resolution: {detail.get(\"refine\", [{}])[0].get(\"ls_d_res_high\", \"N/A\")} Angstrom')
print(f'  View      : https://www.rcsb.org/structure/{ids[0]}')
print(f'  Download  : https://files.rcsb.org/download/{ids[0]}.pdb')
"
```

### 3 — Virtual Screening Library (ZINC)

```bash
# Search ZINC for drug-like compounds
QUERY="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$QUERY")
curl -s "https://zinc.docking.org/substances/search/?q=${ENCODED}&count=10&format=json" \
  | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    compounds = data if isinstance(data, list) else data.get('results', data.get('objects', []))
    for c in compounds[:5]:
        print(f\"  ZINC ID : {c.get('zinc_id', c.get('id', 'N/A'))}\")
        print(f\"  Name    : {c.get('name', 'N/A')}\")
        print(f\"  SMILES  : {c.get('smiles', 'N/A')}\")
        print(f\"  MW      : {c.get('mwt', 'N/A')}\")
        print()
except Exception as e:
    print(f'Error: {e}. Try https://zinc.docking.org/substances/search/?q=YOUR_QUERY')
"
```

### 4 — ADMET Prediction (pkCSM)

```bash
# Predict ADMET properties for a SMILES string
SMILES="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$SMILES")
curl -s -X POST "http://biosig.lab.uq.edu.au/pkcsm/api/predict" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "smiles=${ENCODED}" \
  | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print('=== pkCSM ADMET Predictions ===')
    print('ABSORPTION')
    print(f'  Caco-2 permeability : {data.get(\"Caco2_permeability\", \"N/A\")}')
    print(f'  HIA                 : {data.get(\"HIA_Hou\", \"N/A\")}')
    print(f'  Pgp substrate       : {data.get(\"Pgp_substrate\", \"N/A\")}')
    print('DISTRIBUTION')
    print(f'  BBB penetration     : {data.get(\"BBB_Martins\", \"N/A\")}')
    print('METABOLISM')
    for cyp in [\"CYP2D6_substrate\",\"CYP3A4_substrate\",\"CYP2C9_inhibitor\",\"CYP3A4_inhibitor\"]:
        print(f'  {cyp:<28}: {data.get(cyp, \"N/A\")}')
    print('TOXICITY')
    print(f'  hERG inhibition     : {data.get(\"hERG_Karim\", \"N/A\")}')
    print(f'  AMES mutagenicity   : {data.get(\"AMES\", \"N/A\")}')
    print(f'  Hepatotoxicity      : {data.get(\"hepatotoxicity\", \"N/A\")}')
except Exception as e:
    print(f'API error: {e}. Try https://biosig.lab.uq.edu.au/pkcsm/ manually.')
"
```

### 5 — Similarity Search (PubChem)

```bash
# Find compounds similar to a known active by PubChem CID
CID="$1"
THRESHOLD=90
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/cid/${CID}/cids/JSON?Threshold=${THRESHOLD}&MaxRecords=10" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
cids = data.get('IdentifierList', {}).get('CID', [])
print(f'Similar compounds (Tanimoto >= ${THRESHOLD}%): {len(cids)} found')
print(f'CIDs: {cids[:10]}')
"
```

### 6 — QSAR Data Extraction (ChEMBL)

```bash
# Get IC50 data for a ChEMBL target for QSAR model building
TARGET_ID="$1"
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id=${TARGET_ID}&standard_type=IC50&pchembl_value__isnull=false&limit=20&format=json" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
acts = data.get('activities', [])
print(f'IC50 data for QSAR ({len(acts)} records):')
print(f'  {\"Molecule\":<18} {\"pIC50\":>6}  {\"IC50\":>10}  Assay')
print('  ' + '-'*60)
for a in acts:
    print(f'  {a.get(\"molecule_chembl_id\",\"N/A\"):<18} {str(a.get(\"pchembl_value\",\"N/A\")):>6}  {str(a.get(\"standard_value\",\"N/A\")):>10}  {a.get(\"assay_chembl_id\",\"N/A\")}')
"
```

---

## Reasoning Framework
Is the 3D protein structure known (X-ray/cryo-EM/NMR/AlphaFold)?
YES → Structure-Based (SB): Docking, MD, Virtual Screening (SB), Homology Modelling
NO  → Ligand-Based (LB): QSAR, Pharmacophore, Similarity Searching, Virtual Screening (LB)

When reasoning about a CADD problem:
1. Target known? → AlphaFold or PDB lookup first
2. Binding site identified? → Check literature
3. Known actives available? → Pull from ChEMBL
4. Library to screen? → ZINC (purchasable), PubChem (broad)
5. Computational resource? → Docking (regular PC), MD (GPU needed)

See references/CADD_METHODS.md for full method notes including docking score
interpretation, pLDDT thresholds, and QSAR descriptor guidance.

---

## Quick Reference

| Method | API | Free | Auth |
|--------|-----|------|------|
| Protein structure (predicted) | AlphaFold | Yes | None |
| Protein structure (experimental) | RCSB PDB | Yes | None |
| Gene to UniProt ID | UniProt REST | Yes | None |
| Compound library | ZINC | Yes | None |
| ADMET prediction | pkCSM | Yes | None |
| Similarity search | PubChem | Yes | None |
| QSAR/SAR data | ChEMBL | Yes | None |

Use alongside the drug-discovery skill for bioactivity and wet-lab workflows.
