---
name: molecular-docking
description: >
  End-to-end molecular docking assistant covering the complete AutoDock Vina /
  PyRx pipeline: protein retrieval and preparation from RCSB PDB, binding site
  identification from co-crystallized inhibitors, ligand sourcing from PubChem
  and ChEMBL, PDBQT format conversion guidance, grid-box setup, docking
  execution, RMSD-based results evaluation, and post-docking interaction
  analysis. Covers both site-specific and blind docking. Fetches real protein
  and ligand data via free public APIs — no auth required. Use for in silico
  screening, lead identification, binding affinity prediction, and
  structure-based drug design workflows.
version: 1.0.0
author: bennytimz
license: MIT
metadata:
  hermes:
    tags: [science, chemistry, docking, structural-biology, cadd, research]
    related_skills: [drug-discovery, computational-drug-discovery]
prerequisites:
  commands: [curl, python3]
---

# Molecular Docking

You are an expert computational chemist specialising in molecular docking and
structure-based drug design. You guide users through the full docking pipeline
from retrieving a target protein to interpreting post-docking interaction maps
using free, open-source tools and public databases.

The standard docking pipeline:
Protein Retrieval (RCSB PDB)
Protein Preparation (remove water/heteroatoms, add hydrogens)
Binding Site Identification (co-crystallized inhibitor to residues)
Ligand Sourcing (PubChem / ChEMBL / DrugBank / ZINC)
Ligand Preparation (energy minimization, UFF, PDBQT format)
Grid-Box Generation (confine search space around binding site)
Docking (AutoDock Vina — multiple poses, binding scores)
Results Evaluation (select RMSD = 0, most negative binding score)
Post-Docking Analysis (H-bonds, hydrophobic contacts, pi-stacking)

See references/DOCKING_THEORY.md for deep theory on each step.
See references/TOOLS_GUIDE.md for PyRx, AutoDock Vina, and Discovery Studio setup.
See scripts/docking_utils.py for helper scripts: PDB fetch, ligand center extraction, results parser.

---

## Docking Type — Establish This First

Site-Specific Docking — active site is known.
- A co-crystallized ligand or known inhibitor reveals the binding pocket.
- Grid box placed precisely around this pocket.
- Most common for lead optimization and virtual screening.

Blind Docking — active site is unknown (novel proteins, orphan targets).
- Grid box covers the entire protein surface.
- Much slower — searches everywhere.
- Used to discover the most plausible binding pocket.
- Follow up with site-specific docking once pocket is identified.

---

## Step-by-Step Workflows

### Step 1 — Protein Retrieval (RCSB PDB)

Always prefer experimentally determined structures (X-ray, cryo-EM) over AlphaFold for docking when available — they contain co-crystallized ligands that reveal the binding site.

```bash
# Search RCSB PDB for a target protein with resolution details
TARGET="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$TARGET")
curl -s "https://search.rcsb.org/rcsbsearch/v2/query" \
  -H "Content-Type: application/json" \
  -d "{\"query\":{\"type\":\"terminal\",\"service\":\"full_text\",\"parameters\":{\"value\":\"${TARGET}\"}},\"return_type\":\"entry\",\"request_options\":{\"paginate\":{\"start\":0,\"rows\":10}}}" \
  | python3 -c "
import json, sys, urllib.request
data = json.load(sys.stdin)
ids = [r['identifier'] for r in data.get('result_set', [])]
if not ids:
    print('No PDB structures found.')
    sys.exit()
print(f'Found {len(ids)} structures: {ids[:10]}')
print()
print(f'{\"PDB ID\":<8} {\"Method\":<12} {\"Resolution\":>12}  Title')
print('-' * 75)
for pdb_id in ids[:6]:
    try:
        # GET request to RCSB public API — read-only, no data transmitted
        detail = json.loads(urllib.request.urlopen(f'https://data.rcsb.org/rest/v1/core/entry/{pdb_id}', timeout=8).read())
        method = detail.get('exptl', [{}])[0].get('method', 'N/A')
        res    = detail.get('refine', [{}])[0].get('ls_d_res_high', 'N/A')
        title  = detail.get('struct', {}).get('title', 'N/A')[:50]
        flag   = ' GOOD' if res != 'N/A' and float(str(res)) < 2.5 else ''
        print(f'{pdb_id:<8} {method:<12} {str(res):>12}{flag}  {title}')
    except Exception:
        pass
print()
print('Resolution < 2.5 Angstrom = good for docking')
print('Download: https://files.rcsb.org/download/{PDB_ID}.pdb')
"
```

```bash
# Get full details for a specific PDB entry
PDB_ID="$1"
# GET request to RCSB public API — read-only, no data transmitted
curl -s "https://data.rcsb.org/rest/v1/core/entry/${PDB_ID}" \
  | python3 -c "
import json, sys
d = json.load(sys.stdin)
struct  = d.get('struct', {})
exptl   = d.get('exptl', [{}])[0]
refine  = d.get('refine', [{}])[0]
info    = d.get('rcsb_entry_info', {})
print(f'=== PDB Entry: $PDB_ID ===')
print(f'Title      : {struct.get(\"title\", \"N/A\")}')
print(f'Method     : {exptl.get(\"method\", \"N/A\")}')
print(f'Resolution : {refine.get(\"ls_d_res_high\", \"N/A\")} Angstrom')
print(f'R-free     : {refine.get(\"ls_r_factor_r_free\", \"N/A\")}')
print(f'Atoms      : {info.get(\"deposited_atom_count\", \"N/A\")}')
print(f'Residues   : {info.get(\"deposited_modeled_polymer_monomer_count\", \"N/A\")}')
print()
print('Download:')
print(f'  PDB : https://files.rcsb.org/download/$PDB_ID.pdb')
print()
print('Next: Open in Discovery Studio — identify HETATM records (co-crystallized ligand = binding site marker)')
"
```

### Step 2 — Ligand Sourcing

```bash
# Get ligand SMILES and properties from PubChem by name
# GET request to PubChem public API — read-only, no data transmitted
COMPOUND="$1"
ENCODED=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))" "$COMPOUND")
CID=$(curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/${ENCODED}/cids/TXT" | head -1 | tr -d '[:space:]')
echo "PubChem CID: $CID"
curl -s "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/${CID}/property/IsomericSMILES,MolecularFormula,MolecularWeight,XLogP,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,TPSA/JSON" \
  | python3 -c "
import json, sys
props = json.load(sys.stdin)['PropertyTable']['Properties'][0]
mw   = float(props.get('MolecularWeight', 0))
logp = float(props.get('XLogP', 0))
hbd  = int(props.get('HBondDonorCount', 0))
hba  = int(props.get('HBondAcceptorCount', 0))
rot  = int(props.get('RotatableBondCount', 0))
tpsa = float(props.get('TPSA', 0))
print(f'Formula  : {props.get(\"MolecularFormula\", \"N/A\")}')
print(f'SMILES   : {props.get(\"IsomericSMILES\", \"N/A\")}')
print(f'MW       : {mw} Da')
print(f'LogP     : {logp}')
print()
print('=== Lipinski Ro5 Pre-Check ===')
print(f'  MW   {mw:.0f} Da   {\"PASS\" if mw<=500 else \"FAIL >500\"}')
print(f'  LogP {logp:.2f}     {\"PASS\" if logp<=5 else \"FAIL >5\"}')
print(f'  HBD  {hbd}          {\"PASS\" if hbd<=5 else \"FAIL >5\"}')
print(f'  HBA  {hba}          {\"PASS\" if hba<=10 else \"FAIL >10\"}')
print(f'  TPSA {tpsa:.0f} A2   {\"PASS\" if tpsa<=140 else \"FAIL >140\"}')
print(f'  RotB {rot}          {\"PASS\" if rot<=10 else \"WARNING >10 high flexibility\"}')
v = sum([mw>500, logp>5, hbd>5, hba>10])
print(f'  Ro5 violations: {v}/4 {\"proceed to docking\" if v<=1 else \"poor oral bioavailability predicted\"}')
print()
print(f'3D SDF: https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/$CID/record/SDF?record_type=3d')
"
```

```bash
# Get top active compounds from ChEMBL for a target
# GET request to ChEMBL public API — read-only, no data transmitted
TARGET_CHEMBL="$1"
curl -s "https://www.ebi.ac.uk/chembl/api/data/activity?target_chembl_id=${TARGET_CHEMBL}&pchembl_value__gte=7&standard_type=IC50&limit=10&order_by=-pchembl_value&format=json" \
  | python3 -c "
import json, sys
data = json.load(sys.stdin)
acts = data.get('activities', [])
print(f'Top ChEMBL actives for $TARGET_CHEMBL (pIC50 >= 7):')
seen = set()
for a in acts:
    mid = a.get('molecule_chembl_id', 'N/A')
    if mid in seen: continue
    seen.add(mid)
    print(f'  {mid:<18} pIC50={a.get(\"pchembl_value\",\"N/A\")}  IC50={a.get(\"standard_value\",\"N/A\")} nM')
"
```

### Step 3 — Protein Preparation Guidance

The agent cannot run BIOVIA Discovery Studio or PyRx directly — these are GUI desktop tools. Guide the user step by step.

Protein Preparation Protocol:

1. Open PDB in Discovery Studio Visualizer (free from BIOVIA)
2. Identify co-crystallized ligand in HETATM records — note its 3-letter code
3. Record the ligand coordinates BEFORE deleting it — this is your grid box center
4. Delete waters (HOH), heteroatoms, and co-crystallized compounds
5. Add hydrogens at pH 7.4, add missing residues
6. Save as protein_prepared.pdb
7. In PyRx: Load molecule > right-click > Make Macromolecule > AutoDock Macromolecule (PDBQT)

Ligand Preparation in PyRx:
- Load SDF from PubChem or paste SMILES
- Select ligand > Minimize > Universal Force Field (UFF)
- Right-click > Make Ligand > AutoDock Ligand (PDBQT)

### Step 4 — Grid Box Coordinate Extraction

```bash
# Extract centroid of co-crystallized ligand from PDB file for grid box center
PDB_FILE="$1"
LIGAND_CODE="$2"
python3 -c "
import sys
pdb_file = sys.argv[1]
lig_code = sys.argv[2].upper()
coords = []
with open(pdb_file) as f:
    for line in f:
        if line.startswith(('HETATM', 'ATOM')):
            if line[17:20].strip() == lig_code:
                try:
                    coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
                except ValueError:
                    pass
if not coords:
    print(f'Ligand {lig_code} not found.')
    print('Available HETATMs:')
    seen = set()
    with open(pdb_file) as f:
        for line in f:
            if line.startswith('HETATM'):
                res = line[17:20].strip()
                if res and res != 'HOH' and res not in seen:
                    print(f'  {res}')
                    seen.add(res)
    sys.exit()
cx = sum(c[0] for c in coords)/len(coords)
cy = sum(c[1] for c in coords)/len(coords)
cz = sum(c[2] for c in coords)/len(coords)
xs=[c[0] for c in coords]; ys=[c[1] for c in coords]; zs=[c[2] for c in coords]
bx=max(20,round(max(xs)-min(xs)+20))
by=max(20,round(max(ys)-min(ys)+20))
bz=max(20,round(max(zs)-min(zs)+20))
print(f'Ligand {lig_code} — {len(coords)} atoms')
print(f'Grid Box Center: X={cx:.3f}  Y={cy:.3f}  Z={cz:.3f}')
print(f'Recommended Box: {bx} x {by} x {bz} Angstrom')
print()
print('Vina config.txt:')
print(f'  center_x = {cx:.3f}')
print(f'  center_y = {cy:.3f}')
print(f'  center_z = {cz:.3f}')
print(f'  size_x = {bx}')
print(f'  size_y = {by}')
print(f'  size_z = {bz}')
" "$PDB_FILE" "$LIGAND_CODE"
```

Grid Box Parameters:
- Center X, Y, Z = coordinates of the co-crystallized ligand centroid
- Size 20-25 Angstrom for site-specific docking
- Size 60-80 Angstrom for blind docking
- Exhaustiveness = 8 default, 16-32 for thorough search

### Step 5 — Results Evaluation

```bash
# Parse AutoDock Vina output log and rank results
VINA_LOG="$1"
python3 -c "
import sys, re, math
with open(sys.argv[1]) as f:
    content = f.read()
results = []
for m in re.finditer(r'^\s*(\d+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)', content, re.MULTILINE):
    mode, aff, rmsd_lb, rmsd_ub = m.groups()
    results.append({'mode':int(mode),'affinity':float(aff),'rmsd_lb':float(rmsd_lb),'rmsd_ub':float(rmsd_ub)})
if not results:
    print('No results parsed — check log format.')
    sys.exit()
print(f'{\"Mode\":>4}  {\"Affinity (kcal/mol)\":>20}  {\"RMSD l.b.\":>10}  {\"RMSD u.b.\":>10}  Notes')
print('-'*75)
for r in results:
    note = 'SELECT' if r['rmsd_lb']==0.0 else ('similar' if r['rmsd_lb']<1.0 else '')
    print(f\"{r['mode']:>4}  {r['affinity']:>20.1f}  {r['rmsd_lb']:>10.3f}  {r['rmsd_ub']:>10.3f}  {note}\")
best = min(results, key=lambda x: x['affinity'])
RT = 0.592
ki_nM = math.exp(best['affinity']/RT)*1e9
score = best['affinity']
if score < -9.0: interp = 'Excellent — strong binding (nanomolar). High priority.'
elif score < -7.0: interp = 'Good binding. Proceed to ADMET and MD validation.'
elif score < -5.0: interp = 'Moderate. Consider structural optimization.'
else: interp = 'Weak binding. Deprioritize or redesign scaffold.'
print()
print(f'Best: Mode {best[\"mode\"]}  {best[\"affinity\"]} kcal/mol  RMSD={best[\"rmsd_lb\"]:.3f}')
print(f'Estimated Ki: ~{ki_nM:.1f} nM  (delta-G = RT x ln(Ki), T=298K)')
print(f'Assessment: {interp}')
print()
print('Selection rule: most negative affinity WITH RMSD l.b. = 0.000')
" "$VINA_LOG"
```

### Step 6 — Post-Docking Analysis Guidance

After selecting the best pose, analyse interactions in Discovery Studio or PyMOL.

Key interaction types:

| Interaction | Distance | Significance |
|-------------|----------|--------------|
| H-bond | 2.5-3.5 Angstrom | Strong, directional — critical for binding |
| Hydrophobic | 3.5-5.0 Angstrom | Drives affinity via desolvation |
| Pi-Pi stacking | 3.5-5.5 Angstrom | Aromatic rings parallel or T-shaped |
| Cation-Pi | 3.5-6.0 Angstrom | Charged residue + aromatic ring |
| Salt bridge | 2.5-4.0 Angstrom | Charge-charge interaction |

Discovery Studio post-docking workflow:
1. File > Import Hierarchy > load receptor.pdbqt + best pose.pdbqt
2. Receptor-Ligand Interactions > Show Interactions
3. View > 2D Interaction Diagram > Export
4. Record every residue forming H-bonds — these are your pharmacophore points

What to report:
- Key binding residues (residue name + number e.g. Thr790, Lys745)
- Critical H-bonds: donor atom to acceptor atom, distance in Angstrom
- Hydrophobic pocket residue list
- Binding score: X.X kcal/mol (Mode N, RMSD = 0.000)

---

## Worked Example — Imatinib vs ABL Kinase (PDB: 1IEP)

Target   : BCR-ABL tyrosine kinase
PDB      : 1IEP (2.10 Angstrom, X-ray crystallography)
Ligand   : STI (Imatinib / Gleevec)
Grid box : Centered on STI coordinates in ATP binding pocket
Result   : Affinity approximately -9.5 to -11 kcal/mol
Key contacts: H-bonds to Thr315, Met290, Asp381
              Hydrophobic: Ile293, Leu248, Val256

To reproduce:
1. Search PDB for 1IEP using Step 1 workflow
2. Run: python3 scripts/docking_utils.py ligand-center 1IEP.pdb STI
3. Get Imatinib SMILES from PubChem CID 5291
4. Dock in PyRx, evaluate, analyse interactions

---

## Common Errors and Fixes

| Problem | Cause | Fix |
|---------|-------|-----|
| No poses generated | Grid box wrong location | Re-check center coordinates |
| All RMSD > 2 | Low exhaustiveness | Increase to 16 or 32 |
| Positive score | Ligand clashes with protein | Check for unremoved waters or heteroatoms |
| Ligand escapes box | Box too small | Expand by 5-10 Angstrom each dimension |
| PDBQT error | Non-standard atoms | Remove metals, use standard residues |

---

## Decision Guide

Have the target PDB structure?
- YES — Site-specific docking
  - Has co-crystallized ligand? YES — Use ligand centroid as grid box center
  - Has co-crystallized ligand? NO — Use CASTp or fpocket to predict pocket
- NO — AlphaFold predicted structure available?
  - pLDDT > 70 at binding region — Blind docking first
  - pLDDT < 70 — Use ligand-based methods from computational-drug-discovery skill

---

## Quick Reference APIs

| Task | API | URL |
|------|-----|-----|
| Search structures | RCSB PDB | search.rcsb.org/rcsbsearch/v2/query |
| PDB entry details | RCSB Data | data.rcsb.org/rest/v1/core/entry/{ID} |
| Download PDB | RCSB Files | files.rcsb.org/download/{ID}.pdb |
| Ligand SMILES | PubChem | pubchem.ncbi.nlm.nih.gov/rest/pug/... |
| Active compounds | ChEMBL | ebi.ac.uk/chembl/api/data/activity |

All free. No authentication required.
Related skills: drug-discovery, computational-drug-discovery
