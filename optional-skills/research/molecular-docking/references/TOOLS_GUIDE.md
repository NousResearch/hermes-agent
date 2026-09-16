# Molecular Docking Tools Guide

---

## Tool Stack

Discovery Studio Visualizer  — Protein preparation + post-docking analysis
Open Babel (via PyRx)        — Ligand preparation + format conversion
AutoDock Vina (via PyRx)     — Grid generation + docking engine
PyRx                         — Unified GUI integrating all above

All tools are free. No commercial license required for academic use.

---

## PyRx

Download: https://pyrx.sourceforge.io/downloads

Workflow:
1. File > Load Molecule > protein PDB
2. Right-click > Make Macromolecule > AutoDock Macromolecule (PDBQT)
3. File > Load Molecule > ligand SDF or SMILES
4. Select ligand > Minimize > Universal Force Field (UFF)
5. Right-click > Make Ligand > AutoDock Ligand (PDBQT)
6. AutoDock Vina Wizard tab > select macromolecule and ligand
7. Set Grid Box center from co-crystallized ligand coordinates
8. Run Vina > Export results as CSV

Common issues:
- Cannot load molecule: ensure PDB is clean (no alt conformations)
- Vina not found: reinstall PyRx with admin rights (Windows)
- Blank grid box: switch to AutoDock Vina tab before loading molecules

---

## AutoDock Vina (Command Line)

Download: https://vina.scripps.edu/downloads/

Config file (config.txt):

receptor = protein.pdbqt
ligand   = ligand.pdbqt
center_x = 10.500
center_y = 25.300
center_z = -8.100
size_x = 20
size_y = 20
size_z = 20
exhaustiveness = 8
num_modes = 9
energy_range = 3

Run: vina --config config.txt --out result.pdbqt --log result.log

Batch docking:
```bash
for LIGAND in ligands/*.pdbqt; do
    NAME=$(basename "$LIGAND" .pdbqt)
    vina --receptor protein.pdbqt --ligand "$LIGAND" --config config.txt \
         --out "results/${NAME}_out.pdbqt" --log "results/${NAME}.log"
done
```

---

## Open Babel

Install:
- Ubuntu/Debian: sudo apt-get install openbabel
- macOS: brew install open-babel
- Windows: https://openbabel.org/wiki/Get_Open_Babel

Common conversions:
```bash
obabel -:"CC(=O)Oc1ccccc1C(=O)O" --gen3d -O aspirin.sdf
obabel aspirin.sdf -O aspirin.pdbqt
obabel protein.pdb -O protein.pdbqt -xr
obabel result.pdbqt -O result.sdf
obabel protein.pdb -O protein_H.pdb -p 7.4
```

---

## BIOVIA Discovery Studio Visualizer

Download: https://www.3ds.com/products/biovia/discovery-studio (free Visualizer version)

Protein preparation:
1. File > Open > PDB file
2. Identify HETATM group in left panel — note 3-letter ligand code
3. View > Spreadsheet for X,Y,Z coordinates of ligand atoms
4. Select Water > Delete, Select HETATMs > Delete (after recording coordinates)
5. Structure > Add Hydrogen (pH 7.4) > Add Missing Atoms
6. File > Save As > PDB

Post-docking:
1. Import receptor.pdbqt and best_pose.pdbqt
2. Merge both structures
3. Receptor-Ligand Interactions > Show Interactions
4. View > 2D Interaction Diagram > Export

---

## Web-Based Alternatives (No Installation)

CB-Dock2: https://cadd.labshare.cn/cb-dock2/ — blind docking with pocket prediction
SwissDock: https://www.swissdock.ch/ — AutoDock + CHARMM
DockThor: https://dockthor.lncc.br/ — GPU-accelerated

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| PDBQT has no receptor atoms | Re-run: obabel protein.pdb -O protein.pdbqt -xr |
| Grid box center is 0,0,0 | Run docking_utils.py ligand-center to extract coordinates |
| Positive docking score | Protein preparation error — check for unremoved waters |
| Only 1 mode returned | Increase exhaustiveness, check num_modes setting |
| Out of memory | Reduce grid box size; use machine with more than 8GB RAM |
