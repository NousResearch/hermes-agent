# Molecular Docking Theory Reference

Deep theory behind each step of the docking pipeline.

---

## What Is Molecular Docking?

Molecular docking is an in silico technique that mimics and predicts the complex formation and interactability between biochemical elements. It predicts:

1. The binding pose — orientation and conformation of the ligand in the binding pocket
2. The binding affinity — estimated free energy of binding (delta-G, kcal/mol)

Types by partner:
- Protein-Ligand (primary focus in drug discovery)
- Protein-Protein (PPI modulators)
- Protein-RNA/DNA (nucleic acid-targeting drugs)
- Protein-Peptide (peptidomimetics)

---

## Blind vs Site-Specific Docking

Site-Specific Docking:
- Active site known from co-crystallized inhibitor
- Grid box placed precisely around the known binding pocket
- Faster, more accurate
- Used for: lead optimization, virtual screening, SAR studies

Blind Docking:
- Active site unknown (novel proteins, orphan receptors)
- Grid box covers the entire protein surface
- Much slower — samples full surface
- Used to discover the most plausible binding pocket
- Tools: fpocket, CASTp (web)

---

## PDB Structure Quality for Docking

| Resolution | Quality | For docking? |
|------------|---------|-------------|
| < 1.5 Angstrom | Atomic | Excellent |
| 1.5-2.0 Angstrom | Very high | Excellent |
| 2.0-2.5 Angstrom | High | Good — preferred |
| 2.5-3.0 Angstrom | Moderate | Acceptable |
| > 3.0 Angstrom | Low | Avoid if alternatives exist |

R-free < 0.25 = good quality model.
R-free > 0.30 = poor quality — sidechain positions unreliable.

AlphaFold vs Experimental:
- Experimental preferred — contains actual binding pocket geometry
- AlphaFold in apo state — may not show open pocket
- pLDDT > 90 at binding site: acceptable if no PDB exists
- pLDDT < 70 at binding site: do not use for docking

---

## Protein Preparation — Why Each Step Matters

Remove water molecules (HOH):
- Crystal waters occupy binding site but displaced by ligand physiologically
- Exception: structural/bridging waters — consult literature

Remove heteroatoms:
- Co-crystallized ligands, buffer molecules, detergents
- FIRST: record co-crystallized ligand coordinates (binding site anchor)

Add missing hydrogens:
- PDB files often lack hydrogen atoms
- Critical for H-bond geometry and partial charge calculation
- Standard pH: 7.4

PDBQT format:
- AutoDock format: standard PDB + partial charge (q) + AutoDock atom type (t)
- Protein: rigid (one conformation)
- Ligand: flexible (rotatable bonds identified and marked)

---

## Ligand Preparation — Force Fields

Universal Force Field (UFF):
- PyRx/Open Babel default for ligand minimization
- Good for drug-like organic molecules
- Minimizes bond lengths, angles, torsions, non-bonded terms

Why minimize before docking:
- 2D SMILES have no 3D coordinates
- Energy minimization generates a low-energy 3D conformation
- Starting from strained geometry gives poor docking results

Number of rotatable bonds:
- 5 or fewer: rigid — fast docking
- 6-10: moderate — good balance
- More than 10: highly flexible — slow docking, poor sampling

---

## Grid Box Generation

| Parameter | Site-specific | Blind docking |
|-----------|--------------|--------------|
| Center X,Y,Z | Co-crystallized ligand centroid | Protein geometric center |
| Size X,Y,Z | 20-25 Angstrom | 60-80 Angstrom |
| Spacing | 0.375 Angstrom | 0.375 Angstrom |
| Exhaustiveness | 8-16 | 16-32 |

---

## AutoDock Vina Scoring Function

Components:
1. Gauss steric: Gaussian-shaped steric repulsion
2. Repulsion: penalty for short-range clashes
3. Hydrophobic: attractive term between hydrophobic atoms
4. HBond: directional H-bond term
5. Torsion penalty: penalizes increase in rotatable bond flexibility

Search algorithm:
1. Random starting positions (exhaustiveness controls count)
2. Local optimization via gradient descent
3. Metropolis criterion to escape local minima
4. Clustering of similar poses (RMSD < 2 Angstrom = same cluster)
5. Ranking by score — output top N poses

---

## Results Evaluation

RMSD selection rule: choose most negative affinity WITH RMSD l.b. = 0.000

Binding Affinity to Ki conversion (at 298K, RT = 0.592 kcal/mol):
delta-G = RT x ln(Ki)

| delta-G (kcal/mol) | Ki |
|--------------------|----|
| -6.8 | 10 micromolar |
| -8.2 | 1 micromolar |
| -9.5 | 100 nanomolar |
| -11.5 | 1 nanomolar |

Practical thresholds:
| Score | Interpretation | Action |
|-------|---------------|--------|
| below -9.0 | Excellent | High priority |
| -7 to -9 | Good | Proceed |
| -5 to -7 | Moderate | Optimize scaffold |
| above -5 | Weak | Deprioritize |

Vina scores are estimates. Correlation with experimental IC50 is typically 0.5-0.7.
Always validate with MD simulation then experimental assay.

---

## Post-Docking Interaction Analysis

H-Bond Analysis:
- Identify all H-bond donor-acceptor pairs between ligand and protein
- Modifying ligand to strengthen H-bonds increases affinity

Hydrophobic Analysis:
- Hydrophobic residues (Leu, Ile, Val, Phe, Trp, Met, Ala) lining pocket
- Adding hydrophobic groups increases affinity but may hurt solubility

Pi-Pi Stacking:
- Aromatic rings stacking with Phe, Tyr, Trp, His residues
- Parallel-displaced or T-shaped geometry
- Distance: 3.5-5.5 Angstrom between ring centroids

Pharmacophore Extraction:
- H-bond acceptor at position X — must keep
- Hydrophobic group at position Y — can vary scaffold
- This becomes the pharmacophore model for designing analogs

---

## Standard Reporting Format

Target protein   : [Name] ([PDB ID], [resolution] Angstrom, [method])
Ligand           : [Name] ([ChEMBL ID or PubChem CID])
Docking software : AutoDock Vina via PyRx
Grid center      : X=[x], Y=[y], Z=[z]
Grid dimensions  : [n] x [n] x [n] Angstrom
Exhaustiveness   : [n]

Best binding mode:
  Affinity       : [X.X] kcal/mol
  RMSD l.b.      : 0.000 Angstrom

Key interactions:
  H-bonds        : [Ligand atom] to [Residue, distance Angstrom]
  Hydrophobic    : [Residue list]
  Pi-Pi stacking : [Residue, geometry]

---

## References

- Trott, O. and Olson, A.J. (2010). AutoDock Vina. J. Comput. Chem. 31, 455-461.
- Morris, G.M. et al. (2009). AutoDock4 and AutoDockTools4. J. Comput. Chem. 30, 2785-2791.
- Berman, H.M. et al. (2000). The Protein Data Bank. Nucleic Acids Res. 28, 235.
- Dallakyan, S. and Olson, A.J. (2015). Docking with PyRx. Methods Mol. Biol. 1263, 243-250.
