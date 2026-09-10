# CADD Methods Reference Guide

Eight computational drug design methods organized by the SB/LB framework.

## The SB vs LB Decision

Structure-Based (SB): Requires 3D protein coordinates. Sources: X-ray (RCSB PDB), cryo-EM, NMR, AlphaFold. Advantage: precise — you see how the ligand fits the binding pocket.

Ligand-Based (LB): No protein structure needed. Uses known active compounds to find new ones. Advantage: faster to start.

## Method 1 — Molecular Docking

Purpose: Predict how a ligand binds to a target — orientation, conformation, affinity.

How it works:
1. Obtain protein structure (PDB or AlphaFold)
2. Define the binding site
3. Generate ligand poses
4. Score by estimated binding energy (more negative = better)
5. Rank and select top poses

Docking score interpretation:
- More negative delta-G = more favorable binding
- Typical drug scores: -7 to -12 kcal/mol
- Validate with MD or experiment — scores alone are insufficient

Computer requirement: Regular PC (AutoDock Vina runs on CPU)

## Method 2 — Molecular Dynamics (MD)

Purpose: Simulate time-dependent behavior of protein-ligand complex. Answers: is the docked pose stable over time?

Key outputs:
- RMSD: Root Mean Square Deviation. Low RMSD (<2 Angstrom) = stable complex
- Binding free energy via MM-GBSA/MM-PBSA post-MD
- Conformational sampling reveals induced fit effects

Computer requirement: Strong PC or GPU cluster (GROMACS, AMBER — open source)

Use MD to validate docking hits before synthesis. If RMSD spikes early, the pose was wrong.

## Method 3 — Homology Modelling

Purpose: Build a 3D model of a protein whose structure is experimentally unknown, using a related protein as template.

When to use:
- Target has no PDB entry
- AlphaFold pLDDT < 70 for the binding region
- Need structure for docking but none exists

Quality thresholds:
- Sequence identity > 30% = reliable model
- 20-30% = twilight zone, use with caution
- < 20% = very unreliable, prefer AlphaFold

Free tools: SWISS-MODEL (web), Modeller, RoseTTAFold
API: AlphaFold API is often faster and better

## Method 4 — Virtual Screening

Purpose: Screen large compound libraries (thousands to millions) computationally before lab testing.

SB Virtual Screening (structure known):
- Dock entire library against protein binding site
- Score and rank — select top compounds for testing
- Library sizes: 10,000 to 10,000,000+ compounds

LB Virtual Screening (no structure):
- Compare library to known actives by fingerprint similarity
- Pharmacophore-based filtering
- Faster than SB but less precise

Free resources: ZINC (750M purchasable), PubChem, ChEMBL

## Method 5 — QSAR

Purpose: Mathematical models correlating molecular structure (descriptors) with biological activity.

Principle: Similar structures = similar activities.

Workflow:
1. Collect actives + inactives with measured IC50 from ChEMBL
2. Calculate molecular descriptors (MW, LogP, TPSA, fingerprints)
3. Train ML model (random forest, SVM, neural net)
4. Validate: R2 > 0.6 training, Q2 > 0.5 cross-validation
5. Predict activity of new compounds

Free tools: RDKit (descriptors), scikit-learn (ML)
Data: ChEMBL (activity data), PubChem (properties)

## Method 6 — Pharmacophore Modelling

Purpose: Identify essential 3D chemical features (HBD, HBA, hydrophobic, charge) required for activity.

Structure-based pharmacophore (protein known):
- Features derived from protein-ligand interactions
- More precise — grounded in actual binding geometry

Ligand-based pharmacophore (no protein):
- Align multiple active compounds in 3D
- Find common features across all actives

Pharmacophoric features: HBA, HBD, Hydrophobic, Positive ionizable, Negative ionizable, Aromatic

## Method 7 — Similarity Searching

Purpose: Find new actives by structural similarity to a known hit. Simplest and fastest LB method.

Tanimoto coefficient: 0 (no similarity) to 1 (identical)
Threshold: Tc > 0.85 (conservative), > 0.70 (broader search)

Free tools: PubChem similarity API, ChEMBL similarity search

## pLDDT Confidence Thresholds (AlphaFold)

| pLDDT | Confidence | Use for docking? |
|-------|-----------|-----------------|
| > 90 | Very high | Yes — reliable backbone and sidechains |
| 70-90 | High | Yes — backbone reliable |
| 50-70 | Low | Caution — validate with MD |
| < 50 | Very low | No — likely disordered region |

Always check pLDDT at binding site residues specifically, not just the global average.

## Full CADD Pipeline

Target identification
→ Protein structure retrieval (AlphaFold / RCSB PDB)
→ Binding site identification (literature)
→ Known actives from ChEMBL
→ Virtual Screening (SB: docking) or (LB: pharmacophore/QSAR)
→ ADMET filtering (pkCSM)
→ Similarity search for analogs (PubChem)
→ MD simulation to validate stability
→ Synthesis and wet-lab testing

## References

- Jumper et al. (2021). Highly accurate protein structure prediction with AlphaFold. Nature 596, 583-589.
- Berman et al. (2000). The Protein Data Bank. Nucleic Acids Res. 28, 235-242.
- Irwin & Shoichet (2005). ZINC — a free database of commercially available compounds. J. Chem. Inf. Model. 45, 177-182.
- Pires et al. (2015). pkCSM: predicting small-molecule pharmacokinetic and toxicity properties. J. Med. Chem. 58, 4066.
- Lipinski et al. (1997). Experimental and computational approaches to estimate solubility and permeability. Adv. Drug Deliv. Rev. 23, 3-25.
