# ADMET Reference Guide

Comprehensive reference for Absorption, Distribution, Metabolism, Excretion, and Toxicity (ADMET) analysis in drug discovery.

## Drug-Likeness Rule Sets

### Lipinski's Rule of Five (Ro5)

| Property | Threshold |
|----------|-----------|
| Molecular Weight (MW) | ≤ 500 Da |
| Lipophilicity (LogP) | ≤ 5 |
| H-Bond Donors (HBD) | ≤ 5 |
| H-Bond Acceptors (HBA) | ≤ 10 |

Reference: Lipinski et al., Adv. Drug Deliv. Rev. 23, 3–25 (1997).

### Veber's Oral Bioavailability Rules

| Property | Threshold |
|----------|-----------|
| TPSA | ≤ 140 Å² |
| Rotatable Bonds | ≤ 10 |

Reference: Veber et al., J. Med. Chem. 45, 2615–2623 (2002).

### CNS Penetration (BBB)

| Property | CNS-Optimal |
|----------|-------------|
| MW | ≤ 400 Da |
| LogP | 1–3 |
| TPSA | < 90 Å² |
| HBD | ≤ 3 |

## CYP450 Metabolism

| Isoform | % Drugs | Notable inhibitors |
|---------|---------|-------------------|
| CYP3A4 | ~50% | Grapefruit, ketoconazole |
| CYP2D6 | ~25% | Fluoxetine, paroxetine |
| CYP2C9 | ~15% | Fluconazole, amiodarone |
| CYP2C19 | ~10% | Omeprazole, fluoxetine |
| CYP1A2 | ~5% | Fluvoxamine, ciprofloxacin |

## hERG Cardiac Toxicity Risk

Structural alerts: basic nitrogen (pKa 7–9) + aromatic ring + hydrophobic moiety, LogP > 3.5 + basic amine.

Mitigation: reduce basicity, introduce polar groups, break planarity.

## Common Bioisosteric Replacements

| Original | Bioisostere | Purpose |
|----------|-------------|---------|
| -COOH | -tetrazole, -SO₂NH₂ | Improve permeability |
| -OH (phenol) | -F, -CN | Reduce glucuronidation |
| Phenyl | Pyridine, thiophene | Reduce LogP |
| Ester | -CONHR | Reduce hydrolysis |

## Key APIs

- ChEMBL: https://www.ebi.ac.uk/chembl/api/data/
- PubChem: https://pubchem.ncbi.nlm.nih.gov/rest/pug/
- OpenFDA: https://api.fda.gov/drug/
- OpenTargets GraphQL: https://api.platform.opentargets.org/api/v4/graphql
