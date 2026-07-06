# Test Fixtures: Implementation Annex Generator

This directory contains test fixtures for the `implementation_annex_generator` module.

## ADHD-I Fixture Set

| File | Type | Source |
|------|------|--------|
| `adhd_i_decision_report.md` | Final decision report (contract format) | Real ADHD-I decision, anonymized and trimmed |
| `adhd_i_external_calibration.md` | External calibration | Real ADHD-I calibration, anonymized |
| `adhd_i_execution_annex_expected.md` | Expected generated annex | Hand-curated expected output |

### Usage

The fixtures are designed for the `child_adhd_education` domain profile and
represent a decision about whether to intervene for a 7-year-old child with
ADHD-I, intervention intensity, parent behavior training, and Grade 3
preparation.

### Notes

- Fixtures contain internal evidence-tier labels (e.g., `evidence_supported`)
  and internal terminology that the generator must map to user-facing language.
- Fixtures contain no drug names, dosages, or treatment instructions.
- The expected annex was manually verified against the quality gate.
