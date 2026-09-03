# Validation Test — Cross-Tradition Resemblance Rejected as Historical Equivalence

**Input**: "A scene involves a circular motif and a spiral path. A character suggests it 'means' unity across cultures. Diagnose and repair."

**Expected Behavior**:
- Resemblance noted but tagged correctly (e.g., FORMAL_RESEMBLANCE or MODERN_SYNTHESIS).
- Not treated as HISTORICALLY_DERIVED or equivalent.
- Separate provenance for circle in one tradition (e.g., Eliade sacred space) vs spiral in another (e.g., geometric or alchemical circulation).
- Repair: Use the forms for dramatic function (enclosure vs irreversible change) without claiming shared origin.
- Quality gate "tradition_flattening" or "unsupported equivalence" flagged.

**Pass Criteria**: Explicit relationship type used; no equivalence claimed without evidence.
