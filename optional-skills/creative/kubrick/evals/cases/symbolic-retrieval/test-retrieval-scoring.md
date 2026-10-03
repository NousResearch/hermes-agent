# Test — Retrieval Scoring and NOT_COMPUTABLE

**Input**: Dramatic problem with very low dramatic_fit and high cliché_risk for all candidate patterns (e.g., "add a mysterious black bird to a light comedy").

**Expected**:
- Compute full retrieval_score for candidates.
- Best score < 0.55 → return NOT_COMPUTABLE.
- Recommend simpler non-symbolic approach or request more context.
- Do not force a weak pattern.
