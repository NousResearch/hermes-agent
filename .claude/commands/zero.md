ZERO FALSE POSITIVES / ZERO FALSE NEGATIVES MODE — ACTIVATED

This mode stays active for the rest of the session. Every finding and every fix must meet both standards.

ZERO FALSE POSITIVES:
- Every finding must have file:line evidence from grep or file read.
- No finding is accepted based on inference alone.
- If you're not 100% certain, label it INFERRED and say what would confirm it.
- Before closing any finding, search for competing explanations.

ZERO FALSE NEGATIVES:
- Before concluding an investigation, actively search for what you might have missed.
- Check for: alternate code paths, shadow implementations, dead code that looks live, duplicate helpers, stale specs, feature flags that change behavior, fallback logic, and environment-specific branches.
- For every conditional branch in the target area, verify BOTH branches.
- For every error path, verify what it returns.

REQUIRED CHECKLIST (run before closing any finding):
□ Searched for competing explanations?
□ Checked alternate code paths?
□ Checked for shadow implementations or duplicate helpers?
□ Checked for dead code that might look live?
□ Checked for feature flags that change behavior?
□ Checked for fallback logic?
□ Checked for stale specs or comments that contradict the code?
□ Verified both branches of every conditional?
□ Verified every error/catch path returns the restrictive default?

If any checklist item is unchecked, the investigation is NOT complete.

ENFORCEMENT:
- When presenting findings, label each: CONFIRMED / INFERRED / UNVERIFIED
- When closing a finding, state which checklist items were verified
- When declaring "all clear," state the coverage explicitly: "checked N files, M functions, K error paths"
