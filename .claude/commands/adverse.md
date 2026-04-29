ADVERSARIAL COMPLETENESS CHECK for:

$ARGUMENTS

Score the audited area on three coverage dimensions. This is not just import tracing — it confirms actual execution, actual data flow, actual failure handling. The goal is to identify what was verified, what was inferred, and what could still hide a false negative.

DIMENSION 1 — SECURITY SURFACE COVERAGE:
1. List every catch block in the target area. For each:
   - What does it catch?
   - What does it return? (Check: never succeed from an error path — error states must return the restrictive default)
   - Is the error logged?
2. List every external input (user data, API response, DB read). For each:
   - Is it validated before use?
   - What happens if it's null/undefined/malformed?
3. List every error path. For each:
   - Does it fail restrictive (block/reject) or permissive (allow/accept)?
Score: {N}% of error paths verified

DIMENSION 2 — BUSINESS LOGIC COVERAGE:
1. List every inviolable rule (see CLAUDE.md → Inviolable Rules) that governs this area. For each:
   - Is it implemented as specified?
   - Grep evidence of implementation
2. List every conditional branch (if/else, switch, ternary). For each:
   - Is every branch reachable?
   - Is the else/default case correct?
3. List every pipeline boundary this area touches. For each:
   - What data crosses the boundary?
   - Is anything dropped or transformed?
Score: {N}% of business logic branches verified

DIMENSION 3 — INTEGRATION WIRING COVERAGE:
1. List every function called by this area. For each:
   - Does the called function actually exist and is it on the LIVE path?
   - Does the call site pass the correct arguments in the correct order?
2. List every DB read/write. For each:
   - Does the schema match what the code expects?
   - Are the column names correct?
3. List every event emission/consumption. For each:
   - Is the event name correct?
   - Is the payload shape correct?
Score: {N}% of integration points verified

OVERALL COVERAGE:
Security: {N}% | Business Logic: {N}% | Integration: {N}%
Weighted average: {N}%

GAPS IDENTIFIED:
{List anything NOT verified, with specific recommendations for how to verify it}

CONFIDENCE ASSESSMENT:
- Verified (mechanical proof): {N} items
- Inferred (logic, no grep): {N} items
- Unknown (not checked): {N} items
