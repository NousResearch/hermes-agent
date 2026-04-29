ROOT-CAUSE MATRIX for:

$ARGUMENTS

Do NOT fix anything yet. First produce a ranked root-cause matrix. The goal is to distinguish between correlation and actual causation — the first apparent culprit may not be the real failure point.

STEP 1 — DESCRIBE THE SYMPTOM:
What exactly is the observed behavior? Be precise. Include:
- What happened (actual)
- What should have happened (expected)
- Where in the system (module, file, function)
- How it was discovered (audit, test, query, user report)

STEP 2 — GENERATE CANDIDATE CAUSES:
List every plausible cause. Be generous — include unlikely ones. For each candidate:
- What code path would produce this symptom?
- What data condition would trigger it?
- Is this a code bug, a data bug, a config bug, or a spec bug?

STEP 3 — BUILD THE MATRIX:

| # | Candidate Cause | Evidence FOR | Evidence AGAINST | What Would Falsify | Likelihood |
|---|----------------|-------------|-----------------|-------------------|------------|
| 1 | {cause} | {supporting evidence with file:line} | {contradicting evidence} | {specific test/grep that would disprove} | HIGH/MED/LOW |
| 2 | ... | ... | ... | ... | ... |

STEP 4 — RANK AND RECOMMEND:
1. Which cause has the strongest evidence FOR and weakest evidence AGAINST?
2. Which cause is most dangerous if left unfixed (even if less likely)?
3. What is the recommended investigation order? (check highest-danger first, or easiest-to-falsify first?)

STEP 5 — PROPOSE VERIFICATION PLAN:
For the top-ranked cause:
- What grep would confirm it?
- What query would show it in the data?
- What test would reproduce it?
- What would a fix look like? (do NOT implement — just describe)

Do NOT proceed to implementation. Present the matrix and wait for the operator's decision on which cause to pursue.
