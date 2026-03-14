---
name: one-three-one-rule
description: Enforces the 1-3-1 communication framework for clear, concise, and structured agent responses.
category: communication
---

### 1-3-1 Communication Protocol

This skill ensures all responses adhere to a strict 1-3-1 format: Problem, Options, Recommendation, followed by Definition of Done (DoD) and Implementation Plan tailored to the chosen option.

### Rules:

1.  **Problem:**
    *   Must be a single, concise sentence.
    *   Must state the desired outcome or the core issue.
    *   Must NOT contain the word "and".
    *   Must NOT mention implementation details (e.g., specific tools, data sources).
    *   Example: `Linear issue tickets must be accessible for immediate work.`

2.  **Options:**
    *   Present exactly three distinct, viable options for solving the problem.
    *   Each option (A, B, C) should have a brief description, pros, and cons.

3.  **Recommendation:**
    *   Clearly state the single best option based on current context, user priorities, or efficiency.
    *   This is the agent's primary suggestion.

4.  **Definition of Done (DoD):**
    *   This section details the specific success criteria for the **recommended** option.
    *   It outlines what constitutes a "completed" task for that particular path.
    *   *Note:* If the user selects a different option, the DoD must be updated to reflect the success criteria for *that* newly recommended option.

5.  **Implementation Plan:**
    *   This section details the concrete steps required to achieve the **recommended** option.
    *   It outlines the tooling, commands, and workflow.
    *   *Note:* Similar to the DoD, the Implementation Plan must be updated if a different option becomes the recommendation.

### Skill Usage:

- When presented with a task requiring a structured approach, invoke this skill.
- First, state the clear, concise **Problem**.
- Then, present Options (A, B, C) with their trade-offs.
- State the **Recommendation** (your chosen option).
- Finally, provide the Definition of Done and Implementation Plan specifically aligned with the **recommended** option.
- If the user selects a different option, revise the Recommendation, DoD, and Implementation Plan accordingly.

### Example Interaction:

User: I need to integrate with Linear.

Agent:
1) Problem: Linear issue tickets must be accessible for immediate work.

2) Options:
   - Option A: Full GraphQL bootstrap (projects + all issues + all milestones) in one pass.
     Pros: Complete initial state; one-shot setup.
     Cons: More data fetched upfront; longer bootstrap.
   - Option B: GraphQL bootstrap (projects + issues) + periodic milestone refresh.
     Pros: Fast initial load; data stays fresh with regular refresh.
     Cons: Requires a scheduler; more moving parts.
   - Option C: GraphQL bootstrap (projects + issues) + on-demand milestones (no ongoing refresh).
     Pros: Fastest path to start work; simplest architecture; milestones retrieved only when needed.
     Cons: Milestones not pre-fetched; potential on-demand latency if needed soon after bootstrap.

3) Recommendation: Option C. This is the simplest, fastest path to get real work started and minimizes upfront effort.

4) Definition of Done (DoD) for Option C:
   - GraphQL bootstrap retrieves projects and issues, stored in persistent memory.
   - Memory persists to `~/.hermes/linear_memory.json`.
   - `graphql.py` includes `get-issues --project-id <id>` for on-demand fetching.
   - Live proof shows sample project/issue data and bootstrap counts.
   - Clear path to start work is established.

5) Implementation Plan for Option C:
   - Step 1: Consolidate into `graphql.py`. Keep `mcp.py` dormant.
   - Step 2: Configure `~/.hermes/linear_memory.json` for projects and issues.
   - Step 3: Implement `graphql.py bootstrap`.
   - Step 4: Implement `graphql.py get-issues`.
   - Step 5: Provide live proof.
   - Step 6: Save approach.

### Note on Dynamic Options:
If the user selects Option A or B, the Recommendation, DoD, and Implementation Plan will be adjusted accordingly, based on the chosen option's specifics.
