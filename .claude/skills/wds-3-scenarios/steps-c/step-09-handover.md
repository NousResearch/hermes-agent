---
name: step-09-handover
description: Complete Phase 3 and prepare for Phase 4 UX Design

# File References
workflowFile: '../workflow.md'
---

# Step 9: Handover

## STEP GOAL:

Complete Phase 3 by presenting a final summary, guiding the user through design intent selection for each scenario, explaining what comes next in Phase 4, and updating the design log.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a UX Scenario Facilitator collaborating with the project owner
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring scenario thinking and user journey expertise, user brings their project knowledge, together we create concrete UX scenario outlines
- ✅ Maintain collaborative equal-partner tone throughout

### Step-Specific Rules:

- 🎯 Focus on completion summary, design intent selection, and handover
- 🚫 FORBIDDEN to end without presenting design intent options for each scenario
- 💬 Approach: Celebrate completion while providing clear next steps
- 📋 Save design intent choices to scenario frontmatter

## EXECUTION PROTOCOLS:

- 📋 Present comprehensive completion summary
- 🎯 Guide user through design intent selection per scenario
- 💾 Save design intent and status to scenario files
- 📖 Explain Phase 4 approaches clearly
- 🚫 FORBIDDEN to end workflow without proper completion

## CONTEXT BOUNDARIES:

- Available context: All Phase 3 artifacts, quality scores, design log
- Focus: Phase completion and Phase 4 preparation
- Limits: No scenario modifications, only status updates
- Dependencies: Design log must be updated from Step 8

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present Completion Summary

```
## Phase 3: UX Scenarios Complete ✓

**Project:** [project_name]
**Scenarios Created:** [N]

### Scenario List
| ID | Name | Persona | Pages | Priority |
|----|------|---------|-------|----------|
| 01 | [Name] | [Persona] | [N] | P1 |
| 02 | [Name] | [Persona] | [N] | P1 |
| ... | ... | ... | ... | ... |

### Coverage
- **Total pages:** [N]
- **Pages in scenarios:** [N]
- **Coverage:** [100% / X%]

### Quality
- **All scenarios pass quality review:** [Yes/No]
- **Overall quality:** [Excellent / Good]

### Files Created
- `C-UX-Scenarios/00-ux-scenarios.md` — Scenario index
- `C-UX-Scenarios/01-[slug]/01-[slug].md` — [Scenario name]
- `C-UX-Scenarios/02-[slug]/02-[slug].md` — [Scenario name]
[list all...]
```

### 2. Design Intent Selection

Before handing over to Phase 4, help the user choose a design approach for each scenario.

Present:

```
Your scenarios are ready for design. How would you like to approach each?

| # | Scenario | Approach |
|---|----------|----------|
| 01 | [Name] | [K] [C] [S] [D] [L] |
| 02 | [Name] | [K] [C] [S] [D] [L] |
| ... | ... | ... |

**Approaches:**
[K] Sketch — I will draw it myself, agent interprets later
[C] Discuss — Creative dialog for page design
[S] Suggest — Agent proposes step by step, I confirm each
[D] Dream Up — Agent creates the whole flow, I review the result
[L] Later — Decide when I start Phase 4
```

For each scenario, save the chosen approach as `design_intent` in the scenario output file:
- Add `design_intent: [K|C|S|D|L]` to the scenario frontmatter
- Add `design_status: not-started` to track progress

### 3. What Comes Next

Explain to user:

```
**Next Steps:**

**Phase 4: UX Design** takes each scenario outline and designs it using your chosen approach:
- **Sketch** scenarios wait for your drawings
- **Discuss** scenarios start with a creative dialog for each page
- **Suggest** scenarios let the agent propose step by step
- **Dream Up** scenarios let the agent create autonomously

You can always change approach in Phase 4.

Would you like to continue to Phase 4, or take a break?
```

### 4. Update Design Log (If Exists)

If tracking via design log:
- Mark Phase 3 as complete
- Log scenario count and quality scores
- Note any user adjustments made during the process

### 5. Present MENU OPTIONS

Display: "[M] Main Menu — Return to workflow start"

#### Menu Handling Logic:

- IF M: Load, read entire file, then execute {workflowFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY complete workflow when user selects 'M' or indicates they want to stop
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [M main menu option] is selected and [design intent captured for all scenarios], will the workflow end gracefully with Phase 3 complete and Phase 4 prepared.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Comprehensive completion summary presented
- Design intent selection offered for each scenario
- Design intent and status saved to scenario frontmatter
- Phase 4 approaches clearly explained
- Design log updated if applicable
- User informed of next steps
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Not presenting completion summary
- Skipping design intent selection
- Not saving design intent to scenario files
- Ending without explaining next steps
- Not updating design log when one exists

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
