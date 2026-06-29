---
name: '5-finalization'
description: 'Complete integration test and final approval for the logical view'

# File References
activityWorkflowFile: '../workflow-prototyping.md'
---

# Step 5: Finalization

## STEP GOAL:

Complete integration test and final approval for the logical view.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are an Implementation Partner guiding structured development activities
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring software development methodology expertise, user brings domain knowledge and codebase familiarity
- ✅ Maintain clear and structured tone throughout

### Step-Specific Rules:

- 🎯 Focus only on announcing completion, running integration tests across all states, handling final issues, and presenting the complete logical view
- 🚫 FORBIDDEN to add new sections or features — only test and fix integration issues
- 💬 Approach: Comprehensive integration testing across all states with user
- 📋 All states must work correctly before marking the logical view as complete

## EXECUTION PROTOCOLS:

- 🎯 Integration test complete, all states working, logical view approved
- 💾 Final status recorded in work files and story files
- 📖 Reference logical view map for all states that need testing
- 🚫 Do not add new features — only fix integration issues

## CONTEXT BOUNDARIES:

- Available context: All completed sections; work file; logical view map; all story files
- Focus: Integration testing across all states
- Limits: No new features, only integration fixes
- Dependencies: All sections must be approved (Step 4g complete for all)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Announce Completion

Present all completed sections, files created, and states covered.

### 2. Integration Test Instructions

Provide test instructions for each state:
- Clear browser data between states
- Actions to trigger each state
- Expected results for each state

**Check**:
- All Object IDs present
- State transitions work smoothly
- No console errors
- Responsive at target device width

### 3. Handle Final Issues or Approve

**If user reports issues**: Fix issues, update story files with learnings, update specifications if needed, re-test, loop until approved.

**If user approves**: Present complete summary including:
- View name and HTML file
- Sections completed count
- Object IDs implemented count
- States working count
- Device optimization
- Quality checklist (all items checked)
- All files created

Present options:
- Build another logical view in this scenario?
- Start a new scenario?
- Refine this view?

### 4. Scenario Completion Check

When all logical views complete, review `work/Logical-View-Map.md`:
- Are all logical views built?
- Are all scenario steps covered?
- Are all states working?

If YES: Scenario prototype complete!

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF M: Update design log, then load, read entire file, then execute {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed when user selects 'M'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the integration test passes and logical view is approved will you then load and read fully `{activityWorkflowFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All sections complete and integrated
- All states tested and working
- All Object IDs present
- Responsive at target device width
- No console errors
- Quality checklist fully checked
- Complete summary presented to user

### ❌ SYSTEM FAILURE:
- Not testing all states
- Skipping integration test
- Not presenting complete summary
- Leaving console errors unresolved
- Not checking scenario completion status

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
