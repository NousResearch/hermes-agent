---
name: '3-logical-view-breakdown'
description: 'Select a logical view to build and break it into implementable sections'

# File References
nextStepFile: './4a-announce-and-gather.md'
---

# Step 3: Logical View Selection & Section Breakdown

## STEP GOAL:

Select a logical view to build and break it into implementable sections. This creates the work plan, but NOT the story files yet (those are created just-in-time).

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

- 🎯 Focus only on confirming view selection, gathering objects from specs, proposing section breakdown, and creating the work file
- 🚫 FORBIDDEN to create story files or begin implementing — those are later steps
- 💬 Approach: Collaboratively break down the view into 4-8 sections with user approval
- 📋 Group objects logically, consider all states, estimate time per section

## EXECUTION PROTOCOLS:

- 🎯 Work file created with approved section breakdown
- 💾 Create `work/[View-Name]-Work.yaml` with section plan
- 📖 Read all scenario step specs that use the selected logical view
- 🚫 Do not create story files or write any HTML/JS

## CONTEXT BOUNDARIES:

- Available context: Logical view map from Step 2; all scenario step specifications
- Focus: Section breakdown planning — objects, grouping, estimation
- Limits: No story files, no implementation
- Dependencies: Step 2 must be complete (logical view map exists)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Confirm Logical View Selection

**User says**: "Let's build [Logical View Name]" or selects from list

### 2. Gather All Specifications

**Actions**:

1. **Read all scenario step specs** that use this logical view
2. **Extract all Object IDs** across all states
3. **Identify unique objects** vs **state-specific objects**
4. **Note functional requirements** from all specs

Present objects to user for confirmation.

### 3. User Confirms Objects

**If user says "N"**: Ask what's missing or should be removed, update, re-present
**If user says "Y"**: Continue to section breakdown

### 4. Propose Section Breakdown

**Actions**:

1. **Group objects logically** into 4-8 sections
2. **Consider all states** when grouping
3. **Estimate time** per section

Present breakdown to user for approval.

### 5. User Reviews Section Breakdown

**If user says "N"**: Ask what needs adjustment, revise, re-present

### 6. Create Work File

**When user approves**: Create `work/[View-Name]-Work.yaml` with section details, statuses, and estimates.

### 7. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4a: Announce and Gather"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the section breakdown is approved and work file is created will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Logical view selection confirmed
- All objects extracted from specifications
- User confirmed object list
- Section breakdown approved (4-8 sections)
- Work file created with section plan

### ❌ SYSTEM FAILURE:
- Creating story files before work file is approved
- Beginning implementation before planning
- Not getting user approval on section breakdown
- Not extracting all objects from specs

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
