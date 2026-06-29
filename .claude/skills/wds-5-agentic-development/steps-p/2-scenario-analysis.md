---
name: '2-scenario-analysis'
description: 'Analyze the entire scenario to identify all logical views and map which scenario steps use which views'

# File References
nextStepFile: './3-logical-view-breakdown.md'
---

# Step 2: Scenario Analysis & Logical View Identification

## STEP GOAL:

Analyze the entire scenario to identify all logical views and map which scenario steps use which views. A "logical view" is a conceptual page/screen with multiple states.

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

- 🎯 Focus only on reading all scenario step specs, identifying logical views, getting user confirmation, and creating the logical view map
- 🚫 FORBIDDEN to begin building any views or breaking them into sections — that is the next step
- 💬 Approach: Present logical view mapping to user for review and confirmation
- 📋 Multiple scenario steps can use the same logical view with different states

## EXECUTION PROTOCOLS:

- 🎯 Complete logical view map with all views identified and confirmed by user
- 💾 Create `work/Logical-View-Map.md` with view mapping and build order
- 📖 Read all scenario step specification files
- 🚫 Do not begin section breakdown or implementation

## CONTEXT BOUNDARIES:

- Available context: Prototype folder structure from Step 1; all scenario step specifications
- Focus: Identifying logical views and mapping scenario steps to views
- Limits: No section breakdown, no implementation
- Dependencies: Step 1 must be complete (prototype folder exists)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Read All Scenario Step Specifications

**Actions**:
1. List all scenario step folders in `../[Scenario]/`
2. Read each `[Step].md` specification file
3. Note step names, purposes, and any "inherit from" or "base page" references

### 2. Identify Logical Views

For each scenario step, determine:
- Is this a **new logical view** (new page/screen)?
- Or does it **reuse an existing logical view** (same page, different state)?

**Key indicators of SAME logical view**:
- Spec says "inherit from [other step]"
- Spec says "same structure as [other step]"
- Same page name (e.g., "Family Page" in 1.5, 1.7, 1.9)
- Overlay/modal/confirmation on existing page

**Key indicators of NEW logical view**:
- Completely different page structure
- Different purpose and user context
- No reference to inheriting from another step

Present the mapping to user for confirmation.

### 3. User Reviews & Confirms Mapping

**Wait for response**

**If user says "N"**:
- Ask what needs adjustment
- Update logical view mapping
- Re-present for confirmation

**If user says "Y"**: Proceed to create the map document

### 4. Create Logical View Map Document

Create `work/Logical-View-Map.md` with view details, build order, and notes.

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 3: Logical View Breakdown"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the logical view mapping is confirmed by user and the map document is created will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All scenario step specifications read
- Logical views identified with correct grouping
- User confirmed the mapping
- Logical-View-Map.md created with build order

### ❌ SYSTEM FAILURE:
- Beginning to build views before analysis is complete
- Not reading all scenario step specifications
- Not getting user confirmation on the mapping
- Not creating the map document

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
