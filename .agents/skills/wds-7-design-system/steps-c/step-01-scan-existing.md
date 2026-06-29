---
name: 'step-01-scan-existing'
description: 'Scan existing design system components to find matches for the current component type'

# File References
nextStepFile: './step-02-compare-attributes.md'
---

# Step 1: Scan Existing Components

## STEP GOAL:

Find all components in the design system that match the current component type. Scan the design system folder, extract component metadata, and build a candidate list for comparison.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are the Design System Architect guiding design system creation and maintenance
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design system expertise and component analysis, user brings design knowledge and project context
- ✅ Maintain systematic and analytical tone throughout

### Step-Specific Rules:

- 🎯 Focus ONLY on this step's specific goal — do not skip ahead
- 🚫 FORBIDDEN to jump to later steps before this step is complete
- 💬 Approach: Systematic execution with clear reporting
- 📋 All outputs must be documented and presented to user

## EXECUTION PROTOCOLS:

- 🎯 Execute each instruction in the sequence below
- 💾 Document all findings and decisions
- 📖 Present results to user before proceeding
- 🚫 FORBIDDEN to skip instructions or optimize the sequence

## CONTEXT BOUNDARIES:

- Available context: Previous step outputs and project configuration
- Focus: This step's specific goal only
- Limits: Do not perform actions belonging to subsequent steps
- Dependencies: Requires all previous steps to be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Read Design System Folder

Scan design system components:
- Read all files in `D-Design-System/components/`
- Parse component type from each file
- Filter by matching type

### 2. Extract Component Metadata

For each matching component, extract:
- Component ID (e.g., `btn-001`)
- Variants (e.g., primary, secondary, ghost)
- States (e.g., default, hover, active, disabled)
- Key styling attributes
- Usage count (how many pages use it)

### 3. Build Candidate List

Present matching components to user with full metadata.

### 4. Handle Edge Cases

**No matching components found:** Route to `step-08b-create-new-component.md`

**Design system empty:** Route to `step-08a-initialize-design-system.md`

**Multiple type matches:** Continue to comparison for each candidate.

### 5. Pass Data to Next Step

Pass candidate list to comparison step:
- Component IDs
- Full metadata
- Current component specification

### 6. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Compare Attributes"

#### Menu Handling Logic:

- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#6-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects the appropriate option
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option is selected and scan is complete with candidate list built], will you then load and read fully `{nextStepFile}` to execute the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Step goal achieved completely
- All instructions executed in sequence
- Results documented and presented to user
- User confirmed before proceeding
- Design log updated

### ❌ SYSTEM FAILURE:

- Skipping any instruction in the sequence
- Generating content without user input
- Jumping ahead to later steps
- Not presenting results to user
- Proceeding without user confirmation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
