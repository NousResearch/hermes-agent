---
name: 'step-03-prepare-specifications'
description: 'Locate or create specifications with OBJECT IDs for consistent Figma layer naming'
nextStepFile: './step-04-generate-validate.md'
---

# Step 3: Prepare Specifications

## STEP GOAL:

Locate existing specifications with OBJECT IDs for all components in the export scope, or create them if they do not exist, ensuring consistent Figma layer naming.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a specification analyst ensuring design-code parity through OBJECT IDs
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring specification methodology, user brings project context
- ✅ Maintain a meticulous, detail-oriented tone

### Step-Specific Rules:

- 🎯 Focus ONLY on locating or creating specifications with OBJECT IDs
- 🚫 FORBIDDEN to generate HTML in this step
- 💬 Offer to reverse-engineer specifications from code if none exist
- 📋 Achieve 100% specification coverage before proceeding

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Document specification coverage report
- 📖 Search in `docs/C-UX-Scenarios/` and `docs/D-Design-System/` for existing specs
- 🚫 FORBIDDEN to proceed without OBJECT IDs for all components

## CONTEXT BOUNDARIES:

- Available context: Export scenario type, ID naming pattern from Step 2
- Focus: Finding or creating OBJECT IDs for all components in scope
- Limits: Do not generate HTML — just prepare the ID specifications
- Dependencies: Confirmed scenario type from Step 2

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Search for Specification Documents

Search for specification files containing OBJECT IDs:
- `docs/C-UX-Scenarios/` for scenario specifications
- `docs/D-Design-System/` for component documentation
- Search for files containing "OBJECT ID"
- Look for markdown files matching component/page name

### 2. Handle Found Specifications

If specifications exist with OBJECT IDs: extract all OBJECT ID field values, map to components in code, store mapping for HTML generation.

### 3. Handle Missing Specifications

If no specifications exist, offer to:
1. Analyze the code and reverse-engineer specifications
2. Generate OBJECT IDs following project conventions
3. Create a specification document for review

Reference `../data/figma-spec-preparation.md` for detailed guidance.

### 4. Validate Coverage

For each component in export scope, verify it has an OBJECT ID. Generate a coverage report showing validated components and any gaps.

### 5. Resolve Gaps

If partial coverage: offer to create missing specs or auto-generate IDs. Target 100% coverage before proceeding.

### 6. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Save specification mapping, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#6-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and all components have OBJECT IDs will you load {nextStepFile} to begin generating and validating HTML.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Specification search completed across all relevant locations
- OBJECT IDs found or created for all components
- 100% specification coverage achieved
- Coverage report presented to user

### ❌ SYSTEM FAILURE:

- Starting HTML generation without OBJECT IDs
- Not searching all specification locations
- Proceeding with partial coverage without user acknowledgment
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
