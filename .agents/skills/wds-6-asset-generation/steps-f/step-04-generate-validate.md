---
name: 'step-04-generate-validate'
description: 'Generate Figma-compatible HTML with OBJECT IDs and validate before export'
nextStepFile: './step-05-execute-export.md'
---

# Step 4: Generate and Validate

## STEP GOAL:

Generate clean, Figma-compatible HTML with proper OBJECT IDs from specifications and validate all aspects — specification coverage, ID naming, structure, styling, and content — before the export is executed.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a technical HTML generation specialist for Figma export
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring HTML/CSS-to-Figma expertise, user brings design intent
- ✅ Maintain a precise, quality-focused tone

### Step-Specific Rules:

- 🎯 Focus on generating validated HTML with correct OBJECT IDs
- 🚫 FORBIDDEN to execute the export in this step — validation only
- 💬 Present validation report and resolve errors before proceeding
- 📋 All five validation checks must pass before export

## EXECUTION PROTOCOLS:

- 🎯 Follow the Sequence of Instructions exactly
- 💾 Generate HTML structure with proper IDs and styling
- 📖 Convert all CSS variables to hex, rem/em to px, use Google Fonts only
- 🚫 FORBIDDEN to proceed with validation errors unresolved

## CONTEXT BOUNDARIES:

- Available context: Specification OBJECT IDs, scenario type, ID naming pattern
- Focus: Generating HTML and validating it for Figma compatibility
- Limits: Do not execute the MCP export — just generate and validate
- Dependencies: Complete OBJECT ID mapping from Step 3

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Generate HTML Structure

Create root container, state/variant containers, apply OBJECT IDs from specification mapping, include state labels, use semantic HTML tags.

### 2. Apply Styling Requirements

Convert all styles to Figma-compatible CSS:
- Fonts: Google Fonts only, imported in style block
- Colors: Convert CSS variables to hex values
- Spacing: Convert rem/em to pixels
- Layout: Inline styles or style block, simple flexbox/grid only

### 3. Run Validation Checks

Execute five validation checks:
1. **Specification Coverage:** All components have OBJECT IDs, IDs match exactly, no duplicates
2. **ID Naming Convention:** IDs follow project pattern, consistent naming, correct state suffixes
3. **HTML Structure:** Semantic tags, proper hierarchy, container elements
4. **Styling Compatibility:** Google Fonts, hex colors, pixel values, clean markup
5. **Content Completeness:** Text matches specifications, no placeholder content

### 4. Present Validation Report

Display pass/fail for each check, list any warnings and errors.

### 5. Handle Validation Failures

If errors found: offer auto-fix (CSS variables to hex, rem to px, missing IDs), guide manual fixes (structure issues, missing content), or allow skipping problematic components.

### 6. Present MENU OPTIONS

Display: **"Select an Option:** [C] Continue"

#### Menu Handling Logic:

- IF C: Confirm validation passes, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#6-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions — always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN C is selected and all validation checks pass will you load {nextStepFile} to execute the export.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- HTML generated with correct OBJECT IDs
- All five validation checks pass
- Figma-compatible styling applied
- Validation report presented to user

### ❌ SYSTEM FAILURE:

- Executing export before validation passes
- Using CSS variables instead of hex colors
- Using rem/em instead of pixels
- Proceeding with duplicate IDs
- Not waiting for user input at menu

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
