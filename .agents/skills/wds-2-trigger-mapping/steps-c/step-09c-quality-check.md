---
name: 'step-09c-quality-check'
description: 'Run final quality check on all trigger map documents'

# File References
nextStepFile: './step-09d-create-handover-package.md'
activityWorkflowFile: '../workflow.md'
---

# Step 34: Final Quality Check

## STEP GOAL:

Run final quality verification on all trigger map documents to ensure completeness, consistency, correct cross-references, and proper Mermaid diagram rendering.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - verifying completeness
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on comprehensive final quality verification
- 🚫 FORBIDDEN to approve with known issues
- 💬 Approach: Systematic verification checklist
- 📋 Check: document existence, Mermaid rendering, cross-references, terminology, driving forces, Feature Impact

## EXECUTION PROTOCOLS:

- 🎯 Verify all quality dimensions
- 💾 Fix any issues found
- 📖 Present verification results
- 🚫 Do not approve with unresolved issues

## CONTEXT BOUNDARIES:

- Available context: All generated documents with cross-references
- Focus: Final quality verification
- Limits: Must check all dimensions
- Dependencies: Requires cross-references added

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Run Verification

Ensure:
- All documents exist
- Mermaid diagram renders correctly
- Cross-references work
- Consistent terminology
- No broken links
- All personas have driving forces
- Feature Impact document exists (if workshop was run)

### 2. Fix Any Issues

If issues found, identify and fix before proceeding.

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Handover Package | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. All quality checks must pass before proceeding.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All documents verified as existing
- Mermaid diagram renders correctly
- Cross-references all working
- Terminology consistent across documents
- No broken links
- All personas have complete driving forces
- Feature Impact present if workshop completed

### ❌ SYSTEM FAILURE:
- Missing documents
- Broken Mermaid diagram
- Broken cross-references
- Inconsistent terminology
- Missing driving forces
- Approving with known issues

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
