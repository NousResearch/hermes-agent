---
name: 'step-08h-mermaid-quality'
description: 'Verify Mermaid diagram meets all quality standards before finalization'

# File References
nextStepFile: './step-09a-finalize-hub.md'
activityWorkflowFile: '../workflow.md'
---

# Step 31: Mermaid Diagram Quality Check

## STEP GOAL:

Verify the complete Mermaid diagram meets all quality standards across configuration, node formatting, emoji usage, connections, styling, content quality, and syntax before finalization.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - ensuring diagram quality
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on comprehensive quality verification of complete diagram
- 🚫 FORBIDDEN to approve with known issues or skip any quality dimension
- 💬 Approach: Systematic checklist verification, fix issues before approval
- 📋 Check: configuration, nodes, emojis, driving forces, connections, styling, content, syntax
- 📋 If issues found, fix and re-verify

## EXECUTION PROTOCOLS:

- 🎯 Run through complete quality checklist
- 💾 Fix any issues found during verification
- 📖 Present verification results
- 🚫 Do not approve with unresolved issues

## CONTEXT BOUNDARIES:

- Available context: Complete assembled diagram
- Focus: Quality verification across all dimensions
- Limits: Must check ALL dimensions - no shortcuts
- Dependencies: Requires complete diagram from step-08g

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Configuration & Structure Check

- Mermaid config includes custom font (Inter, system-ui, sans-serif)
- fontSize set to 14px
- Flowchart direction is LR
- Section comments present

### 2. Node Formatting Check

- All nodes start with `<br/>` and end with `<br/><br/>`
- All titles in ALL CAPS
- All line breaks use `<br/>`
- No HTML tags in any node
- All nodes properly closed with `"]`

### 3. Emoji Usage Check

- Each persona has matching emoji in both TG and DF nodes
- Business goals have appropriate emojis
- Platform has appropriate emoji
- WANTS and FEARS headers have NO emojis
- Positive drivers all have checkmark
- Negative drivers all have X

### 4. Driving Forces Check

- Exactly 3 positive drivers per persona
- Exactly 3 negative drivers per persona
- Section headers are WANTS and FEARS (no emojis, ALL CAPS)
- Blank line between sections
- DF emoji matches corresponding TG emoji

### 5. Connections Check

- All BG nodes connect to PLATFORM
- PLATFORM connects to all TG nodes
- Each TG connects to matching DF
- Simple arrows used
- Connection comments present
- No broken connections

### 6. Styling Check

- Light gray styling with dark text applied
- All four classDef statements present
- Colors EXACTLY match specification
- Platform has thicker border (3px vs 2px)
- All nodes assigned to correct classes
- Node counts match actual diagram

### 7. Content & Syntax Check

- Business goals have 3-5 key points
- Platform includes transformation statement
- Target groups have 3-4 profile traits
- Drivers concise (under 40 characters)
- No syntax errors
- All brackets and quotes properly closed
- Node IDs follow patterns

### 8. Fix Issues If Found

If any issues found: Fix identified issues, re-run quality check, repeat until all checks pass.

### 9. Present Results

If all checks pass: "Quality verified - Diagram ready for publication"

The professional Mermaid diagram can now be inserted into 00-Trigger-Map-Poster.md, presentations, documentation, and reports.

### 10. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Finalize Hub | [M] Return to Activity Menu"

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
- All quality dimensions checked
- All issues found were fixed
- Re-verification passed after fixes
- Diagram renders without errors
- Professional, clean, readable appearance
- All specifications exactly met

### ❌ SYSTEM FAILURE:
- Skipping quality dimensions
- Approving with known issues
- Not fixing found issues
- Not re-verifying after fixes
- Diagram has syntax errors
- Colors don't match specification

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
