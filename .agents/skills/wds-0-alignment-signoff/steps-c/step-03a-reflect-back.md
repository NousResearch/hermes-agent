---
name: 'step-03a-reflect-back'
description: 'Synthesize and reflect back understanding of all explored sections before creating the alignment document'

# File References
nextStepFile: './step-03b-synthesize-document.md'
workflowFile: '../workflow.md'
---

# Step 17: Reflect Back What You've Captured

## STEP GOAL:

Reflect back the complete understanding from all explored sections, confirm accuracy with the user before proceeding to document synthesis.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are the Alignment & Signoff facilitator, guiding users to create stakeholder alignment
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring alignment and stakeholder management expertise, user brings their project knowledge
- ✅ Maintain a supportive and clarifying tone throughout

### Step-Specific Rules:

- 🎯 Focus only on reflecting back and confirming understanding
- 🚫 FORBIDDEN to proceed to document synthesis without user confirmation
- 💬 Approach: Summarize each section, ask for corrections
- 📋 This is a quality gate - ensure accuracy before creating the document

## EXECUTION PROTOCOLS:

- 🎯 Reflect back complete understanding for confirmation
- 💾 Note any adjustments or corrections from user
- 📖 Reference all captured content from exploration steps
- 🚫 Do not skip confirmation - this is a critical quality gate

## CONTEXT BOUNDARIES:

- Available context: All explored sections (01a through 02k)
- Focus: Verification and confirmation of captured understanding
- Limits: Reflect only - do not create the document yet
- Dependencies: All exploration steps must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Reflect Back What You've Captured

**After covering all sections** (in whatever order made sense):

Reflect back what you've captured:

"I've explored [list sections covered] with you. Let me reflect back what I understand:

- **The Realization**: [summarize their realization]
- **Why It Matters**: [summarize why it matters and who we help]
- **How We See It Working**: [summarize solution approach]
- **Recommended Solution**: [summarize preferred approach]
- **The Path Forward**: [summarize work approach]
- **The Value We'll Create**: [summarize value and success metrics]
- **Cost of Inaction**: [summarize consequences]
- **Our Commitment**: [summarize resources and risks]
- **Summary**: [summarize key points]

Does this capture what you want in your alignment document? Anything you'd like to adjust or clarify?"

### 2. Handle Adjustments

If user wants adjustments, make them and re-reflect until confirmed.

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-03b-synthesize-document"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has confirmed the reflected understanding is accurate will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Complete understanding is reflected back to user
- User confirms accuracy or adjustments are made
- All sections are represented in the reflection

### ❌ SYSTEM FAILURE:
- Skipping the reflection and jumping to document creation
- Not asking for confirmation
- Missing sections in the reflection
- Proceeding despite user indicating inaccuracies

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
