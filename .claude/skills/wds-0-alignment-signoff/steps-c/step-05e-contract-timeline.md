---
name: 'step-05e-contract-timeline'
description: 'Build Section 5 Timeline with milestones and delivery dates'

# File References
nextStepFile: './step-05f-contract-availability.md'
workflowFile: '../workflow.md'
---

# Step 27: Build Section 5 - Timeline

## STEP GOAL:

Build the Timeline section defining when work will happen, key milestones, and delivery dates.

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

- 🎯 Focus only on building the Timeline section
- 🚫 FORBIDDEN to invent deadlines without user input
- 💬 Approach: Extract from alignment document and gather specifics
- 📋 Sets expectations for delivery dates

## EXECUTION PROTOCOLS:

- 🎯 Build Timeline with milestones and dates
- 💾 Add to contract document
- 📖 Extract from Work Plan or Investment Required from alignment document
- 🚫 Do not create fictional timelines

## CONTEXT BOUNDARIES:

- Available context: Alignment document, business model, contract sections 1-4
- Focus: Contract Section 5 - Timeline
- Limits: Timeline content only
- Dependencies: step-05d must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Section 5: Timeline

**Section 5: Timeline**

**Background**: When work will happen

**What it does**: Defines schedule and milestones

**Purpose**: Sets expectations for delivery dates

**Content**: Extract from Work Plan or Investment Required from alignment document

**What to include**:
- Key milestones
- Delivery dates
- Critical deadlines

### 2. Route Based on Business Model

**If Retainer model**: Continue to step-05f-contract-availability.md (availability section applies)
**If not Retainer**: Continue to step-05g-contract-confidentiality.md (skip availability)

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-05f-contract-availability (or step-05g if not Retainer)"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile} (or step-05g if not Retainer)
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the Timeline section is built and confirmed will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Key milestones and delivery dates are captured
- Timeline is realistic and agreed upon
- Correct routing based on business model

### ❌ SYSTEM FAILURE:
- Inventing deadlines without user input
- Not routing correctly based on business model
- Skipping milestone definition

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
