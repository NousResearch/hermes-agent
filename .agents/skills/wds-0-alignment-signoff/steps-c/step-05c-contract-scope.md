---
name: 'step-05c-contract-scope'
description: 'Build Section 3 Scope of Work with explicit IN scope OUT of scope and deliverables'

# File References
nextStepFile: './step-05d-contract-payment.md'
workflowFile: '../workflow.md'
---

# Step 25: Build Section 3 - Scope of Work

## STEP GOAL:

Build the Scope of Work section with explicit IN scope, OUT of scope, deliverables, and path forward - preventing scope creep and setting clear boundaries.

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

- 🎯 Focus only on building the Scope of Work section
- 🚫 FORBIDDEN to skip IN scope/OUT of scope definitions - this prevents disputes
- 💬 Approach: Ask explicitly about what's included and excluded
- 📋 Adapt scope section based on business model (fixed-price needs very clear boundaries)

## EXECUTION PROTOCOLS:

- 🎯 Build Scope of Work with clear boundaries
- 💾 Add to contract document
- 📖 Pull path forward and deliverables from alignment document
- 🚫 Do not skip explicit IN/OUT scope definitions

## CONTEXT BOUNDARIES:

- Available context: Alignment document, business model, contract sections 1-2
- Focus: Contract Section 3 - Scope of Work
- Limits: Scope definition only
- Dependencies: step-05b must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Section 3: Scope of Work

**Section 3: Scope of Work**

**Background**: Defines exactly what will be delivered and what won't be

**What it does**: Lists path forward, deliverables, explicit IN scope, and explicit OUT of scope

**Purpose**: Prevents scope creep and sets clear boundaries - critical for avoiding disputes

**Why this matters**:
- Most contract disputes arise from unclear scope
- Clear IN/OUT scope prevents misunderstandings
- Protects both parties from scope creep
- Sets expectations upfront

**Content to gather**:
1. **The Path Forward**: Pull from alignment document (path_forward) - how the work will be done
2. **Deliverables**: Pull from alignment document (deliverables_list) - what will be delivered
3. **IN Scope**: Ask user explicitly - "What work is explicitly included? Be specific about what's covered."
4. **OUT of Scope**: Ask user explicitly - "What work is explicitly NOT included? What would require a change order?"

**Important**: Based on business model, adapt scope section:
- **Fixed-Price**: Must have very clear IN scope and OUT of scope (critical for fixed-price - this protects both parties)
- **Hourly**: Can be more flexible, but still define boundaries to prevent misunderstandings
- **Retainer**: Define what types of work are included in retainer vs. project work
- **Hybrid**: Define scope for each component separately

**What to ask user**:
- "Let's be very clear about what's included and what's not. What work is explicitly IN scope for this contract?"
- "What work is explicitly OUT of scope? What would require a change order?"
- "Are there any assumptions about what's included that we should document?"

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-05d-contract-payment"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the Scope of Work section is built with clear IN/OUT scope will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Clear IN scope and OUT of scope definitions
- Deliverables are explicitly listed
- Scope is adapted to business model
- User confirms the scope boundaries

### ❌ SYSTEM FAILURE:
- Skipping IN scope/OUT of scope definitions
- Not adapting scope to business model
- Creating vague scope that invites disputes

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
