---
name: 'step-05i-contract-work-initiation'
description: 'Build Section 9 Work Initiation specifying when work can begin'

# File References
nextStepFile: './step-05j-contract-terms.md'
workflowFile: '../workflow.md'
---

# Step 31: Build Section 9 - Work Initiation

## STEP GOAL:

Build the Work Initiation section specifying exactly when work can begin - protecting both parties from unauthorized work or charges.

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

- 🎯 Focus only on building the Work Initiation section
- 🚫 FORBIDDEN to assume when work should begin without asking
- 💬 Approach: Present options, let user choose
- 📋 Prevents unauthorized work and establishes clear timeline

## EXECUTION PROTOCOLS:

- 🎯 Build Work Initiation section with clear start conditions
- 💾 Add to contract document
- 📖 User selects from work initiation options
- 🚫 Do not assume start conditions

## CONTEXT BOUNDARIES:

- Available context: Business model, contract sections 1-8
- Focus: Contract Section 9 - Work Initiation
- Limits: Work initiation conditions only
- Dependencies: step-05h must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Section 9: Work Initiation

**Section 9: Work Initiation**

**Background**: Specifies exactly when work can begin

**What it does**: Defines start date or conditions before work begins

**Purpose**: Prevents unauthorized work, establishes timeline, protects both parties

**Why this matters**:
- Without this clause, work might begin before contract is fully executed
- Prevents disputes about when work actually started
- Protects contractor from doing unpaid work
- Protects client from unauthorized charges
- Establishes clear timeline expectations

**User options**:
- **Upon contract signing**: Work begins immediately when both parties sign
- **Specific date**: Work begins on a set calendar date
- **After initial payment**: Work begins when first payment/deposit is received
- **After written notice**: Work begins when client sends written authorization
- **Conditional start**: Work begins after specific conditions are met (e.g., materials received, approvals obtained)

**What to ask user**: "When should work begin? Options: immediately upon signing, a specific date, after initial payment, or when you give written notice?"

**Content**: Ask user when work should begin, document the chosen option clearly

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-05j-contract-terms"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the Work Initiation section is built and confirmed will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Clear work initiation conditions are defined
- User's chosen option is documented
- Both parties are protected

### ❌ SYSTEM FAILURE:
- Assuming when work should begin
- Skipping this section
- Not presenting all options

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
