---
name: 'step-05b-contract-business-model'
description: 'Build Section 2 Business Model of the contract based on user selection'

# File References
nextStepFile: './step-05c-contract-scope.md'
workflowFile: '../workflow.md'
---

# Step 24: Build Section 2 - Business Model

## STEP GOAL:

Build the Business Model section based on the user's selected model, explaining payment structure and key terms.

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

- 🎯 Focus only on building the Business Model contract section
- 🚫 FORBIDDEN to change the user's business model selection
- 💬 Approach: Structure the section based on selected model, gather specific terms
- 📋 This is the foundation for all payment-related sections

## EXECUTION PROTOCOLS:

- 🎯 Build Business Model section tailored to selected model
- 💾 Add to contract document
- 📖 Reference the business model selected in step-04b
- 🚫 Do not change the selected business model

## CONTEXT BOUNDARIES:

- Available context: Business model selection from step-04b, alignment document
- Focus: Contract Section 2 - Business Model
- Limits: Business model structure only
- Dependencies: step-05a must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Section 2: Business Model

**Section 2: Business Model**

**Background**: Defines how the service will be paid for - critical foundation for all payment terms

**What it does**: Explains the selected business model and its terms

**Purpose**: Sets clear expectations about payment structure, prevents misunderstandings

**Content**: Based on user's selection from step-04b

**For each business model, include**:

**If Fixed-Price Project**:
- Model name: "Fixed-Price Project"
- Description: "This contract uses a fixed-price model. The contractor commits to deliver the specified scope of work for the agreed price, regardless of actual time or costs incurred."
- Key terms: Total project price, what price includes/excludes, payment structure, not-to-exceed applies

**If Hourly/Time-Based**:
- Model name: "Hourly/Time-Based"
- Description: "This contract uses an hourly billing model. The client pays for actual time worked at the agreed hourly rate."
- Key terms: Hourly rate, estimated hours, estimated total, time tracking method, billing frequency, optional not-to-exceed

**If Retainer**:
- Model name: "Monthly Retainer"
- Description: "This contract uses a retainer model. The client pays a fixed monthly amount for a minimum number of hours, with additional hours billed at the overage rate."
- Key terms: Monthly retainer amount, minimum hours, effective hourly rate, overage rate, availability, retainer period, hour rollover, billing due date

**If Hybrid**:
- Model name: "Hybrid Model"
- Description: "This contract combines multiple payment structures."
- Key terms: Combine terms from each component

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-05c-contract-scope"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the Business Model section is built and confirmed will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Business Model section matches the selected model
- Key terms are clearly defined
- User confirms the section accurately reflects their arrangement

### ❌ SYSTEM FAILURE:
- Changing the user's business model selection
- Missing key terms for the selected model
- Not gathering specific values from the user

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
