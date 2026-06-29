---
name: 'step-05h-contract-not-to-exceed'
description: 'Build Section 8 Not to Exceed Clause conditionally based on business model'

# File References
nextStepFile: './step-05i-contract-work-initiation.md'
workflowFile: '../workflow.md'
---

# Step 30: Build Section 8 - Not to Exceed Clause (Conditional)

## STEP GOAL:

Build the Not-to-Exceed clause based on business model - required for fixed-price, optional for hourly, not applicable for retainer.

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

- 🎯 Focus only on building the Not-to-Exceed clause based on business model
- 🚫 FORBIDDEN to skip this for fixed-price contracts - it's required
- 💬 Approach: Explain why this matters, gather specifics based on model
- 📋 Protects both parties from unexpected cost overruns and scope creep

## EXECUTION PROTOCOLS:

- 🎯 Build Not-to-Exceed clause conditionally based on business model
- 💾 Add to contract document (if applicable)
- 📖 Reference business model from step-04b
- 🚫 Do not skip for fixed-price contracts

## CONTEXT BOUNDARIES:

- Available context: Business model, contract sections 1-7
- Focus: Contract Section 8 - Not to Exceed (conditional)
- Limits: Only applicable based on business model
- Dependencies: step-05g must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Determine Applicability

**Section 8: Not to Exceed Clause** (Conditional - applies to Fixed-Price and optionally Hourly)

**When this section applies**:
- **Fixed-Price Project**: REQUIRED - This is the project price cap
- **Hourly/Time-Based**: OPTIONAL - Can include optional not-to-exceed cap if desired
- **Retainer**: NOT APPLICABLE - Retainer already has monthly cap
- **Hybrid**: CONDITIONAL - May apply to fixed-price components

### 2. Build Section If Applicable

**If applicable, include**:

**Why this matters**:
- Without this clause, costs could spiral out of control (fixed-price)
- Protects your budget from unexpected expenses
- Prevents scope creep by requiring approval for additional work
- Ensures contractor gets paid fairly for extra work through change orders
- Creates clear boundaries that prevent misunderstandings

**User options**:
- **Fixed budget cap**: Hard limit that cannot be exceeded without new agreement
- **Soft cap with buffer**: Cap with small percentage buffer (e.g., 10%) for minor overruns
- **Cap with change order process**: Cap exists, but change orders can adjust it with approval

**What to ask user**:
- **For Fixed-Price**: "The not-to-exceed amount is [total_amount]. This protects both parties from cost overruns. Any work beyond the original scope requires a change order."
- **For Hourly**: "Would you like to set an optional not-to-exceed cap? This protects your budget while still allowing flexibility."

**Content**:
- **Fixed-Price**: Not-to-exceed = total project price
- **Hourly**: Optional cap amount if user wants one

**Fairness note**: "This protects your budget while ensuring I get paid fairly for additional work if needed through the change order process"

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-05i-contract-work-initiation"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the Not-to-Exceed section is handled (built or correctly skipped) will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Clause is included for fixed-price contracts (required)
- Optional cap is offered for hourly contracts
- Retainer correctly skips this section
- Fairness is explained

### ❌ SYSTEM FAILURE:
- Skipping for fixed-price contracts
- Including for retainer contracts
- Not explaining the purpose and fairness of the clause

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
