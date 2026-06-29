---
name: 'step-04b-determine-business-model'
description: 'Determine the business model for external contracts before building contract sections'

# File References
nextStepFile: './step-05a-contract-overview.md'
workflowFile: '../workflow.md'
---

# Step 22: Determine Business Model

## STEP GOAL:

Determine how the service will be paid for before building contract sections - the business model determines contract structure.

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

- 🎯 Focus only on determining the business model
- 🚫 FORBIDDEN to choose the business model for the user
- 💬 Approach: Present all options with clear explanations and examples
- 📋 The selected model determines how all contract sections are structured

## EXECUTION PROTOCOLS:

- 🎯 Help user select the appropriate business model
- 💾 Record the business model selection for contract building
- 📖 This selection affects all subsequent contract sections
- 🚫 Do not begin building contract sections yet

## CONTEXT BOUNDARIES:

- Available context: Accepted alignment document, signoff type selection
- Focus: Business model selection
- Limits: Selection only - do not build contract yet
- Dependencies: step-04a must be completed with external contract selection

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present Business Model Options

**Before building contract sections**, determine the business model:

"First, let's determine the business model - how will this service be paid for? This helps us structure the contract correctly.

**What business model fits this project?**

1. **Fixed-Price Project** - Set price for a defined scope of work
   - Best for: Projects with clear deliverables and scope
   - Includes: Not-to-exceed clause, upfront payment recommended
   - Example: "$50,000 for complete website redesign with 5 pages"

2. **Hourly/Time-Based** - Pay for actual time worked
   - Best for: Ongoing work, uncertain scope, flexible requirements
   - Includes: Hourly rate, time tracking, optional not-to-exceed cap
   - Example: "$150/hour, estimated 200 hours"

3. **Retainer** - Monthly commitment with minimum hours
   - Best for: Ongoing support, regular availability needed
   - Includes: Monthly retainer amount, minimum hours, availability expectations, hourly rate for overage
   - Example: "$5,000/month retainer for 20 hours minimum, $200/hour for additional hours"

4. **Hybrid** - Combination of models (e.g., retainer + project work)
   - Best for: Complex arrangements with multiple work types
   - Includes: Multiple payment structures combined

Which model fits your situation?"

### 2. Confirm Understanding

**Confirm understanding**: "So you've chosen [model]. This means [brief explanation of what this means for the contract]."

**Note the selection**: This will determine which sections we include and how we configure payment terms, not-to-exceed, availability, etc.

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-05a-contract-overview"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the business model is selected and confirmed will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All business model options are clearly presented with examples
- User's selection is captured and confirmed
- Implications for contract structure are understood

### ❌ SYSTEM FAILURE:
- Choosing the business model for the user
- Not explaining what each model means for the contract
- Proceeding without confirmation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
