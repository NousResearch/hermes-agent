---
name: 'step-05d-contract-payment'
description: 'Build Section 4 Payment Terms tailored to the selected business model'

# File References
nextStepFile: './step-05e-contract-timeline.md'
workflowFile: '../workflow.md'
---

# Step 26: Build Section 4 - Our Commitment & Payment Terms

## STEP GOAL:

Build the payment terms section tailored to the selected business model, covering costs, payment schedule, and financial expectations.

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

- 🎯 Focus only on building the Payment Terms section
- 🚫 FORBIDDEN to skip explaining why certain payment structures are fair
- 💬 Approach: Tailor to business model, explain fairness, gather specific terms
- 📋 Transparency builds trust - explain the reasoning behind payment structures

## EXECUTION PROTOCOLS:

- 🎯 Build payment terms tailored to business model
- 💾 Add to contract document
- 📖 Pull commitment info from alignment document, add payment specifics
- 🚫 Do not use generic payment terms - tailor to business model

## CONTEXT BOUNDARIES:

- Available context: Business model, alignment document, contract sections 1-3
- Focus: Contract Section 4 - Our Commitment & Payment Terms
- Limits: Payment terms only
- Dependencies: step-05c must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Section 4: Our Commitment & Payment Terms

**Section 4: Our Commitment & Payment Terms**

**Background**: Financial commitment needed and payment structure - tailored to business model

**What it does**: States costs, payment schedule, and payment terms based on selected business model

**Purpose**: Clear financial expectations - transparency builds trust

**Why this matters**:
- Protects supplier from non-payment risk
- Ensures client commits financially to the project
- Provides cash flow for supplier to deliver quality work
- Prevents disputes about payment timing

**Adapt based on business model**:

**If Fixed-Price Project**:
- **User options for payment structure**:
  - **50% upfront, 50% on completion**: Fair split, supplier gets commitment, client retains leverage
  - **100% upfront**: Full commitment, supplier assumes all risk, client gets best price
  - **30% upfront, 70% on completion**: Lower upfront, more risk for supplier
  - **Milestone payments**: Payments tied to specific deliverables or project phases
  - **Payment on completion**: All payment at end (risky for supplier, not recommended)
- **Why upfront payment is fair**:
  - Supplier assumes risk in fixed-price contracts
  - Cost overruns are supplier's problem
  - Client gets price certainty
  - Upfront payment shows commitment
  - Enables quality delivery
- **What to ask user**: "For fixed-price contracts, upfront payment is fair since you're assuming the risk. Would you like 50% upfront and 50% on completion, or 100% upfront?"

**If Hourly/Time-Based**:
- Billing frequency, payment terms (Net 15, Net 30), time tracking, invoice format
- **What to ask user**: "How often would you like to be invoiced? What payment terms work for you?"

**If Retainer**:
- Monthly retainer amount, due date, minimum hours, overage billing, hour rollover
- **What to ask user**: "When should the retainer be due each month? How should we handle unused hours?"

**If Hybrid**:
- Combine payment terms from each component

**Content**: Pull from alignment document (our_commitment), add payment schedule and terms based on business model

**Fairness note**: Tailor to business model - for fixed-price emphasize upfront payment fairness, for retainer emphasize commitment and availability

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-05e-contract-timeline"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the Payment Terms section is built and confirmed will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Payment terms are tailored to business model
- Specific amounts, schedules, and terms are captured
- Fairness is explained transparently
- User confirms the terms

### ❌ SYSTEM FAILURE:
- Using generic payment terms not tailored to model
- Skipping fairness explanation
- Not gathering specific payment details from user

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
