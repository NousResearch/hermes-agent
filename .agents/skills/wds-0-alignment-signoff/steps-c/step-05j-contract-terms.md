---
name: 'step-05j-contract-terms'
description: 'Build Section 10 Terms and Conditions covering legal framework for the agreement'

# File References
nextStepFile: './step-05k-contract-approval.md'
workflowFile: '../workflow.md'
---

# Step 32: Build Section 10 - Terms and Conditions

## STEP GOAL:

Build the Terms and Conditions section covering the legal framework including changes, termination, IP ownership, dispute resolution, jurisdiction, and contract language.

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

- 🎯 Focus only on building the Terms and Conditions section
- 🚫 FORBIDDEN to skip jurisdiction and contract language questions
- 💬 Approach: Cover all key legal sections, ask about jurisdiction and language
- 📋 Protects both parties legally

## EXECUTION PROTOCOLS:

- 🎯 Build Terms and Conditions with all key legal sections
- 💾 Add to contract document
- 📖 Use standard terms including governing law (see template)
- 🚫 Do not skip jurisdiction and language questions

## CONTEXT BOUNDARIES:

- Available context: Business model, all previous contract sections
- Focus: Contract Section 10 - Terms and Conditions
- Limits: Standard legal terms - not custom legal drafting
- Dependencies: step-05i must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Section 10: Terms and Conditions

**Section 10: Terms and Conditions**

**Background**: Legal framework for the agreement

**What it does**: Covers changes, termination, IP ownership, dispute resolution, jurisdiction

**Purpose**: Protects both parties legally

**Key sections to include**:
- **Changes and Modifications**: How scope changes are handled (change orders)
- **Termination**: How to exit the contract (fair notice, payment for work done)
- **Intellectual Property**: Who owns what (usually client owns deliverables upon payment)
- **Dispute Resolution**: How conflicts are handled (mediation/arbitration/litigation)
- **Governing Law and Jurisdiction**: Which laws apply and where legal proceedings take place
- **Contract Language**: Language of the contract

**Content**: Standard terms including governing law and jurisdiction (see template)

**What to ask user**:
- "Which jurisdiction's laws should govern this contract? (e.g., 'State of California, USA', 'England and Wales')"
- "What language should this contract be in? (e.g., English, Spanish, French)"
- "If the contract is translated, which version should be the official one?"

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-05k-contract-approval"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the Terms and Conditions section is built and confirmed will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All key legal sections are covered
- Jurisdiction and contract language are specified
- User confirms the terms

### ❌ SYSTEM FAILURE:
- Skipping jurisdiction or language questions
- Missing key legal sections (IP, termination, dispute resolution)
- Not confirming terms with user

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
