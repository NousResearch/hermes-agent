---
name: 'step-01c-offer-extract'
description: 'Offer optional step to extract information from existing communications or documents'

# File References
nextStepFile: './step-01d-extract-info.md'
workflowFile: '../workflow.md'
---

# Step 3: Offer to Extract Information from Communications

## STEP GOAL:

Offer the user the optional opportunity to provide existing communications or documents from which key information can be extracted to inform the alignment document.

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

- 🎯 Focus only on offering the extraction option and capturing user decision
- 🚫 FORBIDDEN to force users to provide documents - this is optional
- 💬 Approach: Present the option clearly, respect skip decisions
- 📋 If user provides documents, route to extract step; if not, skip to detect starting point

## EXECUTION PROTOCOLS:

- 🎯 Present the extraction offer clearly
- 💾 Note whether user has communications to share
- 📖 This is an optional enhancement step
- 🚫 Do not pressure user to provide documents

## CONTEXT BOUNDARIES:

- Available context: User's situation and routing from steps 01a-01b
- Focus: Offering document extraction as an optional enhancement
- Limits: Do not begin extraction until user provides documents
- Dependencies: step-01b must be completed with alignment path confirmed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Ask If They Have Relevant Communications or Documents

Ask if they have relevant communications or documents:

"Do you have any email threads, chat conversations, documents, or other materials from clients or stakeholders about this project?

If you do, I can extract key information from them - things like:
- Realizations or observations they've mentioned
- Requirements they've discussed
- Concerns or questions they've raised
- Context about the project
- Existing documentation or specifications

This can help us create a more informed alignment document. You can paste the content here, share the communications, or upload documents, and I'll extract the relevant information."

### 2. Handle Decision Point

**If user provides communications/documents**:
Continue to step-01d-extract-info.md

**If user doesn't have any or skips**:
Skip to step-01e-detect-starting-point.md

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-01d-extract-info (if documents provided) or step-01e-detect-starting-point (if skipping)"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile} (or step-01e if skipping extraction)
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has decided whether to provide documents or skip will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- User is offered the extraction option clearly
- User's decision (provide or skip) is respected
- Correct routing based on user's choice

### ❌ SYSTEM FAILURE:
- Pressuring user to provide documents
- Skipping the offer entirely
- Proceeding without user input on their choice

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
