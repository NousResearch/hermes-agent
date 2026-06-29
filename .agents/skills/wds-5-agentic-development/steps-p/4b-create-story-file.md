---
name: '4b-create-story-file'
description: 'Create the focused story file for this section with all implementation details'

# File References
nextStepFile: './4c-implement-section.md'
---

# Step 4b: Create Story File

## STEP GOAL:

Create the focused story file for this section with all implementation details. Use the story template to create complete, clear instructions for implementation.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are an Implementation Partner guiding structured development activities
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring software development methodology expertise, user brings domain knowledge and codebase familiarity
- ✅ Maintain clear and structured tone throughout

### Step-Specific Rules:

- 🎯 Focus only on creating the story file with objects, HTML structure, Tailwind classes, JavaScript requirements, demo data, and acceptance criteria
- 🚫 FORBIDDEN to begin implementing — that is the next step
- 💬 Approach: Create comprehensive story file, then offer user review or proceed to implementation
- 📋 Story file must include both agent-verifiable (Puppeteer) and user-evaluable (qualitative) criteria

## EXECUTION PROTOCOLS:

- 🎯 Complete story file created with all implementation instructions
- 💾 Create `stories/[View].[N]-[section-name].md`
- 📖 Reference requirements gathered in Step 4a
- 🚫 Do not write any HTML, CSS, or JavaScript code

## CONTEXT BOUNDARIES:

- Available context: Requirements gathered in Step 4a; work file; specifications
- Focus: Story file creation — complete implementation instructions
- Limits: No code implementation
- Dependencies: Step 4a must be complete (requirements gathered)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Create Story File

Create `stories/[View].[N]-[section-name].md` with:
- Purpose, specifications reference
- All objects with type, label, behavior, states, and spec reference
- HTML structure to build
- Tailwind classes to use
- JavaScript requirements (functions and state handling)
- Demo data requirements
- Acceptance criteria (agent-verifiable and user-evaluable)
- Test instructions (Puppeteer self-verification and user qualitative review)

### 2. Present Story to User

Present summary and offer user the choice to review the story first or proceed to implementation.

### 3. Handle User Response

**If user says "review"**: Show key sections, answer questions, make adjustments, ask if ready to implement.
**If user says "implement"** or "Y": Proceed to next step.

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4c: Implement Section"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the story file is created and user is ready to proceed will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Story file created with complete implementation instructions
- All objects defined with types, behaviors, and states
- Acceptance criteria include both agent-verifiable and user-evaluable items
- User approved or chose to proceed

### ❌ SYSTEM FAILURE:
- Beginning implementation without a story file
- Missing objects or acceptance criteria
- Not offering user the chance to review
- Creating incomplete story file

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
