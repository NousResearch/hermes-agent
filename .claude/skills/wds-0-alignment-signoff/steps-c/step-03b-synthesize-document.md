---
name: 'step-03b-synthesize-document'
description: 'Create the alignment document from all explored and confirmed sections'

# File References
nextStepFile: './step-03d-present-approval.md'
workflowFile: '../workflow.md'
---

# Step 18: Synthesize Alignment Document

## STEP GOAL:

Create the alignment document from all explored sections, using framework thinking to build a clear, compelling narrative.

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

- 🎯 Focus only on synthesizing the alignment document from confirmed content
- 🚫 FORBIDDEN to add new content not confirmed in the reflection step
- 💬 Approach: Crystallize into a clear, compelling narrative
- 📋 Format must be clear, brief, readable in 2-3 minutes

## EXECUTION PROTOCOLS:

- 🎯 Create the alignment document using confirmed content
- 💾 Save to `docs/wds-1-project-brief/pitch.md`
- 📖 Use template at `../../wds-1-project-brief/templates/pitch.template.md`
- 🚫 Do not add content that wasn't confirmed in step-03a

## CONTEXT BOUNDARIES:

- Available context: All confirmed content from step-03a reflection
- Focus: Document synthesis and creation
- Limits: Only use confirmed content
- Dependencies: step-03a must be completed with user confirmation

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Crystallize into a Compelling Narrative

**After confirming understanding**:

Help crystallize into a clear, compelling narrative using framework thinking:
- **Realization -> Why It Matters -> How We See It Working -> Value We'll Create**
- **Realization -> Agitate (Cost of Inaction) -> Solve (Solution) -> Commitment**

### 2. Framework Check

**Framework check**: Does the alignment document flow logically?
- Realization is clear and evidence-backed
- Why it matters and who we help is understood
- Solution addresses the realization
- Commitment is reasonable and risks acknowledged
- Cost of inaction makes the case
- Value we'll create justifies the commitment

### 3. Create Alignment Document

**Output file**: `docs/wds-1-project-brief/pitch.md` (deliverable name: "pitch")

**Format**: Clear, brief, readable in 2-3 minutes

**Structure**: Use the 10-section structure covered in the exploration

**Template reference**: `../../wds-1-project-brief/templates/pitch.template.md`

**Ask**: "Does this present your idea in the best light?"

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Present for Approval"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the alignment document is created and user is satisfied will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Alignment document is created with all confirmed content
- Document flows logically and is compelling
- Document is brief and readable in 2-3 minutes
- User confirms the document presents their idea well

### ❌ SYSTEM FAILURE:
- Adding content not confirmed in the reflection step
- Creating a document that's too long or unfocused
- Not saving to the correct output location
- Proceeding without user satisfaction with the document

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
