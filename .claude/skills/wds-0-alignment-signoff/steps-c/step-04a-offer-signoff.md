---
name: 'step-04a-offer-signoff'
description: 'Offer to generate signoff document after alignment acceptance with document type options'

# File References
nextStepFile: './step-04b-determine-business-model.md'
workflowFile: '../workflow.md'
---

# Step 21: Offer to Generate Signoff Document

## STEP GOAL:

Offer to create a signoff document after alignment acceptance, presenting document type options and routing to the correct path.

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

- 🎯 Focus only on offering signoff document options and routing
- 🚫 FORBIDDEN to force signoff creation - user can skip
- 💬 Approach: Present clear options, explain each document type
- 📋 Route to contract path (step-04b) for external, or internal signoff path (step-06a)

## EXECUTION PROTOCOLS:

- 🎯 Present signoff document type options
- 💾 Record user's choice for routing
- 📖 Reference the accepted alignment document
- 🚫 Do not begin building signoff content yet

## CONTEXT BOUNDARIES:

- Available context: Accepted alignment document
- Focus: Signoff document type selection
- Limits: Selection only - do not build content yet
- Dependencies: Alignment document must be accepted (step-03d)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Offer Signoff Document Options

**After the alignment document is accepted by stakeholders**, offer to create a signoff document:

"Great! The alignment document has been accepted and everyone is aligned on the idea, why, what, how, budget, and commitment.

Now let's secure this commitment with a signoff document. This will formalize what everyone has agreed to and ensure everyone stays committed to making this project happen.

**What type of document do you need?**

1. **Project Contract** - If you're a consultant and the client has approved the alignment document
2. **Service Agreement** - If you're a founder/owner and suppliers have approved the alignment document
3. **Project Signoff Document** - If this is an internal company project and stakeholders have approved
   - *Note: If you have an existing company signoff document format, you can upload it and I'll adapt it to match your company's format*
4. **Skip** - If you don't need a formal document right now

Which applies to your situation?

**Remember**: WDS helps with alignment - the alignment document got everyone on the same page, and this signoff document secures the commitment before we start building something that makes the world a better place!"

### 2. Handle Decision Point

**If user chooses "Skip"**:
- Acknowledge: "No problem! The alignment document is ready to share. You can always generate a signoff document later if needed."
- Proceed to Project Brief workflow

**If user chooses Project Contract or Service Agreement**:
Continue to step-04b-determine-business-model.md (for external contracts)

**If user chooses Project Signoff Document**:
Route to step-06a-build-internal-signoff.md (for internal signoff)

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-04b-determine-business-model (or step-06a for internal signoff)"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile} (or step-06a for internal)
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has selected a document type will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All document type options are clearly presented
- User's choice is captured and routing is correct
- Skip option is respected if chosen

### ❌ SYSTEM FAILURE:
- Forcing signoff creation when user wants to skip
- Not presenting all options
- Routing to wrong path based on document type

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
