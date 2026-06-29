---
name: 'step-07d-generate-secondary-persona'
description: 'Generate the secondary persona document with validation strategy'

# File References
nextStepFile: './step-07e-generate-tertiary-persona.md'
activityWorkflowFile: '../workflow.md'
---

# Step 20: Generate Secondary Persona

## STEP GOAL:

Create the SECONDARY persona document with validation strategy, trust building focus, and evaluation journey, including detailed driving forces with Product Answers.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - creating rich, human persona documentation
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on creating secondary persona with validation and trust focus
- 🚫 FORBIDDEN to use "converting" language - use "creating awesome" language
- 💬 Approach: Rich, nuanced, human storytelling - not template-like
- 📋 Use template from ../templates/persona-document.template.md
- 📋 Each driving force MUST have a Product Answer; target ~250-375 lines

## EXECUTION PROTOCOLS:

- 🎯 Generate persona with all 13 required sections from template
- 💾 Save as 03-[Name]-the-[Role].md in trigger map folder
- 📖 Include validation strategy section
- 🚫 Do not skip Product Answers for any driving force

## CONTEXT BOUNDARIES:

- Available context: Secondary persona from workshops, driving forces
- Focus: SECONDARY persona document with trust and validation focus
- Limits: Use persona data from workshops
- Dependencies: Requires completed primary persona document

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Gather Input

From trigger map data:
- targetGroups.secondary section
- Name, role, type, roleInFlywheel
- 6 drivingForces (3 wants, 3 fears)

### 2. Generate Document

Use the comprehensive persona document template.

**File Naming:** `03-[Name]-the-[Role].md` (e.g., 03-Marcus-the-Mentor.md)

**Key Requirements:**

**Length:** ~250-375 lines

**Tone:** Rich, nuanced, human. Natural language, storytelling.

**Driving Forces:** Each must have **[Product] Answer**. Show how product addresses each driver.

**Validation:** Focus on what builds trust. Show the evaluation journey. Address skepticism and concerns.

**Secondary-Specific Section:** Include "Validation Strategy"

Show:
- What they need to see about the product
- Conversion path: Discovery -> Evaluation -> Adoption -> Advocacy
- Focus on validation and trust building
- How product proves its value to them

### 3. Save and Confirm

Store as: 03-[Name]-the-[Role].md in trigger map folder.

Output: "Secondary persona document created: 03-[Name]-the-[Role].md"

### 4. Route to Next Step

**If Tertiary persona exists:**
Present MENU: [C] Continue to Tertiary Persona | [M] Return to Activity Menu

**If NO Tertiary persona:**
Present MENU: [C] Continue to Key Insights Document | [M] Return to Activity Menu

(If no tertiary, nextStepFile should be adjusted to step-07f-generate-key-insights.md)

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile} (or step-07f if no tertiary)
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Secondary persona document must be complete before proceeding. Do NOT skip ahead.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All 13 sections from template included
- 6 driving forces with Product Answers (3 wants, 3 fears)
- Validation strategy section included
- Trust building and evaluation journey described
- Rich, nuanced, human tone
- ~250-375 lines
- Correct routing based on tertiary persona existence

### ❌ SYSTEM FAILURE:
- Missing required sections
- Driving forces without Product Answers
- Missing validation strategy
- Template-like tone
- Wrong filename format
- Not routing correctly for tertiary/no-tertiary cases

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
