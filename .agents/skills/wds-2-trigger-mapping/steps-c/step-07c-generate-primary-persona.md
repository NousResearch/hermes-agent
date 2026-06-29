---
name: 'step-07c-generate-primary-persona'
description: 'Generate the primary persona document with full transformation journey'

# File References
nextStepFile: './step-07d-generate-secondary-persona.md'
activityWorkflowFile: '../workflow.md'
---

# Step 19: Generate Primary Persona

## STEP GOAL:

Create the PRIMARY persona document with full transformation journey, champion creation story, and detailed driving forces with Product Promises.

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

- 🎯 Focus on creating the most detailed persona document (PRIMARY gets most detail)
- 🚫 FORBIDDEN to use "converting" language - use "creating awesome" language
- 💬 Approach: Rich, nuanced, human storytelling - not template-like
- 📋 Use template from ../templates/persona-document.template.md
- 📋 Each driving force MUST have a Product Promise; target ~250-375 lines

## EXECUTION PROTOCOLS:

- 🎯 Generate persona with all 13 required sections from template
- 💾 Save as 02-[Name]-the-[Role].md in trigger map folder
- 📖 Include full BEFORE/AFTER transformation section
- 🚫 Do not skip Product Promises for any driving force

## CONTEXT BOUNDARIES:

- Available context: Primary persona from workshops, driving forces, priorities
- Focus: PRIMARY persona document with champion creation story
- Limits: Use persona data from workshops - rich and human, not template-like
- Dependencies: Requires completed workshops and hub/business goals documents

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Gather Input

From trigger map data:
- targetGroups.primary section
- Name, role, type, roleInFlywheel
- 6 drivingForces (3 wants, 3 fears)

### 2. Generate Document

Use the comprehensive persona document template: `../templates/persona-document.template.md`

This template provides the complete structure for sections 1-13.

**File Naming:** `02-[Name]-the-[Role].md` (e.g., 02-Sarah-the-Student.md)

**Key Requirements:**

**Length:** ~250-375 lines

**Tone:** Rich, nuanced, human. Not "converting" but "creating awesome." Natural language, storytelling.

**Driving Forces:** Each must have **[Product] Promise**. Show how product addresses each driver. Be specific and actionable.

**Transformation:** PRIMARY persona gets full BEFORE/AFTER section. Show emotional journey, not just functional. Use emojis to show emotional states.

**Primary-Specific Section:** Include "Role in Flywheel: Creating Awesome [Personas] Who Become [Champions]"

Show:
- The natural evolution from user to champion
- What they need to see on product page
- Focus on transformation and champion creation
- How they naturally advocate after transformation

### 3. Save and Confirm

Store as: 02-[Name]-the-[Role].md in trigger map folder.

Output: "Primary persona document created: 02-[Name]-the-[Role].md"

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Secondary Persona | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Primary persona document must be complete before proceeding. Do NOT skip ahead.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All 13 sections from template included
- 6 driving forces with Product Promises (3 wants, 3 fears)
- Full BEFORE/AFTER transformation section
- Champion creation story included
- Rich, nuanced, human tone throughout
- "Creating awesome" language (not "converting")
- ~250-375 lines
- Saved with correct filename

### ❌ SYSTEM FAILURE:
- Missing required sections
- Driving forces without Product Promises
- Using "converting" language
- Missing transformation section
- Template-like, not human and rich
- Wrong filename format

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
