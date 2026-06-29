---
name: '4c-implement-section'
description: 'Implement the section following the story file precisely'

# File References
nextStepFile: './4d-present-for-testing.md'
---

# Step 4c: Implement Section

## STEP GOAL:

Implement the section following the story file precisely. Linear code generation is the task.

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

- 🎯 Focus only on implementing the HTML structure, adding object IDs, Tailwind classes, JavaScript, and placeholders per the story file
- 🚫 FORBIDDEN to deviate from the story file instructions or add unplanned features
- 💬 Approach: Follow the story file precisely, implementing section by section
- 📋 For Section 1, create new HTML file; for subsequent sections, update existing file

## EXECUTION PROTOCOLS:

- 🎯 Section implemented with all objects, styles, and JavaScript per story file
- 💾 HTML file created/updated with section implementation
- 📖 Follow story file instructions precisely
- 🚫 Do not add features not in the story file

## CONTEXT BOUNDARIES:

- Available context: Story file from Step 4b; page template (for Section 1)
- Focus: Code implementation following story file
- Limits: Only implement what the story file specifies
- Dependencies: Step 4b must be complete (story file exists and approved)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Begin Implementation

Announce implementation start.

### 2. Create or Update HTML File

**For Section 1 ONLY**:
- Create new HTML file from `templates/page-template.html`
- Name it: `[View].html`
- Place in prototype root folder

**For Sections 2+**:
- Open existing `[View].html` file
- Find insertion point (after previous section or before placeholder)

### 3. Add HTML Structure

**Follow story file precisely**:

1. Add HTML structure with Tailwind classes from story
2. Add all Object IDs on interactive elements
3. Add state-specific classes/attributes
4. Add placeholder content where specified

### 4. Add JavaScript

**If section needs JavaScript**:

1. Add functions specified in story file
2. Add event listeners for interactive elements
3. Add state handling logic
4. Add console logging for debugging
5. Load demo data from `data/demo-data.json`

### 5. Add Placeholder for Remaining Sections

**If more sections remain**: Add a placeholder div at the bottom indicating the next section.

### 6. Final Check

**Before presenting to user, verify**:
- [ ] All Object IDs from story file are present
- [ ] Tailwind classes match story file
- [ ] JavaScript functions implemented
- [ ] Console logging added
- [ ] Code is clean and readable
- [ ] No syntax errors

### 7. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4d: Present for Testing"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the section is fully implemented per the story file will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All Object IDs from story file present
- Tailwind classes match story file
- JavaScript functions implemented
- Code is clean, readable, and error-free
- Placeholder for remaining sections added (if applicable)

### ❌ SYSTEM FAILURE:
- Deviating from story file instructions
- Missing Object IDs
- Adding unplanned features
- Syntax errors in code
- Not following story file precisely

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
