---
name: 'step-04-suggest-scenarios'
description: 'Present scenario plan to user for approval before creating outlines'

# File References
nextStepFile: './step-05-outline-scenario.md'
---

# Step 4: Suggest Scenarios (USER CHECKPOINT)

## STEP GOAL:

Present the complete scenario plan to the user for approval before creating any outlines, ensuring alignment on scenario count, page assignments, naming, and priorities.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a UX Scenario Facilitator collaborating with the project owner
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring scenario thinking and user journey expertise, user brings their project knowledge, together we create concrete UX scenario outlines
- ✅ Maintain collaborative equal-partner tone throughout

### Step-Specific Rules:

- 🎯 Focus only on presenting the scenario plan and getting user approval
- 🚫 FORBIDDEN to proceed to outlining without explicit user approval
- 💬 Approach: Present clearly, handle feedback gracefully, re-present if needed
- 📋 This is a critical USER CHECKPOINT — do not auto-proceed under any circumstances

## EXECUTION PROTOCOLS:

- 📋 Format scenario plan exactly as specified
- ✅ Include coverage check with all four verifications
- 🔄 Handle user feedback and re-present adjusted plan
- 🚫 FORBIDDEN to skip user approval checkpoint

## CONTEXT BOUNDARIES:

- Available context: Strategic context from Step 3, page inventory, Trigger Map data
- Focus: Presenting and getting approval for the scenario plan
- Limits: No scenario outlining, no file creation — only planning approval
- Dependencies: Complete strategic context chains from Step 3

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Format the Scenario Plan

Present to user in this exact format:

```
## Scenario Plan for [Project Name]

### Site Analysis
- **Type:** [Presentation / Dynamic / Mixed]
- **Total Pages:** [N]
- **Format:** [Screen Flow / Storyboard / Mixed]
- **Scenarios:** [N] total

---

### Suggested Scenarios

**01: [Persona Name]'s [Purpose]** ⭐ Priority 1
- **Pages:** [comma-separated list]
- **Persona:** [Name] — [Primary driving force]
- **User Value:** [What user gets — be specific]
- **Business Value:** [What business gets — be measurable]
- **Format:** [Screen Flow / Storyboard]

**02: [Persona Name]'s [Purpose]** ⭐ Priority 1
- **Pages:** [comma-separated list]
- **Persona:** [Name] — [Primary driving force]
- **User Value:** [specific]
- **Business Value:** [measurable]
- **Format:** [Screen Flow / Storyboard]

[Continue for all scenarios...]

---

### Coverage Check
✅ All pages assigned: [Yes/No — if No, list unassigned pages]
✅ No page repetition: [Yes/No]
✅ Primary persona covered: [Yes/No]
✅ Top business goal addressed: [Yes/No]
```

### 2. Naming Rules

Scenario names MUST use persona names:

**Good:**
- "Hasse's Emergency Search"
- "Lars Checks Workshop Hours"
- "Åke Coordinates Fleet Service"

**Bad:**
- "Emergency Booking Flow"
- "Hours Lookup"
- "Service Scheduling"

**Why:** Keeps persona psychology front-of-mind throughout design.

### 3. Scenario ID Convention

- Format: `01`, `02`, `03`, etc.
- Folder slug: `01-hasses-emergency-search` (lowercase, hyphenated)
- File: `01-hasses-emergency-search.md`

### 4. Wait for Approval

**CHECKPOINT — Wait for user response.**

User may:
- **"Looks good, proceed"** → Continue to menu options
- **"Combine X and Y"** → Adjust and re-present
- **"Add a scenario for [purpose]"** → Add scenario chain and re-present
- **"Focus on just [one flow]"** → Apply selective ignorance, re-present
- **"Missing page [X]"** → Add to inventory and assign

**Do NOT proceed to Step 05 until user explicitly approves the scenario plan.**

### 5. Record Approved Plan

After user approval, record:
- Final scenario count
- Final page assignments
- Any user adjustments and reasoning

### 6. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to Outlining Scenarios?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [user has explicitly approved the scenario plan], will you then load and read fully `{nextStepFile}` to execute and begin scenario outlining.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Scenario plan formatted exactly as specified
- All scenarios use persona names in their titles
- Coverage check included and all four items verified
- User explicitly approves the plan before proceeding
- User feedback handled gracefully with re-presentation
- Approved plan recorded with any adjustments noted
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Proceeding without explicit user approval
- Using feature-first naming instead of persona names
- Missing coverage check
- Not handling user feedback (combining, adding, removing scenarios)
- Auto-proceeding past the user checkpoint

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
