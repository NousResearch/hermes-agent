---
name: step-06-generate-overview
description: Create the 00-ux-scenarios.md index file linking all scenarios

# File References
nextStepFile: './step-07-quality-review.md'
---

# Step 6: Generate Scenario Overview

## STEP GOAL:

Create the 00-ux-scenarios.md index file that links all scenarios, includes a coverage matrix, and serves as the master reference for Phase 3 output.

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

- 🎯 Focus only on creating the overview index file with accurate links and data
- 🚫 FORBIDDEN to modify any scenario files during this step
- 💬 Approach: Compile and verify all data from completed scenarios
- 📋 All links must be verified as pointing to correct files

## EXECUTION PROTOCOLS:

- 📋 Follow the exact document structure specified
- 🔗 Verify all file links point to correct folders and files
- ✅ Cross-check coverage matrix against actual scenario content
- 🚫 FORBIDDEN to create the file with broken links or missing scenarios

## CONTEXT BOUNDARIES:

- Available context: All completed scenario outlines from Step 5
- Focus: Index file creation and link verification
- Limits: No scenario modifications, only index compilation
- Dependencies: All scenarios from Step 5 must be complete

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Create Overview File

Create `{output_folder}/C-UX-Scenarios/00-ux-scenarios.md`

### 2. Document Structure

Use the following structure:

```markdown
# UX Scenarios: [Project Name]

> Scenario outlines connecting Trigger Map personas to concrete user journeys

**Created:** [Date]
**Author:** [user_name] with [Agent Name]
**Method:** Whiteport Design Studio (WDS)

---

## Scenario Summary

| ID | Scenario | Persona | Pages | Priority | Status |
|----|----------|---------|-------|----------|--------|
| 01 | [Name] | [Persona] | [count] | ⭐ P1 | ✅ Outlined |
| 02 | [Name] | [Persona] | [count] | ⭐ P1 | ✅ Outlined |
| 03 | [Name] | [Persona] | [count] | P2 | ✅ Outlined |

---

## Scenarios

### [01: Scenario Name](01-slug/01-slug.md)
**Persona:** [Name] — [Driving Force]
**Pages:** [comma-separated list]
**User Value:** [one line]
**Business Value:** [one line]

---

### [02: Scenario Name](02-slug/02-slug.md)
[Same format...]

---

## Page Coverage Matrix

| Page | Scenario | Purpose in Flow |
|------|----------|----------------|
| [Page 1] | 01 | [What user does here] |
| [Page 2] | 01 | [What user does here] |
| [Page 3] | 02 | [What user does here] |
| ... | ... | ... |

**Coverage:** [X/Y] pages assigned to scenarios

---

## Next Phase

These scenario outlines feed into **Phase 4: UX Design** where each page gets:
- Detailed page specifications
- Wireframe sketches
- Component definitions
- Interaction details

---

_Generated with Whiteport Design Studio framework_
```

### 3. Verify Links

- [ ] Each scenario link points to correct folder/file
- [ ] All scenarios from approved plan are listed
- [ ] Page coverage matrix is complete
- [ ] No pages missing from matrix

### 4. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to Quality Review?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [overview file created with all links verified], will you then load and read fully `{nextStepFile}` to execute and begin quality review.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Overview file created at correct path
- All scenarios listed with accurate data
- All links verified and pointing to correct files
- Coverage matrix complete with all pages
- Document structure follows specification exactly
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Missing scenarios from the overview
- Broken file links
- Incomplete coverage matrix
- Incorrect data (wrong persona, wrong page counts)
- Not verifying links before completing

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
