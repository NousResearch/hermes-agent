---
name: step-08-update-design-log
description: Document Phase 3 completion in the project design log

# File References
nextStepFile: './step-09-handover.md'
---

# Step 8: Update Design Log

## STEP GOAL:

Document Phase 3 completion in the project design log, recording all artifacts created, key decisions made, and quality scores achieved.

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

- 🎯 Focus only on updating the design log with accurate Phase 3 data
- 🚫 FORBIDDEN to overwrite existing log entries — only append
- 💬 Approach: Be specific and factual in documentation
- 📋 List every artifact file created — no summarizing with "etc."

## EXECUTION PROTOCOLS:

- 📖 Read the existing design log before making changes
- 📋 Append progress entry after the last existing entry
- ✅ Record key decisions if any were made during Phase 3
- 🚫 FORBIDDEN to use generic summaries — be specific

## CONTEXT BOUNDARIES:

- Available context: All Phase 3 artifacts, quality review results, scenario data
- Focus: Design log documentation only
- Limits: No scenario modifications, only log updates
- Dependencies: Quality review must be complete from Step 7

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Read the Current Log

Read `{output_folder}/_progress/00-design-log.md` to understand existing entries and format.

### 2. Append Progress Entry

Add the following under the `## Progress` section (after the last entry):

```
### [date] — Phase 3: UX Scenarios Complete

**Agent:** Saga (Scenario Outline)
**Scenarios:** [N] scenarios covering [N] pages
**Quality:** [Excellent / Good]

**Artifacts Created:**
- `C-UX-Scenarios/00-ux-scenarios.md` — Scenario index
- `C-UX-Scenarios/01-[slug]/01-[slug].md` — [Scenario name]
- [list ALL scenario files created]

**Summary:** [2-3 sentences: what scenarios were created, key design decisions made during the process, page coverage status]

**Next:** Phase 4 — UX Design
```

**Rules:**
- List every artifact file — do not summarize with "etc."
- Summary must mention specific decisions, not generic statements
- Use the actual date, not a placeholder

### 3. Record Key Decisions

Add rows to the `## Key Decisions` table for any significant choices made during Phase 3:

```
| [date] | [decision] | Phase 3: Scenarios | Saga + [user_name] |
```

Examples of key decisions worth logging:
- Scenario count adjustments (user added/removed scenarios)
- Page assignment changes
- Priority reordering
- Scope decisions (selective ignorance applied)

If no significant decisions were made, skip this section.

### 4. Verify

- [ ] Progress entry appended (not overwriting existing entries)
- [ ] All artifact files listed
- [ ] Summary is specific, not generic
- [ ] Key decisions recorded (if any)

### 5. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to Handover?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [design log updated with all required information], will you then load and read fully `{nextStepFile}` to execute and begin the handover process.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Existing log read before making changes
- Progress entry appended (not overwriting)
- All artifact files listed individually
- Summary is specific with concrete decisions mentioned
- Key decisions recorded where applicable
- Verification checklist passes
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Overwriting existing log entries
- Summarizing artifacts with "etc." instead of listing each
- Using generic summary statements
- Not reading existing log first
- Missing artifact files from the list

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
