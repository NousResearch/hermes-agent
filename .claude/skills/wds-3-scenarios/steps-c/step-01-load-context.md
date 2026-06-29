---
name: 'step-01-load-context'
description: 'Read all prerequisite artifacts and detect project state'

# File References
nextStepFile: './step-02-analyze-scope.md'
---

# Step 1: Load Context & Detect Project State

## STEP GOAL:

Read all prerequisite artifacts (Product Brief, Trigger Map) and detect whether this is a fresh start or resume, establishing the complete project context needed for scenario creation.

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

- 🎯 Focus only on loading context and detecting project state
- 🚫 FORBIDDEN to skip reading any prerequisite artifact
- 💬 Approach: Methodically gather all context before any creative work
- 📋 Present a clear context summary so the user can verify understanding

## EXECUTION PROTOCOLS:

- 📖 Read all prerequisite files completely before summarizing
- 💾 Extract and note key elements from each artifact
- 🔍 Check for existing work to determine fresh start vs resume
- 🚫 FORBIDDEN to proceed without presenting context summary to user

## CONTEXT BOUNDARIES:

- Available context: Project config, Product Brief, Trigger Map artifacts
- Focus: Loading and understanding all prerequisite data
- Limits: No scenario creation, no analysis — only context gathering
- Dependencies: Product Brief (Phase 1) and Trigger Map (Phase 2) must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Read Configuration

Read `{project-root}/_bmad/wds/config.yaml` and extract:
- `project_name`
- `output_folder`
- `user_name`
- `communication_language`
- `document_output_language`

### 2. Read Product Brief

Read `{output_folder}/A-Product-Brief/product-brief.md`

**Extract and note:**
- Site/app type (marketing site, SaaS, booking system, portfolio, etc.)
- Business context and constraints
- Technical platform (WordPress, custom, etc.)
- Number of pages/views mentioned
- Any navigation structure described

### 3. Read Trigger Map

Read `{output_folder}/B-Trigger-Map/trigger-map.md` (the hub document)

**Extract and note:**
- **Business Goals:** Vision statement, all objectives with priority tiers (Primary/Secondary/Tertiary)
- **Personas:** For each persona:
  - Name and role
  - Priority level (Primary/Secondary/Tertiary)
  - Top 3 positive drivers (wants)
  - Top 3 negative drivers (fears)
  - Role in flywheel

**Also read persona documents** if they exist:
- `{output_folder}/B-Trigger-Map/02-*.md` (Primary persona)
- `{output_folder}/B-Trigger-Map/03-*.md` (Secondary persona)
- `{output_folder}/B-Trigger-Map/04-*.md` (Tertiary persona, if exists)

### 4. Check for Existing Work

**Check for resume situation:**
- Does `{output_folder}/C-UX-Scenarios/` exist?
- Are there any scenario files already?
- Is there in-progress work in the design log (`{output_folder}/_progress/00-design-log.md`)?

**If existing work found:**
```
"I see we have existing scenario work:
- [list what exists]

Should I:
1. Continue where we left off
2. Review and adjust existing scenarios
3. Start fresh"
```
Wait for user response before proceeding.

**If starting fresh:** Continue to next instruction.

### 5. Present Context Summary

Present to user:
```
"Here's what I'm working with:

**Project:** [project_name]
**Site Type:** [type from Product Brief]
**Business Goals:** [count] objectives across [tier count] tiers
**Personas:** [list names with priority levels]
**Primary Persona:** [name] — [top driving force]

Ready to analyze the scope."
```

### 6. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to Scope Analysis?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [context summary presented and acknowledged], will you then load and read fully `{nextStepFile}` to execute and begin scope analysis.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All prerequisite artifacts read completely (Product Brief, Trigger Map, persona documents)
- Key elements extracted and noted from each artifact
- Existing work detected and handled appropriately
- Clear context summary presented to user
- User acknowledges understanding before proceeding
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Skipping any prerequisite artifact
- Not detecting existing work when it exists
- Proceeding without presenting context summary
- Generating scenarios or analysis during this step
- Not waiting for user acknowledgment before proceeding

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
