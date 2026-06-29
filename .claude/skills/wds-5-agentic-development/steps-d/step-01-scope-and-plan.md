---
name: 'step-01-scope-and-plan'
description: 'Read the approved spec, extract every deliverable, and create a sequenced implementation plan'

# File References
nextStepFile: './step-02-setup-environment.md'
---

# Step 1: Scope and Plan

## STEP GOAL:

Read the approved spec, extract every deliverable, and create a sequenced implementation plan.

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

- 🎯 Focus only on reading the spec, inventorying work items, mapping dependencies, and creating the implementation plan
- 🚫 FORBIDDEN to begin implementing or coding anything — that is a later step
- 💬 Approach: Walk through the spec with user, extract deliverables collaboratively, and sequence them
- 📋 Every acceptance criterion must be extracted into a standalone checklist

## EXECUTION PROTOCOLS:

- 🎯 Produce a complete, sequenced implementation plan with all work items inventoried
- 💾 Record inventory, implementation order, and acceptance criteria in the dialog file
- 📖 Reference the approved spec document thoroughly
- 🚫 Do not write any implementation code

## CONTEXT BOUNDARIES:

- Available context: Approved specification document
- Focus: Planning and scoping — deliverables, dependencies, order, acceptance criteria
- Limits: No implementation, no environment setup
- Dependencies: An approved spec must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Read the Approved Specification

Open the approved spec (page spec, feature spec, or component spec) and read it end to end. Do not skim. Note:

- Every page, component, and feature described
- All properties, states, and variants for each component
- Responsive behavior and breakpoints
- Data requirements (APIs, models, mock data)
- Acceptance criteria (these are your definition of done)
- Design tokens and design system references

### 2. Inventory All Work Items

Create a flat list of everything that needs to be built. Be specific — not "build the form" but each field, validation rule, and submit behavior. Include:

- Pages / routes
- Components (new and modified)
- Data models or API integrations
- State management additions
- Styling / responsive work
- Tests to write

### 3. Identify Dependencies and Determine Order

Map which items depend on which. Common patterns:

- Shared components before pages that use them
- Data layer before UI that consumes it
- Layout / structure before content sections
- Base styles / tokens before component styles

Order the work so you never build something before its dependencies exist.

### 4. Estimate Effort

For each work item, note relative size (small / medium / large). This is not for time tracking — it is for spotting items that are too large and should be broken down further. Any "large" item should be split into sub-tasks.

### 5. Create the Branch

Create a feature branch from the appropriate base branch. Use a descriptive name that ties to the spec (e.g., `feature/booking-page`, `feature/user-profile-card`).

### 6. List All Acceptance Criteria

Extract every acceptance criterion from the spec into a standalone checklist. This checklist will be used in Step 04 (Verify) to confirm completion. Format:

```markdown
## Acceptance Criteria

- [ ] Criterion from spec
- [ ] Criterion from spec
- [ ] ...
```

Record the inventory, implementation order, and acceptance criteria in the dialog file.

### 7. Verify Checklist

- [ ] Spec read in full — no sections skipped
- [ ] All pages, components, and features inventoried
- [ ] Dependencies mapped, implementation order defined
- [ ] Large items broken into sub-tasks
- [ ] Feature branch created
- [ ] All acceptance criteria extracted into a checklist
- [ ] Plan documented in dialog file

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 2: Setup Environment"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the implementation plan is complete with inventoried work items, sequenced order, and acceptance criteria will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Spec read in full — no sections skipped
- All pages, components, and features inventoried
- Dependencies mapped, implementation order defined
- Large items broken into sub-tasks
- Feature branch created
- All acceptance criteria extracted into a checklist
- Plan documented in dialog file

### ❌ SYSTEM FAILURE:
- Beginning implementation before planning is complete
- Skipping sections of the spec
- Not extracting acceptance criteria
- Not creating a feature branch

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
