---
name: 'step-03-implement'
description: 'Build every feature in the implementation plan, one at a time, following existing codebase patterns'

# File References
nextStepFile: './step-04-verify.md'
---

# Step 3: Implement

## STEP GOAL:

Build every feature in the implementation plan, one at a time, following existing codebase patterns.

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

- 🎯 Focus only on implementing plan items one at a time in dependency order
- 🚫 FORBIDDEN to skip ahead or implement multiple items simultaneously
- 💬 Approach: Work through each item methodically — re-read spec, implement, test, commit
- 📋 Follow existing codebase patterns for consistency; document any deviations

## EXECUTION PROTOCOLS:

- 🎯 Complete every work item from the implementation plan in order
- 💾 Commit incrementally after each logical unit of work
- 📖 Re-read the relevant spec section before implementing each item
- 🚫 Do not accumulate large uncommitted changesets

## CONTEXT BOUNDARIES:

- Available context: Implementation plan from Step 1; environment baseline from Step 2; approved spec
- Focus: Building features one at a time in dependency order
- Limits: Only implement what is in the plan — no scope creep
- Dependencies: Steps 1 and 2 must be complete

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Work Through the Plan Item by Item

Take the first item from your implementation order (Step 01). For each item:

1. Re-read the relevant section of the spec
2. Identify the acceptance criteria that apply to this item
3. Implement it
4. Test it (see section 5 below)
5. Commit it
6. Move to the next item

Do not jump ahead. Do not implement multiple items simultaneously. Finishing one thing completely before starting the next prevents half-done work from accumulating.

### 2. Follow Existing Codebase Patterns

Before writing new code, study how the existing codebase handles the same concerns:

| Concern | Look At |
|---------|---------|
| **File naming** | How are existing components/pages named? |
| **Component structure** | Props interface, state management, lifecycle |
| **Styling approach** | CSS modules, Tailwind, styled-components, etc. |
| **Error handling** | How do existing features handle errors? |
| **Data fetching** | What patterns are used for API calls? |
| **State management** | Local state, context, store — what is the convention? |
| **Testing patterns** | How are existing tests structured? |

Match these patterns. Consistency with the codebase is more important than what you consider "better."

### 3. Reference the Spec for Each Component

For every component you build, check the spec for:

- **Properties:** What props/attributes does it accept?
- **States:** Default, hover, active, disabled, loading, error, empty
- **Responsive behavior:** How does it change at each breakpoint?
- **Content:** Exact text, labels, placeholder copy
- **Interactions:** Click handlers, form submissions, navigation
- **Edge cases:** What happens with long text, missing data, zero items?

Do not rely on memory. Re-read the spec section each time.

### 4. Use Design System Tokens

If the project has design tokens or a design system:

- Use token variables for colors, spacing, typography — never hardcode values
- Use existing components from the design system before creating new ones
- If you need a new component, build it using the same token system
- Check that your implementation visually matches at design-review zoom levels

### 5. Run Tests After Each Major Change

After completing each work item (not just at the end):

- Run the relevant test suite
- If you wrote new tests, confirm they pass
- If existing tests break, determine if it is an intentional change or a regression
- Fix regressions immediately — do not move on with broken tests

For projects using inline testing (Puppeteer), follow the INLINE-TESTING-GUIDE.md in `data/guides/`.

### 6. Commit Incrementally

Each commit should represent one logical unit of work:

- One component added
- One feature wired up
- One set of tests for a feature

Write clear commit messages that reference what was built. Do not accumulate large uncommitted changesets.

### 7. Verify Checklist

- [ ] Working through plan items in dependency order
- [ ] Each item: spec re-read before implementation
- [ ] Following existing codebase patterns (naming, structure, styling)
- [ ] Design tokens used — no hardcoded values
- [ ] All component states implemented (not just the happy path)
- [ ] Responsive behavior implemented per spec
- [ ] Tests run after each major change
- [ ] Commits are incremental and logically scoped
- [ ] Deviations from spec documented with rationale

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4: Verify"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN all plan items are implemented, tested, and committed will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All plan items implemented in dependency order
- Spec re-read before each item implementation
- Existing codebase patterns followed
- Design tokens used consistently
- All component states implemented
- Tests run after each major change
- Commits are incremental and logically scoped

### ❌ SYSTEM FAILURE:
- Implementing items out of order or simultaneously
- Not re-reading spec before implementation
- Hardcoding values instead of using design tokens
- Accumulating large uncommitted changesets
- Moving on with broken tests

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
