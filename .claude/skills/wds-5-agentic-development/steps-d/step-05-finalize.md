---
name: 'step-05-finalize'
description: 'Clean up, run the full test suite, document deviations, and prepare the work for review'

# File References
activityWorkflowFile: '../workflow-development.md'
---

# Step 5: Finalize

## STEP GOAL:

Clean up, run the full test suite, document deviations, and prepare the work for review.

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

- 🎯 Focus only on cleanup, full test suite, deviation documentation, and PR preparation
- 🚫 FORBIDDEN to add new features or make non-cleanup changes
- 💬 Approach: Systematically clean up, test, document, and prepare for review with user
- 📋 Ensure deviations are documented with rationale for reviewer context

## EXECUTION PROTOCOLS:

- 🎯 Code cleaned, tests passing, deviations documented, PR prepared
- 💾 Document spec deviations and PR description in the dialog file
- 📖 Reference test baseline from Step 2 to distinguish regressions from pre-existing failures
- 🚫 Do not add features or refactor beyond cleanup

## CONTEXT BOUNDARIES:

- Available context: Implementation from Step 3; verification from Step 4; test baseline from Step 2; spec
- Focus: Final cleanup, testing, documentation, and review preparation
- Limits: No new features, no major refactoring
- Dependencies: Step 4 must be complete (verification passed)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Clean Up Code

Go through all files you created or modified:

- Remove `console.log`, `debugger`, and other debug statements
- Remove commented-out code (if it was kept for reference, it belongs in a comment on the PR, not in the code)
- Remove unused imports and variables
- Verify file naming follows project conventions
- Check for any TODO comments you left — resolve them or convert to tracked issues

### 2. Run the Full Test Suite

Run the complete test suite, not just the tests for your changes:

```
npm test   # or the project's equivalent
```

- **All pass:** Proceed.
- **New failures:** Determine if they are regressions from your changes or pre-existing. Fix regressions. Pre-existing failures should match the baseline from Step 02.
- **Flaky tests:** Note them but do not ignore them. If your changes made a test flaky, fix it.

If the project has linting or type checking, run those too:

```
npm run lint
npm run typecheck   # if applicable
```

Fix any issues your code introduced.

### 3. Document Deviations from Spec

If you deviated from the spec during implementation (discovered in Step 03 or Step 04), document each deviation:

```markdown
## Spec Deviations

### [Component/Feature Name]
- **Spec said:** [what the spec required]
- **Implementation does:** [what you built instead]
- **Reason:** [why the deviation was necessary]
```

Record this in the dialog file. These deviations become discussion points during review.

### 4. Update Affected Documentation

Check if your changes require documentation updates:

- Component API documentation (new props, changed behavior)
- Route documentation (new pages, changed URLs)
- Environment setup docs (new dependencies, new env vars)
- Storybook stories (if the project uses Storybook)

Update only what your changes affect. Do not create new documentation that was not asked for.

### 5. Prepare PR Description

Write a pull request description that helps the reviewer:

- **Summary:** What was built and why (reference the spec)
- **Changes:** Key files and what changed in each
- **Testing:** How to test the changes (steps to reproduce each feature)
- **Deviations:** Any departures from the spec, with rationale
- **Screenshots:** Before/after if visual changes were made
- **Acceptance criteria:** Copy the checklist from Step 01 with all items checked

### 6. Suggest Acceptance Testing

Based on what was built, recommend next steps:

- If the project uses formal acceptance testing, suggest triggering the [T] Acceptance Testing workflow
- If a designer needs to review visual fidelity, flag that
- If stakeholder demo is needed, note what to demonstrate
- If the feature has user-facing copy, suggest copy review

### 7. Verify Checklist

- [ ] Debug statements removed
- [ ] Unused code and imports removed
- [ ] TODOs resolved or converted to tracked issues
- [ ] Full test suite passes (or pre-existing failures match baseline)
- [ ] Linting and type checking pass
- [ ] Spec deviations documented with rationale
- [ ] Affected documentation updated
- [ ] PR description written with summary, changes, testing steps, and deviations
- [ ] Next steps recommended (acceptance testing, design review, etc.)

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF M: Update design log, then load, read entire file, then execute {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed when user selects 'M'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN all cleanup is done, tests pass, deviations are documented, and PR is prepared will you then load and read fully `{activityWorkflowFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Debug statements removed
- Unused code and imports removed
- Full test suite passes (or pre-existing failures match baseline)
- Linting and type checking pass
- Spec deviations documented with rationale
- PR description written
- Next steps recommended

### ❌ SYSTEM FAILURE:
- Leaving debug statements in code
- Not running the full test suite
- Not documenting spec deviations
- Not preparing a PR description
- Skipping linting or type checking

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
