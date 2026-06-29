---
name: 'step-05-document'
description: 'Document the bug, the fix, and create a clean PR'

# File References
activityWorkflowFile: '../workflow-bugfixing.md'
---

# Step 5: Document

## STEP GOAL:

Document the bug, the fix, and create a clean PR.

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

- 🎯 Focus only on documenting root cause, fix approach, updating tests, creating PR, and flagging similar risks
- 🚫 FORBIDDEN to add new features or make code changes beyond test cleanup
- 💬 Approach: Collaborative documentation and PR creation with user
- 📋 Flag similar risks elsewhere in the codebase for future investigation

## EXECUTION PROTOCOLS:

- 🎯 Complete bug documentation, PR created, similar risks flagged
- 💾 Finalize dialog file with complete bugfix record
- 📖 Reference all previous steps for comprehensive documentation
- 🚫 Do not add features or make non-documentation changes

## CONTEXT BOUNDARIES:

- Available context: Reproduction from Step 1; root cause from Step 2; fix from Step 3; verification from Step 4
- Focus: Documentation, PR creation, risk flagging
- Limits: No feature additions, no code changes beyond test cleanup
- Dependencies: Step 4 must be complete (verification passed)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Document What Caused the Bug

- Write a clear root cause summary in the dialog file
- Explain why the bug existed (design oversight, edge case, regression from another change)
- Include the relevant code context — what was wrong and why

### 2. Document the Fix Approach

- Explain what was changed and why this approach was chosen
- If alternative approaches were considered, note why they were rejected
- Reference the specific files and lines changed

### 3. Update Tests

- Ensure the regression test from Step 03 is complete and well-named
- Add any additional test cases discovered during verification
- Verify all tests pass with the final fix

### 4. Create PR with Clear Description

- Write a PR title that summarizes the fix: `fix: <what was fixed>`
- PR body should include:
  - **Bug:** What the user experienced
  - **Root cause:** Why it happened
  - **Fix:** What was changed
  - **Testing:** How it was verified
- Link to the bug report or issue if one exists

### 5. Flag Similar Risks

- If the root cause suggests similar bugs may exist elsewhere, note this
- Examples: "This pattern of unchecked null access also exists in X and Y"
- Create follow-up issues or add notes to the dialog for future investigation

### 6. Verify Checklist

- [ ] Root cause documented
- [ ] Fix approach documented
- [ ] Tests updated and passing
- [ ] PR created with clear description
- [ ] Similar risks flagged (if any)
- [ ] Dialog file finalized with complete bugfix record

### 7. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF M: Update design log, then load, read entire file, then execute {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed when user selects 'M'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN documentation is complete, PR is created, and similar risks are flagged will you then load and read fully `{activityWorkflowFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Root cause documented
- Fix approach documented
- Tests updated and passing
- PR created with clear description
- Similar risks flagged (if any)
- Dialog file finalized with complete bugfix record

### ❌ SYSTEM FAILURE:
- Not documenting root cause
- Creating PR without clear description
- Not flagging similar risks
- Not finalizing dialog file

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
