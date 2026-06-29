---
name: 'step-05-verify-and-document'
description: 'Comprehensive verification of all new and existing functionality, then document and create PR'

# File References
activityWorkflowFile: '../workflow-evolution.md'
---

# Step 5: Verify and Document

## STEP GOAL:

Comprehensive verification of all new and existing functionality, then document and PR.

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

- 🎯 Focus only on verifying new functionality, running regression, verifying backward compatibility, checking performance, documenting, and creating PR
- 🚫 FORBIDDEN to add new features — only verify, fix issues, and document
- 💬 Approach: Comprehensive end-to-end verification with user, then collaborative PR creation
- 📋 Any regression failure must be fixed before proceeding

## EXECUTION PROTOCOLS:

- 🎯 All new and existing functionality verified, documented, and PR created
- 💾 Finalize dialog file with status, deviations, and PR link
- 📖 Reference the boundary map from Step 1 for backward compatibility checks
- 🚫 Do not add scope — only verify and document

## CONTEXT BOUNDARIES:

- Available context: Scope from Step 1; risks from Step 2; plan from Step 3; implementation from Step 4
- Focus: Final verification, documentation, and PR creation
- Limits: No new features
- Dependencies: Step 4 must be complete (implementation done)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Test All New Functionality

- Walk through every new feature end-to-end
- Verify against the feature spec — does it do what was requested?
- Test happy paths, error paths, and edge cases
- Use Puppeteer for measurable UI criteria (text, colors, layout, interactions)

### 2. Run Regression Suite

- Run the full test suite (unit, integration, end-to-end)
- If no automated suite exists, manually test all core user flows
- Pay special attention to areas identified as "modified" or "integration points" in Step 01
- Any failure here must be fixed before proceeding

### 3. Verify Backward Compatibility

- Test every feature listed as "untouched" in the boundary map
- Confirm that existing users see no change in behavior (unless intended)
- If feature flags were used, test with flag OFF — behavior must match pre-change baseline
- Test with flag ON — new behavior works correctly alongside existing features

### 4. Performance Check

- Does the new feature load within acceptable time?
- Did the changes increase bundle size significantly?
- Are there new network requests that could slow down existing pages?
- Check for obvious performance regressions (slow renders, unnecessary re-renders)

### 5. Document Changes

- Update the dialog file with final status
- Document what was added, what was changed, what was left untouched
- Note any deviations from the original plan and why

### 6. Create PR

- Write a PR title that summarizes the evolution: `feat: <what was added>`
- PR body should include:
  - **What:** New functionality added
  - **Why:** Business reason or user need
  - **How:** Implementation approach (incremental steps, feature flags)
  - **Testing:** How it was verified, including backward compatibility
  - **Risks:** Any known risks or areas to watch
- Link to the feature spec or dialog file

### 7. Verify Checklist

- [ ] All new functionality tested and working
- [ ] Regression suite passes
- [ ] Backward compatibility verified
- [ ] Performance acceptable
- [ ] Changes documented in dialog file
- [ ] PR created with clear description
- [ ] Feature flags documented (if used)
- [ ] Dialog file finalized

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

ONLY WHEN all verification is complete, changes documented, and PR created will you then load and read fully `{activityWorkflowFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All new functionality tested and working
- Regression suite passes
- Backward compatibility verified
- Performance acceptable
- Changes documented in dialog file
- PR created with clear description

### ❌ SYSTEM FAILURE:
- Not running regression suite
- Not verifying backward compatibility
- Skipping performance check
- Creating PR without clear description
- Not documenting deviations from the plan

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
