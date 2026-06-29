---
name: 'step-03-fix'
description: 'Implement the minimal fix targeting the root cause'

# File References
nextStepFile: './step-04-verify.md'
---

# Step 3: Fix

## STEP GOAL:

Implement the minimal fix targeting the root cause.

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

- 🎯 Focus only on writing a failing test, implementing the minimal fix, and verifying the reproduction case passes
- 🚫 FORBIDDEN to refactor surrounding code, add unrelated improvements, or fix other issues noticed nearby
- 💬 Approach: Write test first, then minimal fix, then verify — test-driven bugfixing
- 📋 The fix must be the smallest change that resolves the root cause

## EXECUTION PROTOCOLS:

- 🎯 Bug fixed with minimal change, regression test written, reproduction case passes
- 💾 Update dialog file with fix details
- 📖 Reference root cause from Step 2 and reproduction steps from Step 1
- 🚫 Do not include unrelated changes in the fix

## CONTEXT BOUNDARIES:

- Available context: Reproduction details from Step 1; root cause and proposed fix from Step 2
- Focus: Minimal fix targeting root cause with regression test
- Limits: No refactoring, no unrelated improvements
- Dependencies: Step 2 must be complete (root cause identified)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Create Fix Branch

- Create a branch from the current working branch
- Use a descriptive name: `fix/<short-description>` or `bugfix/<issue-id>`
- Commit the branch before making changes

### 2. Write a Test That Catches the Bug

- Before writing the fix, write a test that reproduces the bug
- The test should fail in the current state (proving the bug exists)
- This ensures the fix is verifiable and the bug cannot silently return
- If automated testing is not set up, document the manual test steps clearly

### 3. Implement the Minimal Fix

- Target the root cause identified in Step 02
- Make the smallest change that resolves the issue
- Do NOT:
  - Refactor surrounding code
  - Add unrelated improvements
  - Change formatting or style in unrelated lines
  - "Fix" other issues you noticed nearby
- If the fix requires more than a few lines, pause and verify scope

### 4. Verify the Fix Resolves the Reproduction Case

- Run the reproduction steps from Step 01
- Confirm the bug no longer occurs
- Run the test from sub-step 2 — it should now pass
- If the fix does not resolve the bug, revisit Step 02

### 5. Verify Checklist

- [ ] Fix branch created
- [ ] Test written that catches the bug (fails before fix, passes after)
- [ ] Minimal fix implemented targeting root cause
- [ ] No unrelated changes included
- [ ] Reproduction case passes with the fix
- [ ] Dialog file updated with fix details

### 6. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4: Verify"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the fix is implemented, test passes, and reproduction case is resolved will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Fix branch created
- Test written that catches the bug (fails before fix, passes after)
- Minimal fix implemented targeting root cause
- No unrelated changes included
- Reproduction case passes with the fix

### ❌ SYSTEM FAILURE:
- Implementing fix without writing a regression test first
- Including refactoring or unrelated improvements
- Fix does not target root cause (treats symptom only)
- Proceeding without verifying reproduction case

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
