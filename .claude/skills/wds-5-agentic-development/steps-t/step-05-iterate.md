---
name: 'step-05-iterate'
description: 'Either iterate with development team to fix issues, or approve the feature for production'

# File References
activityWorkflowFile: '../workflow-acceptance-testing.md'
---

# Step 5: Iterate or Approve

## STEP GOAL:

Either iterate with development team to fix issues, or approve the feature for production.

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

- 🎯 Focus only on two paths: iterate (fix issues and retest) or approve (sign off for production)
- 🚫 FORBIDDEN to approve with unfixed high-severity issues or create endless iteration loops (max 3 iterations)
- 💬 Approach: Guide user through the appropriate path based on test results
- 📋 Maximum 3 iterations before escalation

## EXECUTION PROTOCOLS:

- 🎯 Feature either approved with sign-off document or issues fixed and retested
- 💾 Create sign-off document if approved; create retest report if iterating
- 📖 Reference test report from Step 4 and issues from Step 3
- 🚫 Do not approve with unfixed high-severity issues

## CONTEXT BOUNDARIES:

- Available context: Test report from Step 4; issues from Step 3; all test results
- Focus: Decision — iterate or approve
- Limits: Maximum 3 iterations before escalation
- Dependencies: Step 4 must be complete (test report created)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### Two Paths

#### Path A: Issues Found - Iterate

**If test result was FAIL:**

1. **Wait for Fixes** - Be available for questions, clarify issues, review early feedback
2. **Receive Ready for Retest** notification
3. **Retest** - Focus on:
   - Fixed issues: Verify actually fixed
   - Regression testing: Fixes did not break anything
   - Related areas: Check affected parts
   - Use abbreviated testing (do not rerun all tests)
4. **Update Issues** - Mark fixed issues as Closed with version, date, and verifier
5. **Create Retest Report** - Reference data/issue-templates.md for template
6. **Decision Point**:
   - If all high-severity fixed: proceed to Path B (Approve)
   - If issues remain: Repeat iteration (max 3 total)

#### Path B: No Issues - Approve

**If test result was PASS:**

1. **Create Sign-Off Document** - Reference data/issue-templates.md for template
2. **Notify Development Team** - Formal approval notification
3. **Update Status** - Set delivery status to 'approved' with timestamp and approver

### Iteration Limits

**Maximum iterations:** 3

If after 3 iterations issues persist:
1. Escalate to leads
2. Review requirements
3. Consider scope reduction

### Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF M: Update design log, then load, read entire file, then execute {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed when user selects 'M'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the feature is approved with sign-off document or escalated will you then load and read fully `{activityWorkflowFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All high-severity issues fixed
- Retesting complete
- Sign-off document created
- Development team notified of approval
- Status updated to approved

### ❌ SYSTEM FAILURE:
- Approving with unfixed high-severity issues
- No sign-off document
- Status not updated
- Development team not notified
- Endless iteration loop (more than 3)

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
