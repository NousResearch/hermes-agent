---
name: '4e-handle-issue'
description: 'Fix reported issues in the section, document, and re-verify'
---

# Step 4e: Handle Issue

## STEP GOAL:

Fix reported issues in the section. Identify, fix, document, and re-test.

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

- 🎯 Focus only on acknowledging the issue, fixing it, updating the story file with learning, re-verifying, and re-presenting
- 🚫 FORBIDDEN to add unrelated improvements while fixing an issue
- 💬 Approach: Acknowledge, analyze, fix, document the learning, then re-verify
- 📋 Update story file with what was wrong, why, and what was learned

## EXECUTION PROTOCOLS:

- 🎯 Issue fixed, documented in story file, re-verified with Puppeteer
- 💾 Update story file with changes made section
- 📖 Reference the reported issue and story file
- 🚫 Do not add unrelated features or improvements

## CONTEXT BOUNDARIES:

- Available context: User's issue report; current implementation; story file
- Focus: Issue identification, fix, documentation, re-verification
- Limits: Only fix the reported issue — no scope expansion
- Dependencies: User has reported an issue (from Step 4d)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Acknowledge Issue

Acknowledge the specific problem, analyze why it is happening, and describe the fix.

### 2. Fix the Issue

**Actions**:

1. Identify the root cause
2. Make the specific fix in the code
3. Test the fix mentally (does it solve the problem?)
4. Keep the fix focused and local

### 3. Update Story File with Learning

Add to story file `stories/[View].[N]-[section-name].md`:
- Problem: What was wrong
- Root cause: Why it happened
- Solution: What was changed
- Code change: Specific change made
- Learned: What to do differently next time

### 3.5. Re-Verify with Puppeteer

After fixing the issue, run Puppeteer verification before re-presenting:

1. Open page in browser
2. Verify the fix resolves the reported issue
3. Verify no regressions on previously passing criteria
4. Narrate findings with pass/fail

**Only proceed to re-present when all criteria pass.**

### 4. Re-present for Testing

Present the fix, explain what changed, why it works now, and request re-testing.

**Note**: This may loop multiple times until issue is resolved. After re-presenting, route back to Step 4d for user feedback.

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4d: Present for Testing (re-test)"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute `./4d-present-for-testing.md`
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the issue is fixed and re-verified will you then loop back to present for testing again.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Issue acknowledged and analyzed
- Root cause identified
- Focused fix implemented
- Story file updated with learning
- Re-verified with Puppeteer before re-presenting

### ❌ SYSTEM FAILURE:
- Not acknowledging or analyzing the issue
- Fix does not address root cause
- Not updating story file with learning
- Skipping re-verification
- Adding unrelated improvements during fix

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
