---
name: '4d-present-for-testing'
description: 'Present the implemented section to user with clear test instructions after agent self-verification'
---

# Step 4d: Present Section for Testing

## STEP GOAL:

Present the implemented section to user with clear test instructions after performing agent self-verification.

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

- 🎯 Focus only on agent self-verification with Puppeteer, presenting implementation, and requesting qualitative user review
- 🚫 FORBIDDEN to skip self-verification before presenting to user
- 💬 Approach: Verify first, then present with clear test instructions for qualitative aspects
- 📋 Only present to user when all agent-verifiable criteria pass

## EXECUTION PROTOCOLS:

- 🎯 Agent self-verification complete, section presented to user for qualitative review
- 💾 Record verification results
- 📖 Reference story file acceptance criteria for verification
- 🚫 Do not present to user until self-verification passes

## CONTEXT BOUNDARIES:

- Available context: Implemented section from Step 4c; story file acceptance criteria
- Focus: Self-verification and user presentation
- Limits: No code changes during presentation (unless self-verification fails)
- Dependencies: Step 4c must be complete (section implemented)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 0. Agent Self-Verification (Before Presenting)

**BEFORE presenting to the user, verify your own work with Puppeteer.**

See: [Inline Testing Guide](../data/guides/INLINE-TESTING-GUIDE.md) for full methodology.

**Actions**:

1. Open the page in browser using Puppeteer
2. Set viewport to target device width
3. Verify each agent-verifiable criterion from the story file
4. Narrate findings using the pass/fail pattern (actual vs expected)
5. Fix any failures and re-verify

**If modifying existing features**: Compare against baseline captured before implementation. Confirm only intended changes occurred.

**Only proceed to Step 1 when all agent-verifiable criteria pass.**

### 1. Present Implementation

Present what was built, listing new features with Object IDs and files updated.

### 2. Present Verification Results & Request Qualitative Review

Present Puppeteer verification results, then ask user to evaluate qualitative aspects:
- Feel the flow: Does the interaction feel natural?
- Visual hierarchy: Does your eye go to the right place first?
- Clarity: Is it immediately clear what to do?
- Consistency: Does this section feel like it belongs with the rest?

### 3. Wait for User Feedback

**User will respond with one of**:
- Approved: "Looks good!" / "Y" / "Perfect!" -> Go to `4g-section-approved.md`
- Issue: "The button doesn't..." / "I see a problem with..." -> Go to `4e-handle-issue.md`
- Improvement: "Could we make it..." / "What if we..." -> Go to `4f-handle-improvement.md`

### 4. Present MENU OPTIONS

Display based on user feedback:
- **If approved**: "[C] Continue to Step 4g: Section Approved"
- **If issue reported**: "[C] Continue to Step 4e: Handle Issue"
- **If improvement suggested**: "[C] Continue to Step 4f: Handle Improvement"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute the appropriate next step file
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user has provided feedback will you then load and read fully the appropriate next step file to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Agent self-verification completed before presenting
- All agent-verifiable criteria pass
- Implementation presented clearly with Object IDs
- Qualitative review requested from user
- User feedback captured and routed correctly

### ❌ SYSTEM FAILURE:
- Presenting to user without self-verification
- Skipping Puppeteer verification
- Not requesting qualitative review
- Routing user feedback incorrectly

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
