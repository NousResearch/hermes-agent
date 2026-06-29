---
name: 'step-02-execute'
description: 'Execute all test scenarios defined in the test scenario file and document results'

# File References
nextStepFile: './step-03-document-issues.md'
---

# Step 2: Run Test Scenarios

## STEP GOAL:

Execute all test scenarios defined in the test scenario file and document results.

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

- 🎯 Focus only on executing tests in order: happy path, error states, edge cases, design system validation, and accessibility
- 🚫 FORBIDDEN to skip test categories or rush through tests
- 💬 Approach: Execute each test category methodically, documenting results as you go
- 📋 Happy path must work before moving to error states and edge cases

## EXECUTION PROTOCOLS:

- 🎯 All test categories executed with results documented
- 💾 Document results using templates from data/test-result-templates.md
- 📖 Reference test scenario file for each test case
- 🚫 Do not skip any test category

## CONTEXT BOUNDARIES:

- Available context: All prepared materials from Step 1; test scenario file
- Focus: Test execution and result documentation
- Limits: No issue creation yet — just document pass/fail
- Dependencies: Step 1 must be complete (preparation done)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Happy Path Tests

For each test in TS-XXX.yaml `happy_path` section:

1. Start screen recording
2. Perform action exactly as written
3. Observe result, compare to expected
4. Compare to design reference
5. Mark PASS or FAIL
6. Take screenshot if FAIL (naming: `HP-XXX-step-X-FAIL.png`)
7. Document using template

### 2. Error State Tests

For each test in TS-XXX.yaml `error_states` section:

1. Set up error condition using test data
2. Trigger the error
3. Verify error handling (message, styling, recovery)
4. Check against design spec
5. Document results using template

### 3. Edge Case Tests

For each test in TS-XXX.yaml `edge_cases` section:

1. Set up unusual scenario
2. Perform edge case action
3. Verify graceful handling (no crash, smooth UX)
4. Document results using template

### 4. Design System Validation

For each component in TS-XXX.yaml `design_system_checks` section:

1. Locate all component instances
2. Measure dimensions (height, width, padding)
3. Check colors against design tokens
4. Check typography (size, weight, line height)
5. Check spacing
6. Check all states (default, hover, active, disabled, focus)
7. Document results using template

### 5. Accessibility Tests

#### Screen Reader Testing
- Enable VoiceOver (iOS) or TalkBack (Android)
- Navigate through flow using only screen reader
- Check button labels, form field labels, error announcements

#### Color Contrast Testing
- Use contrast checker tool
- Body text: 4.5:1 minimum (WCAG AA)
- Large text: 3:1 minimum

#### Touch Target Testing
- Measure all interactive elements
- Minimum: 44x44px
- Minimum 8px spacing between targets

### 6. Compile Overall Summary

After all tests complete, create overall test summary:
- Overall result (PASS/FAIL)
- Test coverage percentages
- Issues by severity
- Issues by category
- Next steps

### 7. Verify Checklist

- [ ] All happy path tests executed
- [ ] All error state tests executed
- [ ] All edge case tests executed
- [ ] Design system validation complete
- [ ] Accessibility tests complete
- [ ] All results documented
- [ ] Screenshots captured for issues
- [ ] Screen recordings saved

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 3: Document Issues"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN all test categories are executed and results documented will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All happy path tests executed
- All error state tests executed
- All edge case tests executed
- Design system validation complete
- Accessibility tests complete
- All results documented
- Screenshots captured for issues
- Screen recordings saved

### ❌ SYSTEM FAILURE:
- Skipping test categories
- Not documenting results
- No screenshots for issues
- Not checking design references
- Rushing through tests
- Not measuring design system compliance

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
