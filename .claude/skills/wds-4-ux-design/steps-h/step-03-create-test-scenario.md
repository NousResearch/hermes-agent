---
name: 'step-03-create-test-scenario'
description: 'Define how to validate Design Delivery after implementation'

# File References
nextStepFile: './step-04-handoff-dialog.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-handover.md'
---

# Step 3: Create Test Scenario

## STEP GOAL:

Define how to validate Design Delivery after implementation by creating a Test Scenario file.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Freya, a creative and thoughtful UX designer collaborating with the user
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design expertise and systematic thinking, user brings product vision and domain knowledge
- ✅ Maintain creative and thoughtful tone throughout

### Step-Specific Rules:

- 🎯 Focus on creating a complete Test Scenario file
- 🚫 FORBIDDEN to skip any test type category
- 💬 Approach: Work through each test category sequentially with user input
- 📋 Test Scenario guides validation testing after implementation

## EXECUTION PROTOCOLS:

- 🎯 Build Test Scenario file section by section with user input
- 💾 Save test scenario file to `test-scenarios/TS-XXX-name.yaml`
- 📖 Reference Design Delivery file for test objectives
- 🚫 FORBIDDEN to save incomplete test scenario

## CONTEXT BOUNDARIES:

- Available context: Design Delivery file, scenario specifications, design system
- Focus: Test scenario creation only
- Limits: Do not conduct tests (that is a later phase)
- Dependencies: Design Delivery file must be created (step 02)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Initialize Test Scenario File

- Choose test scenario ID using `TS-XXX` format (matching the DD-XXX number)
- Create file at `test-scenarios/TS-XXX-name.yaml`
- Fill out basic metadata:
  - `id`: TS-XXX
  - `delivery_id`: DD-XXX (link to delivery)
  - `name`: Descriptive test name
  - `status`: draft
  - `created`: Current date
- Define test objectives: what are we validating and why?

### 2. Define Happy Path Tests

For each main user flow in the delivery:
- **Test name**: Descriptive action being tested
- **Steps**: Numbered sequence of user actions
- **Expected result**: What should happen at each step
- **Design reference**: Link to scenario specification

### 3. Define Error State Tests

For each error scenario:
- **Trigger**: What causes the error (invalid input, network failure, etc.)
- **Expected error message**: Exact text or pattern
- **Recovery path**: How the user gets back on track
- **Graceful degradation**: What still works when this fails

### 4. Define Edge Case Tests

For boundary conditions and unusual scenarios:
- **Empty states**: No data, first-time user, cleared history
- **Boundary values**: Max lengths, zero values, special characters
- **Concurrent actions**: Multiple tabs, rapid clicks, interrupted flows
- **Expected behavior**: What should happen in each case

### 5. Define Design System Validation

- List components to validate against design system spec
- Define token verification:
  - Colors match design tokens
  - Typography follows type scale
  - Spacing follows spacing system
- Check component usage matches approved patterns

### 6. Define Accessibility Tests

- **Screen reader**: All content readable, logical order, ARIA labels present
- **Color contrast**: Meets WCAG AA (4.5:1 text, 3:1 large text)
- **Touch targets**: Minimum 44x44px interactive areas
- **Keyboard navigation**: All interactive elements reachable via Tab, operable via Enter/Space

### 7. Define Sign-Off Criteria

- **Pass threshold**: What percentage of tests must pass
- **Must-fix**: Issues that block sign-off (broken flows, accessibility failures)
- **Nice-to-fix**: Issues to track but not blocking (minor visual differences)
- **Approval process**: Who signs off and how

### 8. Validate Test Scenario File

Before proceeding, verify:

- [ ] Test scenario ID matches delivery ID
- [ ] All test types are defined
- [ ] Each test has clear expected results
- [ ] Design system validation is complete
- [ ] Accessibility tests are included
- [ ] Sign-off criteria are clear

<output>Test Scenario file created: `test-scenarios/TS-XXX-name.yaml`</output>

### 9. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Handoff Dialog | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#9-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user selects an option from the menu and the Test Scenario file has been created and validated will you proceed to the next step or return as directed.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Test scenario ID matches delivery ID
- Happy path tests defined for all main flows
- Error state tests defined
- Edge case tests defined
- Design system validation defined
- Accessibility tests included
- Sign-off criteria clear
- Test scenario file saved

### ❌ SYSTEM FAILURE:

- Skipping any test type category
- Saving incomplete test scenario
- Not linking to Design Delivery
- Tests without clear expected results
- Missing accessibility tests
- Generating tests without user input

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
