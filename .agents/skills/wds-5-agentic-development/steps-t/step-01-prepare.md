---
name: 'step-01-prepare'
description: 'Gather all materials and set up testing environment before starting validation'

# File References
nextStepFile: './step-02-execute.md'
---

# Step 1: Prepare for Acceptance Testing

## STEP GOAL:

Gather all materials and set up your testing environment before starting validation.

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

- 🎯 Focus only on gathering test materials, setting up environment, preparing test data, and creating testing workspace
- 🚫 FORBIDDEN to begin executing tests — that is the next step
- 💬 Approach: Systematically walk through preparation with user, ensuring nothing is missed
- 📋 All materials must be gathered and environment verified before testing begins

## EXECUTION PROTOCOLS:

- 🎯 All materials gathered, environment set up, test data prepared, workspace created
- 💾 Document preparation status in dialog file
- 📖 Reference test scenario file and design delivery file
- 🚫 Do not execute any tests during preparation

## CONTEXT BOUNDARIES:

- Available context: Test scenario file, design delivery file, scenario specifications, design system specs
- Focus: Preparation — materials, environment, data, workspace, time estimation
- Limits: No test execution
- Dependencies: Test scenario file and design delivery must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Gather Materials

#### Test Scenario
- Load test scenario file: `test-scenarios/TS-XXX.yaml`
- Review all test cases
- Understand success criteria
- Note any special setup needed

#### Design Delivery
- Load Design Delivery file: `deliveries/DD-XXX.yaml`
- Review user value and success criteria
- Review acceptance criteria
- Understand what "done" looks like

#### Scenario Specifications
- Load all scenario specs from `C-UX-Scenarios/`
- Review each scenario specification
- Note design details
- Understand expected behavior

#### Design System Specs
- Load design system specs from `D-Design-System/`
- Review component specifications
- Review design tokens
- Note exact colors, sizes, spacing

### 2. Set Up Environment

#### Access the Build
- Staging URL, credentials, platform
- Install build if needed (TestFlight, APK, or web staging)

#### Prepare Test Devices
- Primary device: charged, WiFi, screen recording enabled, screenshot tools ready
- Secondary device (if needed): different platform, screen size, OS version

#### Set Up Tools
- Screen recording (QuickTime, built-in, OBS Studio)
- Screenshot tools with annotation
- Markdown editor and note-taking
- Accessibility tools (VoiceOver/TalkBack, contrast checker, zoom)

### 3. Prepare Test Data

Create test accounts and prepare test data:
- Valid and invalid emails
- Strong and weak passwords
- Special characters
- Edge case data (long names, etc.)

### 4. Create Testing Workspace

Create file structure:
```
testing/DD-XXX/
├── screenshots/
├── screen-recordings/
├── notes.md
└── issues-found.md
```

### 5. Review Test Plan

Understand what you are testing:
- Happy Path Tests: count, flows, expected results
- Error State Tests: count, scenarios, error messages
- Edge Case Tests: count, unusual scenarios, expected behavior
- Design System Validation: components to check, specifications
- Accessibility Tests: screen reader, contrast, touch targets

### 6. Time Estimate

Calculate total testing time with 20% buffer.

### 7. Verify Checklist

- [ ] Test scenario loaded and reviewed
- [ ] Design Delivery loaded and reviewed
- [ ] All scenario specs loaded
- [ ] Design system specs loaded
- [ ] Build accessible and working
- [ ] Test devices ready
- [ ] Tools set up (recording, screenshots, notes)
- [ ] Test data prepared
- [ ] Workspace created
- [ ] Time blocked on calendar

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 2: Execute"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN all materials are gathered, environment is set up, and workspace is ready will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All materials gathered
- Environment set up and accessible
- Test devices ready
- Tools configured
- Test data prepared
- Workspace created
- Test plan reviewed
- Time estimated

### ❌ SYSTEM FAILURE:
- Starting testing without materials
- Cannot access staging environment
- Test devices not ready
- No screen recording capability
- No test data prepared
- No time estimate

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
