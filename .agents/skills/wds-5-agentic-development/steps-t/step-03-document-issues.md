---
name: 'step-03-document-issues'
description: 'Document all problems found during testing as issue tickets'

# File References
nextStepFile: './step-04-report.md'
---

# Step 3: Create Issues

## STEP GOAL:

Document all problems found during testing as issue tickets that can be fixed.

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

- 🎯 Focus only on creating issue files with proper severity, description, screenshots, and design references
- 🚫 FORBIDDEN to create vague or non-actionable issues
- 💬 Approach: Help user document each issue with specific details, design references, and actionable recommendations
- 📋 Every issue must include severity, steps to reproduce, expected vs actual, screenshot, and design reference

## EXECUTION PROTOCOLS:

- 🎯 All issues documented as numbered issue files with proper severity
- 💾 Create issue files in `issues/ISS-XXX-description.md` and issues summary
- 📖 Reference test results from Step 2 and issue templates
- 🚫 Do not create vague descriptions

## CONTEXT BOUNDARIES:

- Available context: Test results from Step 2; screenshots; design specifications
- Focus: Issue creation — specific, actionable, with design references
- Limits: No fixing issues — just documenting them
- Dependencies: Step 2 must be complete (all tests executed)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Issue Creation Process

For each issue found, create issue file: `issues/ISS-XXX-description.md`

Numbering: Start at ISS-001, increment for each issue, use leading zeros.

Reference: [data/issue-templates.md](data/issue-templates.md) for complete issue template

### 2. Assign Severity Levels

| Severity | Description | Fix Timeline |
|----------|-------------|--------------|
| **Critical** | App crashes, data loss, security | Immediate |
| **High** | Major functionality broken | This release |
| **Medium** | Feature wrong, confusing UX | This release |
| **Low** | Minor polish, nice to have | Future release |

### 3. Issue Writing Best Practices

**Be specific:**
- Not "Button looks wrong"
- Instead "Primary button background #3B82F6, should be #2563EB per tokens/colors.json"

**Be actionable:**
- Not "Fix the transition"
- Instead "Add 300ms fade transition per specifications.md line 45"

**Be visual:**
- Include screenshots
- Annotate key areas
- Show expected vs actual

### 4. Create Issues Summary

After creating all issues, create summary with total count and breakdown by severity.

### 5. Verify Checklist

- [ ] All issues documented with correct template
- [ ] Severity levels assigned appropriately
- [ ] Design references included
- [ ] Screenshots attached
- [ ] Recommendations provided
- [ ] Issues summary created

### 6. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 4: Report"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN all issues are documented with proper severity and design references will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All issues documented with correct template
- Severity levels assigned appropriately
- Design references included
- Screenshots attached
- Recommendations provided
- Issues summary created

### ❌ SYSTEM FAILURE:
- Vague descriptions
- Missing severity
- No screenshots
- No design reference
- No steps to reproduce
- Not actionable

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
