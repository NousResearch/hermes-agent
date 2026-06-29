---
name: step-02-navigation-patterns
description: Verify that all scenario shortest paths follow WDS navigation conventions

# File References
nextStepFile: './step-03-outline-completeness.md'
---

# Validation Step 2: Navigation Patterns

## STEP GOAL:

Verify that all scenario shortest paths follow WDS navigation conventions, page naming is consistent across scenarios, and navigation flows are logical with no impossible jumps or dead ends.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a Validation Specialist reviewing scenario quality, coverage, and consistency
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring validation expertise and quality standards knowledge, user brings project context, together we ensure scenario quality meets WDS standards
- ✅ Maintain thorough analytical tone throughout

### Step-Specific Rules:

- 🎯 Focus only on navigation pattern verification and page naming consistency
- 🚫 FORBIDDEN to modify any scenario files during validation
- 💬 Approach: Build a page registry and check for conflicts systematically
- 📋 Report navigation conflicts with specific details

## EXECUTION PROTOCOLS:

- 📋 Check page naming consistency across all scenarios
- 🔗 Verify navigation flow rules for each scenario
- 📊 Build cross-scenario page registry
- 🚫 FORBIDDEN to skip any scenario during verification

## CONTEXT BOUNDARIES:

- Available context: All scenario outlines with their shortest paths
- Focus: Navigation pattern verification and page naming consistency
- Limits: No scenario modifications, only verification and reporting
- Dependencies: All scenario files must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Page Naming Consistency

For each scenario shortest path:
- [ ] Page names are consistent across scenarios (same page = same name everywhere)
- [ ] Page names are descriptive and user-facing (not technical identifiers)
- [ ] No duplicate page names with different meanings

### 2. Navigation Flow Rules

For each scenario:
- [ ] Path is truly linear — zero "if" statements, zero branches
- [ ] First step is a landing/entry page (not an internal page)
- [ ] Last step ends with a success state (marked with ✓)
- [ ] Each step transitions naturally to the next (no impossible jumps)
- [ ] No dead ends — every page has a clear next action

### 3. Cross-Scenario Page Registry

Build a page registry from all scenarios:

```
## Page Registry

| Page Name | Used In Scenarios | Role |
|-----------|-------------------|------|
| [Name] | 01, 03 | Landing |
| [Name] | 01, 02, 03 | Service Detail |

**Total unique pages:** [N]
**Shared pages:** [N] (appear in 2+ scenarios)
```

### 4. Navigation Conflicts

Check for conflicts:
- [ ] No scenario routes FROM the same page TO different pages without clear context
- [ ] Shared pages serve consistent purposes across scenarios
- [ ] Entry points are reachable from the described discovery method

### 5. Generate Report

```
## Navigation Pattern Report

**Scenarios checked:** [N]
**Unique pages:** [N]
**Shared pages:** [N]
**Conflicts found:** [N]

[List any issues with severity]
```

### 6. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to Outline Completeness validation?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [navigation pattern report generated], will you then load and read fully `{nextStepFile}` to execute and begin outline completeness validation.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All scenario paths checked for navigation rules
- Page naming consistency verified across all scenarios
- Page registry built with shared page tracking
- Navigation conflicts identified and reported
- Report generated with all findings
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Skipping any scenario during navigation check
- Not building the page registry
- Missing navigation conflicts
- Not verifying page naming consistency
- Modifying scenario files during validation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
