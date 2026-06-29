---
name: step-01-scenario-coverage
description: Verify that all strategic context chains from the Trigger Map are covered by at least one scenario

# File References
nextStepFile: './step-02-navigation-patterns.md'
---

# Validation Step 1: Scenario Coverage

## STEP GOAL:

Verify that all strategic context chains from the Trigger Map are covered by at least one scenario, with Priority 1 chains having dedicated scenarios.

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

- 🎯 Focus only on strategic-context-to-scenario coverage verification
- 🚫 FORBIDDEN to modify any scenario files during validation
- 💬 Approach: Systematic cross-referencing of Trigger Map strategic context against scenarios
- 📋 Report findings with clear severity levels

## EXECUTION PROTOCOLS:

- 📖 Load both Trigger Map and all scenario files
- 🔗 Cross-reference every strategic context chain against scenario coverage
- 📊 Report with severity levels (Critical/Warning/Pass)
- 🚫 FORBIDDEN to skip any chain during verification

## CONTEXT BOUNDARIES:

- Available context: Trigger Map, all scenario outlines, scenario index
- Focus: Strategic context coverage verification only
- Limits: No scenario modifications, only verification and reporting
- Dependencies: All scenario files must exist from Phase 3 creation workflow

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Load Trigger Map Data

Read `{output_folder}/B-Trigger-Map/trigger-map.md` and extract all strategic context chains (Business Goal → Persona → Driving Force chains).

### 2. Load All Scenario Files

Read all scenario outlines from `{output_folder}/C-UX-Scenarios/`.

### 3. Cross-Reference

For each strategic context chain, verify:
- [ ] At least one scenario addresses this chain
- [ ] The scenario Trigger Map Connections section explicitly references the strategic context components
- [ ] Priority 1 chains have dedicated scenarios (not just secondary coverage)

### 4. Generate Report

```
## Coverage Report

| Chain | Persona | Driving Force | Scenario(s) | Status |
|-----|---------|---------------|-------------|--------|
| [Goal] | [Name] | [Force] | [Scenario ID] | ✅/⚠️/❌ |

**Coverage: [X]/[Total] chains covered ([X]%)
**Gaps: [list uncovered chains]]
```

**Severity:**
- ❌ Critical: Priority 1 chain with no scenario
- ⚠️ Warning: Priority 2-3 chain with no scenario
- ✅ Pass: Chain covered by at least one scenario

### 5. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to Navigation Patterns validation?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [coverage report generated with all chains checked], will you then load and read fully `{nextStepFile}` to execute and begin navigation patterns validation.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All strategic context chains from Trigger Map identified and cross-referenced
- Every chain checked against scenario coverage
- Severity levels correctly assigned
- Coverage report generated with clear gaps identified
- Priority 1 chains verified for dedicated scenario coverage
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Missing any chain from the cross-reference
- Not loading all scenario files
- Incorrect severity assignment
- Not identifying coverage gaps
- Modifying scenario files during validation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
