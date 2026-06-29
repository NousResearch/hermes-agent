---
name: step-04-cross-scenario-consistency
description: Verify scenarios are consistent with each other with no contradictions and balanced coverage

# File References
nextStepFile: './step-05-seo-keyword-alignment.md'
---

# Validation Step 4: Cross-Scenario Consistency

## STEP GOAL:

Verify scenarios are consistent with each other — no contradictions, proper page sharing, balanced persona and business goal coverage, and no duplicate scenarios.

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

- 🎯 Focus only on cross-scenario consistency and balance verification
- 🚫 FORBIDDEN to modify any scenario files during validation
- 💬 Approach: Compare scenarios against each other systematically
- 📋 Check for contradictions, duplicates, and coverage imbalances

## EXECUTION PROTOCOLS:

- 🔗 Check shared page consistency across scenarios
- 📊 Verify persona and business goal balance
- 🔍 Identify any duplicate or overlapping scenarios
- ✅ Validate scenario index accuracy
- 🚫 FORBIDDEN to skip any consistency check

## CONTEXT BOUNDARIES:

- Available context: All scenario outlines, scenario index, Trigger Map
- Focus: Cross-scenario consistency and balance
- Limits: No scenario modifications, only verification and reporting
- Dependencies: All scenario files and index must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Shared Page Consistency

For pages that appear in multiple scenarios:
- [ ] Same page name = same page purpose everywhere
- [ ] Page descriptions are compatible (not contradictory)
- [ ] If a page serves different personas, it should handle both needs

### 2. Persona Balance

- [ ] Priority 1 personas have the most scenarios
- [ ] No persona is over-represented relative to their priority
- [ ] Each primary persona has at least one dedicated scenario

### 3. Business Goal Coverage

- [ ] Each business goal is addressed by at least one scenario
- [ ] High-priority goals have more scenario coverage
- [ ] No business goal is orphaned (referenced but no scenario)

### 4. Scenario Overlap

Check for:
- [ ] No two scenarios are essentially duplicates (same path, different name)
- [ ] Overlapping scenarios have distinct user intents
- [ ] Shared pages are intentional, not accidental

### 5. Scenario Index Verification (00-ux-scenarios.md)

- [ ] Index lists all scenarios
- [ ] Priority assignments are consistent with Trigger Map priorities
- [ ] Coverage matrix is accurate
- [ ] Page count matches actual pages in scenarios

### 6. Generate Report

```
## Cross-Scenario Consistency Report

**Scenarios analyzed:** [N]
**Shared pages:** [N]
**Contradictions found:** [N]
**Duplicate concerns:** [N]

**Persona coverage:**
| Persona | Priority | Scenarios | Status |
|---------|----------|-----------|--------|
| [Name] | P1 | 01, 03 | ✅ |

**Business goal coverage:**
| Goal | Scenarios | Status |
|------|-----------|--------|
| [Goal] | 01, 02 | ✅ |
```

### 7. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to SEO Keyword Alignment validation?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [cross-scenario consistency report generated], will you then load and read fully `{nextStepFile}` to execute and begin SEO keyword alignment validation.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Shared page consistency verified across all scenarios
- Persona balance checked against Trigger Map priorities
- Business goal coverage verified
- Scenario overlap and duplicates checked
- Scenario index accuracy verified
- Consistency report generated with all findings
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Skipping any consistency check
- Not verifying the scenario index accuracy
- Missing contradictions between scenarios
- Not checking persona or business goal balance
- Modifying scenario files during validation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
