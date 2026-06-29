---
name: step-03-outline-completeness
description: Verify every scenario outline has all 7 required components with sufficient quality

# File References
nextStepFile: './step-04-cross-scenario-consistency.md'
---

# Validation Step 3: Outline Completeness

## STEP GOAL:

Verify every scenario outline has all 7 required components with sufficient quality, scoring each component and identifying specific gaps.

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

- 🎯 Focus only on validating the 7 required components of each scenario
- 🚫 FORBIDDEN to modify any scenario files during validation
- 💬 Approach: Check each component systematically with specific quality criteria
- 📋 Score each component and provide actionable gap descriptions

## EXECUTION PROTOCOLS:

- 📋 Check all 7 components for each scenario
- ✅ Score each component with pass/warning/fail
- 📊 Generate completeness report with specific gaps
- 🚫 FORBIDDEN to skip any component or any scenario

## CONTEXT BOUNDARIES:

- Available context: All scenario outlines
- Focus: Component completeness and quality verification
- Limits: No scenario modifications, only verification and reporting
- Dependencies: All scenario files must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Validate Each Scenario

For **each scenario**, validate all 7 components:

#### Component 1: Scenario Name & ID
- [ ] Name includes persona name
- [ ] ID assigned (01, 02, etc.)
- [ ] Slug follows format: `NN-descriptive-name`

#### Component 2: Core Feature
- [ ] Stated as user purpose (not feature name)
- [ ] Aligned to a specific business goal from Trigger Map

#### Component 3: Entry Point
- [ ] Device specified (mobile/desktop/tablet)
- [ ] Context described (where user is, what they are doing)
- [ ] Discovery method specified (search, link, ad, bookmark, etc.)
- [ ] Realistic — not "user opens app"

#### Component 4: Mental State
- [ ] Trigger present and specific (what just happened)
- [ ] Hope present and specific (what they want)
- [ ] Worry present and specific (what they fear)
- [ ] All three are visceral, not generic

#### Component 5: Success Goals
- [ ] User success defined and measurable
- [ ] Business success defined and measurable
- [ ] Both are specific — not "get more customers"

#### Component 6: Shortest Path
- [ ] Linear — zero "if" statements
- [ ] Each step has page name + purpose
- [ ] Minimum viable steps (each justifies existence)
- [ ] Final step marked with ✓

#### Component 7: Trigger Map Connections
- [ ] Persona referenced (with priority level)
- [ ] Positive driving force(s) linked
- [ ] Negative driving force(s) linked
- [ ] Business goal referenced (with objective number)

### 2. Generate Report

```
## Outline Completeness Report

| Scenario | Name | Feature | Entry | Mental | Success | Path | TM Links | Score |
|----------|------|---------|-------|--------|---------|------|----------|-------|
| 01 | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | 6.5/7 |

**All scenarios complete:** [Yes/No]
**Issues found:** [list specific gaps]
```

### 3. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to Cross-Scenario Consistency validation?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [completeness report generated for all scenarios], will you then load and read fully `{nextStepFile}` to execute and begin cross-scenario consistency validation.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All 7 components checked for every scenario
- Each component scored with clear pass/warning/fail
- Specific gaps identified with actionable descriptions
- Completeness report generated with scores
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Skipping any component or any scenario
- Not providing specific gap descriptions
- Giving pass scores without thorough checking
- Modifying scenario files during validation
- Not generating the completeness report

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
