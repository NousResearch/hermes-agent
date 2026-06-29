---
name: step-07-quality-review
description: Self-review all scenarios against the quality rubric

# File References
nextStepFile: './step-08-update-design-log.md'

# Data References
qualityChecklist: '../data/quality-checklist.md'
---

# Step 7: Quality Review

## STEP GOAL:

Self-review all scenarios against the quality rubric across four dimensions (completeness, quality criteria, mistakes avoided, best practices), fix any failing items, and present a review summary.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are a UX Scenario Facilitator collaborating with the project owner
- ✅ If you already have been given a name, communication_style and identity, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring scenario thinking and user journey expertise, user brings their project knowledge, together we create concrete UX scenario outlines
- ✅ Maintain collaborative equal-partner tone throughout

### Step-Specific Rules:

- 🎯 Focus only on reviewing quality against the rubric — no new content creation
- 🚫 FORBIDDEN to skip any dimension of the quality review
- 💬 Approach: Be honest and thorough in self-review, fix gaps before proceeding
- 📋 Present clear summary with scores for each scenario

## EXECUTION PROTOCOLS:

- 📖 Load the full quality checklist before starting review
- ✅ Score each scenario across all four dimensions
- 🔧 Fix any failing items before presenting summary
- 🚫 FORBIDDEN to proceed if minimum thresholds are not met

## CONTEXT BOUNDARIES:

- Available context: All completed scenario outlines and overview file
- Focus: Quality verification and gap remediation
- Limits: Only fix quality issues, do not add new scenarios
- Dependencies: All scenarios and overview must be complete from Steps 5-6

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Load Checklist

Load the full checklist: `{qualityChecklist}`

### 2. Review Each Scenario

For **each scenario**, verify these four dimensions:

#### Dimension 1: Completeness (7 components)

- [ ] Core Feature defined (aligned to business goal)
- [ ] Entry Point realistic (device + context + discovery)
- [ ] Mental State with Trigger/Hope/Worry (all three specific)
- [ ] Success Goals mutual (business + user, both measurable)
- [ ] Shortest Path linear (numbered steps, zero branches)
- [ ] Scenario Name includes persona name + ID assigned
- [ ] Trigger Map Connections explicit (persona, forces, goal)

**Score: [X]/7**

#### Dimension 2: Quality Criteria (7 checks)

- [ ] Persona-specific (not generic "user")
- [ ] Mental state is visceral (not "interested" or "curious")
- [ ] Both successes are measurable (not "get more customers")
- [ ] Path has zero "if" statements
- [ ] Minimum viable steps (each step justifies existence)
- [ ] Entry point is realistic (not "user opens app")
- [ ] Business goal connection is explicit (not assumed)

**Score: [X]/7**

#### Dimension 3: Mistakes Avoided (6 checks)

- [ ] No edge cases in sunshine path
- [ ] Goal-first, not feature-first naming
- [ ] Mental state present (not just actions)
- [ ] Page descriptions include purpose (not just page name)
- [ ] Uses Trigger Map persona (not invented user)
- [ ] Business value explicitly defined

**Score: [X]/6 avoided**

#### Dimension 4: Best Practices (4 checks)

- [ ] Scenario named after persona
- [ ] Started with highest-value persona
- [ ] One job-to-be-done per scenario
- [ ] Driving forces explicitly linked

**Score: [X]/4**

### 3. Check Thresholds

**Minimum (must meet to proceed):**
- Completeness: 6/7
- Quality: 5/7
- Mistakes avoided: 6/6 (all must be avoided)
- Best practices: 2/4

**Excellent:**
- All scores maxed

### 4. Fix Failing Items

If any scenario fails:
1. Identify which scenario(s) fail which checks
2. Go back to the scenario file and fix the specific gaps
3. Re-verify after fixing

**If still failing after corrections:** Note remaining gaps and present to user for guidance.

### 5. Present Review Summary

```
## Quality Review Summary

**Scenarios Reviewed:** [N]

| Scenario | Complete | Quality | Mistakes | Practices | Status |
|----------|----------|---------|----------|-----------|--------|
| 01 | 7/7 | 7/7 | 6/6 | 4/4 | ✅ Excellent |
| 02 | 7/7 | 6/7 | 6/6 | 3/4 | ✅ Good |

**Overall:** [Excellent / Good / Needs Work]
**Gaps:** [list any, or "None"]
```

### 6. Present MENU OPTIONS

Display: "Are you ready to [C] Continue to Updating the Design Log?"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- After other menu items execution, return to this menu
- User can chat or ask questions - always respond and then end with display again of the menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [C continue option] is selected and [all scenarios meet minimum quality thresholds], will you then load and read fully `{nextStepFile}` to execute and begin updating the design log.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All scenarios reviewed across all four dimensions
- Quality checklist loaded and applied thoroughly
- Failing items identified and fixed before proceeding
- Clear summary with scores presented to user
- All scenarios meet minimum quality thresholds
- Menu presented and user input handled correctly

### ❌ SYSTEM FAILURE:

- Skipping any review dimension
- Not loading the quality checklist
- Proceeding with scenarios below minimum thresholds
- Not presenting the review summary
- Rubber-stamping without thorough checking

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
