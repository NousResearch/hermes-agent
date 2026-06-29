---
name: 'step-04-report'
description: 'Create a comprehensive test report summarizing all testing results'

# File References
nextStepFile: './step-05-iterate.md'
---

# Step 4: Create Test Report

## STEP GOAL:

Create a comprehensive test report summarizing all testing results.

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

- 🎯 Focus only on creating the test report with summary, coverage, issues, sign-off recommendation, and attachments
- 🚫 FORBIDDEN to modify test results or issues — only compile them into the report
- 💬 Approach: Compile all results into a clear, actionable report with user
- 📋 Report must include clear PASS/FAIL determination with criteria

## EXECUTION PROTOCOLS:

- 🎯 Complete test report created with all sections
- 💾 Save report to `testing/DD-XXX/TR-XXX-[flow-name].md`
- 📖 Reference test results from Step 2 and issues from Step 3
- 🚫 Do not modify test results or issues

## CONTEXT BOUNDARIES:

- Available context: Test results from Step 2; issues from Step 3; screenshots and recordings
- Focus: Report compilation — summary, coverage, issues, recommendation
- Limits: No test result modification
- Dependencies: Step 3 must be complete (all issues documented)

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Create Test Report File

File: `testing/DD-XXX/TR-XXX-[flow-name].md`

Reference: [data/issue-templates.md](data/issue-templates.md) for complete test report template

### 2. Report Sections

1. **Summary** - Overall result, total issues, blocking status
2. **Test Coverage** - Pass/fail by category
3. **Issues Found** - Table of all issues
4. **Sign-Off Recommendation** - Ready or needs fixes
5. **Next Steps** - What happens next
6. **Attachments** - Recordings, screenshots, issue files

### 3. Overall Result Determination

**PASS if:**
- All Critical issues: 0
- All High issues: Fixed or accepted risk
- Happy path: 100% pass
- Design system: > 95% compliant

**FAIL if:**
- Any Critical issues unfixed
- Any High issues blocking
- Happy path failures
- Design system < 95% compliant

### 4. Attach Supporting Files

Organize testing folder with report, screenshots, recordings, and test data.

### 5. Verify Checklist

- [ ] Test report created with all sections
- [ ] Test coverage complete
- [ ] Issues list accurate
- [ ] Clear recommendation
- [ ] All attachments organized

### 6. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 5: Iterate"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the test report is complete with all sections and clear recommendation will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Test report created with all sections
- Test coverage complete
- Issues list accurate
- Clear recommendation
- All attachments organized

### ❌ SYSTEM FAILURE:
- Missing test categories
- Incorrect issue counts
- Unclear recommendation
- Missing attachments
- Incomplete coverage data

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
