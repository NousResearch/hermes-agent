---
name: 'step-10-final-validation'
description: 'Cross-reference all sections, verify sketch coverage, check for broken links, and generate comprehensive quality report'

# File References
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-validate.md'
---

# Step 10: Final Validation & Quality Report

## STEP GOAL:

Cross-reference all sections, verify sketch coverage, check for broken links, validate naming conventions, and generate comprehensive quality report.

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

- 🎯 Focus on comprehensive cross-referencing and quality report generation
- 🚫 FORBIDDEN to skip sketch coverage or link validation
- 💬 Approach: Synthesize findings from all previous steps, perform final cross-checks
- 📋 Final validation catches inconsistencies before handoff and ensures production-readiness

## EXECUTION PROTOCOLS:

- 🎯 Perform final cross-checks and generate comprehensive quality report
- 💾 Optionally add audit stamp to page spec for handoff tracking
- 📖 Reference findings from all previous validation steps (1-9)
- 🚫 FORBIDDEN to generate incomplete quality report

## CONTEXT BOUNDARIES:

- Available context: Page specification, all previous validation results, design system, sketches
- Focus: Final comprehensive validation and quality report
- Limits: This is a synthesis step — do not repeat detailed checks from earlier steps
- Dependencies: Steps 1-9 should be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Cross-Reference Sections

Verify:
- Cross-references between sections are consistent
- All sketch elements are documented in Page Sections
- All Object IDs in sections appear in Object Registry
- Internal links are not broken
- Naming conventions are consistent throughout
- Page numbers match across all references
- No orphaned or undocumented elements

### 2. Verify Sketch Coverage

Confirm every element visible in the sketch has corresponding documentation in Page Sections.

### 3. Validate Internal Links

Check all internal links work (navigation links, design system references, etc.).

### 4. Check Naming Consistency

Verify naming consistency (page numbers, Object ID format, component names) across the entire document.

### 5. Generate Quality Report

Synthesize findings from Steps 1-9 into comprehensive quality report:

```markdown
# Page Specification Quality Report

**Page:** [Page Number] [Page Name]
**Audit Date:** [Date]
**Overall Status:** PASS / NEEDS WORK / CRITICAL ISSUES

## Executive Summary
[Brief overview of specification quality]

## Critical Issues (Must Fix Before Handoff)
[List critical issues from all steps]

## Warnings (Should Fix)
[List warnings from all steps]

## Info (Nice to Have)
[List informational items]

## Coverage Metrics
- Object Registry Coverage: X%
- Sketch Coverage: X%
- Design System References: X%

## Recommendations
[Prioritized list of fixes]

## Next Steps
[What to do next based on findings]
```

### 6. Record Final Validation Result

```yaml
final_validation_complete:
  cross_references_consistent: [true/false]
  sketch_coverage_complete: [true/false]
  links_validated: [true/false]
  naming_conventions_consistent: [true/false]
  no_orphaned_content: [true/false]
  quality_report_generated: [true/false]
  overall_status: [pass/needs_work/critical]
```

### 7. Present MENU OPTIONS

Display: "**Select an Option:** [F] Fix issues | [S] Add audit stamp to page spec | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF F: Help user implement fixes for identified issues, then [Redisplay Menu Options](#7-present-menu-options)
- IF S: Add audit stamp to page specification for handoff tracking, then [Redisplay Menu Options](#7-present-menu-options)
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#7-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user selects an option from the menu and the quality report has been generated will you proceed accordingly. This is the last step in the Validate activity.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All cross-references validated
- Sketch coverage verified
- Internal links checked
- Naming conventions validated
- Comprehensive quality report generated with executive summary
- Coverage metrics calculated
- User presented with fix/stamp/return options

### ❌ SYSTEM FAILURE:

- Skipping cross-reference validation
- Not checking sketch coverage
- Not validating internal links
- Generating incomplete quality report
- Not calculating coverage metrics
- Not synthesizing findings from all previous steps

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
