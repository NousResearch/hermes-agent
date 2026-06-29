# Diagnostic Report Template

**Use this template for generating diagnostic reports during page specification validation.**

---

## Step-Specific Diagnostic Report

```markdown
🔍 [Step Name] Audit

**Status:** ✅ PASS / ⚠️ WARNING / ❌ CRITICAL

**Issues Found:**
1. [Issue type] [Description]
   - Location: Line X-Y
   - Current: [what exists now]
   - Should be: [what it should be]
   - Why: [explanation of why this matters]

2. [Issue type] [Description]
   - Location: Line X-Y
   - Current: [what exists now]
   - Should be: [what it should be]
   - Why: [explanation of why this matters]

**Recommendation:**
[Specific actionable fix with examples]

**Example of Correct Format:**
```[language]
[code example showing correct implementation]
```

Would you like me to fix this?
```

---

## Validation Checklist Format

```yaml
[section_name]_validated:
  field_1: [true/false]
  field_2: [true/false]
  field_3: [true/false]
  status: [pass/warning/critical]
```

---

## Issue Severity Levels

### ✅ PASS
- All checks passed
- No issues found
- Specification meets standards

### ⚠️ WARNING
- Non-critical issues found
- Specification functional but could be improved
- Recommended fixes, not required

### ❌ CRITICAL
- Critical issues that must be fixed
- Missing required sections
- Specification incomplete or non-compliant
- Blocks developer handoff

---

## Common Issue Types

### Missing Section
```markdown
❌ Missing required section: [Section Name]
   - Location: Should appear after [Previous Section]
   - Why: [Explanation of why this section is required]
   - Example: [Show what the section should look like]
```

### Incorrect Format
```markdown
⚠️ Incorrect format: [Element Name]
   - Location: Line X
   - Current: [what's there now]
   - Should be: [correct format]
   - Why: [Explanation of why format matters]
```

### Missing Object ID
```markdown
❌ Missing Object ID: [Component Name]
   - Location: Line X
   - Current: Component has no OBJECT ID declaration
   - Should be: **OBJECT ID**: `component-name`
   - Why: Object IDs enable traceability from spec → code → Figma
```

### Design System Violation
```markdown
❌ Design System violation: CSS details in page spec
   - Location: Line X-Y
   - Current: Contains hex codes, pixel values, CSS classes
   - Should be: Component references with Design System links
   - Why: Page specs focus on WHAT/WHY, Design System handles HOW
```

### Incomplete Coverage
```markdown
⚠️ Incomplete Object Registry coverage
   - Missing: [list of Object IDs not in registry]
   - Orphaned: [list of Object IDs in registry but not in sections]
   - Coverage: X% (should be 100%)
   - Why: Registry must be single source of truth for all elements
```

---

## Recommendation Format

### Simple Fix
```markdown
**Recommendation:**
Add the missing section after [Previous Section]:

```markdown
## [Section Name]

[Content template]
```

Would you like me to add this section?
```

### Complex Fix
```markdown
**Recommendation:**
1. Extract CSS details to Design System documentation
2. Replace inline styles with component references
3. Add Design System links for colors/typography
4. Keep page-specific layout notes (mobile vs desktop behavior)

**Next Steps:**
- Move color values to `Design-System/Foundation/Colors/`
- Move typography to `Design-System/Foundation/Typography/`
- Update page spec to reference Design System components

Would you like me to help extract these styles to the Design System?
```

---

## Final Validation Report Format

```markdown
# Page Specification Quality Report

**Page:** [Page Number] [Page Name]
**Audit Date:** [Date]
**Overall Status:** ✅ PASS / ⚠️ NEEDS WORK / ❌ CRITICAL ISSUES

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
- Platform Metadata: Complete/Incomplete

## Recommendations
[Prioritized list of fixes]

## Next Steps
[What to do next based on findings]
```

---

## Usage Guidelines

1. **Be Specific:** Always include line numbers and exact examples
2. **Be Helpful:** Explain WHY each issue matters
3. **Be Actionable:** Provide clear recommendations with examples
4. **Be Conversational:** Use friendly, collaborative tone
5. **Be Respectful:** Let designer decide whether to implement fixes
6. **Be Thorough:** Don't skip issues, but group related problems

---

## Example Complete Report

```markdown
🔍 Page Metadata Audit

**Status:** ⚠️ WARNING

**Issues Found:**
1. ⚠️ Missing scenario inheritance reference (Line 17-23)
   - Location: Page Metadata section
   - Current: All platform fields present but no inheritance link
   - Should be: "**Inherits From:** Scenario 03 Platform Strategy"
   - Why: Creates explicit traceability from Product Brief → Scenario → Page

**Recommendation:**
Add inheritance reference after Navigation Context:

```markdown
**Navigation Context**: Authenticated - overlays calendar page

**Inherits From**: Scenario 03 Platform Strategy (see scenario overview)
```

This creates explicit traceability chain and ensures platform context is properly inherited.

Would you like me to add this reference?
```
