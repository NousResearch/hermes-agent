# Specification Audit Report

**Date:** {YYYY-MM-DD}  
**Auditor:** {Name/Agent}  
**Scope:** {Scenario name or Page name}  
**Audit Level:** {Quick/Standard/Complete}  
**Project:** {Project name}

---

## Executive Summary

**Overall Status:** {✅ Pass / ⚠️ Pass with Issues / ❌ Fail}

**Issue Counts:**
- 🔴 Critical Issues: {count}
- 🟡 Warnings: {count}
- 🔵 Suggestions: {count}

**Recommendation:** {Ready for development / Needs fixes before development / Major rework required}

---

## Level 0: Specification Formatting & Standards

**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

### Markdown Structure
**Checklist:**
- [ ] Proper heading hierarchy (H1 → H2 → H3 → H4)
- [ ] Only one H1 per page
- [ ] No skipped heading levels

**Issues Found:**
- {Issue description, line number, and severity}

---

### Area Label Format
**Checklist:**
- [ ] Format: `**AREA LABEL**: `{label}``
- [ ] Naming convention: `{page}-{section}-{element}`
- [ ] Consistent throughout

**Issues Found:**
- {Issue description, line number, and severity}

---

### Translation Format
**Checklist:**
- [ ] Each language on separate line
- [ ] Format: `- {LANG}: "{content}"`
- [ ] All product languages present
- [ ] Consistent language order
- [ ] No inline translations

**Issues Found:**
- {Issue description, line number, and severity}

---

### List & Code Formatting
**Checklist:**
- [ ] Use `-` for bullets (not `*` or `+`)
- [ ] Consistent indentation
- [ ] Code blocks have language specified
- [ ] Proper spacing

**Issues Found:**
- {Issue description, line number, and severity}

---

### Section Organization
**Checklist:**
- [ ] Sections in standard order
- [ ] No missing required sections
- [ ] Logical flow maintained

**Issues Found:**
- {Issue description, line number, and severity}

---

### File Naming
**Checklist:**
- [ ] Follows WDS naming conventions
- [ ] No generic names (README.md, GUIDE.md)
- [ ] Descriptive and specific

**Issues Found:**
- {Issue description and severity}

---

## Level 1: Scenario-Level Findings

### Strategic Foundation
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

**Checklist:**
- [ ] User situation clearly defined
- [ ] Usage context documented
- [ ] Strategic context (Trigger Map) defined and linked
- [ ] Scenario purpose stated
- [ ] Success criteria defined

**Issues Found:**
- {Issue description and severity}

---

### Navigation Flow
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

**Checklist:**
- [ ] All pages in scenario identified
- [ ] Entry points documented for each page
- [ ] Exit points documented for each page
- [ ] User can navigate through all pages
- [ ] Navigation paths logical and complete

**Issues Found:**
- {Issue description and severity}

---

## Level 2: Page-Level Findings

### Structure & Organization
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

**Checklist:**
- [ ] Page purpose clearly stated
- [ ] Success criteria defined
- [ ] Trigger Map reference present
- [ ] Sections properly separated and named
- [ ] Section purposes defined
- [ ] Page layout logical and flows well

**Structural Area Labels:**
- [ ] Page container (`{page-name}-page`)
- [ ] Header section (`{page-name}-header`)
- [ ] Main content area (`{page-name}-main`)
- [ ] Form container (`{page-name}-form`)
- [ ] Section containers (`{page-name}-{section}-section`)
- [ ] Section header bars (`{page-name}-{section}-header-bar`)

**Issues Found:**
- {Issue description and severity}

---

### Visual-Spec Alignment
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

**Checklist:**
- [ ] Sketch/visualization exists
- [ ] Sketch linked in specification
- [ ] All objects in sketch documented in spec
- [ ] All objects in spec visible in sketch
- [ ] Visual hierarchy matches spec structure

**Misalignments Found:**
- **Objects in sketch but missing from spec:**
  - {Object name and location}
- **Objects in spec but missing from sketch:**
  - {Object name and location}
- **Visual discrepancies:**
  - {Description of mismatch}

---

### Area Label Coverage
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

**Checklist:**
- [ ] All interactive elements have Area Labels
- [ ] Labels follow naming convention (`{page}-{section}-{element}`)
- [ ] Labels are unique within page
- [ ] ARIA labels match Area Labels

**Missing Area Labels:**
- {Element description and suggested label}

**Naming Convention Issues:**
- {ID that doesn't follow pattern and suggested fix}

---

## Level 3: Component-Level Findings

### Componentization
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

**Checklist:**
- [ ] Reusable sections identified
- [ ] Components properly separated from page specs
- [ ] Component specifications exist
- [ ] Component references valid and linked

**Issues Found:**
- **Components needing extraction:**
  - {Component name and pages where it appears}
- **Missing component specs:**
  - {Component name}
- **Broken component references:**
  - {Reference location and issue}

---

### Design System Integration
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail / N/A}

**Checklist:**
- [ ] All components added to design system
- [ ] Components at proper hierarchy level
- [ ] Design tokens applied
- [ ] Figma components linked

**Issues Found:**
- {Issue description and severity}

---

## Level 4: Feature-Level Findings

### Shared Functionality
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

**Checklist:**
- [ ] Common features identified
- [ ] Feature files created and documented
- [ ] Feature references consistent across pages
- [ ] Validation rules centralized

**Issues Found:**
- **Features needing extraction:**
  - {Feature name and pages where it appears}
- **Inconsistent implementations:**
  - {Feature name and inconsistency description}

---

## Level 5: Content Audit Findings

### Text Content
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

**Checklist:**
- [ ] All content defined (no placeholders)
- [ ] Multi-language content complete
- [ ] Field labels present and clear
- [ ] Button text defined
- [ ] Error messages in all languages
- [ ] Success messages in all languages
- [ ] Empty state messages defined
- [ ] Loading state messages defined
- [ ] Meta content (page title, meta description) for public pages
- [ ] Social sharing content (title, description, image) for public pages

**Missing Content:**
- {Element and missing content type}

**Language Gaps:**
- {Content that's missing in specific languages}

**Meta Content Issues:**
- {Missing or incomplete meta tags for public pages}

---

### Accessibility Content
**Status:** {✅ Pass / ⚠️ Warning / ❌ Fail}

**ARIA Labels:**
- [ ] All interactive elements have aria-label
- [ ] ARIA labels descriptive and meaningful

**Missing ARIA Labels:**
- {Element description}

**Images:**
- [ ] All images have alt text
- [ ] Alt text descriptive

**Missing Alt Text:**
- {Image location and suggested alt text}

**Forms:**
- [ ] All inputs have labels
- [ ] Required fields marked
- [ ] Field instructions present

**Form Issues:**
- {Issue description}

**Keyboard Navigation:**
- [ ] Tab order documented
- [ ] Focus management specified
- [ ] Skip links present

**Keyboard Issues:**
- {Issue description}

**Screen Reader Support:**
- [ ] Semantic HTML specified
- [ ] Heading hierarchy logical
- [ ] ARIA live regions for dynamic content

**Screen Reader Issues:**
- {Issue description}

**Visual Accessibility:**
- [ ] Color contrast meets WCAG AA
- [ ] Information not color-dependent
- [ ] Focus indicators visible

**Visual Accessibility Issues:**
- {Issue description}

**WCAG Compliance:**
- [ ] Target level documented
- [ ] Known issues documented

**WCAG Issues:**
- {Issue description}

---

## Summary of Issues

### 🔴 Critical Issues (Must Fix Before Development)

1. **{Issue Title}**
   - **Location:** {Page/Section}
   - **Problem:** {Description}
   - **Impact:** {Why this is critical}
   - **Recommended Fix:** {Specific action to take}

2. **{Issue Title}**
   - **Location:** {Page/Section}
   - **Problem:** {Description}
   - **Impact:** {Why this is critical}
   - **Recommended Fix:** {Specific action to take}

---

### 🟡 Warnings (Should Fix)

1. **{Issue Title}**
   - **Location:** {Page/Section}
   - **Problem:** {Description}
   - **Impact:** {Why this matters}
   - **Recommended Fix:** {Specific action to take}

2. **{Issue Title}**
   - **Location:** {Page/Section}
   - **Problem:** {Description}
   - **Impact:** {Why this matters}
   - **Recommended Fix:** {Specific action to take}

---

### 🔵 Suggestions (Nice to Have)

1. **{Issue Title}**
   - **Location:** {Page/Section}
   - **Problem:** {Description}
   - **Impact:** {Why this would improve quality}
   - **Recommended Fix:** {Specific action to take}

2. **{Issue Title}**
   - **Location:** {Page/Section}
   - **Problem:** {Description}
   - **Impact:** {Why this would improve quality}
   - **Recommended Fix:** {Specific action to take}

---

## Recommendations

### Immediate Actions
1. {Action item with priority and owner}
2. {Action item with priority and owner}

### Before Development Handoff
1. {Action item with priority and owner}
2. {Action item with priority and owner}

### Future Improvements
1. {Action item with priority and owner}
2. {Action item with priority and owner}

---

## Next Steps

- [ ] Fix critical issues
- [ ] Address warnings
- [ ] Consider suggestions
- [ ] Re-audit after fixes
- [ ] Update specifications
- [ ] Update sketches if needed
- [ ] Notify development team when ready

---

## Audit Metrics

**Specification Completeness:** {percentage}%
- Structural Area Labels: {X/Y complete}
- Interactive Area Labels: {X/Y complete}
- Content defined: {X/Y complete}
- Accessibility: {X/Y complete}

**Quality Score:** {percentage}%
- Based on critical issues, warnings, and suggestions

**Development Readiness:** {Ready / Not Ready / Needs Review}

---

**Audit Completed:** {YYYY-MM-DD HH:MM}  
**Next Audit Scheduled:** {YYYY-MM-DD or "After fixes"}

---

_Generated using WDS Specification Audit Workflow_
