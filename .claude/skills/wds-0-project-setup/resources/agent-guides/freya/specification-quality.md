# Freya's Specification Quality Guide

**When to load:** Before creating any page spec, component definition, or scenario documentation

---

## Core Principle

**If I can't explain it logically, it's not ready to specify.**

Gaps in logic become bugs in code. Clear specifications = confident implementation.

---

## The Logical Explanation Test

Before you write any specification, ask:

**"Can I explain WHY this exists and HOW it works without hand-waving?"**

- ✅ "This button triggers the signup flow, serving users who want to feel prepared (driving force)"
- ❌ "There's a button here... because users need it?"

**If you can't explain it clearly, stop and think deeper.**

---

## Area Label Structure & Hierarchy

**Area Labels follow a consistent hierarchical pattern to identify UI locations across sketch, specification, and code.**

### Structural Area Labels (Containers)
These define the page architecture and visual grouping:

- `{page-name}-page` - Top-level page wrapper
- `{page-name}-header` - Header section container
- `{page-name}-main` - Main content area
- `{page-name}-form` - Form element wrapper
- `{page-name}-{section}-section` - Section containers
- `{page-name}-{section}-header-bar` - Section header bars

**Purpose:** Organize page structure, enable Figma layer naming (via aria-label), support testing selectors (via id attribute)

### Interactive Area Labels (Components)
These identify specific interactive elements:

- `{page-name}-{section}-{element}` - Standard pattern
- `{page-name}-input-{field}` - Form inputs
- `{page-name}-button-{action}` - Buttons
- `{page-name}-error-{field}` - Error messages

**Purpose:** Enable user interaction, form validation, accessibility, and location tracking across design and code

**Note:** Area Labels become both `id` and `aria-label` attributes in HTML implementation.

### Purpose-Based Naming

**Name components by FUNCTION, not CONTENT**

### Good (Function)
- `hero-headline` - Describes its role on the page
- `primary-cta` - Describes its function in the flow
- `feature-benefit-section` - Describes what it does
- `form-validation-error` - Describes when it appears

### Bad (Content)
- `welcome-message` - What if the message changes?
- `blue-button` - What if we change colors?
- `first-paragraph` - Position isn't purpose
- `email-error-text` - Too specific, not reusable

**Why this matters:**
- Content changes, function rarely does
- Makes specs maintainable
- Helps developers understand intent
- Enables component reuse
- Supports Figma html.to.design layer naming

---

## Clear Component Purpose

**Every component needs a clear job description:**

### Template
```markdown
### [Component Name]

**Purpose:** [What job does this do?]
**Triggers:** [What user action/state causes this?]
**Serves:** [Which driving force or goal?]
**Success:** [How do we know it worked?]
```

### Example
```markdown
### Primary CTA Button

**Purpose:** Initiate account creation flow
**Triggers:** User clicks after reading value proposition
**Serves:** User's desire to "feel prepared" (positive driving force)
**Success:** User enters email and moves to step 2
```

---

## Section-First Workflow

**Understand the WHOLE before detailing the PARTS**

### Wrong Approach (Bottom-Up)
1. Design individual components
2. Try to arrange them into sections
3. Hope the page makes sense
4. Realize it doesn't flow logically
5. Start over

### Right Approach (Top-Down)
1. **Define structural containers** - Page, header, main, sections
2. **Assign structural Area Labels** - `{page}-page`, `{page}-header`, etc.
3. **Identify page sections** - What major areas exist?
4. **Define section purposes** - Why does each section exist?
5. **Confirm flow logic** - Does the story make sense?
6. **Detail each section** - Now design components
7. **Specify components** - With clear purpose and context
8. **Assign interactive Area Labels** - `{page}-{section}-{element}`

**Result:** Logical flow, no gaps, confident specifications, complete Area Label coverage

### Area Label Coverage Checklist
- [ ] Page container (`{page}-page`)
- [ ] Header section (`{page}-header`)
- [ ] Main content area (`{page}-main`)
- [ ] Form container if applicable (`{page}-form`)
- [ ] Section containers (`{page}-{section}-section`)
- [ ] Section header bars if visible (`{page}-{section}-header-bar`)
- [ ] All interactive elements (`{page}-{section}-{element}`)

---

## Multi-Language from the Start

**Never design in one language only**

### Grouped Translations
```markdown
#### Hero Headline

**Content:**
- EN: "Stop losing clients to poor proposals"
- SE: "Sluta förlora kunder på dåliga offerter"
- NO: "Slutt å miste kunder på dårlige tilbud"

**Purpose:** Hook Problem Aware users by validating frustration
```

### Why This Matters
- Prevents "English-first" bias
- Reveals translation issues early
- Shows if message works across cultures
- Keeps translations coherent (grouped by component)

---

## Specification Quality Checklist

Before marking a spec "complete":

### Core Quality
- [ ] **Logical Explanation** - Can I explain WHY and HOW?
- [ ] **Purpose-Based Names** - Named by function, not content?
- [ ] **Clear Purpose** - Every component has a job description?
- [ ] **Section-First** - Whole page flows logically?
- [ ] **Multi-Language** - All product languages included?
- [ ] **No Hand-Waving** - No "probably" or "maybe" or "users will figure it out"?

### Area Labels
- [ ] **Structural Area Labels** - Page, header, main, sections all have labels?
- [ ] **Interactive Area Labels** - All buttons, inputs, links have labels?
- [ ] **Area Label Hierarchy** - Labels follow `{page}-{section}-{element}` pattern?
- [ ] **Figma-Ready** - Area Labels support html.to.design layer naming?

### Accessibility
- [ ] **ARIA Labels** - All interactive elements have aria-label attributes?
- [ ] **Alt Text** - All images have descriptive alt attributes?
- [ ] **Form Labels** - All inputs have associated labels?
- [ ] **Keyboard Navigation** - Tab order and focus management documented?
- [ ] **Screen Reader Support** - Semantic HTML and ARIA attributes specified?
- [ ] **Color Contrast** - WCAG AA compliance (4.5:1 for text)?
- [ ] **Error Announcements** - Error messages accessible to screen readers?
- [ ] **Heading Hierarchy** - Logical H1-H6 structure documented?

### SEO (Public Pages)
- [ ] **H1 Present** - Exactly one H1 on the page, contains primary keyword?
- [ ] **Heading Hierarchy** - Logical H1 → H2 → H3, no skipped levels?
- [ ] **URL Slug** - Defined, keyword-rich, matches project brief keyword map?
- [ ] **Primary Keyword** - Identified and placed in H1, title tag, meta description?
- [ ] **Meta Title** - ≤ 60 chars, includes primary keyword + brand?
- [ ] **Meta Description** - 150-160 chars, includes keyword + CTA?
- [ ] **Image Alt Text** - All images have descriptive alt text in all languages?
- [ ] **Internal Links** - At least 2 links to other pages on the site?
- [ ] **Structured Data** - Schema type specified per project brief plan?

### Content Completeness
- [ ] **All Text Defined** - No placeholder content?
- [ ] **Error Messages** - All error states have messages in all languages?
- [ ] **Success Messages** - Confirmation messages defined?
- [ ] **Empty States** - Messages for no-data scenarios?
- [ ] **Loading States** - Loading indicators and messages?
- [ ] **Meta Content** - Page title and meta description for public pages?
- [ ] **Social Sharing** - Social media title, description, and image for public pages?

### Implementation Ready
- [ ] **Developer-Ready** - Could someone build this confidently?
- [ ] **Component References** - All design system components linked?
- [ ] **API Endpoints** - Data requirements documented?
- [ ] **Validation Rules** - Form validation clearly specified?

---

## Red Flags (Stop and Rethink)

🚩 **Vague language:** "Something here to help users understand..."  
🚩 **Content-based names:** "blue-box", "top-paragraph"  
🚩 **Missing purpose:** "There's a button... because buttons are good?"  
🚩 **Illogical flow:** "This section comes after that one... because?"  
🚩 **English-only:** "We'll translate later..."  
🚩 **Gaps in logic:** "Users will just know what to do here"  
🚩 **Missing accessibility:** "We'll add ARIA labels during development..."  
🚩 **No alt text:** Images without descriptive alternatives
🚩 **Unlabeled inputs:** Form fields without associated labels
🚩 **No SEO section:** Public page without URL slug, keywords, or meta content

**When you spot these, pause and dig deeper.**

---

## The Developer Trust Test

**Imagine handing your spec to a developer who:**
- Has never seen your sketches
- Doesn't know the business context
- Speaks a different language
- Lives in a different timezone

**Could they build this confidently?**

- ✅ Yes → Good spec
- ❌ No → More work needed

---

## Related Resources

- **File Naming:** `skill:wds-0-project-setup` → FILE-NAMING-CONVENTIONS.md
- **Language Config:** `skill:wds-0-project-setup` → language-configuration-guide.md
- **Page Spec Template:** `./resources/wds-4-ux-design/templates/page-specification.template.md`

---

*Quality specifications are the foundation of confident implementation.*

