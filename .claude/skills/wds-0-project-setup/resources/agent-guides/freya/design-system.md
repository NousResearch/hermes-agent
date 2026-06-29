# Freya's Design System Guide

**When to load:** When Phase 7 (Design System) is enabled and component questions arise

---

## Core Principle

**Design systems grow organically - discover components through actual work, never create speculatively.**

---

## Three Design System Modes

### Mode A: No Design System
**What it means:**
- All components stay page-specific
- No component extraction
- AI/dev team handles consistency
- Faster for simple projects

**When this workflow doesn't run:**
- Phase 7 is disabled
- Components reference page context only

**Agent behavior:**
- Create components as page-specific
- Use clear, descriptive class names
- No need to think about reusability

---

### Mode B: Custom Figma Design System
**What it means:**
- Designer defines components in Figma
- Components extracted as discovered during Phase 4
- Figma MCP endpoints for integration
- Component IDs link spec ↔ Figma

**Workflow:**
1. Designer creates component in Figma
2. Component discovered during page design
3. Agent links to Figma via Component ID
4. Specification references Figma source

**See:** `skill:wds-6-asset-generation` → workflow-figma.md

---

### Mode C: Component Library Design System
**What it means:**
- Uses shadcn/Radix/similar library
- Library chosen during setup
- Components mapped to library defaults
- Variants customized as needed

**Workflow:**
1. Component needed during page design
2. Check if library has it (button, input, card, etc.)
3. Map to library component
4. Document customizations (if any)

---

## The Design System Router

**Runs automatically during Phase 4 component specification**

**For each component:**
1. Check: Design system enabled? (Mode B or C)
2. If NO → Create page-specific, continue
3. If YES → Call design-system-router.md

**Router asks:**
- Is this component new?
- Is there a similar component?
- Should we create new or use/extend existing?

**See:** `skill:wds-7-design-system` → design-system-router.md

---

## Never Create Components Speculatively

### ❌ Wrong Approach
"Let me create a full component library upfront..."

**Why bad:**
- You don't know what you'll actually need
- Over-engineering
- Wasted effort on unused components

---

### ✅ Right Approach
"I'm designing the landing page hero... oh, I need a button."

**Process:**
1. Design the button for this specific page
2. When another page needs a button → Opportunity!
3. Assess: Similar enough to extract?
4. Extract to Design System if makes sense

**Result:** Components emerge from real needs.

---

## Opportunity/Risk Assessment

**When similar component exists, run assessment:**

**See:** `skill:wds-7-design-system` → assessment/

**7 Micro-Steps:**
1. Scan existing components
2. Compare attributes (visual, behavior, states)
3. Calculate similarity score
4. Identify opportunities (reuse, consistency)
5. Identify risks (divergence, complexity)
6. Present decision to designer
7. Execute decision

**Outcomes:**
- **Use existing** - Component is close enough
- **Create variant** - Extend existing with new state
- **Create new** - Too different, warrants separate component
- **Update existing** - Existing is too narrow, expand it

---

## Foundation First

**Before any components:**

### 1. Design Tokens
```
Design tokens = the DNA of your design system

Colors:
- Primary, secondary, accent
- Neutral scale (50-900)
- Semantic (success, warning, error, info)

Typography:
- Font families
- Font scales (h1-h6, body, caption)
- Font weights
- Line heights

Spacing:
- Spacing scale (xs, sm, md, lg, xl)
- Layout scales

Effects:
- Border radius scale
- Shadow scale
- Transitions
```

**Why first:** Tokens ensure consistency across all components.

---

### 2. Atomic Design Structure

**Organize from simple → complex:**

```
atoms/
├── button.md
├── input.md
├── label.md
├── icon.md
└── badge.md

molecules/
├── form-field.md (label + input + error)
├── card.md (container + content)
└── search-box.md (input + button + icon)

organisms/
├── header.md (logo + nav + search + user-menu)
├── feature-section.md (headline + cards + cta)
└── form.md (multiple form-fields + submit)
```

**Why this structure:** Clear dependencies, easy to understand, scales well.

---

## Component Operations

**See:** `skill:wds-7-design-system` → operations/

### 1. Initialize Design System
**First component triggers auto-initialization**
- Creates folder structure
- Creates design-tokens.md
- Creates component-library-config.md (if Mode C)

### 2. Create New Component
- Defines component specification
- Assigns Component ID
- Documents states and variants
- Notes where used

### 3. Add Variant
- Extends existing component
- Documents variant trigger
- Updates component spec

### 4. Update Component
- Modifies existing definition
- Increments version
- Documents change rationale

---

## Component Specification Template

```markdown
# [Component Name] [COMP-001]

**Type:** [Atom|Molecule|Organism]
**Library:** [shadcn Button|Custom|N/A]
**Figma:** [Link if Mode B]

## Purpose
[What job does this component do?]

## Variants
- variant-name: [When to use]
- variant-name: [When to use]

## States
- default
- hover
- active
- disabled
- loading (if applicable)
- error (if applicable)

## Props/Attributes
| Prop | Type | Default | Description |
|------|------|---------|-------------|
| size | sm\|md\|lg | md | Button size |
| variant | primary\|secondary | primary | Visual style |

## Styling
[Design tokens or Figma reference]

## Used In
- [Page name] ([Component purpose in context])
- [Page name] ([Component purpose in context])

## Version History
- v1.0.0 (2024-01-01): Initial creation
```

---

## Integration with Phase 4

**Phase 4 (UX Design) → Phase 7 (Design System) flow:**

```
User creates page specification
├── Component 1: Button
│   ├── Check: Design system enabled?
│   ├── YES → Router checks existing components
│   ├── Similar button found → Opportunity/Risk Assessment
│   └── Decision: Use existing primary button variant
├── Component 2: Input
│   ├── Check: Design system enabled?
│   ├── YES → Router checks existing components
│   ├── No similar input → Create new
│   └── Add to Design System
└── Component 3: Custom illustration
    ├── Check: Design system enabled?
    └── NO extraction (one-off asset)
```

**Result:**
- Page spec contains references + page-specific content
- Design System contains component definitions
- Clean separation maintained

---

## Common Mistakes

### ❌ Creating Library Before Designing
"Let me make 50 components upfront..."
- **Instead:** Design pages, extract components as needed

### ❌ Over-Abstracting Too Early
"This button might need 10 variants someday..."
- **Instead:** Start simple, add variants when actually needed

### ❌ Forcing Reuse
"I'll make this work even though it's awkward..."
- **Instead:** Sometimes a new component is better than a forced variant

### ❌ No Design Tokens
"I'll define colors per component..."
- **Instead:** Tokens first, components second

---

## Quality Checklist

Before marking a component "complete":

- [ ] **Clear purpose** - What job does it do?
- [ ] **Design tokens** - Uses tokens, not hard-coded values?
- [ ] **All states** - Default, hover, active, disabled documented?
- [ ] **Variants** - Each variant has clear use case?
- [ ] **Reusability** - Can be used in multiple contexts?
- [ ] **Documentation** - Specification is complete?
- [ ] **Examples** - Shows where it's actually used?

---

## Related Resources

- **Phase 7 Workflow:** `skill:wds-7-design-system`
- **Figma Integration:** `skill:wds-6-asset-generation` → workflow-figma.md
- **Shared Knowledge:** design-system data (tokens, naming, states, validation, boundaries)

---

*Components emerge from real needs. Design systems grow organically.*

