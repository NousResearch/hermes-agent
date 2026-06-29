---
name: 'step-08e-generate-catalog'
description: 'Generate or update the interactive HTML catalog showcasing all design system components'

# File References
activityWorkflowFile: '../workflow-create.md'
---

# Step 8e: Generate Catalog

## STEP GOAL:

Generate or update the interactive HTML catalog from design system data. Load template, gather project info, generate navigation, token sections, component sections, and changelog.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are the Design System Architect guiding design system creation and maintenance
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring design system expertise and component analysis, user brings design knowledge and project context
- ✅ Maintain systematic and analytical tone throughout

### Step-Specific Rules:

- 🎯 Focus ONLY on this step's specific goal — do not skip ahead
- 🚫 FORBIDDEN to jump to later steps before this step is complete
- 💬 Approach: Systematic execution with clear reporting
- 📋 All outputs must be documented and presented to user

## EXECUTION PROTOCOLS:

- 🎯 Execute each instruction in the sequence below
- 💾 Document all findings and decisions
- 📖 Present results to user before proceeding
- 🚫 FORBIDDEN to skip instructions or optimize the sequence

## CONTEXT BOUNDARIES:

- Available context: Previous step outputs and project configuration
- Focus: This step's specific goal only
- Limits: Do not perform actions belonging to subsequent steps
- Dependencies: Requires all previous steps to be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

## Input

**Design System Files:**

- `D-Design-System/components/*.md` - All component specifications
- `D-Design-System/design-tokens.md` - Design token definitions
- `D-Design-System/figma-mappings.md` - Figma references (if Mode B)
- `D-Design-System/component-library-config.md` - Library config (if Mode C)

**Project Config:**

- Project name
- Design system mode
- Version number
- Creation date

---

## Output

**Generated File:**

- `D-Design-System/catalog.html` - Interactive HTML catalog

**Features:**

- Fixed sidebar navigation
- Live component previews
- Interactive state toggles
- Code examples
- Design token swatches
- Changelog
- Figma links (if Mode B)
- Responsive design

---

## Step 1: Load Template

<action>
Load catalog template:
</action>

**File:** `workflows/wds-7-design-system/templates/catalog.template.html`

**Template variables:**

```
{{PROJECT_NAME}}
{{PROJECT_ICON}}
{{PROJECT_DESCRIPTION}}
{{PROJECT_OVERVIEW}}
{{VERSION}}
{{COMPONENT_COUNT}}
{{DESIGN_SYSTEM_MODE}}
{{CREATED_DATE}}
{{LAST_UPDATED}}
{{INSTALLATION_INSTRUCTIONS}}
{{USAGE_EXAMPLE}}
{{COMPONENT_NAVIGATION}}
{{DESIGN_TOKENS_CONTENT}}
{{COLOR_TOKENS}}
{{TYPOGRAPHY_TOKENS}}
{{SPACING_TOKENS}}
{{COMPONENTS_CONTENT}}
{{CHANGELOG_CONTENT}}
{{FIGMA_LINKS}}
```

---

## Step 2: Gather Project Information

<action>
Extract project metadata:
</action>

**From project config:**

```yaml
project_name: 'Dog Week'
project_icon: '🐕'
project_description: 'Family dog care coordination platform'
design_system_mode: 'custom' # or "library" or "none"
created_date: '2024-09-15'
version: '1.0.0'
```

**Calculate:**

```
component_count: Count files in D-Design-System/components/
last_updated: Current date/time
```

---

## Step 3: Generate Navigation

<action>
Build component navigation from component files:
</action>

**Scan components:**

```
D-Design-System/components/
├── button.md [btn-001]
├── input.md [inp-001]
├── card.md [crd-001]
└── ...
```

**Group by category:**

```
Interactive:
- Button [btn-001]
- Link [lnk-001]

Form:
- Input [inp-001]
- Select [sel-001]

Display:
- Card [crd-001]
- Badge [bdg-001]
```

**Generate HTML:**

```html
<div class="nav-section">
  <h4 class="nav-section-title">Interactive</h4>
  <ul class="nav-list">
    <li><a href="#button" class="nav-link">Button</a></li>
    <li><a href="#link" class="nav-link">Link</a></li>
  </ul>
</div>

<div class="nav-section">
  <h4 class="nav-section-title">Form</h4>
  <ul class="nav-list">
    <li><a href="#input" class="nav-link">Input</a></li>
    <li><a href="#select" class="nav-link">Select</a></li>
  </ul>
</div>
```

**Replace:** `{{COMPONENT_NAVIGATION}}`

---

## Step 4: Generate Design Tokens Section

<action>
Read and format design tokens:
</action>

**Load:** `D-Design-System/design-tokens.md`

**Parse tokens:**

```yaml
Colors:
  primary-500: #3b82f6
  primary-600: #2563eb
  gray-900: #111827

Typography:
  text-display: 3.75rem
  text-heading-1: 3rem
  text-body: 1rem

Spacing:
  spacing-2: 0.5rem
  spacing-4: 1rem
  spacing-6: 1.5rem
```

**Generate color swatches:**

```html
<div class="component-card">
  <h3 class="text-lg font-semibold mb-4">Primary Colors</h3>
  <div class="variant-grid">
    <div>
      <div class="token-swatch" style="background: #3b82f6;"></div>
      <p class="text-sm font-mono mt-2">primary-500</p>
      <p class="text-xs text-gray-500">#3b82f6</p>
    </div>
    <div>
      <div class="token-swatch" style="background: #2563eb;"></div>
      <p class="text-sm font-mono mt-2">primary-600</p>
      <p class="text-xs text-gray-500">#2563eb</p>
    </div>
  </div>
</div>
```

**Generate typography examples:**

```html
<div class="component-card">
  <h3 class="text-lg font-semibold mb-4">Typography Scale</h3>
  <div class="space-y-4">
    <div>
      <p class="text-sm text-gray-500 mb-1">text-display (3.75rem)</p>
      <p style="font-size: 3.75rem; font-weight: 800; line-height: 1.1;">Display Text</p>
    </div>
    <div>
      <p class="text-sm text-gray-500 mb-1">text-heading-1 (3rem)</p>
      <p style="font-size: 3rem; font-weight: 700; line-height: 1.2;">Heading 1</p>
    </div>
  </div>
</div>
```

**Replace:** `{{COLOR_TOKENS}}`, `{{TYPOGRAPHY_TOKENS}}`, `{{SPACING_TOKENS}}`

---

## Step 5: Generate Component Sections

<action>
For each component, generate interactive showcase:
</action>

**For each file in `D-Design-System/components/`:**

### Parse Component

**Read component file:**

```markdown
# Button Component [btn-001]

**Type:** Interactive
**Category:** Action

## Variants

- primary
- secondary
- ghost
- outline

## States

- default
- hover
- active
- disabled
- loading

## Sizes

- small
- medium
- large
```

### Generate Component Section

**HTML structure:**

```html
<section id="button" class="mb-16" style="scroll-margin-top: 2rem;">
  <h2 class="text-3xl font-bold text-gray-900 mb-6">
    Button
    <span class="version-badge">[btn-001]</span>
    <span class="usage-badge">Used in 12 pages</span>
  </h2>

  <!-- Component Description -->
  <div class="component-card">
    <p class="text-gray-700 mb-4">{{COMPONENT_DESCRIPTION}}</p>
    <div class="flex gap-2">
      <span class="text-sm text-gray-600"> <strong>Type:</strong> Interactive </span>
      <span class="text-sm text-gray-600"> <strong>Category:</strong> Action </span>
    </div>
  </div>

  <!-- Variants -->
  <div class="component-card">
    <h3 class="text-xl font-semibold mb-4">Variants</h3>
    <div class="component-preview">
      <div class="variant-grid">{{VARIANT_EXAMPLES}}</div>
    </div>
  </div>

  <!-- States -->
  <div class="component-card">
    <h3 class="text-xl font-semibold mb-4">States</h3>
    <div class="component-preview">
      <div class="state-grid">{{STATE_EXAMPLES}}</div>
    </div>

    <!-- Interactive State Toggle -->
    <div class="mt-4">
      <p class="text-sm text-gray-600 mb-2">Try it:</p>
      <div class="flex gap-2 mb-4">
        <button onclick="toggleState(this, 'demo-button', 'default')" class="px-3 py-1 rounded text-sm bg-blue-500 text-white">
          Default
        </button>
        <button onclick="toggleState(this, 'demo-button', 'hover')" class="px-3 py-1 rounded text-sm bg-gray-200 text-gray-700">
          Hover
        </button>
        <button onclick="toggleState(this, 'demo-button', 'active')" class="px-3 py-1 rounded text-sm bg-gray-200 text-gray-700">
          Active
        </button>
        <button onclick="toggleState(this, 'demo-button', 'disabled')" class="px-3 py-1 rounded text-sm bg-gray-200 text-gray-700">
          Disabled
        </button>
      </div>
      <div class="component-preview">
        <button id="demo-button" class="state-default">Interactive Button</button>
      </div>
    </div>
  </div>

  <!-- Code Example -->
  <div class="component-card">
    <h3 class="text-xl font-semibold mb-4">Code Example</h3>
    <div class="code-block">
      <pre>{{CODE_EXAMPLE}}</pre>
    </div>
  </div>

  <!-- Usage Guidelines -->
  <div class="component-card">
    <h3 class="text-xl font-semibold mb-4">Usage Guidelines</h3>
    {{USAGE_GUIDELINES}}
  </div>

  <!-- Figma Link (if Mode B) -->
  {{FIGMA_COMPONENT_LINK}}
</section>
```

### Generate Variant Examples

**For each variant:**

```html
<div>
  <button class="btn-primary btn-medium">Primary Button</button>
  <p class="text-xs text-gray-500 mt-2">primary</p>
</div>

<div>
  <button class="btn-secondary btn-medium">Secondary Button</button>
  <p class="text-xs text-gray-500 mt-2">secondary</p>
</div>
```

### Generate State Examples

**For each state:**

```html
<div>
  <button class="btn-primary btn-medium state-default">Default</button>
  <p class="text-xs text-gray-500 mt-2">default</p>
</div>

<div>
  <button class="btn-primary btn-medium state-hover">Hover</button>
  <p class="text-xs text-gray-500 mt-2">hover</p>
</div>
```

### Generate Code Example

**Extract from component spec:**

```tsx
import { Button } from '@/components/button';

<Button variant="primary" size="medium">
  Click me
</Button>

<Button variant="secondary" size="large" disabled>
  Disabled
</Button>
```

**Replace:** `{{COMPONENTS_CONTENT}}`

---

## Step 6: Generate Changelog

<action>
Build changelog from component version histories:
</action>

**Scan all components for version history:**

```markdown
## Version History

**Created:** 2024-09-15
**Last Updated:** 2024-12-09

**Changes:**

- 2024-09-15: Created component
- 2024-10-01: Added loading state
- 2024-12-09: Updated hover animation
```

**Generate changelog HTML:**

```html
<div class="space-y-4">
  <div class="border-l-4 border-blue-500 pl-4">
    <p class="font-semibold">December 9, 2024</p>
    <ul class="text-sm text-gray-600 mt-1">
      <li>• Button: Updated hover animation</li>
      <li>• Input: Added success state</li>
    </ul>
  </div>

  <div class="border-l-4 border-gray-300 pl-4">
    <p class="font-semibold">October 1, 2024</p>
    <ul class="text-sm text-gray-600 mt-1">
      <li>• Button: Added loading state</li>
    </ul>
  </div>
</div>
```

**Replace:** `{{CHANGELOG_CONTENT}}`

---

## Step 7: Add Figma Links (Mode B)

<action>
If Mode B, add Figma component links:
</action>

**Load:** `D-Design-System/figma-mappings.md`

**Parse mappings:**

```yaml
Button [btn-001] → figma://file/abc123/node/456:789
Input [inp-001] → figma://file/abc123/node/456:790
```

**Generate Figma section:**

```html
<h3 class="text-lg font-semibold mb-4">Figma Components</h3>
<div class="space-y-2">
  <a href="figma://file/abc123/node/456:789" class="flex items-center gap-2 text-blue-600 hover:text-blue-800">
    <svg><!-- Figma icon --></svg>
    Button [btn-001]
  </a>
  <a href="figma://file/abc123/node/456:790" class="flex items-center gap-2 text-blue-600 hover:text-blue-800">
    <svg><!-- Figma icon --></svg>
    Input [inp-001]
  </a>
</div>
```

**Replace:** `{{FIGMA_LINKS}}`

---

## Step 8: Generate Installation Instructions

<action>
Create mode-specific installation instructions:
</action>

**Mode B (Custom/Figma):**

```bash
# Install dependencies
npm install

# Import design tokens
import '@/styles/design-tokens.css';

# Import components
import { Button } from '@/components/button';
```

**Mode C (Component Library):**

```bash
# Install component library
npm install shadcn-ui

# Configure library
npx shadcn-ui init

# Import components
import { Button } from '@/components/ui/button';
```

**Replace:** `{{INSTALLATION_INSTRUCTIONS}}`, `{{USAGE_EXAMPLE}}`

---

## Step 9: Write Catalog File

<action>
Save generated HTML:
</action>

**File:** `D-Design-System/catalog.html`

**Content:** Fully populated template with all sections

**Validation:**

- All template variables replaced
- Valid HTML structure
- All component sections included
- Navigation links work
- Interactive demos functional

---

## Step 10: Update Git

<action>
Version control the catalog:
</action>

**Git operations:**

```bash
git add D-Design-System/catalog.html
git commit -m "Update design system catalog - [component changes]"
```

**Commit message format:**

```
Update design system catalog - Added Button loading state

- Button [btn-001]: Added loading state variant
- Updated catalog with interactive demo
- Version: 1.0.1
```

---

## Output Format

**Success message:**

```
✅ Design system catalog generated

File: D-Design-System/catalog.html
Components: 12
Last updated: 2024-12-09 14:30

View catalog:
file:///path/to/D-Design-System/catalog.html

Changes committed to git.
```

---

## Error Handling

### Missing Template

**Error:** Catalog template not found

**Action:**

```
⚠️ Catalog template missing

Expected: workflows/wds-7-design-system/templates/catalog.template.html

Please ensure WDS is properly installed.
```

### Invalid Component Spec

**Error:** Component file has invalid format

**Action:**

```
⚠️ Invalid component specification

File: D-Design-System/components/button.md
Issue: Missing required sections

Skipping component in catalog.
Please fix component specification.
```

### No Components

**Error:** No components in design system

**Action:**

```
⚠️ No components found

Design system appears empty.
Catalog will include only foundation (tokens).

Add components to populate catalog.
```

---

## Automation

**Catalog is automatically regenerated:**

- After creating new component
- After adding variant
- After updating component
- After updating design tokens

**Manual regeneration:**

```
Agent: Regenerate design system catalog
```

---

## Best Practices

### DO ✅

- Regenerate after every component change
- Commit catalog with component changes
- Include all variants and states
- Add interactive demos
- Keep changelog updated

### DON'T ❌

- Don't manually edit catalog.html
- Don't skip catalog regeneration
- Don't forget to commit changes
- Don't remove interactive features

---

**The interactive catalog is the living documentation of your design system, always up-to-date and version controlled.**

### 11. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF M: Return to {activityWorkflowFile} activity menu
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#11-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects the appropriate option
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN [M is selected and catalog is generated and saved], will you then return to the activity workflow menu.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Step goal achieved completely
- All instructions executed in sequence
- Results documented and presented to user
- User confirmed before proceeding
- Design log updated

### ❌ SYSTEM FAILURE:

- Skipping any instruction in the sequence
- Generating content without user input
- Jumping ahead to later steps
- Not presenting results to user
- Proceeding without user confirmation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
