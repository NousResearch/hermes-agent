---
name: 'step-01-sketch-analysis'
description: 'AI reads entire sketch, identifies sections, interprets function/purpose, user confirms before detailed specification'

# File References
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-sketch.md'
---

# Step 1: Sketch Analysis

## STEP GOAL:

AI reads entire sketch, identifies sections, interprets function and purpose. User confirms structure before detailed specification begins. This balances AI enhancement with user control.

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

- 🎯 Focus on interpreting sketch structure, sections, objects, and purpose
- 🚫 FORBIDDEN to generate detailed specifications without user confirmation of structure
- 💬 Approach: Read holistically first, then section-by-section with user validation
- 📋 Cross-reference with previous pages for consistency and design system patterns

## EXECUTION PROTOCOLS:

- 🎯 Analyze sketch holistically before breaking into sections
- 💾 Store confirmed interpretations for specification generation
- 📖 Reference established patterns from previously analyzed pages
- 🚫 FORBIDDEN to proceed to specification without user confirmation of section structure

## CONTEXT BOUNDARIES:

- Available context: User's sketch (image, description, or file reference), previous page analyses, design system
- Focus: Interpreting sketch into structured sections and objects
- Limits: Do not generate final specifications — that is the Specify activity (steps-p/)
- Dependencies: User must provide sketch input

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Holistic Sketch Reading

<output>**Let me read your entire sketch and identify the main sections...**</output>

<ask>Please share your sketch:

- Describe it to me
- Paste/upload an image
- Reference filename in Sketches/ folder

Your sketch:</ask>

<action>Analyze entire sketch holistically:

1. **Establish Scale First:**
   - Check if other pages in project have been analyzed
   - Look for established text styles (body text, buttons, headings)
   - Identify UI anchors (browser chrome, scrollbars, buttons, icons)
   - Use previous pages + UI elements to calibrate scale

2. **Identify Sections:**
   - Identify logical sections (header, hero, features, cards, footer, etc.)
   - Determine section boundaries (whitespace, visual grouping, layout)
   - Recognize section purposes from visual context
   - Count objects/elements in each section

3. **Cross-Page Pattern Matching:**
   - Check if sections look like components from previous pages
   - Compare text styles to established patterns (e.g., "thin lines, icon-sized spacing = 16px body text")
   - Note potential design system components
     </action>

<output>**I've analyzed your sketch. Here's what I see:**

{{#if has_previous_pages}}
**Cross-Page References Detected:**
{{#each established_patterns}}

- {{pattern_name}}: {{specification}} (from {{source_page}})
  {{/each}}

I'll use these as reference points for consistency.
{{/if}}

---

**{{page_name}}** contains **{{section_count}} main sections:**

## {{#each identified_sections}}

**Section {{index}}: {{section_name}}** ({{location}})

- **Purpose:** {{interpreted_purpose}}
- **Contains:** {{object_count}} objects/elements
- **Layout:** {{layout_description}}
  {{#if looks_like_previous_component}}
- **Component?** Similar to {{component_name}} from {{previous_page}}
  {{/if}}
  {{#if matches_established_pattern}}
- **Pattern Match:** Text styles match {{pattern_name}} from {{source_page}}
  {{/if}}
  {{/each}}

---

This is my interpretation of the structure. Does this look right?</output>

<ask>Section structure:

1. **Confirm** - Yes, this is correct!
2. **Adjust** - I need to refine the section breakdown
3. **Add sections** - I see more sections
4. **Remove/merge sections** - Some sections should be combined

Choice [1/2/3/4]:</ask>

<check if="choice != 1">
  <ask>**How should I adjust the sections?**

Current breakdown:
{{#each identified_sections}}
{{index}}. {{section_name}} - {{interpreted_purpose}}
{{/each}}

Your changes:</ask>

<action>Update section structure based on feedback</action>
<output>**Updated structure:**

{{#each updated_sections}}
{{index}}. {{section_name}} - {{interpreted_purpose}}
{{/each}}

Does this look better?</output>

<action>Loop until user confirms structure</action>
</check>

---

### 2. Component Identification

<check if="any_sections_look_like_components">
  <output>**I noticed some sections might be reusable components:**

  {{#each potential_components}}
  - **{{section_name}}** looks similar to **{{component_name}}** from {{previous_page}}
  {{/each}}
  </output>

  <ask>Should these be components (reusable across pages)?

1. **Yes, make them components** - Define once, reference later
2. **No, keep them as page-specific** - Each page has unique version
3. **Let me decide section-by-section** - I'll choose as we go

Choice [1/2/3]:</ask>

<action>Mark sections as components or page-specific based on user choice</action>
</check>

---

### 3. Section-by-Section AI Interpretation

<output>**Perfect! Now I'll analyze each section in detail, one at a time.**

I'll interpret the objects, functions, and content for each section. You can confirm or refine my interpretation before I generate the spec.

---

**Section {{current_index}}/{{total_sections}}: {{section_name}}**</output>

#### 3A: AI Reads & Interprets Section (Recursive)

<action>For current section, identify objects **Top-Left to Bottom-Right**:

1. **Identify Top-Level Containers** (e.g., Cards, Rows, Groups)
   - IF container has children -> Dive in and identify child elements
   - IF repeating group (e.g., 3 Feature Cards) -> Identify as "Repeating Pattern"

2. **Handle Repeating Objects:**
   - **Fixed Count (e.g., 3 Cards):** Name individually (`card-01`, `card-02`, `card-03`)
   - **Dynamic List:** Define as Pattern + Data Source

3. **Determine Object Hierarchy:**
   - Parent: `feature-card-01`
   - Child: `feature-card-01-icon`, `feature-card-01-title`

4. **Interpret Attributes:**
   - Type (Button, Text, Input)
   - Function & Purpose
   - Text Content (Actual vs. Markers)
   - Visual Hierarchy
     </action>

<output>**My interpretation of "{{section_name}}":**

**Section Purpose:** {{interpreted_section_purpose}}

**Hierarchy I see:**

{{#each interpreted_objects}}
{{object_index}}. **{{interpreted_type}}** ({{hierarchy_level}})

- **Object ID:** `{{suggested_object_id}}`
  {{#if is_container}}
- **Contains:**
  {{#each children}}
  - {{child_type}}: `{{child_object_id}}`
    {{/each}}
    {{/if}}
- **Function:** {{interpreted_function}}
- **Purpose:** {{interpreted_purpose}}
  {{#if has_actual_text}}
- **Text in sketch:** "{{extracted_text}}"
  {{/if}}
  {{/each}}

**Overall Function:** {{section_function_summary}}</output>

#### 3B: User Refinement Dialog

<ask>**Does this interpretation look right?**

1. **Yes, looks good!** - Move to content/translations
2. **Adjust interpretations** - I need to correct some things
3. **Add missing objects** - You missed something
4. **Remove objects** - Something isn't an object

Choice [1/2/3/4]:</ask>

<check if="choice == 2">
  <ask>**Which interpretations need adjustment?**

  {{#each interpreted_objects}}
  {{object_index}}. {{interpreted_type}} - {{interpreted_function}}
  {{/each}}

  Your corrections:</ask>

  <action>Update interpretations based on user feedback</action>
</check>

<check if="choice == 3">
  <ask>**What did I miss?**

  Describe the missing object(s):</ask>

  <action>Add missed objects to interpretation</action>
</check>

<check if="choice == 4">
  <ask>**Which objects should I remove?**

  {{#each interpreted_objects}}
  {{object_index}}. {{interpreted_type}}
  {{/each}}

  Remove numbers:</ask>

  <action>Remove specified objects</action>
</check>

<action>Re-display updated interpretation for confirmation</action>
<action>Loop until user confirms: "Yes, looks good!"</action>

---

### 4. Content & Translation Gathering

<output>**Great! Now let's gather the content for all text elements in this section.**

I'll suggest translations for everything at once.</output>

## {{#each text_objects}}

**{{object_purpose}}** (`{{object_id}}`)

{{#if has_actual_text}}
<output>I found text in your sketch: "{{extracted_text}}"

Let me suggest translations...</output>

<action>Generate translations for all product_languages</action>

<output>**Suggested content:**

{{#each product_languages}}
{{this}}: {{suggested_translation}}
{{/each}}
</output>

<ask>For "{{object_purpose}}":

1. **Use these translations**
2. **Adjust translations**
3. **Manual input**

Choice [1/2/3]:</ask>

{{else}}
<ask>**Content for "{{object_purpose}}":**

{{primary_language}}:</ask>

<action>After receiving primary language, suggest other languages</action>

<output>**Translation suggestions:**

{{#each remaining_languages}}
{{this}}: {{suggested_translation}}
{{/each}}

Use these? [1] Yes [2] Adjust [3] Manual</output>

{{/if}}

## <action>Store confirmed content for this object</action>

{{/each}}

---

### 5. Batch Specification Generation

<output>**Perfect! I have everything I need for "{{section_name}}".**

Let me generate the complete section specification...</output>

<action>Generate section spec:

1. Section header with purpose
2. All objects with full details
3. All translations grouped by object
4. Component references if applicable
5. Interactions and behaviors
6. States if applicable
7. Validation rules if applicable
   </action>

<output>**Section "{{section_name}}" specification generated!**

```markdown
### {{Section_Name}}

**Purpose**: {{section_purpose}}

{{#each objects}}

#### {{Object_Purpose_Title}}

**OBJECT ID**: `{{object_id}}`

- **Component**: {{component_type}}
- **Position**: {{position}}
- **Style**: {{style_specs}}
  {{#if has_behavior}}
- **Behavior**: {{behavior}}
  {{/if}}
  {{#if is_text}}
- **Content**:
  {{#each product_languages}}
  - {{this}}: "{{content}}"
    {{/each}}
    {{/if}}
    {{#if has_states}}
- **States**: {{states}}
  {{/if}}

{{/each}}
```

**Next:** {{#if more_sections}}Section {{next_index}}: {{next_section_name}}{{else}}Complete page!{{/if}}</output>

<check if="more_sections">
  <action>Move to next section</action>
  <action>Repeat from step 3 for next section</action>
</check>

<check if="!more_sections">
  <output>**All sections complete!**

  Your page specification includes:
  - {{total_sections}} sections
  - {{total_objects}} objects
  - {{total_text_elements}} text elements with {{language_count}} languages
  - {{component_count}} reusable components identified

  Ready to generate prototype!</output>

  <action>Proceed to specification generation</action>
</check>

---

### 6. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#6-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has completed sketch analysis for all sections and chosen to return to the menu will you proceed accordingly. This is the only step in the Sketch activity.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- Sketch analyzed holistically with scale calibration
- All sections identified and confirmed by user
- Cross-page patterns detected and referenced
- Section-by-section interpretation completed with user validation
- Content and translations gathered for all text elements
- Batch specification generated for each confirmed section
- Component reuse opportunities identified

### ❌ SYSTEM FAILURE:

- Generating specifications without user confirmation of structure
- Skipping holistic analysis and jumping to details
- Not cross-referencing with previous page analyses
- Proceeding without user confirming section breakdown
- Missing objects or sections in the interpretation
- Not gathering translations for all supported languages
- Ignoring repeating patterns or component opportunities

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
