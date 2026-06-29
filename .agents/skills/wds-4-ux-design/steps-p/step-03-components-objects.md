---
name: 'step-03-components-objects'
description: 'Identify all interactive elements, route to object-specific instructions, and assign Object IDs'

# File References
nextStepFile: './step-04-content-languages.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow-specify.md'
---

# Step 3: Components & Object IDs

## STEP GOAL:

Identify all interactive elements in each section, route to object-specific instructions for detailed documentation, and assign Object IDs.

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

- 🎯 Focus on systematic component identification: top-to-bottom, left-to-right per section
- 🚫 FORBIDDEN to skip sections or miss components
- 💬 Approach: Work through each section, routing to object-type templates
- 📋 Use object-router for type-specific documentation

## EXECUTION PROTOCOLS:

- 🎯 Work through sections systematically, identifying all components
- 💾 Store component specs with Object IDs for each
- 📖 Reference object-types/ templates for consistent documentation
- 🚫 FORBIDDEN to skip design system check after component spec

## CONTEXT BOUNDARIES:

- Available context: page_basics, layout_sections
- Focus: Component identification and Object ID assignment
- Limits: Do not specify content/languages yet (next step)
- Dependencies: Layout sections must be defined

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Identify Components

<output>**Let's identify and document every component systematically.**

We'll work through each section, going **top-to-bottom, left-to-right** within each section, documenting each object using specialized instructions.</output>

### 2. For Each Section

<action>For each section identified in step 02:

<output>**Section: {{section_name}}**

Starting from top-left corner of this section...</output>

### 3. For Each Object in Section

<action>Loop through objects in section (top-to-bottom, left-to-right):

<output>**Next object in {{section_name}}:**</output>

<ask>What is the first/next object in this section (from top-left)?

Describe what you see:</ask>

<action>Store object_description</action>

#### Route to Object-Type Instructions

<action>Load and execute `object-types/object-router.md`</action>

<action>Object-router will: 1. Ask user to identify object type 2. Load appropriate object-type instruction file 3. Guide through complete object documentation 4. Generate specification with Object ID 5. Return here when complete
</action>

#### Design System Check (If Enabled)

<action>After component specification complete: 1. Check project config: Is design system enabled? 2. If YES: Load and execute `workflows/wds-7-design-system/design-system-router.md` 3. Design system router will: - Check for similar components - Run opportunity/risk assessment if needed - Extract component-level info to design system - Return component reference - Update page spec with reference 4. If NO: Keep complete specification on page 5. Continue to next object
</action>

<ask>**More objects in {{section_name}}?**

1. **Yes** - Document next object (move right, then down)
2. **No** - Section complete

Choice [1/2]:</ask>

  <check if="choice == 1">
    <action>Loop back to document next object in section</action>
  </check>

  <check if="choice == 2">
    <output>**Section {{section_name}} complete!**</output>
    <action>Move to next section</action>
  </check>

</action>

</action>

### 4. All Sections Complete

<output>**All components identified and documented!**

**Summary:**

- **Sections processed:** {{section_count}}
- **Total components:** {{component_count}}
- **Components by type:**
  {{#each component_type}}
  - {{type_name}}: {{count}}
    {{/each}}

**Object IDs assigned:**
{{#each component}}

- `{{object_id}}` ({{component_type}})
  {{/each}}

**Next:** We'll specify the content and languages.</output>

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Content & Languages | [M] Return to Activity Menu"

#### Menu Handling Logic:

- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options](#5-present-menu-options)

#### EXECUTION RULES:

- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions — always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user selects an option from the menu and all components have been documented with Object IDs will you proceed to the next step or return as directed.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:

- All sections processed systematically
- All components documented with Object IDs
- Object-type routing used for consistent documentation
- Design system check performed after each component
- Component registry complete

### ❌ SYSTEM FAILURE:

- Skipping sections or components
- Not using object-type routing for documentation
- Missing Object IDs
- Skipping design system check
- Proceeding with incomplete component registry

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
