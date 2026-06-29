---
name: 'step-02-structure'
description: 'Configure project settings create folder structure and generate project outline'

# File References
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 2: Project Configuration & Structure

## STEP GOAL:

Configure all project settings (name, complexity, tech stack, component library, brief level, strategic analysis, working relationship), create the folder structure, and generate the project outline.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are the Project Setup facilitator, configuring the WDS project
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring WDS methodology expertise, user brings their project knowledge
- ✅ Maintain a helpful and efficient tone throughout

### Step-Specific Rules:

- 🎯 Focus only on gathering project configuration and creating structure
- 🚫 FORBIDDEN to skip configuration questions or assume answers
- 💬 Approach: Ask each configuration question, respect user choices
- 📋 Configuration determines the entire project workflow path

## EXECUTION PROTOCOLS:

- 🎯 Gather all configuration settings and create project structure
- 💾 Save project outline to `{{root_folder}}/_progress/wds-project-outline.yaml`
- 📖 Follow the configuration sequence exactly
- 🚫 Do not skip questions or assume defaults without offering choice

## CONTEXT BOUNDARIES:

- Available context: Project type (greenfield/brownfield) from step 0.1
- Focus: Project configuration and structure creation
- Limits: Configuration only - do not begin Phase 1 work
- Dependencies: step-01-welcome must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Project Name

**What is your project name?**

### 2. What Are You Building?

**What type of product is this?**

[A] **Landing Page** - Single page, marketing focused -> Simplified workflow, minimal phases
[B] **Website** - Multiple pages, content focused -> Standard workflow, most phases
[C] **Web Application** - Complex features, user interactions -> Full workflow, all phases
[D] **Mobile App** - iOS/Android application -> Full workflow + platform considerations

Store as `product_complexity`: A=simple, B=standard, C=complex, D=complex+mobile

### 3. Tech Stack (Optional)

**Tech stack?** [A] React/Next [B] Vue/Nuxt [C] WordPress [D] HTML [E] Other [F] Skip

Store as `tech_stack` (or null if F)

### 4. Component Library (Optional)

**Component library?**

[A] **shadcn/ui** -> Skip Phase 5
[B] **Tailwind only** -> Phase 5 optional
[C] **Material UI** -> Skip Phase 5
[D] **Custom** -> Phase 5 recommended
[E] **Skip** - Decide later

Store as `component_library`. If A/C -> `skip_design_system: true`

### 5. Root Folder Name

**Where should design process files live?**

[A] **design-process** (Recommended)
[B] **docs**
[C] **Custom name**

Store as `root_folder`: A=design-process, B=docs, C=custom

### 6. Existing Materials (Optional Context)

**Do you have any existing materials for this project?**

[A] **Yes** - I have some materials to share
[B] **No** - Starting fresh

If Yes: Review materials, store in outline under `existing_materials`
If No: Store `existing_materials.has_materials: false`

### 7. Brief Level

**Greenfield**: Always use Complete Brief (`brief_level: complete`)

**Brownfield**: Ask how thorough the Project Brief should be:
[A] **Complete** - Full strategic documentation
[B] **Simplified** (Recommended) - Document what exists + what to change

Store as `brief_level`: complete | simplified

### 8. Strategic Analysis Level (Greenfield only)

**How deep should the user/market analysis go?** (Only ask if greenfield AND not simple)

[A] **Full Trigger Map** (Recommended for complex) -> Phase 2 enabled
[B] **Simplified** -> Strategic context in Phase 1, skip Phase 2
[C] **Skip** (Not recommended) -> Skip Phase 2

Store as `strategic_analysis`: full | simplified | skip

### 9. Working Relationship Context

**What are the stakes of this project?**

[A] **Personal/Hobby** -> Encouragement-focused, minimal documentation
[B] **Small Business Investment** -> Balanced rationale, professional
[C] **Departmental/Organizational** -> Comprehensive justification, evidence-based
[D] **Enterprise/High Stakes** -> Rigorous documentation, proof for every point

**How involved do you want to be?**
[A] Highly collaborative [B] Balanced [C] Autonomous execution

**What is your role?**
[A] Client/Stakeholder [B] Product Owner [C] Design Partner [D] Project Manager [E] Other

**How should I present recommendations?**
[A] Suggest options [B] Recommend with rationale [C] Direct guidance

If stakes C/D: Ask about stakeholders and political sensitivities.

Store all as `project_context` and `working_relationship` in outline.

### 10. Create Structure & Outline

**Check existing:** Look for `{{root_folder}}/` and outline file
**If exists:** Ask to keep or reset

**Create folder structure:**
1. Create root folder: `{{root_folder}}/`
2. Create progress folder: `{{root_folder}}/_progress/`
3. Create agent experiences folder: `{{root_folder}}/_progress/agent-experiences/`
4. Create phase folders (greenfield vs brownfield)
5. Create D-Design-System subfolders
6. Install folder guide files from templates

**Generate `{{root_folder}}/_progress/wds-project-outline.yaml`** with all configuration values.

**Fill in `00-design-log.md`** with initial Phase 0 entry.

### 11. Summary & Next Steps

**Project configured!** Display summary table of all settings.

**Greenfield routing:**
[A] Start Phase 1 now
[B] Hand off to Saga (specialist)

**Brownfield routing:**
[A] Create Limited Brief now
[B] Scan my codebase first
[C] I know what to improve - go

### 12. Routing

**Greenfield:** [A] -> Phase 1 workflow, [B] -> Hand off to Saga
**Brownfield:** [A] -> Limited Brief, [B] -> Scan codebase, [C] -> Phase 8

### 13. Present MENU OPTIONS

Display: "**Select an Option:** [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the project is fully configured and structure is created will you present the return to Activity Menu option.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- All configuration questions are answered
- Project outline YAML is generated correctly
- Folder structure is created
- Correct routing to next phase based on project type and configuration
- User understands what comes next

### ❌ SYSTEM FAILURE:
- Skipping configuration questions
- Assuming defaults without offering choice
- Not creating folder structure
- Not generating project outline YAML
- Routing to wrong phase

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.

---

_Phase 0: Project Setup - Step 02: Configuration & Structure_
