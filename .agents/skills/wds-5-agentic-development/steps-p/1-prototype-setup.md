---
name: '1-prototype-setup'
description: 'Set up the prototype environment for an entire scenario (one-time setup)'

# File References
nextStepFile: './2-scenario-analysis.md'
---

# Step 1: Prototype Setup

## STEP GOAL:

Set up the prototype environment for an entire scenario (one-time setup). This assumes the scenario specification already exists.

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

- 🎯 Focus only on running the initiation dialog, creating folder structure, and setting up demo data
- 🚫 FORBIDDEN to begin building any pages or components — that is a later step
- 💬 Approach: Ask the 4 initiation questions, then create the folder structure with user
- 📋 Skip this phase if scenario already has `data/demo-data.json` and `PROTOTYPE-ROADMAP.md`

## EXECUTION PROTOCOLS:

- 🎯 Prototype folder structure created with demo data and roadmap
- 💾 Create demo-data.json and PROTOTYPE-ROADMAP.md
- 📖 Reference PROTOTYPE-INITIATION-DIALOG.md for exact conversation scripts
- 🚫 Do not build any pages or UI during this step

## CONTEXT BOUNDARIES:

- Available context: Scenario specification (from scenario-init workflow)
- Focus: Environment setup — folder structure, demo data, configuration
- Limits: No page building, no UI work
- Dependencies: Scenario specification must exist

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. User Requests Scenario Setup

**User says**: "Create interactive prototypes for Scenario [N]: [Scenario Name]"

**Your response**: Follow the **Scenario Initiation Dialog** in `PROTOTYPE-INITIATION-DIALOG.md`

### 2. Run Initiation Dialog

**Ask 4 questions**:

1. **Device Compatibility** (Mobile-Only / Mobile+Tablet / Fully Responsive / Desktop-Only)
2. **Design Fidelity** (Gray Model / Design System / Figma Integration)
3. **Languages** (if project supports multiple languages)
4. **Demo Data** (Create realistic test family data)

**See**: `PROTOTYPE-INITIATION-DIALOG.md` for exact conversation scripts

### 3. Create Prototype Folder Structure

**Actions**:

1. **Create prototype folder**: `[Scenario-Number]-[Scenario-Name]-Prototype/`
2. **Create all subfolders**:
   - `data/` - Demo data JSON files
   - `work/` - Planning/work files (one per page)
   - `stories/` - Section implementation guides (created just-in-time)
   - `shared/` - Shared JavaScript (utilities, API abstraction)
   - `components/` - Reusable UI components
   - `pages/` - Page-specific scripts (if complex)
   - `assets/` - Images, icons, etc.
3. **Create `data/demo-data.json`** with demo family
4. **Create `PROTOTYPE-ROADMAP.md`** with scenario overview
5. **Record device compatibility and design approach** in roadmap

**Folder structure created**:
```
[Scenario-Number]-[Scenario-Name]-Prototype/
├── PROTOTYPE-ROADMAP.md
├── data/
│   └── demo-data.json
├── work/              (empty, will be filled per-page)
├── stories/           (empty, created just-in-time)
├── shared/            (empty, add as needed)
├── components/        (empty, add as needed)
├── pages/             (empty, add if needed)
└── assets/            (empty, add as needed)

HTML files will be placed in root as they're created.
```

### 4. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Step 2: Scenario Analysis"

#### Menu Handling Logic:
- IF C: Update design log, then load, read entire file, then execute {nextStepFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the prototype folder structure is created with demo data and roadmap will you then load and read fully `{nextStepFile}` to execute.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Initiation dialog completed (4 questions answered)
- Prototype folder structure created with all subfolders
- demo-data.json created with realistic test data
- PROTOTYPE-ROADMAP.md created with scenario overview
- Device compatibility and design approach recorded

### ❌ SYSTEM FAILURE:
- Beginning page building before setup is complete
- Skipping initiation dialog questions
- Not creating demo data
- Not creating the roadmap

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
