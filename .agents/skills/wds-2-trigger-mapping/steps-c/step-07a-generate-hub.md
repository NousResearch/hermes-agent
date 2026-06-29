---
name: 'step-07a-generate-hub'
description: 'Generate the 00-trigger-map.md hub document with Mermaid diagram and navigation'

# File References
nextStepFile: './step-07b-generate-business-goals.md'
activityWorkflowFile: '../workflow.md'
---

# Step 17: Generate Hub Document

## STEP GOAL:

Create the main entry point document (00-trigger-map.md) with Mermaid diagram, on-page summaries, navigation menu, and strategic overview including the key transformation and flywheel.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - creating comprehensive trigger map documentation
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on creating the hub document with all required sections
- 🚫 FORBIDDEN to skip Mermaid diagram or on-page summaries
- 💬 Approach: Generate structured document following template
- 📋 Include: Header, Mermaid diagram, Summary, Detailed Documentation menu, How to Read, Footer
- 📋 Target ~220-250 lines total

## EXECUTION PROTOCOLS:

- 🎯 Generate Mermaid diagram by loading step-08a-mermaid-init-structure.md sequence
- 💾 Save as 00-trigger-map.md in trigger map folder
- 📖 Include on-page summaries for each section (visible without clicking)
- 🚫 Do not skip any required section

## CONTEXT BOUNDARIES:

- Available context: All workshop outputs, personas, driving forces, priorities
- Focus: Hub document creation with diagram and navigation
- Limits: Follow template structure exactly
- Dependencies: Requires all workshop outputs available

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 0. Data Extraction (MANDATORY BEFORE GENERATION)

Before generating ANY content, extract structured data from all workshop outputs:

**Read and extract from workshop data:**

1. **From Business Goals workshop:**
   - `vision_statement` = exact vision text (character-for-character)
   - `objectives[]` = each SMART objective with metric, target, timeline

2. **From Target Groups workshop:**
   - `target_groups[]` = each group with name, priority, persona summary
   - `positive_drivers[]` per group (specific wants)
   - `negative_drivers[]` per group (specific fears)

3. **From Prioritization workshop:**
   - `focus_statement` = strategic focus
   - `top_group` = primary design target
   - `must_address_drivers[]` and `should_address_drivers[]`

**Store these as variables. When filling the hub document, use EXACT values from these variables. Do NOT paraphrase or summarize workshop outputs.**

**Validation rule:** Vision statement in the hub MUST be character-for-character identical to the vision from Business Goals workshop. If you cannot find the exact text, ask the user rather than inventing a paraphrase.

---

### 1. Generate Header Section

Create header with project name, date, author, and methodology credit.

### 2. Generate Mermaid Diagram

Load and execute the mermaid generation sequence starting with step-08a-mermaid-init-structure.md.

**Requirements:**
- Light gray professional styling (consistent for all business goals)
- All nodes have proper padding (`<br/>`)
- Emojis: checkmark for wants, X for fears
- Exactly 3 drivers per persona
- Proper connections: BG->PLATFORM->TG->DF

### 3. Generate Summary Section

Include:
- Primary Target with one-line transformation
- The Flywheel (numbered steps with priority emojis)
- Key Transformation statement

### 4. Generate Detailed Documentation Menu

For each section, provide:
- Link to document with description
- On-page content (key information visible without clicking)
- Proper formatting with bold, bullets, clear structure

Include sections for:
- Business Strategy (01-Business-Goals.md)
- Target Users (persona documents)
- Strategic Implications (05-Key-Insights.md)

### 5. Generate How to Read Section

Explain diagram reading: left-to-right flow, top-to-bottom priority, driving force symbols.

### 6. Generate Footer

Include WDS framework credit and Effect Mapping methodology credits.

### 6b. Cross-Validation Check

Before saving, verify data consistency:
- [ ] Vision in hub matches vision from Business Goals workshop exactly
- [ ] Persona names in hub match names used in individual persona documents
- [ ] Driver count in Mermaid diagram matches drivers in per-persona workshop outputs
- [ ] Priority ordering in hub matches prioritization workshop output

If any mismatch found: correct the hub document to match the source workshop data.

### 7. Save and Confirm

Store as: 00-trigger-map.md in trigger map folder.

Output: "Hub document created with diagram and navigation!"

### 8. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Business Goals Document | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Hub document must be generated before proceeding. Do NOT skip ahead.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Hub document created with all required sections
- Mermaid diagram generated with proper styling
- On-page summaries included for all sections
- Navigation links to all sub-documents
- Summary with transformation and flywheel
- How to Read section explaining diagram
- Footer with methodology credits
- Document saved to correct location
- ~220-250 lines total

### ❌ SYSTEM FAILURE:
- Missing Mermaid diagram
- Missing on-page summaries (requiring clicks to see content)
- Missing navigation links
- Missing transformation or flywheel in summary
- Not following template structure
- Document not saved

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
