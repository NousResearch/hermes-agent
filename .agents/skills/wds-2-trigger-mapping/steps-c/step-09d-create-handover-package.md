---
name: 'step-09d-create-handover-package'
description: 'Create handover summary package for UX Design phase'

# File References
nextStepFile: './step-09e-update-design-log.md'
activityWorkflowFile: '../workflow.md'
---

# Step 35: Create Handover Package

## STEP GOAL:

Create a summary handover package for the UX Designer (Freya) with primary focus, must-address drivers, feature priorities, and design implications.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - preparing handover for UX Design phase
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on creating clear, actionable handover for UX Designer
- 🚫 FORBIDDEN to omit primary focus or must-address drivers
- 💬 Approach: Clear, structured handover with all critical information
- 📋 Include: documentation structure, primary focus, must-address drivers, feature priorities, design implications
- 📋 Show complete file tree of created documents

## EXECUTION PROTOCOLS:

- 🎯 Present comprehensive handover summary
- 💾 No new files to save - summary presentation
- 📖 Include all critical information for UX Designer
- 🚫 Do not skip any handover component

## CONTEXT BOUNDARIES:

- Available context: All verified documents, quality check results
- Focus: Handover summary for UX Design phase
- Limits: Focus on what UX Designer needs to know
- Dependencies: Requires quality check passed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Present Completion Summary

Output:
"**Trigger Map Phase Complete!**

**All Documentation Created:**

```
B-Trigger-Map/
  00-trigger-map.md          - Start here: Visual overview
  01-Business-Goals.md
  02-{{primary_persona}}.md
  03-{{secondary_persona}}.md
  04-{{tertiary_persona}}.md  (if exists)
  05-Key-Insights.md
  06-Feature-Impact.md        (if completed)
```"

### 2. Present Handover Summary

"**Handover Summary for UX Design:**

**Primary Focus:**
- **Who:** {{primary_persona_name}} ({{primary_persona_role}})
- **Transformation:** {{transformation_summary}}

**Must Address:**
(top 3 positive drivers with checkmarks)

**Must Avoid:**
(top 3 negative drivers with X marks)

**Feature Priority:** (if available, top 3 features; otherwise note not yet analyzed)

**Design Implications:**
(3 key design implications)

**Ready for Phase 4: UX Design**"

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Update Design Log | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Handover must be presented before proceeding to design log update.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Complete file tree shown
- Primary focus clearly stated (who, transformation)
- Must-address positive drivers listed
- Must-avoid negative drivers listed
- Feature priorities shown (if available)
- Design implications included
- Clear readiness signal for Phase 4

### ❌ SYSTEM FAILURE:
- Missing file tree
- Missing primary focus
- Missing must-address drivers
- Incomplete handover information
- Not indicating Phase 4 readiness

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
