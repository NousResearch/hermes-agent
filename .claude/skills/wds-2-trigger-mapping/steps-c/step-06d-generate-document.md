---
name: 'step-06d-generate-document'
description: 'Generate the complete Feature Impact Analysis document'

# File References
nextStepFile: './step-06e-feature-wrap-up.md'
activityWorkflowFile: '../workflow.md'
---

# Step 15: Generate Feature Impact Document

## STEP GOAL:

Generate the complete Feature Impact Analysis document with the confirmed assessment, including prioritization tiers, detailed rationale, strategic implications, and questions for the designer.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are Saga the Analyst - documenting strategic priorities
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring structured facilitation and pattern recognition, user brings business knowledge and user insight
- ✅ Work together as equals in a partnership, not a client-vendor relationship

### Step-Specific Rules:

- 🎯 Focus on generating complete, well-structured document
- 🚫 FORBIDDEN to save without user review of summary
- 💬 Approach: Generate document, present summary, iterate if needed
- 📋 Use template from ../templates/feature-impact.template.md
- 📋 Include Must Have/Consider/Defer prioritization tiers

## EXECUTION PROTOCOLS:

- 🎯 Generate document following template structure
- 💾 Save to {output_folder}/B-Trigger-Map/feature-impact-analysis.md
- 📖 Present summary to user for review
- 🚫 Do not proceed until user confirms document

## CONTEXT BOUNDARIES:

- Available context: Confirmed assessment scores, personas, driving forces
- Focus: Document generation with proper structure and rationale
- Limits: Use confirmed scores only - do not change assessment
- Dependencies: Requires confirmed assessment from step-06c

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Generate Document

Use the template: `../templates/feature-impact.template.md`

Include:
1. **Header** with project name, date, and scoring legend
2. **Prioritized Features Table** with all scores
3. **Feature Details & Rationale** for each feature (especially top scorers)
4. **Strategic Implications** section
5. **Questions for Designer** section

**Prioritization Logic:**

**Must Have MVP:**
- Any feature where Primary Persona scored HIGH (5 pts)
- OR features with score >= (max_possible - 3)

**Consider for MVP:**
- Mid-range scores
- Strategic value but not critical

**Defer:**
- Low scores
- Minimal strategic value

### 2. Save Document

Save as: `{output_folder}/B-Trigger-Map/feature-impact-analysis.md`

### 3. Present Summary

Output:
"**Feature Impact Analysis Document Generated!**

**Saved to:** B-Trigger-Map/feature-impact-analysis.md

**Summary:**

**Must Have MVP Features ({{must_have_count}}):**
{{#each must_have}}
- {{this.name}} (Score: {{this.score}})
{{/each}}

**Consider for MVP ({{consider_count}}):**
{{#each consider}}
- {{this.name}} (Score: {{this.score}})
{{/each}}

**Key Insights:**
- [Strategic insight 1]
- [Strategic insight 2]
- [Strategic insight 3]

This Feature Impact Analysis serves as your **Design Brief** - it guides:
- **Phase 4: UX Design** - Which scenarios to design first
- **Phase 6: PRD/Development** - Epic and story prioritization

The document includes detailed rationale for each feature's scoring.

**Would you like to make any final adjustments, or are we good to proceed?**"

### 4. Handle Feedback

If user requests changes: Update document, regenerate, show summary again.

### 5. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to Workshop Wrap-Up | [M] Return to Activity Menu"

#### Menu Handling Logic:
- IF C: Load and execute {nextStepFile}
- IF M: Return to {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN user selects [C] will you load the next step file. Document must be generated and user must confirm before proceeding.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Document generated following template structure
- All sections included (header, table, rationale, implications, questions)
- Prioritization tiers applied correctly (Must Have/Consider/Defer)
- Document saved to correct location
- Summary presented to user
- User confirmed or adjustments made and re-confirmed

### ❌ SYSTEM FAILURE:
- Document missing required sections
- Incorrect prioritization tier assignment
- Not saving to correct location
- Not presenting summary for user review
- Proceeding without user confirmation

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
