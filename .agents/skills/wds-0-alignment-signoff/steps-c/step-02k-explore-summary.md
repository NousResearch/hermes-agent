---
name: 'step-02k-explore-summary'
description: 'Help user create a summary of key points from all explored alignment sections'

# File References
nextStepFile: './step-03a-reflect-back.md'
workflowFile: '../workflow.md'

# Data References
sectionRoutingFile: '../data/02-explore-sections-routing.md'
---

# Step 16: Explore Summary

## STEP GOAL:

Help the user create a summary of key points from all explored alignment sections - the main takeaways stakeholders should remember.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:

- 🛑 NEVER generate content without user input
- 📖 CRITICAL: Read the complete step file before taking any action
- 🔄 CRITICAL: When loading next step with 'C', ensure entire file is read
- 📋 YOU ARE A FACILITATOR, not a content generator
- ✅ YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:

- ✅ You are the Alignment & Signoff facilitator, guiding users to create stakeholder alignment
- ✅ If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- ✅ We engage in collaborative dialogue, not command-response
- ✅ You bring alignment and stakeholder management expertise, user brings their project knowledge
- ✅ Maintain a supportive and clarifying tone throughout

### Step-Specific Rules:

- 🎯 Focus only on exploring the summary of key points
- 🚫 FORBIDDEN to write the summary for the user without their input
- 💬 Approach: Help identify the main takeaways
- 📋 Keep it brief - summary of key points, let readers draw their own conclusion

## EXECUTION PROTOCOLS:

- 🎯 Help user identify key takeaways from all explored sections
- 💾 Capture summary for the alignment document
- 📖 Reference `{sectionRoutingFile}` (Section 10: Summary)
- 🚫 Do not create an overly long summary

## CONTEXT BOUNDARIES:

- Available context: All explored alignment sections (02a through 02j)
- Focus: Summary section of the alignment document
- Limits: Brief key points only
- Dependencies: All exploration steps (02a-02j) must be completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Explore the Summary

Explore the summary.

**Reference**: `{sectionRoutingFile}` (Section 10: Summary)

**Questions to explore**:
- "What are the key points?"
- "What should stakeholders remember?"
- "What's the main takeaway?"

**Keep it brief** - Summary of key points (let readers draw their own conclusion)

### 2. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-03a-reflect-back"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the user has identified key summary points will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Key takeaways from all sections are identified
- Summary is brief and compelling
- User feels all sections are well represented

### ❌ SYSTEM FAILURE:
- Writing the summary without user input
- Creating an overly long summary
- Missing key points from explored sections

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
