---
name: 'step-06a-build-internal-signoff'
description: 'Build internal signoff document as an alternative to external contract for internal company projects'

# File References
nextStepFile: './step-06b-finalize-signoff.md'
workflowFile: '../workflow.md'
---

# Step 35: Build Internal Signoff Document

## STEP GOAL:

Build an internal signoff document for company projects, covering goals, budget, ownership, approval workflow, and timeline - as an alternative to external contracts.

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

- 🎯 Focus only on building the internal signoff document sections
- 🚫 FORBIDDEN to use external contract language - this is an internal document
- 💬 Approach: Focus on goals, ownership, approval workflow - not detailed scope/hours
- 📋 Offer to adapt to company's existing signoff format if they have one

## EXECUTION PROTOCOLS:

- 🎯 Build internal signoff document with all required sections
- 💾 Save to `docs/wds-1-project-brief/signoff.md`
- 📖 Focus on internal company needs (goals, budget, ownership, approval)
- 🚫 Do not use external contract language

## CONTEXT BOUNDARIES:

- Available context: Accepted alignment document, internal signoff selection
- Focus: Internal signoff document
- Limits: Internal company document - not external contract
- Dependencies: step-04a must be completed with internal signoff selection

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Build Internal Signoff Document

**For Internal Signoff Document**:

**Focus areas** (not detailed scope/hours):
- Goals and success metrics
- Budget estimates
- Ownership and responsibility
- Approval workflow
- Timeline and milestones

**Section 1: Project Overview**
- The Realization
- Recommended Solution

**Section 2: Goals and Success Metrics**
- What we're trying to accomplish
- Success metrics
- How we'll measure success
- Key performance indicators (KPIs)

**Section 3: Budget and Resources**
- Budget allocation (total budget estimate)
- Budget breakdown (if applicable)
- Resources needed (high-level)
- Not-to-exceed budget cap (if applicable)

**Section 4: Ownership and Responsibility**
- Project Owner
- Process Owner
- Key Stakeholders
- Decision-Making Authority

**Section 5: Approval and Sign-Off**
- Who needs to approve
- Approval stages
- Sign-off process
- Timeline for approval

**Section 6: Timeline and Milestones**
- Key milestones
- Delivery dates
- Critical deadlines

**Section 7: Optional Sections**
- Risks and considerations (optional)
- Confidentiality (optional)
- The Path Forward (optional - high-level overview)

### 2. Company Signoff Format (Optional)

**Company Signoff Format (Optional)**:
- If user has existing company format, ask: "Do you have an existing company signoff document format? You can upload it and I'll adapt it to match your format."

### 3. Present MENU OPTIONS

Display: "**Select an Option:** [C] Continue to step-06b-finalize-signoff"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- ONLY proceed to next step when user selects 'C'
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE

ONLY WHEN the internal signoff document is built and user is satisfied will you then load and read fully `{nextStepFile}` to execute and begin the next step.

---

## 🚨 SYSTEM SUCCESS/FAILURE METRICS

### ✅ SUCCESS:
- Internal signoff document covers all required sections
- Document is adapted to company format if provided
- Focus is on goals, ownership, and approval - not contract language
- User confirms the document

### ❌ SYSTEM FAILURE:
- Using external contract language for internal document
- Skipping ownership and approval sections
- Not offering to adapt to company format
- Missing key sections

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
