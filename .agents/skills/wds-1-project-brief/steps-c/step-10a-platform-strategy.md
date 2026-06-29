---
name: 'step-10a-platform-strategy'
description: 'Define platform and device strategy'

# File References
nextStepFile: './step-11-tone-of-voice.md'
workflowFile: '../workflow.md'
activityWorkflowFile: '../workflow.md'
---

# Step 10A: Define Platform & Device Strategy

## STEP GOAL:
Establish the technical platform strategy and device support requirements that will shape all design and development decisions.

## MANDATORY EXECUTION RULES (READ FIRST):

### Universal Rules:
- NEVER generate content without user input
- CRITICAL: Read the complete step file before taking any action
- CRITICAL: When loading next step with 'C', ensure entire file is read
- YOU ARE A FACILITATOR, not a content generator
- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

### Role Reinforcement:
- You are a Strategic Business Analyst helping user make critical architectural decisions about platforms and devices
- If you already have been given a name, communication_style and persona, continue to use those while playing this new role
- We engage in collaborative dialogue, not command-response
- You bring structured thinking and facilitation skills, user brings domain expertise and product vision
- Maintain collaborative and strategic tone throughout

### Step-Specific Rules:
- Focus: Platform choice, device support, interaction models, platform rationale
- FORBIDDEN: Do not make technology decisions without user input
- Approach: Present options with trade-offs, guide user to informed decision

## EXECUTION PROTOCOLS:
- Primary goal: Platform strategy documented with rationale
- Save/document outputs appropriately
- Avoid generating content without user input

## CONTEXT BOUNDARIES:
- Available context: All previous steps (vision, positioning, Trigger Map, business model, users, success criteria, competitive landscape, constraints)
- Focus: Platform and device strategy
- Limits: Not detailed technical specs - strategic platform direction
- Dependencies: Steps 1-10 completed

## Sequence of Instructions (Do not deviate, skip, or optimize)

### 1. Guide Platform Strategy Definition
Help user define their platform strategy by asking about primary platform choice, supported devices, device priority, interaction models needed, offline functionality requirements, native device features needed, and platform rationale including constraints and future plans.

**Common Platform Options:**

1. **Responsive Web Application** - Single codebase, works across all devices, fastest time to market, no app store approval, limited native features
2. **Native Mobile Apps (iOS/Android)** - Best performance and UX, full device features, requires separate codebases, app store approval process
3. **Progressive Web App (PWA)** - Web app with native-like features, offline capable, installable, good balance of web and native
4. **Desktop Application** - Windows/Mac/Linux apps, full system integration, best for power users and complex workflows
5. **Cross-Platform (React Native, Flutter, Electron)** - Single codebase for multiple platforms, near-native performance, faster than separate native apps
6. **Multi-Platform Strategy** - Different platforms for different use cases (e.g., web for setup/admin, mobile for daily use), higher complexity but optimized per context

**Device Priority Options:**

- **Mobile-first** - Design for phones, scale up to tablets/desktop
- **Desktop-first** - Design for desktop, scale down to tablets/mobile
- **Equal priority** - All devices equally important, universal design

**Interaction Models:**

- Touch (mobile, tablets)
- Mouse and keyboard (desktop)
- Voice commands
- Gesture controls
- Accessibility devices (screen readers, switch controls)

### 2. Capture and Validate
Capture platform strategy, validate alignment with vision and constraints, and document in Product Brief under "Platform & Device Strategy" section including primary platform, supported devices, device priority with rationale, interaction models, technical requirements (offline, native features), platform rationale, constraints considered, future plans, and design/development implications.

### 3. Design Log Update
**Mandatory:** Append to `dialog/decisions.md` if key decisions were made.

**Record:**
- Platform/device strategy chosen
- Responsive vs native vs hybrid decision
- Technical approach and rationale

**Then:** Mark Step 10a complete in `dialog/progress-tracker.md` progress tracker

### N. Present MENU OPTIONS
Display: "**Select an Option:** [C] Continue to next step"

#### Menu Handling Logic:
- IF C: Load, read entire file, then execute {nextStepFile}
- IF M: Return to {workflowFile} or {activityWorkflowFile}
- IF Any other comments or queries: help user respond then [Redisplay Menu Options]

#### EXECUTION RULES:
- ALWAYS halt and wait for user input after presenting menu
- User can chat or ask questions - always respond and then redisplay menu options

## CRITICAL STEP COMPLETION NOTE
ONLY WHEN step objectives are met and user confirms will you then load and read fully `{nextStepFile}`.

---

## SYSTEM SUCCESS/FAILURE METRICS

### SUCCESS:
- Platform strategy captured with clear rationale
- Device priority defined
- Interaction models identified
- Alignment with vision and constraints validated
- User confirmed

### FAILURE:
- Made technology decisions without user input
- Skipped platform rationale
- Generated content without user collaboration

**Master Rule:** Skipping steps, optimizing sequences, or not following exact instructions is FORBIDDEN and constitutes SYSTEM FAILURE.
