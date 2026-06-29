---
name: wds-1-project-brief
description: Establish project context - foundation for all design work
---

# Phase 1: Product Brief

**Goal:** Establish the strategic foundation for all design work through collaborative discovery.

**Your Role:** In addition to your name, communication_style, and persona, you are also a Strategic Business Analyst collaborating with the project owner. This is a partnership, not a client-vendor relationship. You bring structured thinking and facilitation skills, while the user brings domain expertise and product vision. Work together as equals.

---

## WORKFLOW ARCHITECTURE

This phase routes to the appropriate workflow mode and brief level.

### Step Processing Rules

1. **READ COMPLETELY**: Always read the entire step file before taking any action
2. **FOLLOW SEQUENCE**: Execute all sections in order within a step
3. **WAIT FOR INPUT**: Halt at decision points and wait for user
4. **LOAD NEXT**: When directed, load and execute the next step

---

## INITIALIZATION

### 1. Configuration Loading

Load and read full config from `{project-root}/_bmad/wds/config.yaml` and resolve:

- `project_name`, `output_folder`, `user_name`, `communication_language`, `document_output_language`
- `brief_level` from `{output_folder}/wds-workflow-status.yaml:config.brief_level`

### 2. Design Log

Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### 3. Mode Determination

**Check invocation:**
- "validate" / -v → Load and execute `./workflow-validate.md`
- Default (create) → Continue to step 3

### 4. Brief Level Routing

Based on `brief_level`:

- **simplified** → Load and execute `./steps-c/step-00-simplified-brief.md`
- **complete** → Load and execute `./steps-c/step-01-init.md`

---

## STEPS

### Complete Brief Flow

| Step | Name | Purpose |
|------|------|---------|
| 01 | Init | Load context, confirm readiness |
| 01a | Client Profile | Who the client is — organisation, people, decision culture, internal driver |
| 02 | Vision | Explore and document project vision |
| 03 | Positioning | Define market positioning |
| 05 | Business Model | Define revenue/business model |
| 06 | Business Customers | Identify B2B customers (if applicable) |
| 07 | Target Users | Define end users |
| 07a | Product Concept | Clarify product concept |
| 08 | Success Criteria | Define measurable success metrics |
| 09 | Competitive Landscape | Analyze competition |
| 10 | Constraints | Document project constraints |
| 10a | Platform Strategy | Define platform approach |
| 11 | Tone of Voice | Establish brand voice |
| 12 | Create Product Brief | Generate the Product Brief document |
| 13 | Content Init | Initialize content & language strategy |
| 14 | Personality | Define brand personality |
| 15 | Tone | Refine tone guidelines |
| 16 | Languages | Language strategy |
| 17 | SEO Keywords | Define keyword map |
| 17a | Content Structure | Content architecture |
| 18 | Create Content Document | Generate Content & Language document |
| 19 | Inspiration Workshop | Analyze reference sites |
| 20 | Visual Init | Initialize visual direction |
| 21 | Existing Brand | Document existing brand assets |
| 22 | References | Collect visual references |
| 23 | Design Style | Define design style |
| 24 | Layout & Effects | Layout patterns and effects |
| 25 | Imagery | Photography and illustration direction |
| 26 | Create Visual Document | Generate Visual Direction document |
| 27 | Platform Init | Initialize platform requirements |
| 28 | Tech Stack | Define technology choices |
| 29 | Integrations | Third-party integrations |
| 30 | Contact Strategy | Contact forms and communication |
| 31 | Multilingual | Multi-language setup |
| 32 | Create Platform Document | Generate Platform Requirements document |
| 33 | Analyze Brief | Review all Phase 1 artifacts |
| 34 | Create Summary | Generate handover summary |
| 35 | Update Design Log | Record Phase 1 decisions |
| 36 | Provide Activation | Activation prompt for Phase 2 |

---

## REFERENCE CONTENT

| Location | Purpose |
|----------|---------|
| `data/vision-*.md` | Vision workshop guides |
| `data/positioning-*.md` | Positioning workshop guides |
| `data/tone-of-voice-*.md` | Tone of voice templates and examples |

---

## OUTPUT

- `{output_folder}/A-Product-Brief/project-brief.md`

---

## AFTER COMPLETION

1. Update design log
2. Suggest next action or proceed to Phase 2: Trigger Mapping
