---
name: wds-0-alignment-signoff
description: Create alignment around your idea before starting the project
---

# Alignment & Signoff Workflow

**Purpose**: Create alignment around your idea before starting the project

**When to Use**:
- ✅ **Use Alignment & Signoff** if you need alignment with others:
  - Consultant proposing a solution to a client
  - Business hiring consultants/suppliers to build software
  - Manager/employee seeking approval for an internal project
  - Any scenario where stakeholders need to agree before starting

- ⏭️ **Skip Alignment & Signoff** if you're doing it yourself:
  - You have full autonomy and don't need approval
  - Go straight to the Project Brief workflow

---

## WORKFLOW ARCHITECTURE

### Step Processing Rules

1. **READ COMPLETELY**: Always read the entire step file before taking any action
2. **FOLLOW SEQUENCE**: Execute all sections in order within a step
3. **WAIT FOR INPUT**: Halt at decision points and wait for user
4. **LOAD NEXT**: When directed, load and execute the next step

---

## INITIALIZATION

### 1. Configuration Loading

Load and read full config from `{project-root}/_bmad/wds/config.yaml` and resolve:
- `project_name`, `output_folder`, `user_name`, `communication_language`

### 2. Design Log

Read `{output_folder}/_progress/00-design-log.md`. Check Current and Backlog for context.

### 3. Start

Load and execute `./steps-c/step-01a-understand-situation.md`

---

## STEPS

### Phase 1: Start & Understand (step-01*)

| Step | Name | Purpose |
|------|------|---------|
| 01a | Understand Situation | Assess what the user needs |
| 01b | Determine If Needed | Check if alignment workflow is appropriate |
| 01c | Offer Extract | Offer to extract from existing communications |
| 01d | Extract Info | Pull information from shared documents |
| 01e | Detect Starting Point | Route to appropriate explore section |

### Phase 2: Explore Sections (step-02*)

Explore 10 alignment document sections (flexible order):

| Step | Section | Topic |
|------|---------|-------|
| 02a | 1 | The Realization |
| 02b | - | The Solution |
| 02c | 2 | Why It Matters |
| 02d | 3 | How We See It Working |
| 02e | 4 | Paths We Explored |
| 02f | 5 | Recommended Solution |
| 02g | 6 | The Path Forward |
| 02h | 7 | The Value We'll Create |
| 02i | 8 | Cost of Inaction |
| 02j | 9 | Our Commitment |
| 02k | 10 | Summary |

### Phase 3: Synthesize & Present (step-03*)

| Step | Name | Purpose |
|------|------|---------|
| 03a | Reflect Back | Synthesize understanding, confirm |
| 03b | Synthesize Document | Create alignment document |
| 03d | Present for Approval | Share with stakeholders |

### Phase 4: Generate Signoff (step-04*)

| Step | Name | Purpose |
|------|------|---------|
| 04a | Offer Signoff | Present signoff options |
| 04b | Determine Business Model | Route to appropriate signoff type |

### Phase 5: Build Contract (step-05*)

| Step | Section | Topic |
|------|---------|-------|
| 05a | 1 | Project Overview |
| 05b | 2 | Business Model |
| 05c | 3 | Scope of Work |
| 05d | 4 | Payment Terms |
| 05e | 5 | Timeline |
| 05f | 6 | Availability |
| 05g | 7 | Confidentiality |
| 05h | 8 | Not to Exceed |
| 05i | 9 | Work Initiation |
| 05j | 10 | Terms and Conditions |
| 05k | 11 | Approval |
| 05l | - | Finalize Contract |

### Phase 6: Build Internal Signoff (step-06*)

| Step | Name | Purpose |
|------|------|---------|
| 06a | Build Internal Signoff | Create internal approval document |
| 06b | Finalize Signoff | Complete and save |

---

## REFERENCE CONTENT

| Location | Purpose |
|----------|---------|
| `data/01-start-understand-routing.md` | Start & understand routing |
| `data/02-explore-sections-routing.md` | Explore section frameworks |
| `data/03-synthesize-present-routing.md` | Synthesize & present routing |
| `data/04-generate-signoff-routing.md` | Signoff generation routing |
| `data/05-build-contract-routing.md` | Contract building routing |
| `data/06-build-signoff-internal-routing.md` | Internal signoff routing |

---

## OUTPUT

- **Alignment Document**: `{output_folder}/A-Product-Brief/pitch.md`
- **Signoff Document**: `{output_folder}/A-Product-Brief/contract.md` (or `service-agreement.md` or `signoff.md`)

---

## AFTER COMPLETION

1. Update design log
2. Proceed to Project Brief workflow:
   → `skill:wds-1-project-brief`
