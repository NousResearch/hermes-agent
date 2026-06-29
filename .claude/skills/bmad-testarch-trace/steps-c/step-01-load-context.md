---
name: 'step-01-load-context'
description: 'Resolve coverage oracle, load knowledge base, and gather related artifacts'
nextStepFile: '{skill-root}/steps-c/step-02-discover-tests.md'
knowledgeIndex: './resources/tea-index.csv'
outputFile: '{test_artifacts}/traceability-matrix.md'
---

# Step 1: Resolve Coverage Oracle & Load Knowledge Base

## STEP GOAL

Resolve the best available coverage oracle, capture confidence and provenance, and gather supporting artifacts for traceability.

## MANDATORY EXECUTION RULES

- 📖 Read the entire step file before acting
- ✅ Speak in `{communication_language}`

---

## EXECUTION PROTOCOLS:

- 🎯 Follow the MANDATORY SEQUENCE exactly
- 💾 Record outputs before proceeding
- 📖 Load the next step only when instructed

## CONTEXT BOUNDARIES:

- Available context: config, source tree, loaded artifacts, and knowledge fragments
- Focus: this step's goal only
- Limits: do not execute future steps
- Dependencies: prior steps' outputs (if any)

## MANDATORY SEQUENCE

**CRITICAL:** Follow this sequence exactly. Do not skip, reorder, or improvise.

## 1. Resolve Coverage Oracle

At least one of the following must be usable:

- Formal requirements (story/epic acceptance criteria, PRD, test design)
- Contract/spec artifacts (OpenAPI, GraphQL schema, protobuf, etc.)
- External pointers to a requirements source that can be resolved through installed adapters/MCPs
- Analyzable source code that supports synthetic journey/requirement inference

Tests exist OR gaps are explicitly acknowledged.

Resolve the oracle in this order:

1. **Formal requirements first**
   - Story/epic acceptance criteria
   - PRD / test design / tech spec
   - Inline requirements provided by the user

2. **Contract/spec artifacts second**
   - OpenAPI / Swagger
   - GraphQL schema or SDL
   - Other machine-readable contract definitions

3. **External pointers third**
   - Placeholder files that point to external trackers or docs such as Jira, Linear, Confluence, shared docs, or other systems of record
   - Follow the pointer automatically only when a compatible adapter/plugin/MCP is available in the active runtime
   - Record `externalPointerStatus` as one of: `not_used`, `resolved`, `skipped`, or `unavailable`

4. **Synthetic oracle last**
   - If no formal oracle exists and `allow_synthetic_oracle` is enabled, inspect `{source_dir}` to infer a provisional trace target
   - For UI apps, infer journeys from:
     - routes/pages/screens/layout entry points
     - navigation flows and feature entry links
     - forms, submit actions, create/update/delete paths
     - auth/session/logout/role-gated flows
     - loading, empty, validation, error, and permission-denied states
     - feature flags and major conditional branches
   - Deduplicate the inferred items into a compact, traceable list (prefer 5-12 items)
   - Assign stable IDs such as `J-01`, `J-02`, etc.
   - Assign provisional priorities using `test-priorities-matrix.md`
     - `P0`: auth, checkout/payment, destructive data changes, revenue-critical, hard blockers to core use
     - `P1`: primary user journeys and common CRUD paths
     - `P2`: secondary workflows and edge scenarios
     - `P3`: low-risk polish or optional flows

Record the resolved oracle metadata in step output/frontmatter using consistent keys:

- `coverageBasis` (`acceptance_criteria` | `synthetic_requirements` | `openapi_endpoints` | `user_journeys`) — the type of oracle selected for coverage tracing
- `oracleResolutionMode` (`formal_requirements` | `spec_artifact` | `external_pointer` | `synthetic_source`) — how the oracle was discovered/resolved
- `oracleConfidence` (`high` | `medium` | `low`) — confidence in the resolved oracle as a coverage source
- `oracleSources` — list of artifact paths, URIs, or references used to resolve the oracle
- `externalPointerStatus` (`not_used` | `resolved` | `skipped` | `unavailable`) — status of external pointer resolution when pointer files are present

If none of the four oracle types can be resolved, **HALT** and request the smallest missing clarification needed to continue.

---

## 2. Load Knowledge Base

From `{knowledgeIndex}` load:

- `test-priorities-matrix.md`
- `risk-governance.md`
- `probability-impact.md`
- `test-quality.md`
- `selective-testing.md`

---

## 3. Load Artifacts

If available:

- Story file and acceptance criteria
- Test design doc (priorities)
- Tech spec / PRD
- OpenAPI or similar contract/spec files
- Placeholder files that reference external requirements systems
- Route maps, page/screen registries, and other source files used for synthetic journey inference

Summarize what was found and explicitly state the resolved oracle, its confidence, and why that oracle was selected.

---

### 4. Save Progress

**Save this step's accumulated work to `{outputFile}`.**

- **If `{outputFile}` does not exist** (first save), create it using the workflow template (if available) with YAML frontmatter:

  ```yaml
  ---
  stepsCompleted: ['step-01-load-context']
  lastStep: 'step-01-load-context'
  lastSaved: '{date}'
  coverageBasis: '{resolved coverage_basis}'
  oracleConfidence: '{resolved oracle_confidence}'
  oracleResolutionMode: '{resolved oracle_resolution_mode}'
  oracleSources: ['{resolved oracle source 1}', '{resolved oracle source 2}']
  externalPointerStatus: '{resolved external_pointer_status}'
  ---
  ```

  Then write this step's output below the frontmatter.

- **If `{outputFile}` already exists**, update:
  - Add `'step-01-load-context'` to `stepsCompleted` array (only if not already present)
  - Set `lastStep: 'step-01-load-context'`
  - Set `lastSaved: '{date}'`
  - Set `coverageBasis` to the resolved oracle basis
  - Set `oracleConfidence` to the resolved oracle confidence
  - Set `oracleResolutionMode` to the resolved oracle resolution mode
  - Set `oracleSources` to the resolved oracle sources
  - Set `externalPointerStatus` to the resolved external pointer status
  - Append this step's output to the appropriate section of the document.

Load next step: `{nextStepFile}`

## 🚨 SYSTEM SUCCESS/FAILURE METRICS:

### ✅ SUCCESS:

- Step completed in full with required outputs

### ❌ SYSTEM FAILURE:

- Skipped sequence steps or missing outputs
  **Master Rule:** Skipping steps is FORBIDDEN.
