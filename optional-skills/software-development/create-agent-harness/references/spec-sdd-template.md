# SPEC SDD Template

Reference for `create-agent-harness` Phase 3 / `plan` sub-agent.

> **Base:** Spec-Driven Development. The `plan` sub-agent copies this template to `.specs/SPEC-{YYYYMMDD}-{feature-name}.md` and fills it before any implementation. An LLM must be able to implement the feature by reading only this SPEC. All fields must be filled; use `[A DEFINIR]` only when the user explicitly declines to answer. No implementation before the SPEC `Status` in section 0 is `Approved`.

```markdown
# SPEC-{YYYYMMDD}-{feature-name}

## 0. Metadata

| Field | Value |
| --- | --- |
| Feature | `[feature-name]` |
| Type | `[Feature | Bugfix | Refactor | API | Infra | Frontend]` |
| Stack | `[Angular | .NET | Java | Terraform | Python | OpenAPI | Docs]` |
| Repository | `[repo path]` |
| Branch | `feature/{AgentLLM}-{YYYYMMDD}-{short-description}` |
| Ticket | `[TICKET-123]` |
| Status | `[Draft | Approved | In implementation | Done]` |

## 1. User Story

**As a** `[persona / user]`
**I want** `[capability / action]`
**So that** `[benefit / business value]`.

**Problem context:**
`[What exists today, limitation, risk or opportunity.]`

## 2. Scope

**In scope:**
- `[item]`
- `[item]`

**Out of scope:**
- `[item]`
- `[item]`

## 3. Technical Context

**Where the change happens** (architecture, layers, integrations):
`[describe objectively]`

**Files to read before implementing:**
- `CLAUDE.md` · `.claude/rules/global-rules.md`
- `[e.g. .claude/rules/{stack}.md]`
- `[e.g. src/{module}/...]`

**Files to create or modify:**
```text
[list paths]
```

## 4. Requirements

> Number every requirement (RF-001, RF-002...). Must be objective and verifiable.

### RF-001: `[name]`
- **Description:** `[The system must...]`
- **Rules:** `[rule 1; rule 2]`
- **Input → Output:** `[input] → [output]`

### RF-002: `[name]`
- **Description:** `[...]`

**Business rules / invariants:**
- `[e.g. session cannot be created without valid token]`
- `[e.g. unique identifier required to identify entity]`

## 5. API Contract (if applicable)

> Fill only if the feature exposes/consumes an API. Document required headers, auth and correlation strategies used by the organization.

**Endpoint:** `[METHOD] /api/v{n}/{resource}`
**Auth:** `[Bearer | JWT | API key | mTLS]`

**Request:**
```json
{ }
```

**Response (success):**
```json
{ "data": {} }
```

**Expected errors:** `[400 validation | 401 auth | 404 not found | 409 conflict | 429 rate limit]` — always generic error format, no PII.

## 6. Acceptance Criteria

> Write as testable scenarios in BDD: "Given...when...then" (or pt-BR "Dado...quando...então"). Each criterion becomes a test.

- [ ] **Given** `[context]` **when** `[action]` **then** `[observable result]`
- [ ] **Given** `[context]` **when** `[action]` **then** `[observable result]`

**Edge cases:**

| Scenario | Input | Expected behavior |
| --- | --- | --- |
| `[e.g. invalid input]` | `[...]` | `[e.g. 400 with message]` |
| `[e.g. missing resource]` | `[...]` | `[e.g. 404]` |

## 7. Task Plan (agent execution)

> Ordered, small tasks the agent executes. Mark `[ ]` as done.

- [ ] **T1 — Discovery:** read context files (section 3) and understand existing patterns.
- [ ] **T2 — Implementation:** `[describe the main change]`
- [ ] **T3 — Verification:** apply the validation strategy for this stack/type (section 7.1) and cover every acceptance criterion (section 6).
- [ ] **T4 — Validation:** run the appropriate build / lint / plan / validate commands locally and fill the DoD (section 9).
- [ ] **T5 — Done + PR:** when the DoD is complete, set SPEC `Status = Done` and immediately open the PR on branch `feature/...`.

**7.1 Validation strategy by type/stack**

> Match the Type/Stack in the metadata. Do not enforce unit-test coverage for artifacts that do not contain application code.

| Type / Stack | Required evidence |
|---|---|
| **Angular** | Unit tests (Jest) for business rules and components; integration tests for flows; minimum **90%** coverage. |
| **.NET** | Unit tests for business rules; integration tests for API/data flows; minimum **80%** coverage. |
| **Java** | Unit tests for business rules; integration tests for API/data flows; minimum **85%** coverage. |
| **Python** | Unit + integration tests; minimum **80%** coverage. |
| **API / OpenAPI** | Contract validation (`swagger-codegen`, `spectral`, `prism mock`); consumer/producer integration tests when the API is implemented. |
| **Terraform / Infra** | `terraform fmt`, `terraform validate`, `terraform plan` (read-only), security scan (`checkov`/`tfsec`), cost/impact review. |
| **Docs** | Markdown lint, link checking, spell check, peer review. |
| **Refactor** | Full existing test suite must pass; coverage must not decrease; no behavioral change without a new test. |
| **Bugfix** | Reproduction test first, then fix; regression test included. |

- Unit tests for business rules and requirements (when the stack has code).
- Integration tests for API/data flows (when the stack has code).
- Coverage threshold as defined in the table above.

## 8. Organization Guardrails (mandatory when provided)

- **Branches:** never commit to `main`, `master` or `develop`. Use `feature/{AgentLLM}-{YYYYMMDD}-{short-description}`.
- **Workflows:** do not modify `.github/workflows/` (protected by branch rule).
- **Registries:** use only approved corporate/organization registries — never public registries unless explicitly authorized.
- **API headers:** use the organization's standard correlation / authentication headers as defined by the API governance team.
- **Security:** do not log PII/tokens/identifiers; store secrets via the organization's secret manager; no `.env` in commit.
- **Scope:** do not invent requirements or expand scope. Stop and ask on ambiguity.
- **Architecture:** no business logic in controllers/components; domain does not access infrastructure.

## 9. Definition of Done

> Filled **during and after implementation**, not at approval time. The SPEC is approved with `Status = Approved` in section 0 and becomes a living document while `Status = In implementation`. The implementing agent checks off the items below; the DoD is complete when **all items are checked**. Immediately after that, set `Status = Done` in section 0 and open the PR. Setting `Status = Done` and opening the PR are **not DoD items** — they are the actions that follow the DoD.

- [ ] All requirements (section 4) implemented.
- [ ] All acceptance criteria (section 6) covered by passing tests or equivalent validation evidence for the stack (section 7.1).
- [ ] Edge cases handled.
- [ ] Build, lint and tests / `terraform validate` / contract lint / docs lint pass locally; minimum coverage or equivalent quality gate reached.
- [ ] Guardrails in section 8 respected.
- [ ] Logs contain no PII/tokens; errors use the organization's generic error format.

**Next action after DoD is complete:** set `Status = Done` in section 0 and open the PR on branch `feature/...` referencing the ticket.

## Open Questions / Pending Ambiguity

- `[list blocking questions — resolve before approving]`
```

## Agent loop for SPEC approval and delivery

After `plan` writes the SPEC, the parent agent must:

1. Read the SPEC and confirm every section is filled and coherent.
2. Ask the user for approval or revision.
3. When approved, set `Status` to `Approved` in section 0. Only after **Status = Approved** may implementation begin.
4. When implementation begins, set `Status` to `In implementation` and keep it updated as a living document.
5. During implementation, fill the DoD checklist (section 9). The DoD is complete when **all items are checked**.
6. When the DoD is complete, set `Status` to `Done` in section 0 and immediately open the PR on branch `feature/...`.
7. Any change during implementation must update the SPEC first.
