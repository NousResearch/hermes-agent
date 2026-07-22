# DATA_SCHEMA_V1

## Synthetic-only, no live intake

## Entities
- deal
- document
- contact
- task
- approval_request

## Schema design

### deal
- id: text primary key
- name: text required
- stage: enum required
- risk: enum required
- sector: text optional
- owner: text required
- estimated_value: integer optional
- created_at: iso timestamp required
- updated_at: iso timestamp required
- status: enum required
- tier: enum required

### document
- id: text primary key
- deal_id: text optional
- title: text required
- extension: text required
- sha256: text required
- tier: enum required
- state: enum required
- stored_path: text required
- created_at: iso timestamp required
- updated_at: iso timestamp required

### contact
- id: text primary key
- name: text required
- role: text optional
- email: text synthetic-only
- phone_e164: text synthetic-only
- tier: enum required
- state: enum required

### task
- id: text primary key
- deal_id: text optional
- assignee: text required
- status: enum required
- priority: enum required
- due_at: iso timestamp optional
- created_at: iso timestamp required
- source: enum required

### approval_request
- id: text primary key
- resource_type: text required
- resource_id: text required
- requester: text required
- approver: text required
- decision: enum required
- decided_at: iso timestamp optional
- notes: text optional

## Schema invariants
- Every record carries `tier`.
- Every record carries `state` with values:
  - `synthetic`, `quarantine`, `validated`, `approved`, `archived`
- No confidential field allowed until Arthur authorizes live intake.

## Evolution
- Schema changes require new version filename.
- Backwards compatibility policy approved by Arthur.
