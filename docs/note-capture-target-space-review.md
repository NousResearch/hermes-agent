# Note Capture Target Space Review

This review maps current known capture and ingress flows onto the revised target
space model and identifies gaps for backlog work.

## Target Classes In Scope

### Current Obsidian Vault

Observed live store:

- `/Users/neilrobinson/Library/Mobile Documents/iCloud~md~obsidian/Documents/vault`

Observed current top-level structures include:

- `1.Inbox`
- `2.Projects`
- `3.Areas`
- `4.Resources`
- `5.Archive`
- `Interactions`
- `memory`
- `attachments`
- `second-brain-infra`
- `Projects`
- `Brand`

Observed structural realities that should inform the contract:

- the vault is not a pure PARA tree
- some domains exist both inside and outside the numbered PARA structure
- there are operational and code-bearing folders inside the vault
- there are duplicate or near-duplicate concepts across current paths

Examples:

- interaction material currently exists in multiple live places, including:
  - top-level `Interactions`
  - `4.Resources/Interactions`
  - approved canonical target: `3.Areas/Relationships`
- household and finance concepts already exist in multiple live places:
  - `3.Areas/household`
  - `3.Areas/finance`
  - `3.Areas/Personal/Finance/Invoices and Receipts`
  - `4.Resources/Household`
- review material already has concrete live structure:
  - `4.Resources/Reviews/Daily Notes`
  - `4.Resources/Reviews/Weekly Notes`
  - `4.Resources/Reviews/Annual Notes`
- content and knowledge targets already exist:
  - `4.Resources/Content Library`
  - `4.Resources/Content Queue`
  - `4.Resources/knowledge`
- infrastructure and attachment-like targets already exist:
  - `second-brain-infra`
  - `attachments`
  - `4.Resources/attachments`

### Non-Vault User Filesystem

Observed live stores include:

- `/Users/neilrobinson/HermesData/projections/second-brain`
- `/Users/neilrobinson/HermesData/projections/second-brain-derived`
- `/Users/neilrobinson/HermesData/daily-notes`
- other user-space folders under `/Users/neilrobinson/`

### Future Obsidian Targets

Not yet created, but explicitly requested as part of the target model.

Examples to plan for:

- additional thematic vaults
- reorganized vault structures
- domain-specific note spaces such as household operations

Modeling note:

- these are still represented with `target_class: obsidian_vault`
- their lifecycle is expressed via `target_status`, such as `deferred` or
  `unavailable`

## Current Known Capture / Ingress Flows

### 1. Vault PARA triage

Source:

- `hermes_cli/vault_para_triage.py`
- optional skill `vault-para-triage`

Current state:

- captures markdown inbox notes
- routes to vault-scoped PARA targets
- stages outputs per enabled store

Mapped target class:

- current Obsidian vault

Gap:

- target identity is still a vault-relative folder string, not a registry-backed
  canonical target id
- approved canonical choice:
  - approved canonical target path: `3.Areas/Relationships`
  - current target status: `pending_migration`
  - current live alternatives:
    - top-level `Interactions`
    - `4.Resources/Interactions`

### 2. Daily note capture

Source:

- runtime plugin `daily_note_capture`
- runtime script `daily_memo_preparer_v2.py`

Current state:

- special-case direct daily-note workflow
- writes to daily notes in the live second-brain vault

Mapped target class:

- current Obsidian vault

Gap:

- still bypasses the generic note-capture projection contract

### 3. Meeting classification

Source:

- runtime plugin `google_workspace_cli`
- tool `run_meeting_classification`

Current state:

- classifies calendar meetings
- writes meeting-classification artifacts associated with the vault review area

Mapped target class:

- current Obsidian vault

Gap:

- artifacts are not yet expressed through the broader target registry model

### 4. Research capture

Source:

- runtime plugin `research_capture`
- Gmail-backed ingest methods

Current state:

- approved runtime plugin
- newsletter ingest and local materialization

Mapped target class:

- currently local/runtime materialization first

Gap:

- no reviewed canonical mapping yet into the broader target registry across
  vault and non-vault destinations

### 5. Payments review

Source:

- runtime plugin `payments_review`
- Gmail-backed invoice/payment-request ingest

Current state:

- captures likely invoices and payment instructions into deterministic local
  state and materialized review artifacts
- explicitly avoids initiating any transfer

Mapped target class:

- currently local/runtime materialization first

Gap:

- not yet mapped to a canonical household-finance or finance-review target in
  the approved target registry
- current live candidates already exist and should be reviewed explicitly, for
  example:
  - `3.Areas/Personal/Finance/Invoices and Receipts`
  - `3.Areas/finance/Property Expenses`
  - `4.Resources/Household/Purchases`

Current approved path correction:

- point-two approved canonical target path:
  `3.Areas/Personal/Finance/Invoices and Receipts`
- current target status: `pending_migration`
- current live predecessor path:
  `3.Areas/Personal/Invoices and receipts`

### 6. Expense submission

Source:

- runtime memory points to `expense_submit.py`

Current state:

- operational workflow with ledger and email send

Mapped target class:

- non-vault operational state today

Gap:

- no reviewed projection target for durable note/file capture output

## Backlog Candidates

### High-value target registry work

1. Introduce a target registry that maps stable `target_id` values to current
   live target paths and target classes.
2. Add reorganisation support so existing target ids survive path changes.
3. Support planned future targets through `target_status` values such as
   `deferred` and `unavailable` before folders exist.

### Capture coverage gaps

1. Household bills automated capture from email into a household-finance target.
   Current live candidates to review:
   - `3.Areas/Personal/Finance/Invoices and Receipts`
     (approved canonical target path, current status `pending_migration`)
   - `3.Areas/Personal/Invoices and receipts`
     (current live predecessor path)
   - `3.Areas/finance/Property Expenses`
   - `4.Resources/Household/Purchases`
   Recommended outcome:
   use `3.Areas/Personal/Finance/Invoices and Receipts` as the approved
   canonical target path and treat the current live predecessor path as a
   migration source unless a later reorganisation promotes a different
   approved target id.
2. Payment-request and invoice capture from Gmail into a reviewed finance
   workflow target rather than only local materialized review state.
3. Research/newsletter capture into approved knowledge/library targets across
   vault and non-vault stores.
4. Attachment capture policy for invoices, statements, bills, and research
   documents.
5. Non-markdown file projection rules for PDFs, receipts, and other binary
   assets.

### Runtime/ops gaps

1. Runtime wording still teaches vault-centric paths instead of the broader
   target-space model.
2. Daily-note capture remains a special-case direct write path.
3. `research_capture` exists in runtime but requires reviewed enablement and
   target mapping decisions.

## Review Guidance

Before runtime rollout, review and approve:

1. target classes:
   `obsidian_vault`, `filesystem`
2. trust boundaries for each target class
3. the move from hard-coded folder strings toward canonical `target_id`
4. which current live vault paths become canonical when duplicates or competing
   locations exist
5. the initial backlog additions for household finance, payments, research, and
   attachments
6. target lifecycle states:
   `active`, `deferred`, `pending_migration`, `unavailable`

Current decisions already incorporated:

- `3.Areas/Relationships` is the approved canonical target path for
  interactions with `target_status: pending_migration`
- `3.Areas/Personal/Finance/Invoices and Receipts` is the approved canonical
  target path for the invoice-and-receipts point with
  `target_status: pending_migration`
