# Note Capture Projection Contract

This document defines the canonical storage mapping for Hermes note capture and
the rules used to derive projected locations in downstream target spaces.

## Purpose

Hermes writes captured notes and files into canonical internal data structures
first. Projection into user-visible stores happens later and may cross trust
boundaries such as iCloud sync, Obsidian app state, or the broader user
filesystem under `/Users/neilrobinson/`.

The contract is:

1. Hermes classifies a capture to one canonical target identity.
2. Hermes stores canonical content and routing metadata in an event log.
3. Hermes stages one projected artifact per enabled downstream store.
4. External sync or projection workers materialize those staged artifacts into
   live targets.

## Current Scope

This contract applies to generic captured notes and files routed through the
canonical note-capture flow.

Current runtime exception:

- daily-note creation still uses a dedicated plugin-backed write path in the
  live runtime

That daily-note path should be treated as a transitional exception until it is
migrated deliberately with its downstream consumers.

## Canonical Storage

For each captured item, Hermes writes:

- Canonical event:
  `.hermes/note-capture/events/<entry_id>.json`
- Source archive:
  `.hermes/note-capture/source-archive/<entry_id>/<original-relative-path>`
- Projection status summary:
  `.hermes/note-capture/status/latest.json`

The canonical event is the source of truth. Projected files are derived
artifacts.

## Target Model

The target model must survive target-space reorganisation. Therefore the
contract separates:

- canonical target identity
- target class
- logical target path
- store-specific materialization path

Recommended canonical target fields:

- `target_id`: stable identifier such as
  `vault.second_brain.resources.content_library`
- `target_class`: one of:
  - `obsidian_vault`
  - `filesystem`
- `logical_path`: reorg-friendly logical location such as
  `resources/content_library`
- `display_path`: current human-facing path such as
  `4.Resources/Content Library`
- `trust_boundary`: where live writes occur, such as
  `external_sync`, `icloud_obsidian`, or `user_filesystem`
- `target_status`: one of:
  - `active`
  - `deferred`
  - `pending_migration`
  - `unavailable`

Current implementation note:

- the repo code still uses a vault-scoped `target` string for current routing
- the broader target-registry model in this contract is the approved direction
  for making projection comprehensive and reorg-safe

## Target Classes

### 1. Current Obsidian Vault Targets

This is the current live Obsidian second-brain store under the iCloud vault
path:

`/Users/neilrobinson/Library/Mobile Documents/iCloud~md~obsidian/Documents/vault`

Observed current top-level target families include:

- numbered PARA buckets:
  - `1.Inbox`
  - `2.Projects/...`
  - `3.Areas/...`
  - `4.Resources/...`
  - `5.Archive/...`
- additional live top-level families:
  - `Interactions`
  - `memory`
  - `attachments`
  - `second-brain-infra`
  - `Projects`
  - `Brand`

Important observed nuance:

- the live vault is not a perfectly normalized PARA tree
- some concepts appear in more than one place, but the contract can still pick
  one canonical live path and treat the others as legacy or non-canonical
  paths
- some top-level folders are operational or infrastructure-oriented rather than
  classic notes, for example:
  - `second-brain-infra`
  - `attachments`
  - `memory`

The contract must therefore model the current vault structure as it exists,
while still allowing future normalization and reorganisation.

Trust-boundary note:

- Even though this vault sits under the user home directory, it should be
  treated as a distinct target because it is mediated by Obsidian and iCloud
  behavior, not just ordinary filesystem writes.

### 2. Non-Vault User Filesystem Targets

These are live targets under `/Users/neilrobinson/` that are not the main
Obsidian vault, including but not limited to:

- `/Users/neilrobinson/HermesData/...`
- `/Users/neilrobinson/Documents/...`
- `/Users/neilrobinson/Downloads/...`
- other managed working folders or filing locations

This class may also include projection directories such as:

- `/Users/neilrobinson/HermesData/projections/second-brain`
- `/Users/neilrobinson/HermesData/projections/second-brain-derived`

Trust-boundary note:

- These are still downstream targets, but they do not share the exact sync and
  application semantics of the live Obsidian/iCloud vault.

### 3. Future Obsidian Vault Targets

The contract must support planned Obsidian targets that do not exist yet.

Examples:

- a new vault for household operations
- a dedicated work vault
- a reorganized second-brain vault with different top-level buckets

These targets should be registerable in the target registry before the folders
or vaults are created, using `target_status: deferred` or
`target_status: unavailable` until they become live.

## Canonical Target Selection

Hermes should classify a capture to a canonical `target_id`, not merely to a
current folder string.

Examples:

- `vault.second_brain.resources.content_library`
- `vault.second_brain.resources.interactions`
- `vault.second_brain.areas.relationships`
- `vault.second_brain.areas.personal.finance.invoices_and_receipts`
- `vault.second_brain.areas.finance.property_expenses`
- `vault.second_brain.resources.household.purchases`
- `vault.second_brain.reviews.daily_notes`
- `vault.second_brain.infrastructure.second_brain_infra`
- `filesystem.hermesdata.projections.second_brain`
- `obsidian.future.household_ops.invoices`

The target registry should then resolve that canonical identity into the
current store-specific path.

Where the current vault already contains multiple plausible live locations for a
concept, the target registry should pick one approved canonical target id and
record any legacy or alternate live paths as aliases during migration.

Current approved canonical examples from the live vault:

- interaction notes:
  - approved canonical target path: `3.Areas/Relationships`
  - current target status: `pending_migration`
  - current live alternatives include:
    - top-level `Interactions`
    - `4.Resources/Interactions`
- invoice and receipt filing:
  - approved canonical target path:
    `3.Areas/Personal/Finance/Invoices and Receipts`
  - current target status: `pending_migration`
  - current live predecessor path:
    `3.Areas/Personal/Invoices and receipts`

## Projected Path Derivation

For each enabled projection store:

```text
projected_relative_path = <store.path_prefix>/<resolved-target-path>/<filename>
```

Rules:

- `store.path_prefix` may be empty.
- `resolved-target-path` is derived from the canonical target identity through
  the active target registry.
- `filename` is the captured artifact filename.
- Path separators are normalized to `/`.

Current compatibility rule:

- For vault-scoped targets already implemented in the repo, `resolved-target-path`
  is currently the routed vault-relative folder string.

## Staging Layout

Projected artifacts are staged under:

```text
.hermes/note-capture/staging/<store>/<entry_id>/<projected-relative-path>
```

This guarantees:

- one canonical event can fan out to multiple stores
- each staged output is tied to a stable `entry_id`
- sync jobs can reconcile by store and event id
- Hermes never needs direct write access to the live target spaces

## Status Semantics

Per-store projections start in `pending` state inside the event log. Hermes
summarizes aggregate store state in `.hermes/note-capture/status/latest.json`.

Current state meanings:

- `pending`: staged and waiting for external projection

Future projection workers may extend this with states such as `projected`,
`failed`, `superseded`, or store-local materialization states, but the
canonical event remains authoritative. Canonical target lifecycle is tracked by
`target_status`, not by per-store projection state.

## Feedback and Corrections

Reviewer feedback changes routing metadata and staged projections, not the live
target space directly.

- `approve`: keep the canonical target and clear feedback-needed state
- `correct`: update the canonical target, rewrite staged projections, and
  remove obsolete staged files
- `ignore`: record reviewer action without changing staging

Corrections preserve the same `entry_id`. The event is updated in place because
it is the canonical record for that captured item.

## Trust Boundaries

Hermes is responsible for:

- routing
- canonical content rendering
- event logging
- staging
- audit/status updates

External sync or projection workers are responsible for:

- copying staged artifacts into the live Obsidian vault
- copying staged artifacts into non-vault user folders
- projecting into future vaults when they exist
- target-space migration and cleanup during reorganisation
- projection health notifications and reconciliation

## Reorganisation Rule

The target space can be reorganised. Reorganisation must not require changing
historical capture events by hand.

Best-practice rule:

- historical events keep their canonical `target_id`
- the target registry is updated to map that `target_id` to a new live path
- projection workers re-materialize from canonical staging or canonical content
  as needed

This is why the contract prefers canonical target identities over hard-coded
folder strings.
