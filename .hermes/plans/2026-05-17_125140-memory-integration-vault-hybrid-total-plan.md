# Hermes `memory-integration` Vault-Hybrid Total Implementation Plan

> **For Hermes:** This is a planning artifact only. Do not implement from this plan until the user explicitly greenlights it. When implementation starts, use `subagent-driven-development` task-by-task with read-only/spec review gates.

**Goal:** Build a bundled Hermes `memory-integration` provider that uses an Obsidian/LLM-wiki-compatible Markdown vault as the canonical approved semantic memory layer, SQLite as the operational/index/control plane, and the existing `obsidian-vault-adapter` style path-resolution primitive for reliable vault discovery.

**Acceptance Criteria / Done Means:**
- Hermes can select `memory-integration` as a bundled memory provider.
- Approved semantic memory is stored in human-readable Markdown pages in an Obsidian-compatible vault.
- SQLite remains authoritative for operational state: page registry, patch queue, locks, reference mappings, search projection, event metadata, idempotency, and migrations.
- The provider is headless-safe: it does not require the Obsidian desktop app or Obsidian CLI to run.
- The provider can optionally integrate with an Obsidian Headless sync workflow and the existing vault adapter for vault path resolution, but correctness does not depend on sync being active.
- All semantic memory mutations are patch-first and approval-gated unless an operation is explicitly defined as immutable raw/event capture.
- The provider preserves Hermes built-in `MEMORY.md` / `USER.md` behavior and does not mirror raw built-in memory content by default.
- Background contexts (`cron`, `flush`, `subagent`) cannot apply semantic writes by default.
- The implementation has deterministic unit tests using temporary Hermes homes and fixture vaults, with no network dependency.

**Architecture:**
- **Markdown vault:** canonical source for approved semantic memory pages and immutable event/source artifacts.
- **SQLite sidecar:** operational authority for indexing, locking, patch workflow, reference resolution, retrieval metadata, and migrations.
- **Vault locator adapter:** use or mirror the `obsidian-vault-adapter` fail-loud discovery semantics for locating and validating the vault root.
- **Obsidian Headless:** supported as an optional sync/draw-vault operational mode, not as the memory database and not as a required runtime dependency.

**Tech Stack:** Python stdlib-first; `sqlite3`; Markdown/YAML-ish frontmatter parsing implemented minimally or via existing project conventions if already available; no network service or external API requirement in v1. Optional integration with the existing `vault-adapter` package/CLI when installed.

---

### Opus adversarial review amendments incorporated

Claude Code Opus max-effort read-only adversarial reviews were run on this plan on 2026-05-17 and 2026-05-18. The following amendments are incorporated before implementation:

- Use an explicit `vault.mode: dedicated | shared`; do **not** auto-detect vault layout from existing directories. If mode is unset, fail closed and ask/bootstrap explicitly.
- Follow-up Opus review reversed the adapter recommendation: the strategic fix is to harden `obsidian-vault-adapter` into a stable shared library and import it, rather than reimplementing locator behavior in every product. V1 should improve/pin the adapter first if that can be time-boxed; only use an internal temporary locator if adapter hardening misses the time-box, with a deletion issue/date.
- Treat Obsidian Headless as deferred operational sync infrastructure. V1 status may report sync as not configured/unsupported, but provider correctness must not depend on headless sync and Hermes must not manage `ob sync --continuous` in v1.
- Hardcode the SQLite sidecar default to `$HERMES_HOME/memory-integration/memory_integration.db` for v1; remove inside-vault SQLite as a v1 option.
- Keep `memory_integration_status` in v1 as a bounded read-only JSON tool; do not add full semantic lint in v1 unless trivial.
- Add memory-root ownership marker enforcement via `_system/OWNER`. Refuse to take ownership of a root owned by another provider/project.
- Add single-writer protection for Markdown mutations using a process/file lock under `$HERMES_HOME/memory-integration/`; SQLite locks alone are not enough because Markdown writes are not transactional.
- Reindex must ignore Obsidian Sync conflict files such as `*.conflict-*.md`, count/surface them in status, and avoid duplicate-ID crashes from conflict copies.
- Define vault disappearance behavior: never silently bootstrap a new vault after a previously configured vault disappears; fail closed with a typed/status error.
- Defer FTS/embeddings from v1; use parameterized `LIKE` with wildcard escaping unless deterministic availability is proven and review-approved.
- Add idempotency for event recording based on an explicit natural key such as source + actor + timestamp + body hash.
- Reindex policy: lightweight per-page freshness checks/lazy projection updates are allowed; full reindex is explicit/status-triggered only and not performed by background contexts.
- Provider rename away from `memory-integration` is a non-blocking naming concern to confirm with the user, not an implementation assumption.
- Workstream 2B is a deliberately small read-only provider skeleton: discovery, config parsing, adapter-backed vault root reporting, sidecar path reporting, and `memory_integration_status` only. It must not create SQLite files, create directories, write Markdown, create `_system/OWNER`, expose write/propose/search tools, or introduce Hashline/runtime npm dependencies.
- `obsidian-vault-adapter` Workstream 1 is complete at commit `bd51129`; the public API is `resolve_vault_path(...)`. The memory provider should depend on that root-resolution contract and must not vendor/copy/shell out to alternative locator implementations in v1.
- Hashline / `@angdrew/opencode-hashline-plugin` is reference/design inspiration only for future native Python patch-reference semantics. Do not install it and do not make it a Hermes runtime dependency.

## Evidence Log

**Requirements/specs read:**
- Current existing plan: `.hermes/plans/2026-05-17_120125-memory-integration-provider.md`.
- Workstream 2A plumbing reconnaissance and Claude Code Opus maintainability/design-debt review on 2026-05-18.
- Hermes skill docs for memory provider lifecycle and contribution constraints.
- `llm-wiki` skill for Karpathy-style Markdown KB shape: `SCHEMA.md`, `index.md`, `log.md`, `raw/`, entity/concept pages, provenance, linting.
- `obsidian` skill and Obsidian CLI reference, then user correction: v1 should *not* focus on Obsidian CLI specifically; it should account for Obsidian Headless sync/draw-vault workflows and the existing path-resolution adapter.
- Existing adapter repo docs:
  - `/home/a01/repos/obsidian-vault-adapter/docs/superpowers/specs/12-05-2026-obsidian-vault-adapter-design.md`
  - `/home/a01/repos/obsidian-vault-adapter/docs/superpowers/plans/12-05-2026-obsidian-vault-adapter-impl.md`

**Code/tests inspected:**
- Hermes repo status on 2026-05-18: `main...origin/main`, untracked `.hermes/` only.
- `plugins/memory/__init__.py` provider discovery/loading conventions.
- `agent/memory_provider.py` provider lifecycle and kwargs.
- Existing memory provider/test conventions under `plugins/memory/*` and `tests/plugins/memory/*`.
- `obsidian-vault-adapter` docs, file layout, and API at commit `bd51129`.

**Commands run and results:**
- `git status --short --branch` in Hermes repo on 2026-05-18: `## main...origin/main`, `?? .hermes/`.
- `search_files` found adapter repo at `/home/a01/repos/obsidian-vault-adapter`.
- Read adapter spec/API sections showing:
  - commit `bd51129` is current in `/home/a01/repos/obsidian-vault-adapter`.
  - public API: `resolve_vault_path(explicit_path=None, *, use_env=True, env_var="OBSIDIAN_VAULT_PATH", config_namespace="vault-adapter") -> Path`.
  - `OBSIDIAN_VAULT_PATH` env var authoritative if set.
  - fallback config file at `$XDG_CONFIG_HOME/vault-adapter/location` or `~/.config/vault-adapter/location`.
  - fail-loud errors for not configured / invalid path / structure problems.
  - `resolve_vault_path()` has no process-global cache; `get_vault_path()` does cache and should not be used for provider tests.
  - no import-time side effects.
  - adapter validation is structural for llm-wiki layout only; memory-integration schema/content validation is separate.
- Claude Code Opus review returned `GREEN_AFTER_PATCH`; blockers were W2B scope ambiguity, stale adapter-dependency ambiguity, and explicit `vault.mode` vs auto-detection contradiction.

**Existing conventions discovered:**
- Hermes bundled memory providers live under `plugins/memory/<provider-name>/`.
- Memory provider tool schemas must be available before `initialize()`.
- Hermes paths must use `get_hermes_home()` or `hermes_home` kwarg, not hardcoded `~/.hermes`.
- Current repo planning convention uses `.hermes/plans/`.

**Assumptions:**
- `obsidian-vault-adapter` commit `bd51129` is the root-resolution contract for v1. Hermes integration may need dependency packaging/import-path plumbing, but not locator redesign.
- Obsidian Headless is for sync/transport and vault availability, not for canonical mutation semantics.
- Workstream 2B is status-only/read-only; SQLite creation, vault bootstrap, `_system/OWNER`, patch queues, semantic writes, and search/retrieval are later slices.

**Open questions / unresolved risks:**
- Confirm the exact installed Python import name for the adapter package in Hermes packaging (`vault_adapter`) and where the dependency should be declared. If dependency wiring is not yet available, stop rather than vendoring or shelling out.
- Confirm whether provider activation should fail hard when the adapter import is missing, or provider discovery should remain available while `memory_integration_status` reports `adapter_unavailable`. For maintainability, prefer a single thin import boundary and explicit typed diagnostic rather than scattered try/excepts.
- Confirm exact real-world Obsidian Sync conflict-file naming before pinning conflict detection beyond the provisional `*.conflict-*.md` pattern.
- Confirm whether Hermes already passes `agent_context` into memory provider tool-call handling before implementing write suppression beyond status-only W2B.

**Out of Scope for v1:**
- No Obsidian desktop CLI dependency as the primary interface.
- No requirement that Obsidian app is open.
- No Dataview/Bases/plugin dependency.
- No semantic contradiction sweeps across the entire vault.
- No embeddings/vector service requirement.
- No remote sync setup automation unless explicitly added later.
- No full multi-agent concurrent editing guarantee beyond lock/conflict detection.
- No migration of all built-in `MEMORY.md` / `USER.md` contents into the vault.

**Known Caveats / Risks:**
- Markdown is not transactional; multi-file vault writes need careful conflict detection and recovery.
- Sync systems can create conflict files or mutate files externally.
- Vault and SQLite can drift unless reindex/rebuild is a first-class operation.
- A bundled Hermes provider should avoid adding fragile package dependencies unless optional and test-gated.
- Headless sync status can be useful but must not become a hidden precondition for memory correctness.

---

## Final Decisions So Far

### Decision 0: Memory-integration vault namespacing only

For v1, define only `memory-integration` behavior. Cross-project standardization for `ai-feed-wiki`, `fin-adv`, and other llm-wiki writers is valuable but out of scope for this provider implementation and should be handled as follow-up documentation once this provider's boundaries are proven.

Memory-integration v1 supports two explicit modes:

- `dedicated`: the resolved vault root is the memory root.
- `shared`: the resolved vault root is a shared llm-wiki/Obsidian vault and `memory_integration.vault.memory_subdir` identifies the provider-owned memory root, defaulting to `wiki/memory-integration`.

No mode is inferred from existing directories. Generated files must stay under the computed memory root and later writers must enforce path containment before writes.

### Decision 1: Hybrid canonical model

Use a hybrid model:

- **Markdown vault is canonical for approved semantic memory.**
- **SQLite is canonical for operational state and indexes.**

Do not use SQLite as the only durable semantic store. Do not use Markdown as the only operational store.

### Decision 2: Obsidian Headless, not Obsidian desktop CLI, is the relevant v1 integration

The prior “include Obsidian CLI from v1” framing was corrected.

V1 should support the existence of an Obsidian Headless workflow for syncing/drawing vaults, but the memory provider should remain filesystem/SQLite correct even if Obsidian Headless is stopped or absent.

Practical v1 interpretation:
- detect/report sync adapter status if configured;
- never require sync for local writes/tests;
- avoid Obsidian desktop active-file/active-vault features in the core provider;
- keep future sync hooks behind a small interface.

### Decision 3: Existing vault adapter belongs in v1 as vault-location primitive

Use the existing `obsidian-vault-adapter` design as a first-class input:

- fail loudly when a configured vault path is invalid;
- no silent fallback from explicit env var to config file;
- no import-time I/O;
- lazy resolution with process-lifetime caching;
- structural validation separate from schema/content lint;
- machine-level config fallback to solve cron/systemd/container env-var drift.

This should be integrated as a `VaultLocator`, not confused with the full semantic memory adapter.

### Decision 4: Patch-first semantic writes

Semantic pages (`entities/`, `projects/`, `decisions/`) are changed only through patch proposal + decision by default.

`record_event` may directly create immutable event/source artifacts when explicitly invoked and allowed by context, because events are append-only/raw provenance, not canonical semantic conclusions.

### Decision 5: Built-in memory preservation

`MEMORY.md` and `USER.md` remain untouched. V1 does not mirror raw built-in memory into the vault. If future import/sync is desired, it must be explicit and provenance-tagged.

### Decision 6: Background write suppression

Write paths are disabled by default when `agent_context in ("cron", "flush", "subagent")`.

Allowed in those contexts:
- search/read;
- lint/readiness checks;
- propose patches only if proposal creation is explicitly deemed non-semantic and safe;
- never apply semantic patches by default.

### Decision 7: Reindexability is mandatory

If Markdown is canonical, SQLite must be rebuildable from vault pages for all projected state.

Non-rebuildable SQLite state should be explicitly classified as operational/ephemeral or exported into auditable patch/event logs where needed.

---

## Proposed V1 Vault Layout

Use a memory-specific sub-tree that remains Obsidian-compatible and can coexist with existing `llm-wiki` layouts.

Default option:

```text
<resolved-vault-root>/
  SCHEMA.md
  index.md
  log.md
  entities/
  projects/
  decisions/
  events/
    YYYY/
      MM/
  raw/
  _system/
    README.md
```

Compatibility option if using the existing adapter’s expected top-level layout:

```text
<resolved-vault-root>/
  raw/
  daily/
  posted/
  wiki/
    memory-integration/
      SCHEMA.md
      index.md
      log.md
      entities/
      projects/
      decisions/
      events/
      raw/
      _system/
```

**Implementation recommendation:** do not auto-detect layout. `vault.mode` is required. In `dedicated` mode the memory root is the resolved vault root and `memory_subdir` is ignored/rejected. In `shared` mode the memory root is `<resolved-vault-root>/<memory_subdir>`, with `memory_subdir` defaulting to `wiki/memory-integration` only because the user selected shared mode, not because directories were detected.

---

## Proposed SQLite Sidecar Location

Default:

```text
$HERMES_HOME/memory-integration/memory_integration.db
```

Rationale:
- avoids Obsidian Sync/git conflict churn;
- keeps operational state private/local by default;
- preserves profile isolation through `get_hermes_home()`.

For v1, do not support an inside-vault SQLite option. A future explicit opt-in may place operational exports under `<memory-root>/_system/`, but the SQLite DB itself remains outside the sync folder unless a later plan and review approve the additional conflict surface.

Workstream 2B computes and reports this path only. It must not create the directory or database.

---

## Provider Config Shape

Add provider-specific config under a stable key. Exact config plumbing should follow existing Hermes config conventions discovered during implementation.

```yaml
memory:
  provider: memory-integration

memory_integration:
  vault:
    mode: shared            # required: shared | dedicated; no auto-detect in v1
    path: null              # optional explicit absolute llm-wiki/memory vault root
    memory_subdir: wiki/memory-integration  # used only in shared mode
    require_existing: true  # fail closed unless explicit bootstrap path is approved
  writes:
    require_approval_for_semantic_pages: true
    allow_event_capture_in_interactive: true
    allow_background_apply: false
  status:
    include_absolute_paths: true  # may be set false in privacy-sensitive contexts
```

YAGNI note: implement only the config fields needed for v1 behavior; document the rest as future extension if plumbing all of this is too much.

---

## Markdown Schemas

### Entity / project / decision page frontmatter

```yaml
---
schema_version: 1
id: entity:example
kind: entity # entity | project | decision
title: Example
status: current
confidence: medium
created_at: 2026-05-17T00:00:00Z
updated_at: 2026-05-17T00:00:00Z
sources:
  - event:2026-05-17-example
aliases:
  - Example Alias
contradictions: []
---
```

### Required sections

```markdown
# Example

## Current summary

## Known facts

## Open questions

## Provenance

## Change log
```

### Event page frontmatter

```yaml
---
schema_version: 1
id: event:2026-05-17-example
kind: event
created_at: 2026-05-17T00:00:00Z
source: hermes-session
session_id: optional
actor: user | assistant | tool | system
sensitivity: normal | private | secret-adjacent
sha256: body-hash
---
```

Events are immutable. Corrections create follow-up events or semantic page patches.

---

## SQLite Tables

V1 minimum:

- `schema_migrations`
- `page_registry`
  - `page_id`, `kind`, `path`, `sha256`, `schema_version`, `title`, `status`, `updated_at`
- `events`
  - event metadata and path/hash; body stays in Markdown event file unless bounded metadata only
- `patches`
  - pending/approved/rejected patches, expected page hashes, rationale, risk flags, metadata JSON
- `memory_references`
  - aliases, unresolved references, target page IDs, confidence, approval status
- `provenance_refs`
  - links between pages/patches/events/sessions/tool calls
- `locks`
  - write leases / conflict protection

Keep previous review correction: table name must be `memory_references`, not `references`. Defer FTS/embeddings/search-index tables from v1; use parameterized `LIKE` with wildcard escaping only when search is implemented in a later slice.

Workstream 2B introduces no tables and creates no SQLite file. SQLite sidecar creation/migrations begin in a later slice with their own TDD and review gates.

---

## Tool Surface

Keep v1 small:

1. Workstream 2B: `memory_integration_status` only.
2. Later slice: `memory_integration_record_event`.
3. Later slice: `memory_integration_propose_patch`.
4. Later slice: `memory_integration_decide_patch`.
5. Later slice: `memory_integration_resolve_reference`.
6. Later slice: `memory_integration_search`.
7. Later slice: `memory_integration_lint` only if structural lint needs a separate surface after status is proven insufficient.

Include `memory_integration_status` first as a bounded read-only JSON tool because it reduces ambiguity around vault/SQLite readiness. Defer all write/propose/search tools until their own reviewed slices.

Bounded status output shape:

```json
{
  "ok": true,
  "vault": {
    "configured": true,
    "mode": "shared",
    "memory_root": "/abs/path/or/redacted",
    "writable": true,
    "structural_errors": []
  },
  "sqlite": {
    "path": "/abs/path/or/redacted",
    "schema_version": 1,
    "size_bytes": 12345,
    "last_reindex_at": "2026-05-17T12:34:56+08:00"
  },
  "counts": {
    "entities": 0,
    "projects": 0,
    "decisions": 0,
    "events": 0,
    "pending_patches": 0,
    "unresolved_references": 0,
    "conflict_files": 0,
    "stale_locks": 0
  },
  "context": {
    "agent_context": "interactive",
    "semantic_writes_allowed": true
  },
  "sync": {"configured": false}
}
```

Status output must not include arbitrary vault page content, unbounded lists, or tracebacks. Detailed diagnostics go to logs.

Status read-only constraints for Workstream 2B:
- Must not create directories.
- Must not create or migrate SQLite.
- Must not create `_system/OWNER` or bootstrap vault files.
- Must not probe writability by test-writing into the vault or Hermes home.
- May stat existing files and read existing config.
- May open an existing SQLite DB read-only in a later slice; in W2B, absent SQLite should be reported as `not_initialized` with the intended path.
- Must work before or without `initialize()` so tool schemas and diagnostics remain available during provider setup.

All tools:
- OpenAI-compatible schema;
- JSON-string return;
- bounded inputs;
- no shell execution of external content;
- provenance for state changes;
- fail closed on ambiguous canonical mutation.

---

## Vault Locator Design

Use `obsidian-vault-adapter` as the shared vault-locator library. Workstream 1 is complete at adapter commit `bd51129`; the public contract for this provider is root resolution via `resolve_vault_path(...)`. Do not vendor/copy adapter code, shell out to adapter CLIs, or keep a second internal locator implementation in v1.

Public API to use:

```python
from vault_adapter import resolve_vault_path

root = resolve_vault_path(
    explicit_path=None,
    use_env=True,
    env_var="OBSIDIAN_VAULT_PATH",
    config_namespace="vault-adapter",
)
```

Implementation should keep the adapter import at one thin boundary so packaging/import errors produce one typed diagnostic, not scattered fallback behavior.

Resolution order should preserve existing adapter semantics:

1. Explicit provider config path, if set.
2. `OBSIDIAN_VAULT_PATH`, if enabled and non-empty.
3. Config-file fallback matching adapter convention:
   - `$XDG_CONFIG_HOME/vault-adapter/location`
   - `~/.config/vault-adapter/location`
4. Not configured error.

Important semantics:

- If an explicit source is set and invalid, fail loudly. Do not silently fall through to weaker sources.
- The locator resolves a root only; it does not decide whether the root is dedicated or shared. `vault.mode` does that explicitly.
- Structural validation is memory-integration-specific unless the adapter grows an explicitly parameterized validation API.
- If adapter import/package wiring is unavailable in Hermes, stop and fix dependency packaging or report `adapter_unavailable`; do not implement a temporary internal locator.

---

## Obsidian Headless Integration Boundary

Obsidian Headless sync management is deferred from v1.

V1 behavior:

- provider correctness is filesystem/SQLite based and does not depend on a running Obsidian app, Obsidian desktop CLI, or Obsidian Headless process;
- `memory_integration_status` may include `sync: {"configured": false}` or a similarly bounded placeholder;
- Hermes does not log into Obsidian Sync, create remote vaults, start/stop `ob sync --continuous`, or manage a systemd sync service in v1;
- sync conflict artifacts are handled defensively by reindex/status, not by invoking sync tooling.

---


## Ownership, Locking, and Patch-Reference Design Boundaries

### `_system/OWNER` format

`_system/OWNER` belongs to the computed memory root, not necessarily the vault root. It is introduced in a later bootstrap/readiness slice, not Workstream 2B. Proposed YAML format:

```yaml
provider: memory-integration
version: 1
created_at: 2026-05-18T00:00:00Z
hermes_home_fingerprint: sha256:<hash-of-canonical-hermes-home-path>
memory_root: <relative-or-redacted-path>
```

Takeover rule: refuse to write if an existing OWNER has `provider` other than `memory-integration`; report typed status and require explicit manual migration.

### Single-writer lock semantics

Markdown mutation slices must define a concrete lock before writes are implemented: lockfile path under `$HERMES_HOME/memory-integration/`, `fcntl.flock` or equivalent implementation, timeout, PID/host metadata, stale-lock detection, and crash cleanup. Workstream 2B does not create or acquire locks.

### Hashline-inspired patch references

Hashline / `@angdrew/opencode-hashline-plugin` is design inspiration only. Do not install it, import it, configure OpenCode with it as part of Hermes runtime, or make it a dependency. Later native Python patch proposals should borrow the concept: stable page/block refs, page revision tokens, expected page hash, expected block/ref hash, stale-write rejection, and safe reapply only when a moved block remains uniquely identifiable.

Hash scope must be specified before patch application is implemented: whole file vs body-only, frontmatter normalization, line ending normalization, and whether metadata-only changes invalidate pending patches.

---
## Patch Application Semantics

`propose_patch`:
- validates target page ID/path;
- records expected current hash;
- stores patch diff or full replacement proposal in SQLite;
- includes rationale, provenance, sensitivity, risk flags;
- does not mutate semantic pages.

`decide_patch approve`:
- rejects if context is `cron`, `flush`, or `subagent` unless explicit override exists;
- verifies `_system/OWNER` matches `provider=memory-integration`;
- acquires the provider single-writer file lock under `$HERMES_HOME/memory-integration/`;
- verifies expected hash still matches current file after lock acquisition;
- acquires SQLite lock;
- writes via temp file + rename where practical;
- updates `page_registry` hash/projection;
- appends `log.md` entry;
- records provenance;
- releases lock.

If hash mismatch:
- do not overwrite;
- mark patch `needs_rebase`;
- return conflict details.

`resolve_reference`:
- by default updates `memory_references` only;
- does not mutate Markdown aliases/frontmatter unless explicitly proposed/approved as a semantic patch.

---

## Context Building / Retrieval

The model should never receive raw vault dumps.

Retrieval flow:
1. Normalize query/current turn.
2. Use SQLite projection/search/reference mappings to select candidate page IDs/events.
3. Load bounded Markdown pages/snippets for current canonical semantics.
4. Include unresolved references/pending patches only when relevant.
5. Treat raw/event/source text as untrusted data, not instructions.

Bounded output sections:
- active project state;
- relevant entities/decisions;
- recent high-signal events;
- unresolved references;
- pending patches requiring user decision.

---

## Tests / Validation Strategy

### Unit tests — provider lifecycle
- provider discoverability under `plugins/memory/memory-integration/`;
- `name == "memory-integration"`;
- `get_tool_schemas()` works before `initialize()`;
- Workstream 2B status path does not create SQLite sidecar, directories, OWNER, or bootstrap files; later slices add explicit creation tests;
- no hardcoded `~/.hermes`.

### Unit tests — vault locator
- explicit `vault.mode` required; no content-based auto-detection;
- explicit config path success/failure;
- `OBSIDIAN_VAULT_PATH` success/failure;
- config-file fallback success/failure;
- invalid explicit source does not silently fall through;
- no import-time I/O;
- cache semantics where applicable;
- missing adapter package does not break non-adapter modes.

### Unit tests — vault schemas
- bootstrap `SCHEMA.md`, `index.md`, `log.md`;
- valid entity/project/decision/event pages parse;
- invalid frontmatter rejected;
- duplicate IDs detected;
- broken wikilinks reported;
- stale index reported.

### Unit tests — patch workflow
- propose entity create patch;
- reject writes when `_system/OWNER` belongs to another provider;
- reject/serialize concurrent writers through provider file lock;
- approve patch writes Markdown + updates SQLite registry;
- reject patch remains auditable;
- external file edit between proposal and approval causes conflict;
- `resolve_reference` updates only SQLite mapping unless semantic alias update is separately approved;
- background contexts cannot apply semantic writes.

### Unit tests — reindex
- rebuild SQLite projection from fixture vault;
- ignore and count `*.conflict-*.md` files without duplicate-ID crashes;
- stale/missing sidecar recovers from Markdown;
- non-rebuildable operational state classified correctly.

### Unit tests — search/context
- search returns current page projection, not stale event fragments;
- SQL uses parameterized queries and escapes LIKE wildcards;
- bounded results include provenance IDs/paths/hashes.

### Optional integration tests — headless/sync
- no required adapter package/CLI tests in v1;
- optional future headless sync smoke tests remain skipped/deferred;
- v1 status should report sync as unsupported/not configured without failing provider readiness.

---

## Workstream 2B — Minimal Read-Only Provider Skeleton

**Goal:** land the first maintainable vertical slice without creating semantic or operational state. This slice answers “is the provider installed/configured, what vault root would it use, and where would its sidecar live?”

**Allowed files:**
- `plugins/memory/memory-integration/__init__.py`
- `plugins/memory/memory-integration/plugin.yaml`
- `plugins/memory/memory-integration/provider.py`
- `plugins/memory/memory-integration/config.py`
- `plugins/memory/memory-integration/status.py`
- `plugins/memory/memory-integration/README.md`
- `tests/plugins/memory/memory_integration/__init__.py`
- `tests/plugins/memory/memory_integration/conftest.py`
- `tests/plugins/memory/memory_integration/test_discovery.py`
- `tests/plugins/memory/memory_integration/test_config.py`
- `tests/plugins/memory/memory_integration/test_status_readonly.py`

If implementation proves a loader/test helper outside this allowlist is required, stop and update the plan before proceeding.

**Forbidden in Workstream 2B:**
- no SQLite file creation or migrations;
- no directory creation under `$HERMES_HOME/memory-integration/` from status or initialization;
- no Markdown writes or vault bootstrap files;
- no `_system/OWNER` creation;
- no locks;
- no tools beyond `memory_integration_status`;
- no semantic write/propose/apply/search/retrieval behavior;
- no mutation of `MEMORY.md` / `USER.md`;
- no Hashline / npm / OpenCode plugin dependency or installation;
- no Obsidian desktop, Obsidian Headless, network, or sync-process dependency.

**Implementation notes:**
- Provider directory should be `plugins/memory/memory-integration/` so `memory.provider: memory-integration` maps to the existing loader. Tests should load via `plugins.memory.load_memory_provider("memory-integration")` rather than normal imports from a hyphenated package.
- `get_tool_schemas()` must work before `initialize()`.
- `initialize()` may store session/context/hermes_home values in memory, but must not create persistent files in W2B.
- Status should return bounded JSON and typed diagnostics for `mode_required`, `adapter_unavailable`, `vault_not_configured`, `vault_path_invalid`, `not_initialized`, and similar conditions.
- Sidecar path default is reported as `$HERMES_HOME/memory-integration/memory_integration.db`, but absent DB is not an error in W2B.

**Verification gates:**
1. Targeted tests pass: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/plugins/memory/memory_integration/ -q`.
2. Discovery works through the Hermes memory plugin loader.
3. `get_tool_schemas()` returns exactly the status tool and works before `initialize()`.
4. Status returns valid bounded JSON for missing config, invalid configured vault, valid configured vault, and absent SQLite sidecar.
5. Tests snapshot temp `$HERMES_HOME` and fixture vaults before/after status calls to prove no files/directories were created.
6. New files contain no hardcoded `~/.hermes`; use `hermes_home` kwarg or `get_hermes_home()`.
7. New files contain no `hashline`, `opencode-hashline`, npm, Obsidian desktop, or Obsidian Headless runtime dependency.

---

## Implementation Tasks

### Task 0: Preflight and final scope confirmation

**Objective:** Confirm final v1 scope, especially adapter/headless boundaries, before implementation.

**Files:**
- Read: this plan
- Read: existing memory provider/plugin code
- Read: `obsidian-vault-adapter` package API if using it directly

**Steps:**
1. Confirm the adapter import path and packaging location for `vault_adapter.resolve_vault_path` in Hermes.
2. Confirm `vault.mode` behavior: required, no auto-detection; `memory_subdir` only applies in shared mode.
3. Confirm `memory_integration_status` is the only Workstream 2B tool.
4. Confirm Workstream 2B is read-only and creates no persistent files/directories.
5. Confirm no implementation starts until this patched plan is green.

**Verification:** final user-approved scope statement.

### Task 1: Inspect existing Hermes memory provider interfaces

**Objective:** Ground implementation in existing provider lifecycle and tests.

**Files:**
- Read existing `agent/memory_provider`-related files.
- Read `plugins/memory/` loader and existing provider tests.

**Deliverable:** short implementation note or plan patch with exact interfaces/imports.

### Task 2: Add provider scaffold

**Objective:** Add discoverable bundled `memory-integration` plugin with static tool schemas.

**Files:**
- Create: `plugins/memory/memory-integration/__init__.py`
- Create: `plugins/memory/memory-integration/plugin.yaml`
- Create: `plugins/memory/memory-integration/README.md`
- Create/modify tests under `tests/plugins/memory/`

**TDD:** first add failing tests for discovery, name, availability, static schemas.

### Task 3: SQLite sidecar schema and migrations — later slice, not W2B

**Objective:** Implement operational DB creation and migration baseline.

**Files:** same provider/test files.

**TDD:** tests for tables, migrations idempotency, temp Hermes home pathing.

### Task 4: Vault locator and bootstrap — later slice, not W2B bootstrap

**Objective:** Resolve vault root/subdir using explicit config/env/adapter-backed fallback and bootstrap minimal vault files only when allowed.

**TDD:** fixture directories for all resolution cases; invalid explicit source fail-loud; no import-time I/O.

### Task 5: Markdown schema parser/validator

**Objective:** Parse and validate v1 frontmatter/body shape for pages/events.

**TDD:** valid/invalid fixture pages, duplicate IDs, schema version handling.

### Task 6: Reindex from vault to SQLite

**Objective:** Rebuild `page_registry`, event metadata, and search projection from Markdown.

**TDD:** stale sidecar rebuilt from fixture vault.

### Task 7: Event recording

**Objective:** Implement immutable event capture with context write guards.

**TDD:** record event writes event Markdown + SQLite metadata; blocked in disallowed contexts unless explicitly permitted.

### Task 8: Patch proposal/decision

**Objective:** Implement semantic patch queue and hash-checked application.

**TDD:** propose/approve/reject/conflict/background-write tests.

### Task 9: Reference resolution

**Objective:** Implement `memory_references` mapping and resolution semantics.

**TDD:** approval updates mapping only; alias/frontmatter mutation requires separate patch.

### Task 10: Search/context retrieval

**Objective:** Implement bounded SQLite-backed search that returns current semantic pages with provenance.

**TDD:** LIKE escaping, current page join/projection, bounded results, raw content treated as data.

### Task 11: Expanded status/lint/readiness — later slice after W2B status skeleton

**Objective:** Add read-only provider status/lint for vault, SQLite, adapter, and optional headless sync.

**TDD:** status reports missing adapter/headless as degraded/optional, not fatal; structural lint reports actionable issues.

### Task 12: Docs and examples

**Objective:** Document configuration, vault layout, adapter/headless role, privacy, and failure modes.

**Files:** provider README and possibly docs site if appropriate.

### Task 13: Adversarial review and final verification

**Objective:** Run read-only reviews before merging/committing implementation.

**Review focus:** lifecycle, safety/privacy, transactionality, sync/conflict handling, test adequacy, YAGNI.

---

## Review Gates Before Implementation

Before coding, run at least one more targeted read-only critique on this total plan with these questions:

1. Is the adapter/headless distinction clear enough to implement without overbuilding?
2. Is the default vault layout compatible with the existing `obsidian-vault-adapter` ecosystem?
3. Are we accidentally creating two canonical sources of truth between Markdown and SQLite?
4. Is the patch workflow too large for v1?
5. Is `memory_integration_status` worth adding in v1 or should it be deferred?
6. What tests would catch the most damaging data-loss/silent-drift bugs?

Patch this plan after critique before implementation. The 2026-05-18 Opus review returned `GREEN_AFTER_PATCH`; this revision addresses its blockers/high findings by adding explicit W2B scope, removing adapter ambiguity, removing layout auto-detection, and making status read-only.

---

## Follow-up Filing

Deferred items to file after v1 scope confirmation:

1. **Title:** Add managed Obsidian Headless sync support for memory-integration
   **Body:** Explore optional management of `ob sync --continuous`, service status, and setup docs. Must remain optional; memory correctness cannot depend on sync process availability.

2. **Title:** Add semantic contradiction lint for memory vault
   **Body:** Build LLM-assisted contradiction/staleness review over Markdown semantic pages after structural lint and reindex are stable.

3. **Title:** Add embeddings-backed retrieval cache for memory-integration
   **Body:** Add optional local/cloud embeddings only after deterministic SQLite/Markdown search is proven insufficient.

4. **Title:** Import selected built-in MEMORY/USER facts into vault through explicit patch workflow
   **Body:** Design explicit, provenance-rich import of selected existing memories. Do not mirror raw built-in memory automatically.

---

## Execution Handoff

Plan complete. Do not implement yet.

After user greenlight:
1. Use this patched plan and verify the W2B scope/allowlist.
2. Execute Workstream 2B using `subagent-driven-development`:
   - one focused implementer per task;
   - spec compliance review;
   - code quality review;
   - controller-side verification;
   - meaningful checkpoints only if commits are explicitly authorized.
