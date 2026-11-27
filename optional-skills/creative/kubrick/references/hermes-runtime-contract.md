# Hermes Runtime Contract

## Purpose

Kubrick is distributed and executed as a self-contained Hermes skill directory. This contract defines what the skill may assume, read, write, invoke, and hand off.

## Runtime assumptions

Kubrick may assume:

- Hermes can load `SKILL.md` and relative references from the installed skill root.
- A Python 3 interpreter may be available for deterministic helpers.
- The skill directory may be read-only after installation.

The Bash installer must validate a staged copy before activation, keep backups outside `skills/`, atomically swap the validated directory, restore the prior installation if activation fails, and write an install receipt. Dry-run performs no writes; rollback preserves the displaced installation as a backup.

Kubrick must not assume:

- the Git repository is present,
- an editable Python package is installed,
- Continuity Forge is installed,
- an MCP server is connected,
- a particular current working directory,
- network access,
- write access inside the installed skill directory.

## Path resolution

Every bundled script resolves resources from its own file location:

```text
script directory → skill root → references / schemas / evals
```

User inputs and outputs must accept explicit paths. Runtime artifacts should default to `./out/kubrick/` relative to the invoking project, not inside the skill corpus.

## Dependency tiers

The canonical manifest names four runtime profiles: `prose`, `stdlib`, `validation`, and `dev`. The tiers below define their guarantees. Kubrick remains a Hermes skill directory in every profile and is never required to install as a Python package.

### Tier 0 — Hermes prose runtime

No external dependencies. All creative routing, diagnosis, drafting, revision, and symbolic translation must remain possible through `SKILL.md` and bundled references.

### Tier 1 — Standard-library helpers

Install validation, graph structural checks, receipt hashing, cache keys, gap reporting, and basic JSON operations should use Python's standard library wherever practical.

### Tier 2 — Optional local dependencies

Examples: PyYAML or jsonschema. A helper that needs one must:

1. detect absence explicitly,
2. print the exact missing package,
3. preserve a useful degraded path where feasible,
4. never imply that the entire Hermes skill is unavailable.

### Tier 3 — Optional companion systems

Continuity Forge, MCP servers, model APIs, and rendering providers are optional extensions. Their absence cannot block local creative work.

### Development profile

The `dev` profile adds test and static-analysis tools. It exists for repository maintenance only and is not an installed-skill runtime requirement.

## Operator surface (intent router)

The unified CLI (`scripts/kubrick.py`) is the authoritative local operator surface.

**Primary form (prefer for Hermes and docs):**

```text
python scripts/kubrick.py do <intent> [--action <action>] [flags]
```

**Intents:** `compile`, `retrieve`, `ledger`, `design`, `storyboard`, `adapt`, `visual`, `learn`, `check`, `operate`, `mcp`, `bundle`

**MCP:** optional stdio server exposes a single tool, `kubrick_do`, over the same router. MCP is never authoritative.

**Legacy aliases** (soft cutover): flat names such as `forge-signals`, `closed-loop-qa`, `adapt-flux`, `validate-skill`, `mcp-server`, `design-build` still resolve through the router.

Registry and resolve logic: `scripts/intent_router.py`.  
Canonical registry: `kubrick.manifest.yaml`. The router loads this file with the Python standard library, so routing does not depend on PyYAML.
Design: `docs/superpowers/specs/2026-07-30-operator-intent-router-design.md`.

See root `README.md`, `QUICKSTART.md`, and `docs/README.md` for workflows and the docs index.

## OpenClaw edition

This contract describes the **Hermes** skill on `main`. An OpenClaw Agent Skill packaging of Kubrick is maintained on the permanent branch `openclaw` (work by Prabu / @prabu-openclaw). That edition uses different install roots, external state directories, and packaging metadata. See `docs/OPENCLAW.md` and https://github.com/scrimshawlife-ctrl/Kubrick/tree/openclaw .

## Artifact classes

| Artifact | Default status | Write location |
|---|---|---|
| creative draft | `PROPOSED` | user project or response |
| symbolic intent | `PROPOSED` | user project or response |
| motif registry | `PROPOSED` | user project or response |
| motif graph | `PROPOSED` | `out/kubrick/` or explicit path |
| retrieval receipt | observational | `out/kubrick/receipts/` or explicit path |
| evolution receipt | maintenance proposal | explicit maintenance output |
| Forge receipt | authoritative according to returned status | project continuity store |

References, schemas, pattern sidecars, and eval fixtures are immutable during ordinary use.

Every compile receipt records deterministic runtime identity fields: `kubrick_version`, `corpus_version`, `corpus_digest`, `schema_bundle_version`, `provider_adapter_version`, `command`, `command_digest`, and `normalized_input_digest`. Digests use relative paths and canonical semantic inputs. They exclude timestamps, absolute paths, file metadata, and environment details.

Every provider packet includes a `preservation_report` conforming to `schemas/provider-preservation-report.schema.json`. Graph identity, required observable content, ownership, geometry, state change, residue, continuity, and negative constraints are critical invariants. Any loss marks the adapter output `INVALID`; provider syntax may change, semantic requirements may not.

## Canon policy

Kubrick may preserve approved facts and generate proposals. It must not promote content to `LOCKED` on its own.

Promotion requires an explicit authoritative receipt from a connected canonical system. When no such system is present, the highest valid status is `PROPOSED`.

## Hidden architecture policy

Esoteric, archetypal, and mythic source terms are private structural metadata by default. Final scripts and prompts should expose:

- material states,
- behavior,
- geometry,
- rhythm,
- blocking,
- framing,
- light,
- sound,
- residue,
- transformation.

They should not expose source labels unless the user explicitly requests analysis or visible occult content.

## Failure policy

Return `NOT_COMPUTABLE` when:

- observed evidence is missing,
- dramatic function is unresolved,
- retrieval confidence remains below threshold,
- cultural boundaries cannot be resolved,
- graph convergence is unsupported,
- requested continuity contradicts locked facts,
- production constraints make the encoding nonviable.

A failure result must include a reason vector and the minimum missing information needed to proceed.

The unified CLI uses stable exit codes from `kubrick.manifest.yaml`: `1` execution or validation failure, `2` invalid command or arguments, `3` unavailable optional dependency with no degraded operation, and `4` domain-level `NOT_COMPUTABLE`. Set `KUBRICK_DIAGNOSTICS=json` for diagnostics conforming to `schemas/diagnostic.schema.json`; the default remains concise human-readable stderr.

## Maintenance policy

Evolution and corpus modification are separate from ordinary runtime.

- Retrieval may write receipts outside the skill directory.
- Outcomes require explicit evidence.
- Confidence changes require receipts.
- Structural changes require human review.
- Deprecation is proposed before removal.

## Optional companion behavior

When `hermes-continuity-forge` is connected, Kubrick may prepare and submit handoff artifacts. The companion owns authorization, leases, ingestion, and canonical receipts.

When it is absent, Kubrick continues locally and labels outputs `PROPOSED`.
