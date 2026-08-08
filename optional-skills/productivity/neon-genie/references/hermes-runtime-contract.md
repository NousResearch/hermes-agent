# Hermes Runtime Contract

## Purpose

Neon Genie is distributed and executed as a self-contained Hermes skill directory. This contract defines what the skill may assume, read, write, invoke, and hand off.

Neon Genie is **not** Kubrick. It does not perform cinematic or motif engineering. Kubrick is an optional sibling skill with a separate mission and must never be treated as a load-time dependency.

## Runtime assumptions

Neon Genie **may** assume:

- Hermes can load `SKILL.md` and relative references from the installed skill root.
- A Python 3 interpreter may be available for deterministic helpers (smoke validation).
- The skill directory may be read-only after installation.
- Host tools for research (web search, page fetch, indexes, docs) may be available — they are optional.

Neon Genie **must not** assume:

- the Git repository is present,
- an editable Python package is installed,
- Kubrick is installed,
- Wayfinder (or any execution runtime) is installed,
- an MCP server is connected,
- a particular current working directory,
- network access,
- write access inside the installed skill directory.

## Path resolution

Every bundled script resolves resources from its own file location:

```text
script directory → skill root → references / schemas / evals / profiles / templates
```

User inputs and outputs must accept explicit paths. Runtime artifacts should default to `./out/neon-genie/` relative to the invoking project, not inside the skill corpus.

## Dependency tiers

### Tier 0 — Hermes prose runtime

No external dependencies. Full OPEN → ALIGN → ASCEND → CLEAR → SEAL advisory work, profile routing, claim labeling, and packet drafting must remain possible through `SKILL.md`, `profiles/`, `schemas/`, and `references/`.

### Tier 1 — Standard-library helpers

Install validation, path checks, version consistency, profile routing suggestions, packet required-field checks, and receipt hashing use Python's standard library only.

```bash
python scripts/neon_genie.py do check
python scripts/neon_genie.py do validate --packet path.json --type opportunity
python scripts/neon_genie.py do route --request templates/request.yaml
python scripts/neon_genie.py do receipt --profiles core --out ./out/neon-genie/receipt.json
# equivalent direct entry:
python scripts/validate_hermes_skill.py
```

These helpers never invent opportunities, never run host research, and never grant execution authority.

### Tier 2 — Optional local dependencies

Helpers that need third-party packages (e.g. jsonschema) must:

1. detect absence explicitly,
2. print the exact missing package,
3. preserve a useful degraded path where feasible,
4. never imply that the entire Hermes skill is unavailable.

Wave 1 does not require Tier 2 packages.

### Tier 3 — Optional companion systems

Host research tools and Wayfinder are optional extensions. Their absence cannot block local advisory work.

| Companion | Role | Required? |
|-----------|------|-----------|
| Host research tools | Close decision-critical evidence gaps | No |
| Wayfinder | Execution handoff consumer | No |
| Kubrick | Sibling creative skill | No |

## Research policy

- Research is disabled by default (`local_only`). Operator may explicitly select `external_research_allowed` for purpose-bound, minimized public research. Briefs may carry `privacy:` configuration (mode, approved domains/tools, purpose-bound consents only — never a global privacy disable). Configuration flows into receipt, envelope, and local learning-ledger disclosure.
- `research.enabled=false` or `offline: true` permits no Neon-Genie-initiated external research action.
- Operator-supplied and workspace sources always outrank live fetches.
- Model prior without fetch is at most `SPECULATIVE`.
- Missing tooling or failed fetch → `NOT_COMPUTABLE` with attempted query — never fabricate `OBSERVED` claims.

### Evidence Request Protocol

When a material fact is missing, follow **find → request → NOT_COMPUTABLE**:

1. **Find** — public (or likely-public) gaps: attempt host research; cite or drop.
2. **Request** — operator/private gaps or undeclared access: emit a `DataRequest` (`schemas/data-request.schema.json`) instead of inventing.
3. **NOT_COMPUTABLE** — only after find was attempted (or correctly skipped offline) and/or a DataRequest is open or unanswered.
4. **Never** mark model prior as `OBSERVED`.

Required DataRequest fields: `field`, `why_decision_critical`, `sensitivity` (`public`|`operator`|`private`), `suggested_source`, `blocks_promotion`, `status` (`open`|`satisfied`|`waived`).

CLEAR fails on Gates P (skip find), Q (skip request), and R (silent private invent). SEAL receipts list `data_requests`, `open_blocking_requests`, and `research_attempts` (may be empty). Full protocol: `SKILL.md` § Evidence Request Protocol; anti-overclaim gates A–R: `references/anti-overclaim-patterns.md`.

## Authority

Neon Genie authority is **advisory_only** (`manifest.json`).

**May:** research (when tools allow), infer, generate, compare, score, model, audit, specify, route, draft, recommend.

**May not** without explicit downstream authorization: spend or transfer money; submit applications; contact targets; publish content; modify repositories; execute irreversible workflows; promote artifacts to canon; represent forecasts as facts; mutate runtime state.

No output packet grants execution authority.

## Artifact policy

- Treat local outputs as drafts / proposals until a human or authorized downstream system accepts them.
- Label material claims: `OBSERVED`, `INFERRED`, `SPECULATIVE`, `NOT_COMPUTABLE`.
- Never modify `references/`, `profiles/`, `schemas/`, or other skill corpus files during ordinary runs.
- Write generated operational artifacts under a user-selected project directory or `./out/neon-genie/`.
- See `PRIVACY.md` and `references/privacy-contract.md`: host/provider retention is `NOT_COMPUTABLE` unless independently established.

## Wayfinder handoff

When a Wayfinder execution packet is requested:

- Neon Genie owns product intent (what/why/user/boundary/success/proof).
- Wayfinder owns decomposition, dependency sequence, milestones, eng validation, implementation status.
- Proposed product-intent changes return to Neon Genie as change requests.
- Absence of Wayfinder never blocks emission of a local handoff packet as advisory text/JSON.

## Validation and release smoke

Minimum integrity check (Wave 1):

```bash
python scripts/validate_hermes_skill.py
```

Minimum pass conditions:

- frontmatter parses and required fields exist,
- `name` is `neon-genie`,
- VERSION matches SKILL frontmatter and `manifest.json`,
- required profiles, schemas, references, evals, examples, and install script exist,
- scripts compile,
- skill docs do not require editable package installs or machine-absolute home paths.

## Final constraint

Neon Genie is a Hermes skill first. External systems may extend research, memory, or execution handoff, but core opportunity and product intelligence must remain portable inside the skill directory without Kubrick, Wayfinder, or network access.
