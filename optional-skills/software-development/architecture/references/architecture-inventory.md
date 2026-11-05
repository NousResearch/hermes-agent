# `docs/architecture/` Inventory Contract

Canonical layout and naming for every artifact the `architecture` skill owns. Run `scripts/architecture-doctor.sh` for a live inventory before creating or updating files.

## Canonical layout

```text
docs/architecture/
├── README.md                            # Optional index of ADRs + diagrams
├── AD-NNNN-<slug>.md                    # Architecture Decision Records
├── architecture-design.md               # From mermaid-architecture/assets/architecture-design-template.md
├── system-design.md                     # From assets/system-design-template.md
├── api-design.md                        # From assets/api-design-template.md
├── database-design.md                   # From assets/database-design-template.md
├── feature-<name>-design.md             # From assets/feature-design-template.md
├── system-architecture.md               # Markdown doc with embedded Mermaid blocks
├── <name>_<type>_<title>.mmd            # Mermaid source (mermaid-architecture)
├── <name>.drawio                        # Editable draw.io source (drawio-architecture)
├── <name>.png | .svg | .pdf             # Exported renders (mmdc or drawio CLI)
├── <name>.json                          # Archify candidate spec
└── <name>.html                          # Archify interactive artifact (deliver output)
```

## Naming rules

| Artifact | Pattern | Notes |
| --- | --- | --- |
| ADR | `AD-NNNN-<kebab-slug>.md` | Sequential, zero-padded (`AD-0001`, `AD-0002`…). Never renumber — supersede with a new AD. |
| Mermaid source | `<name>_<type>_<title>.mmd` | `type` ∈ `flow`, `seq`, `er`, `class`, `state`, `arch` |
| draw.io source | `<name>.drawio` | Keep `type="device"` so the file opens from disk |
| Exports | same basename as source | `.drawio.png` keeps the PNG re-importable into draw.io |
| Archify spec | `<name>.json` | Frozen at `deliver` time — never edit after a passing validation |
| Archify artifact | `<name>.html` | Standalone output of `archify.mjs deliver` |

## ADR lifecycle

1. `AD-NNNN` numbers come from the highest existing record + 1 (`architecture-doctor.sh` prints `next:`).
2. Status lives inside the record body (`Accepted`, `Superseded by AD-XXXX`) — not in the filename.
3. Superseding writes a **new** AD that links back; the old file keeps its number.
4. When `docs/architecture/README.md` exists, keep a chronological AD index there (number, title, status, date).
5. `docs/adr/` (created by `scaffold-mvp` in some projects) may hold future ADRs — if it exists, keep both locations consistent and prefer `docs/architecture/` as canonical.

## What does NOT belong here

- SPEC SDDs → `.specs/` (active) or `docs/specs/` (archived after delivery)
- Session/agent memory → `.claude/memory/`
- Gap-analysis reports → `.claude/memory/gap-analysis-*.md`
- Code-level API docs → generated docs (docfx, TypeDoc, etc.)
- Secrets, `.env`, credentials — never

## Design doc templates

Templates ship inside `mermaid-architecture/assets/` (they are not duplicated here — forward-only dependency). Populate and save to `docs/architecture/` per the table above.
