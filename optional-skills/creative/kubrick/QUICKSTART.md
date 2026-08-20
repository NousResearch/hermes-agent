# Kubrick Quickstart

Kubrick is a **standalone Hermes skill** on branch **`main`** (v0.14.0). Continuity Forge, MCP servers, model APIs, and external generation providers remain optional.

**OpenClaw?** Use the permanent [`openclaw`](https://github.com/scrimshawlife-ctrl/Kubrick/tree/openclaw) branch (OpenClaw Agent Skill packaging by **Prabu**). See [`docs/OPENCLAW.md`](docs/OPENCLAW.md) — install target is `~/.openclaw/skills/kubrick`, not Hermes.

## 1. Install (Hermes / `main`)

```bash
./install.sh
# or: cp -R . ~/.hermes/skills/kubrick
```

The installer validates a staged copy before activation. Use `./install.sh --dry-run` to preview, `./install.sh --version` to inspect the source version, and `./install.sh --rollback` to restore the most recent external backup.

Install runtime validation dependencies when working from the repository:

```bash
python -m pip install pyyaml jsonschema
```

Validate:

```bash
python scripts/kubrick.py do check --action skill
```

### OpenClaw install (other branch)

```bash
openclaw skills install git:scrimshawlife-ctrl/Kubrick@openclaw --global
# or: git clone --branch openclaw --single-branch …
```

Details and credits: [`docs/OPENCLAW.md`](docs/OPENCLAW.md).

## 2. Unified CLI

Primary surface:

```bash
python scripts/kubrick.py do <intent> [--action <action>] [flags]
```

### Intents

```text
compile      Full brief-to-packet compile
retrieve     Registry-aware pattern retrieval
ledger       Project symbolic ledger (init / audit / mutate / …)
design       Design-specification compilation
storyboard   Multi-frame state (propagate / compare)
adapt        Neutral packet and provider adaptation
visual       Visual QA loop (normalize / compare / correct / govern / closed-loop)
learn        Outcomes and multi-signal evolution (outcome / evolve / forge-signals)
check        Validation and regression (skill / corpus / coverage / eval / smoke / …)
operate      Graph/ledger operators (saturation, counterpoint, lock, audit, export)
mcp          Optional stdio MCP server (single tool: kubrick_do)
bundle       Grok review bundle
```

Sugar:

```bash
python scripts/kubrick.py help <intent>
python scripts/kubrick.py recipe <name>    # e.g. storyboard-example, verify
python scripts/kubrick.py aliases
```

### Aliases

Legacy peer command names still work as soft aliases (soft cutover). Prefer `do <intent>`.

Examples:

```bash
# Preferred
python scripts/kubrick.py do check --action skill
python scripts/kubrick.py do visual --action closed-loop --expected … --observation-input … --out …

# Legacy (still supported)
python scripts/kubrick.py validate-skill
python scripts/kubrick.py closed-loop-qa --expected … --observation-input … --out …
```

Full map: `python scripts/kubrick.py aliases`.

## 3. Compile a single frame

```bash
python scripts/kubrick.py do compile \
  --brief project/brief.yaml \
  --ledger project/symbolic-ledger.yaml \
  --mode single-frame \
  --out project/out/kubrick
```

## 4. Compile a storyboard for a provider

```bash
python scripts/kubrick.py do compile \
  --brief project/brief.yaml \
  --ledger project/symbolic-ledger.yaml \
  --mode storyboard \
  --storyboard-plan project/storyboard-plan.yaml \
  --provider grok-imagine \
  --out project/out/kubrick
```

**Providers:** `none` · `generic` · `grok-imagine` · `flux` · `sd3` · `midjourney`

Pipeline:

```text
retrieval
→ private graph
→ schema validation
→ structured symbolic audit
→ audience translation
→ storyboard propagation
→ transition comparison
→ neutral adapter packet
→ provider prompt packet (syntax only)
→ compile receipt
```

### Canonical example

```bash
python scripts/kubrick.py do compile \
  --brief examples/authority-transfer-storyboard/brief.yaml \
  --ledger examples/authority-transfer-storyboard/symbolic-ledger.yaml \
  --mode storyboard \
  --storyboard-plan examples/authority-transfer-storyboard/storyboard-plan.yaml \
  --provider grok-imagine \
  --out out/kubrick/authority-transfer
```

Or: `python scripts/kubrick.py recipe storyboard-example`

A successful storyboard run emits:

```text
brief.normalized.yaml
retrieval-receipt.yaml
graph-input.yaml
motif-graph.private.yaml
structured-symbolic-packet.yaml
structured-anti-slop-report.json
audience-constraints.yaml
text-anti-slop-report.json
storyboard-symbolic-state.yaml
storyboard-transition-report.json
model-adapter-packet.yaml
<provider>-prompt-packet.yaml
schema-graph.json
schema-storyboard.json
schema-adapter.json
compile-receipt.json
```

Weak retrieval, invalid graph state, schema drift, anti-slop failure, storyboard reset, residue loss, or adapter failure returns `NOT_COMPUTABLE` and exits nonzero.

## 5. Project ledger

```bash
python scripts/kubrick.py do ledger init \
  --project-id my-film \
  --out project/symbolic-ledger.yaml

python scripts/kubrick.py do ledger audit \
  --ledger project/symbolic-ledger.yaml

python scripts/kubrick.py do ledger mutate \
  --ledger project/symbolic-ledger.yaml \
  --motif-id cracked-badge \
  --observed-form "a cracked access badge" \
  --state "worn by the former subordinate" \
  --mutation "ownership and access function transferred" \
  --pattern-link interface_badge_authority_transfer

python scripts/kubrick.py do ledger export-retrieval \
  --ledger project/symbolic-ledger.yaml \
  --out project/ledger-retrieval-snapshot.yaml
```

Local ledger state remains `PROPOSED` unless explicitly ingested by Continuity Forge or another authorized canonical system.

## 6. Forge feedback and multi-signal evolution

```bash
python scripts/kubrick.py do learn --action forge-signals \
  --project-id myfilm \
  --input path/to/forge-ledger-diff.yaml \
  --output out/forge-bundle.yaml

# Example fixture:
# references/examples/forge-signals/ledger-before-after.yaml

python scripts/kubrick.py do ledger apply-forge \
  --ledger project/symbolic-ledger.yaml \
  --forge-bundle out/forge-bundle.yaml

python scripts/kubrick.py do learn --action evolve \
  --pattern-id interface_badge_authority_transfer \
  --receipt out/use-receipt.yaml \
  --forge-bundle out/forge-bundle.yaml \
  --output out/evolution-proposal.yaml \
  --receipt-output out/evolution-proposal.multi-signal.yaml
```

Rules:

- Forge signal bundles are `OBSERVATION` only (`forge_canonical: true`).
- Every evolution event emits a deterministic multi-signal receipt.
- Large confidence deltas and structural/lifecycle mutations require human review.
- `automatic_application_allowed` is always `false`.

## 7. Closed-loop visual QA

```bash
python scripts/kubrick.py do visual --action closed-loop \
  --expected out/storyboard-symbolic-state.yaml \
  --observation-input observations/frame-001.json \
  --source-graph-id <graph-id> \
  --frame-id frame-001 \
  --out out/qa/frame-001
```

Produces:

```text
visual-observation.yaml
visual-fidelity-report.json
visual-correction-packet.yaml
closed-loop-qa-receipt.yaml
```

The receipt reports **geometry**, **state**, **residue**, and **convergence** fidelity separately. Graph identity mismatch fails closed (`NOT_COMPUTABLE`).

## 8. Multi-provider adapters

```bash
python scripts/kubrick.py do adapt \
  --graph out/motif-graph.private.yaml \
  --storyboard out/storyboard-symbolic-state.yaml \
  --provider generic \
  --output out/model-adapter-packet.yaml

python scripts/kubrick.py do adapt --action provider \
  --packet out/model-adapter-packet.yaml \
  --provider flux \
  --output out/flux-prompt-packet.yaml

python scripts/kubrick.py do adapt --action provider \
  --packet out/model-adapter-packet.yaml \
  --provider sd3 \
  --output out/sd3-prompt-packet.yaml

python scripts/kubrick.py do adapt --action provider \
  --packet out/model-adapter-packet.yaml \
  --provider midjourney \
  --output out/midjourney-prompt-packet.yaml
```

Adapters change syntax only. They do not call external APIs, require credentials, or rewrite graph identity / private pattern links.

## 9. Operators and optional MCP

```bash
python scripts/kubrick.py do operate saturation-score \
  --ledger project/symbolic-ledger.yaml

python scripts/kubrick.py do operate counterpoint \
  --packet out/structured-symbolic-packet.yaml

python scripts/kubrick.py do operate convergence-lock \
  --graph out/motif-graph.private.yaml

python scripts/kubrick.py do operate surface-occult-audit \
  --input out/audience-constraints.yaml

python scripts/kubrick.py do operate symbolic-architecture-export \
  --graph out/motif-graph.private.yaml \
  --ledger project/symbolic-ledger.yaml \
  --output out/symbolic-architecture-export.yaml

# Optional MCP surface — single tool kubrick_do, never authoritative
python scripts/kubrick.py do mcp
```

Every operator emits an auditable receipt and fails closed on weak evidence.

## 10. Design specification

```bash
python scripts/kubrick.py do design --help
```

Use when consolidating design intent into a schema-valid specification (`schemas/design-specification.schema.yaml`). See:

- `references/design-specification-compiler.md`
- `templates/design-specification.yaml`

## 11. Validate an artifact

```bash
python scripts/kubrick.py do check --action artifact \
  --artifact out/kubrick/motif-graph.private.yaml \
  --schema schemas/motif-structure-graph.schema.yaml
```

Validation errors include exact artifact paths.

## 12. Full verification

```bash
python scripts/kubrick.py do check --action skill
python scripts/kubrick.py do check --action corpus
python scripts/kubrick.py do check --action coverage
python scripts/kubrick.py do check --action eval
python scripts/test_outcome_governance.py
python scripts/test_wave2_wave3.py
python scripts/test_design_specification.py
python scripts/kubrick.py do check --action repeatability --output out/kubrick/repeatability-report.json
python scripts/audit_release_version.py --strict
```

Shortcut: `python scripts/kubrick.py recipe verify` (runs `do check --action smoke`).

## Runtime rules

- observed form before interpretation
- dramatic function before symbolism
- mandatory mutation or justified stagnation
- one governing grammar and at most two supporting patterns
- one or two convergence sites
- private pattern and lexicon links never enter audience output
- adapters may translate syntax but may not rewrite graph identity
- local output is `PROPOSED` / `OBSERVATION` / `PROPOSAL`
- evolution never applies automatically
- large confidence or structural changes require human review
- weak evidence returns `NOT_COMPUTABLE`
- Continuity Forge remains optional and canonical when connected
- MCP remains optional and never authoritative

## Further reading

| Doc | Purpose |
|---|---|
| `README.md` | Public overview and architecture |
| `SKILL.md` | Hermes operating contract |
| `docs/ROADMAP-v0.14.md` | Current roadmap |
| `docs/RELEASE-NOTES-v0.14.md` | What shipped in 0.14 |
| `docs/README.md` | Full docs index |
| `references/continuity-forge-integration.md` | Forge handoff + feedback |
| `references/hermes-model-adapters.md` | Adapter invariants |
| `references/hermes-visual-qa.md` | Visual QA contract |
