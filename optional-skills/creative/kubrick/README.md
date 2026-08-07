<p align="center">
  <img src="assets/hero.jpg" alt="Kubrick — Symbolic Cinematic Narrative Engineering" width="100%">
</p>

<h1 align="center">Kubrick</h1>

<p align="center">
  <strong>Hermes-Native Symbolic Cinematic Engineering</strong><br>
  <em>0.14.0 — Deterministic Contract Consolidation</em>
</p>

<p align="center">
  <a href="https://github.com/scrimshawlife-ctrl/Kubrick/releases"><img src="https://img.shields.io/github/v/release/scrimshawlife-ctrl/Kubrick?color=5a6a8a&style=flat-square" alt="Latest Release"></a>
  <a href="https://github.com/scrimshawlife-ctrl/Kubrick"><img src="https://img.shields.io/github/stars/scrimshawlife-ctrl/Kubrick?style=flat-square" alt="GitHub Stars"></a>
  <a href="https://github.com/scrimshawlife-ctrl/Kubrick/issues"><img src="https://img.shields.io/github/issues/scrimshawlife-ctrl/Kubrick?style=flat-square" alt="Issues"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-1a1f2e?style=flat-square" alt="License"></a>
  <a href="#installation"><img src="https://img.shields.io/badge/Hermes-Standalone_Skill-8b5cf6?style=flat-square" alt="Hermes Standalone Skill"></a>
</p>

<p align="center">
  <strong>Observed form first. Dramatic function first. Hidden architecture made executable.</strong>
</p>

---

**Kubrick** is a self-contained Hermes skill for screenplay development, scene diagnosis, motif engineering, cinematic encoding, storyboard continuity, generative prompt construction, closed-loop visual QA, multi-signal outcome learning, and governed design-specification compilation.

It converts dramatic pressure into observable geometry, behavior, rhythm, material state, light, sound, convergence, and residue. Esoteric and archetypal source systems remain latent by default: audience-facing packets expose enforceable cinematic constraints rather than named occult concepts.

## How to use

### What Kubrick is for

Use Kubrick when you are building a story or image sequence and care about **what the audience can see and feel** — objects, light, geometry, ownership, recurrence, residue — not about dumping named symbols or occult labels into the frame.

Typical jobs:

- develop a premise, scene, or storyboard with consistent motifs  
- turn a brief into a structured symbolic plan and image prompts  
- check that a generated frame still matches the intended state  
- keep a project’s motif ledger as you revise  

You do **not** need Continuity Forge, an image API, or an MCP server to start. Those are optional add-ons.

### Option A — Talk to Hermes (normal creative use)

1. Install Kubrick as a Hermes skill (see [Installation](#installation)).
2. In Hermes, load the **kubrick** skill (or ask in a way that triggers it: screenplay, motif, storyboard, image prompt, continuity).
3. Describe the project in plain language: format, pressure, what must change, what is on screen.
4. Hermes uses Kubrick’s rules and, when useful, the deterministic tools under the hood.
5. Treat local output as **draft / proposal** until you explicitly approve it or hand it to a canonical system (e.g. Continuity Forge).

You mostly work in conversation. You only need the CLI when you want scripts, CI, or batch pipelines.

### Option B — Command line (tools and pipelines)

From a clone of this repo (or the installed skill directory):

```bash
# See the short list of jobs Kubrick can run
python scripts/kubrick.py --help

# Sanity-check the skill install
python scripts/kubrick.py do check --action smoke

# Run the built-in storyboard example end-to-end
python scripts/kubrick.py recipe storyboard-example
```

The pattern is always:

```text
python scripts/kubrick.py do <job> [options…]
```

Think of `<job>` as *what you want done*, not a pile of script names:

| Job | In plain English |
|---|---|
| `compile` | Turn a brief (+ optional ledger) into graph, storyboard, and image-ready packets |
| `retrieve` | Find matching cinematic patterns for a dramatic problem |
| `ledger` | Start or update the project’s motif checklist |
| `adapt` | Reword a neutral packet for Grok Imagine, Flux, SD3, or Midjourney |
| `visual` | Compare what you expected in a frame to what you observed; suggest fixes |
| `learn` | Record outcomes or propose (never auto-apply) pattern improvements |
| `check` | Validate the skill, corpus, or run smoke tests |
| `operate` | Score saturation, lock convergence, audit for named esoterica, export architecture |
| `design` | Compile a governed design specification from project evidence |
| `storyboard` | Propagate or compare multi-frame state |
| `bundle` | Package a Grok-oriented review workflow |
| `mcp` | Optional machine interface over the same tools |

Ask for help on one job:

```bash
python scripts/kubrick.py help adapt
```

### A simple path: idea → storyboard → image prompt

1. **Write a short brief** (dramatic problem, what changes, what is visible).  
   Example layout: `examples/authority-transfer-storyboard/brief.yaml`
2. **Optionally keep a ledger** of active motifs for the project.  
   Example: `examples/authority-transfer-storyboard/symbolic-ledger.yaml`
3. **Compile** (this is the main pipeline):

```bash
python scripts/kubrick.py do compile \
  --brief examples/authority-transfer-storyboard/brief.yaml \
  --ledger examples/authority-transfer-storyboard/symbolic-ledger.yaml \
  --mode storyboard \
  --storyboard-plan examples/authority-transfer-storyboard/storyboard-plan.yaml \
  --provider grok-imagine \
  --out out/kubrick/authority-transfer
```

4. **Open the output folder.** You get a private motif graph, storyboard state, and a provider prompt packet ready for your image tool.  
5. **After you generate images**, you can feed an observation back through `do visual` to see what drifted (geometry, ownership, residue, etc.) and what to regenerate.  
6. **Do not expect the corpus to rewrite itself.** Outcomes can create *proposals*; a human still decides.

Or skip the long form and run:

```bash
python scripts/kubrick.py recipe storyboard-example
```

### Things that stay true no matter how you use it

- **Show, don’t name** — audience-facing prompts avoid named esoterica unless you explicitly ask.  
- **If the evidence is weak, Kubrick stops** — it returns a clear failure rather than inventing structure.  
- **Local work is not “locked canon”** — proposals and observations stay local until you promote them.  
- **Old command names still work** if you used them before; the preferred form is `do <job>`.

### Option C — OpenClaw (separate permanent branch)

If your agent host is **OpenClaw** rather than Hermes, use the dedicated OpenClaw edition maintained with work by **Prabu** ([@prabu-openclaw](https://github.com/prabu-openclaw)). It lives on the permanent git branch **`openclaw`**, not on `main`.

```bash
openclaw skills install git:scrimshawlife-ctrl/Kubrick@openclaw --global
```

That edition uses OpenClaw skill packaging, installs under `~/.openclaw/skills/kubrick`, and keeps mutable runtime state outside the skill package. The creative corpus and laws stay aligned in spirit; installer, state paths, and some tooling differ.

Full detail, install paths, and credits: **[`docs/OPENCLAW.md`](docs/OPENCLAW.md)**  
Branch: https://github.com/scrimshawlife-ctrl/Kubrick/tree/openclaw  
Upstream PR (history): https://github.com/scrimshawlife-ctrl/Kubrick/pull/1

### Where to go next

| Need | Go here |
|---|---|
| Copy-paste workflows | [`QUICKSTART.md`](QUICKSTART.md) |
| Hermes operating rules | [`SKILL.md`](SKILL.md) |
| Official / community Hermes publication | [`docs/HERMES-OFFICIAL-SUBMISSION.md`](docs/HERMES-OFFICIAL-SUBMISSION.md) |
| OpenClaw edition (Prabu) | [`docs/OPENCLAW.md`](docs/OPENCLAW.md) |
| Full command table | [Operator Commands](#operator-commands) |
| Design of the command surface | [`docs/superpowers/specs/2026-07-30-operator-intent-router-design.md`](docs/superpowers/specs/2026-07-30-operator-intent-router-design.md) |

## 0.14.0 Highlights

Kubrick v0.14 consolidates the standalone Hermes skill around one manifest, deterministic identities, structured diagnostics, semantic provider-preservation reports, and atomic installation. Wave 2 and Wave 3 remain intact beneath this contract layer.

| Area | What landed |
|---|---|
| **Forge multi-signal feedback** | Ledger diffs, revisions, saturation, collisions, ingestion, and payoff outcomes → deterministic observation bundles |
| **Multi-signal evolution** | Confidence, mutation success, feasibility, anti-slop, cultural boundaries, payoff → proposal-only evolution with human review gates |
| **First-class project ledgers** | Persistent motif state, pattern history, rehydrate/apply-forge, retrieval snapshots; Forge remains canonical |
| **Multi-provider adapters** | Grok Imagine, Flux, SD3, Midjourney share one latent graph; adapters change **syntax only** |
| **Closed-loop visual QA** | Observe → normalize → differential score → correct; geometry / state / residue / convergence reported separately |
| **CLI + optional MCP operators** | Intent router (`do <intent>`), soft aliases, recipes; MCP tool `kubrick_do`; saturation, counterpoint, lock, occult audit, export |
| **Cultural-signal packs** | Time-sensitive contemporary memetic patterns with provenance and validity windows |
| **Design specification compiler** | Heterogeneous evidence → schema-valid `design.md` candidate without silent authority promotion |
| **Fail-closed governance** | Weak evidence → `NOT_COMPUTABLE`; no structural change applies automatically |

## Core Philosophy

- **Observed first, meaning second** — every motif begins as concrete form, behavior, relation, or material state.
- **Dramatic function before symbolism** — symbolic design must change pressure, agency, causality, or interpretation.
- **Mandatory mutation** — recurrence changes scale, ownership, orientation, material, rhythm, framing, context, or consequence.
- **Three-channel symbolism** — diegetic, dramaturgical, and cinematic channels cross without explanatory dialogue.
- **Constraint over citation** — source traditions become geometry, rhythm, threshold, role persistence, transformation, and residue.
- **Fail closed** — weak evidence or unresolved boundaries return `NOT_COMPUTABLE`.
- **Human-governed evolution** — local outputs remain observations or proposals until explicitly approved.
- **Forge-canonical when connected** — Continuity Forge owns committed project state; Kubrick never auto-promotes authority.

## Installation

```bash
git clone https://github.com/scrimshawlife-ctrl/Kubrick.git
cd Kubrick
./install.sh
```

The installer copies to staging, validates the manifest, Hermes skill, and pattern corpus, then atomically activates the new directory. Existing installs are backed up outside the discoverable skills tree and can be restored:

```bash
./install.sh --dry-run
./install.sh --version
./install.sh --rollback
```

Install and rollback operations write JSON receipts beneath `~/.hermes/receipts/`.

Default destination:

```text
~/.hermes/skills/kubrick
```

Install repository validation dependencies when developing locally:

```bash
python -m pip install pyyaml jsonschema
```

Validate the installed skill:

```bash
python ~/.hermes/skills/kubrick/scripts/kubrick.py do check --action skill
```

Continuity Forge, MCP servers, generation APIs, and vision APIs remain optional.

**OpenClaw users:** install from the permanent [`openclaw`](https://github.com/scrimshawlife-ctrl/Kubrick/tree/openclaw) branch — see [`docs/OPENCLAW.md`](docs/OPENCLAW.md). This `main` branch is the Hermes edition.

## Unified Pipeline

```bash
python scripts/kubrick.py do compile \
  --brief examples/authority-transfer-storyboard/brief.yaml \
  --ledger examples/authority-transfer-storyboard/symbolic-ledger.yaml \
  --mode storyboard \
  --storyboard-plan examples/authority-transfer-storyboard/storyboard-plan.yaml \
  --provider grok-imagine \
  --out out/kubrick/authority-transfer
```

Providers: `none` · `generic` · `grok-imagine` · `flux` · `sd3` · `midjourney`

```text
brief
→ registry-aware retrieval
→ private motif graph
→ structural and anti-slop audit
→ audience constraints
→ storyboard state propagation
→ transition comparison
→ neutral model-adapter packet
→ provider prompt packet (syntax-only)
→ schema receipts
→ compile receipt
```

## Operator Commands

Primary surface:

```bash
python scripts/kubrick.py do <intent> [--action <action>] [flags]
```

| Intent | Purpose |
|---|---|
| `compile` | Full brief-to-packet symbolic compile |
| `retrieve` | Registry-aware deterministic retrieval |
| `ledger` | Project symbolic ledger (init / audit / mutate / rehydrate / apply-forge) |
| `design` | Governed design-specification compilation |
| `storyboard` | Multi-frame state propagate / compare |
| `adapt` | Neutral adapter packet and provider syntax translation |
| `visual` | Visual QA loop (normalize / compare / correct / govern / closed-loop) |
| `learn` | Outcome receipts, multi-signal evolution, Forge signal extraction |
| `check` | Validation and regression (skill, corpus, coverage, eval, smoke, …) |
| `operate` | Saturation, counterpoint, lock, surface-occult audit, architecture export |
| `mcp` | Optional stdio MCP wrapper — single tool `kubrick_do` (never authoritative) |
| `bundle` | Package the complete Grok review workflow |

Sugar: `python scripts/kubrick.py help <intent>`, `recipe <name>`, `aliases`.

Legacy peer names (`adapt-flux`, `closed-loop-qa`, `validate-skill`, …) remain soft aliases.

Step-by-step recipes: [`QUICKSTART.md`](QUICKSTART.md).

## Common Workflows

### Multi-provider adaptation

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
```

Adapters preserve `source_graph_id` and never rewrite canonical symbolic intent.

### Forge feedback → evolution proposal

```bash
python scripts/kubrick.py do learn --action forge-signals \
  --project-id myfilm \
  --input references/examples/forge-signals/ledger-before-after.yaml \
  --output out/forge-bundle.yaml

python scripts/kubrick.py do ledger apply-forge \
  --ledger project/symbolic-ledger.yaml \
  --forge-bundle out/forge-bundle.yaml

python scripts/kubrick.py do learn --action evolve \
  --pattern-id interface_badge_authority_transfer \
  --forge-bundle out/forge-bundle.yaml \
  --output out/evolution-proposal.yaml
```

Every evolution event emits a multi-signal receipt. Structural changes and large confidence deltas require human review. Nothing applies automatically.

### Closed-loop visual QA

```bash
python scripts/kubrick.py do visual --action closed-loop \
  --expected out/storyboard-symbolic-state.yaml \
  --observation-input observations/frame-001.json \
  --source-graph-id <graph-id> \
  --frame-id frame-001 \
  --out out/qa/frame-001
```

### Operators and optional MCP

```bash
python scripts/kubrick.py do operate saturation-score --ledger project/symbolic-ledger.yaml
python scripts/kubrick.py do operate surface-occult-audit --input out/audience-constraints.yaml
python scripts/kubrick.py do operate symbolic-architecture-export \
  --graph out/motif-graph.private.yaml \
  --ledger project/symbolic-ledger.yaml \
  --output out/symbolic-architecture-export.yaml

# Optional — single MCP tool kubrick_do; fail closed on weak evidence
python scripts/kubrick.py do mcp
```

### Design specification

```bash
python scripts/kubrick.py do design --help
```

See `references/design-specification-compiler.md` and `templates/design-specification.yaml`.

## Verification

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

CI (`.github/workflows/hermes-evals.yml`) runs Hermes evals, outcome governance, Wave 2/3 smoke tests, multi-provider storyboard compiles, repeatability, and strict release-version audit.

## Documentation

| File | Purpose |
|---|---|
| [`SKILL.md`](SKILL.md) | Canonical Hermes operating contract |
| [`QUICKSTART.md`](QUICKSTART.md) | Installation and command routing |
| [`CHANGELOG.md`](CHANGELOG.md) | Release history |
| [`docs/ROADMAP-v0.14.md`](docs/ROADMAP-v0.14.md) | Current roadmap and next priorities |
| [`docs/RELEASE-NOTES-v0.14.md`](docs/RELEASE-NOTES-v0.14.md) | v0.14 release notes |
| [`docs/RELEASE-CHECKLIST-v0.14.md`](docs/RELEASE-CHECKLIST-v0.14.md) | Release gates and procedure |
| [`docs/README.md`](docs/README.md) | Docs index (incl. intent-router design/plan) |
| [`docs/OPENCLAW.md`](docs/OPENCLAW.md) | OpenClaw Agent Skill edition (Prabu; branch `openclaw`) |
| [`docs/superpowers/specs/2026-07-30-operator-intent-router-design.md`](docs/superpowers/specs/2026-07-30-operator-intent-router-design.md) | Operator intent-router design |
| [`references/hermes-runtime-contract.md`](references/hermes-runtime-contract.md) | Runtime, dependency, artifact, and canon policy |
| [`references/hermes-model-adapters.md`](references/hermes-model-adapters.md) | Provider adapter contract |
| [`references/hermes-visual-qa.md`](references/hermes-visual-qa.md) | Visual QA contract |
| [`references/continuity-forge-integration.md`](references/continuity-forge-integration.md) | Forge handoff and feedback |
| [`references/design-specification-compiler.md`](references/design-specification-compiler.md) | Design-spec compiler |
| [`references/patterns/`](references/patterns/) | Executable pattern sidecars |
| [`schemas/`](schemas/) | Machine-readable artifact contracts |
| [`evals/`](evals/) | Regression and adversarial specifications |

## Architecture (v0.14)

```text
                    ┌─────────────────────┐
                    │  Project brief +    │
                    │  symbolic ledger    │
                    └──────────┬──────────┘
                               │
              retrieve → graph → audit → audience
                               │
                    storyboard propagate / compare
                               │
              neutral adapter packet (shared latent graph)
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
      grok-imagine           flux / sd3        midjourney
      (syntax only)        (syntax only)     (syntax only)
                               │
                    closed-loop visual QA
                    (geometry · state · residue · convergence)
                               │
              outcome receipt → multi-signal evolution proposal
                               │
                    human review gate (never auto-apply)
                               │
              optional Forge signals ←── Continuity Forge (canonical)
```

## Version

**0.14.0 — Deterministic Contract Consolidation**

This release unifies the Hermes runtime contract, reproducibility identities, failure diagnostics, provider preservation, installer safety, and official-skill packaging. See the v0.14 release notes for verified gates and remaining post-release work.

See [CHANGELOG.md](CHANGELOG.md) for full release history.

---

<p align="center">
  <em>Symbolism should alter the conditions under which a scene is interpreted—without requiring the audience to consciously identify it.</em>
</p>
