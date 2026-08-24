<p align="center">
  <img src="docs/ares-workbench.svg" width="100%" alt="Ares architecture: an isolated Hermes-compatible runtime feeds explicit plugins, MCP services, and an evidence boundary with optional governed integrations.">
</p>

<!-- last-verified: 2026-08-21 -->

# Ares

<p align="center"><strong>An evidence-native, Hermes-compatible AI workbench for bounded execution, inspectable state, and explicit operator control.</strong></p>

<p align="center">
  <a href="https://github.com/RecursiveIntell/Ares">Repository</a> ·
  <a href="https://recursiveintell.github.io/hermes-agent/docs/">Documentation</a> ·
  <a href="https://github.com/NousResearch/hermes-agent">Upstream Hermes</a> ·
  <a href="SECURITY.md">Security</a>
</p>

> [!IMPORTANT]
> **Ares is not regular Hermes and is not an official Nous Research product.** Ares is a RecursiveIntell downstream distribution of [Hermes Agent](https://github.com/NousResearch/hermes-agent). It preserves the Hermes Python package and `hermes` CLI for compatibility while adding an isolated `ares` launcher, managed runtime releases, and explicit evidence-oriented integration boundaries.

## What Ares is

Ares is for operators who want the familiar Hermes agent experience without treating a chat transcript, a registered tool, or a successful-looking model response as proof that work completed correctly.

Ares keeps Hermes’s normal conversation, model routing, tools, plugins, skills, MCP, gateway, TUI, and desktop surfaces. Its fork-owned layer adds a separate runtime control plane around that compatible base:

| Surface | Hermes compatibility | What Ares adds or changes |
|---|---|---|
| Agent process | Existing Python package and `hermes` CLI remain available | An `ares` launcher selects a stable Ares runtime and defaults to the independent `~/.ares` home. |
| Runtime lifecycle | Hermes can be installed and updated through its normal flows | Ares materializes releases, switches them atomically, keeps current/previous pointers, and supports `doctor`, `status`, and rollback. |
| Release custody | Hermes update behavior remains available inside the selected runtime | Ares-owned candidate custody binds release artifacts, identities, inventories, lifecycle events, authorization, and rollback state. See [`docs/ares-candidate-custody.md`](docs/ares-candidate-custody.md). |
| Governed execution | Hermes approvals and toolsets remain the normal agent boundary | The optional Recursive Agent plugin submits one bounded operation through local authenticated IPC and returns daemon-derived verification facts. |
| RecursiveIntell integrations | Normal Hermes providers, MCP, plugins, and skills remain opt-in | Optional transports and external services can be admitted independently: `llm-pipeline`, `context-governor`, `agent-graph`, `poly-kv`, Semantic Memory, Claim Ledger, CEA Graph, and Pilot Bridge. Source presence is not activation. |
| Documentation | Hermes-compatible reference material remains useful | Ares documents which surfaces are fork-owned, inherited, optional, verified, or still unverified. |

### The core distinction

Ares separates four states that are easy to confuse:

```text
selected  →  registered  →  exposed  →  exercised
   │            │             │            │
 config       tool/MCP      current       real run,
 choice       discovery     session       result/receipt
```

A component is not proven merely because it appears in source, configuration, a tool listing, or a successful registration step.

## Capability map

This is the compact Ares-versus-Hermes map. “Inherited” means the surface comes from the Hermes-compatible runtime; “gated” means an operator must install, configure, or verify another component first.

| Capability | Ares state | Hermes relationship | Where to go next |
|---|---|---|---|
| Interactive CLI / TUI | Inherited and launched through `ares chat` / `ares tui` | Compatible Hermes surface | [Hermes CLI documentation](https://hermes-agent.nousresearch.com/docs/user-guide/cli) |
| Desktop | Managed by `ares desktop` after a desktop-capable runtime build | Compatible Hermes desktop surface with Ares branding/runtime selection | [`apps/desktop/README.md`](apps/desktop/README.md) |
| Gateway | Managed by `ares gateway ...` and `ares-gateway.service` | Compatible Hermes gateway and platform adapters | [Messaging documentation](https://hermes-agent.nousresearch.com/docs/user-guide/messaging) |
| Providers and model routing | Inherited | Hermes configuration and provider system | [Provider documentation](https://hermes-agent.nousresearch.com/docs/integrations/providers) |
| Tools, toolsets, plugins, and skills | Inherited, with Ares home isolation | Hermes extension model | [Tools](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) · [Skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) |
| MCP | Inherited | Hermes MCP client and configured servers | [MCP documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp) |
| Cron and scheduled work | Inherited | Hermes scheduler and delivery model | [Cron documentation](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron) |
| Stable runtime lifecycle | Ares-owned | Not a replacement for Hermes’s normal update path | [`ares update`](#runtime-operations) · [`ares rollback`](#runtime-operations) |
| Candidate custody | Ares-owned and separately persisted | No claim that upstream Hermes provides this Ares custody layer | [`docs/ares-candidate-custody.md`](docs/ares-candidate-custody.md) |
| Recursive Agent execution | Gated | Separate plugin and daemon | [`docs/ares-recursive-agent.md`](docs/ares-recursive-agent.md) |
| Rust-backed RecursiveIntell transports | Gated and environment-dependent | Not part of the normal Hermes compatibility guarantee | [Transport boundaries](#recursiveintell-integrations) |
| Semantic Memory, Agent Graph, Claim Ledger, CEA Graph, Pilot Bridge | External and opt-in | Separate services/projects | [Integration boundaries](#integration-boundaries) |

Ares does **not** claim that every optional service is installed, that every native extension is active, or that every provider/platform combination has been tested on every host.

## Quick start

### Prerequisites

- Git
- [uv](https://docs.astral.sh/uv/)
- Python **3.11–3.14**
- A model provider configured through the normal Hermes setup flow

The Ares bootstrap targets Unix-like shells: Linux, macOS, and WSL. The upstream `scripts/install.ps1` remains in the tree for Hermes compatibility testing; it is not an Ares-isolated PowerShell bootstrap.

### Install from the Ares fork

Review the installer before executing it, then run:

```bash
git clone https://github.com/RecursiveIntell/Ares.git Ares
cd Ares
uv sync --locked --extra all
.venv/bin/ares setup --source "$PWD"
```

A successful setup creates or selects:

- independent Ares configuration under `~/.ares/`;
- stable runtime releases under `~/.ares/runtime/releases/<commit>/`;
- atomic `current` and `previous` runtime pointers;
- Ares control state under `~/.ares/runtime-state/`;
- a launcher at `~/.local/bin/ares`;
- the `ares-gateway.service` user unit unless gateway installation is disabled.

If `~/.local/bin` is not on `PATH`, add it through your shell profile. Then check the selected runtime:

```bash
ares --version
ares status
ares doctor
```

The expected first-success signal is a selected Ares revision followed by `PASS` checks from `ares doctor`. Provider credentials are still your responsibility; setup does not create credentials or silently authorize external services.

### Choose the Ares runtime surface

```bash
ares chat                 # Hermes-compatible interactive CLI
ares tui                  # Hermes-compatible TUI
ares desktop              # Launch the selected desktop build
ares gateway status       # Inspect the Ares gateway service
```

## The `ares` command reference

The launcher is defined in [`ares_runtime/local_runtime.py`](ares_runtime/local_runtime.py). Run `ares --help` on an installed runtime for the live parser output.

| Command | Purpose | Important options |
|---|---|---|
| `ares setup` | Build and select a stable runtime from a checkout | `--source PATH`, `--seed-from PATH`, `--no-desktop`, `--no-gateway`, `--upstream-remote URL`, `--upstream-branch NAME` |
| `ares update` | Build and atomically select the configured remote candidate | `--no-desktop` |
| `ares rollback` | Return to the previous stable runtime | None |
| `ares doctor` | Check runtime pointers, imports, configuration, native integrations, and gateway state | None |
| `ares status` | Show selected runtime, remote, and gateway information | None |
| `ares desktop` | Launch the selected Ares Desktop application | `--rebuild` |
| `ares tui` | Launch the selected Hermes-compatible TUI | Pass-through TUI arguments are accepted |
| `ares chat` | Launch the selected Hermes-compatible CLI | Pass-through CLI arguments are accepted |
| `ares gateway` | Manage the Ares gateway service | `start`, `stop`, `restart`, `status`, or `foreground` |
| `ares --version` | Print the selected stable runtime revision | None |

### Runtime operations

```bash
ares status
ares update
ares doctor
ares rollback
```

`ares update` stages a configured Hermes upstream revision, applies the Ares downstream state, builds the candidate, and switches only after the candidate succeeds. If the build or activation path fails, the active release is intended to remain selected. `ares rollback` returns to the previous stable release when one exists.

The source-backed custody details are deliberately kept out of this quick-start block. Read [`docs/ares-candidate-custody.md`](docs/ares-candidate-custody.md) before treating candidate certification, audit state, or rollback state as an authority decision: certification and candidate-bundled activation input are explicitly non-authorizing until the CandidateStore-owned activation transition occurs.

## Integration boundaries

Ares does not bundle authority into a product slogan. The layers remain separate:

```text
operator
   │
   ▼
Ares launcher ──> stable Hermes-compatible runtime ──> tools / plugins / MCP
                                      │
                                      ├── optional Recursive Agent plugin
                                      │       └── local authenticated IPC
                                      │             └── bounded daemon run + receipt chain
                                      │
                                      └── optional external services
                                              ├── Semantic Memory
                                              ├── Agent Graph
                                              ├── Claim Ledger
                                              ├── CEA Graph
                                              └── Pilot Bridge
```

- **Hermes-compatible runtime** owns conversation, provider routing, tool selection, approvals, plugins, and normal persistence.
- **Ares** owns downstream identity, installer behavior, isolated home selection, runtime lifecycle, documentation boundaries, and integration policy in this repository.
- **Recursive Agent** owns its run contract, state machine, receipt chain, and verification result. The plugin does not manufacture evidence or bypass the daemon.
- **MCP services** remain separate processes or services. A registered tool is not proof that its backend is reachable or that a real operation succeeded.

### RecursiveIntell integrations

The repository includes optional transport modules for `llm-pipeline`, `context-governor`, `agent-graph`, and `poly-kv`. The code also exposes the Hermes/Ares `/llm-pipeline` control surface for inspecting or changing the transport state. These paths are **gated**, not unconditional promises:

1. select the relevant provider or engine;
2. install or materialize the required native/runtime component;
3. run `ares doctor` or the integration-specific checks;
4. exercise a real request in the target environment;
5. retain the returned evidence before making a capability claim.

The presence of source modules, a config key, or a registered MCP server does not establish any of those steps.

### Optional Recursive Agent plugin

The Recursive Agent integration is a standalone plugin, not a bundled core tool. It requires a separately built and running local Recursive Agent daemon.

From an existing `RecursiveIntell/recursive-agent` checkout:

```bash
bash install.sh --with-recursive-agent-source /path/to/recursive-agent
```

This installs the plugin package into `~/.ares/plugins/recursive-agent-native`. It does **not** build, configure, start, or grant authority to the daemon. Start a fresh Ares/Hermes session after plugin installation so discovery can occur.

Read [`docs/ares-recursive-agent.md`](docs/ares-recursive-agent.md) for the socket contract, operation envelope, receipts, and verification semantics.

## Configuration and data ownership

Ares preserves the Hermes-compatible configuration format but uses `~/.ares` as its independent agent home. Ares and an existing Hermes installation can therefore have different providers, skills, plugins, sessions, and gateway lifecycles on the same machine.

Keep these boundaries explicit:

- provider secrets belong in the supported local secret mechanism, never in this repository or shell history;
- MCP server mappings and argument lists are typed YAML, not ad-hoc strings;
- plugins and hooks run with agent-process authority and must be reviewed before installation;
- restart or start a fresh session after changing plugin, toolset, MCP, or credential configuration because tool schemas are session-scoped;
- prove a capability at the correct layer: selected, registered, exposed, then exercised.

The bootstrap installer accepts:

| Installer option | Effect |
|---|---|
| `--branch NAME` | Clone or update a specific branch. |
| `--dir PATH` | Select the source checkout directory. |
| `--hermes-home PATH` | Select the Ares data directory. |
| `--ares-bin-dir PATH` | Select where the `ares` launcher is written. |
| `--no-venv` | Use the active Python environment instead of a managed virtual environment. |
| `--with-recursive-agent-source PATH` | Install only the standalone Recursive Agent plugin from an existing checkout. The daemon remains operator-managed. |

Run `bash install.sh --help` for the authoritative installer contract. The bootstrap refuses to update a dirty existing checkout and refuses to overwrite a non-Ares launcher.

## Security and trust boundaries

Ares inherits Hermes’s fundamental security posture: **the operating system or an explicit whole-process sandbox is the real boundary against adversarial model output.** Approval prompts, tool allowlists, plugin review, redaction, and receipts are useful controls; they are not containment.

Important consequences:

- a plugin runs with the authority of the agent process;
- a local IPC socket or verified receipt does not contain a compromised process;
- do not give an agent access to files, credentials, network destinations, or destructive tools you would not delegate to it;
- use a whole-process wrapper or deliberately constrained account for untrusted content or higher-risk workloads.

Read [`SECURITY.md`](SECURITY.md) before exposing Ares to untrusted inputs or shared environments.

## Repository map

| Path | Role |
|---|---|
| `install.sh` | Ares bootstrap installer. |
| `ares_runtime/` | Stable runtime selection, materialization, activation, rollback, gateway handoff, and launcher implementation. |
| `agent/transports/ri_*.py` | Optional RecursiveIntell transport integrations. |
| `docs/ares-candidate-custody.md` | Candidate custody, lifecycle, audit, authorization, and garbage-collection contract. |
| `docs/ares-recursive-agent.md` | Recursive Agent boundary and operator guide. |
| `website/` | Ares documentation front door plus Hermes-compatible reference material. |
| `tests/test_ares_distribution.py` | Fork identity and installer-scope contract tests. |

## Development and validation

Ares is a large Python, TypeScript, and desktop codebase. Start with [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

Useful bounded checks from this checkout:

```bash
bash -n install.sh
bash install.sh --help
python3 -m pytest -q tests/test_ares_distribution.py
```

For broader validation, use the repository-owned test entry point:

```bash
scripts/run_tests.sh
```

The commands above validate installer syntax/help and the Ares distribution contract. They do not prove that a model provider, optional daemon, native extension, or production deployment works on every host.

## Deeper Hermes-compatible documentation

Ares intentionally does not duplicate the entire Hermes manual. Use these references for inherited capabilities:

- [Hermes quickstart](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)
- [CLI and configuration](https://hermes-agent.nousresearch.com/docs/user-guide/cli)
- [Providers and models](https://hermes-agent.nousresearch.com/docs/integrations/providers)
- [Tools and toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)
- [Skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)
- [Memory](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)
- [MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)
- [Cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)
- [Messaging gateway](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)
- [Security](https://hermes-agent.nousresearch.com/docs/user-guide/security)

Where a page names upstream URLs or support channels, treat those as Hermes reference material—not as an Ares release, support, or universal compatibility guarantee.

## Status and claim boundary

**Source review performed 2026-08-21 at commit `e2a870a7e2c0b4028965735bad53e190473f673c`.** The source and targeted contract tests establish the documented fork identity, installer boundaries, Ares launcher command surface, and the presence of the runtime/custody/integration code described above.

That source review does **not** establish cross-platform support, public packaging of the Recursive Agent daemon, a managed service installer for every optional service, production readiness, security certification, performance superiority, or universal provider/platform support. Treat those as separate verification projects.

## Upstream provenance, contributions, and license

Ares is derived from [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent). The canonical downstream repository is [RecursiveIntell/Ares](https://github.com/RecursiveIntell/Ares); the historical [`RecursiveIntell/hermes-agent`](https://github.com/RecursiveIntell/hermes-agent) repository path remains a compatibility reference for existing documentation and issue links. Preserve upstream attribution and license notices when redistributing or contributing changes.

- Security reporting: [`SECURITY.md`](SECURITY.md)
- Contribution process: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- License: [`MIT`](LICENSE)
