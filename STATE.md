# Hermes Agent State

Snapshot date: 2026-07-24
Last verification time: 2026-07-24 04:01 PDT (UTC-07:00)

## Canonical snapshot policy

This file is the canonical operational snapshot for this checkout. Every future
Codex session must read `STATE.md` before making changes, use it to establish
the current baseline, and update it whenever work materially changes the system
inventory, completed milestones, blockers, priorities, recommended next task,
or verification time. Live repository and machine evidence takes precedence if
the snapshot is stale; reconcile any discovered drift back into this file.

## Current system inventory

### Host

- Hostname: `bbl`
- Operating system: Ubuntu 24.04.4 LTS (Noble Numbat)
- Kernel: Linux 7.0.0-28-generic, x86_64
- CPU: AMD Ryzen 5 3600, 6 cores / 12 threads, up to 4.2 GHz
- CPU cache: 384 KiB L1, 3 MiB L2, 32 MiB L3
- Memory: 62 GiB total, 52 GiB available at snapshot time
- Swap: 8 GiB
- Root storage: 916 GiB ext4, 73 GiB used, 797 GiB available
- NVIDIA GPU tooling: not available (`nvidia-smi` is not installed)

### Development runtimes

- Python: 3.12.3
- Node.js: 22.23.1
- npm: 10.9.8
- Git: 2.43.0
- Docker Engine: 29.1.3
- Docker Compose: 5.1.2
- Ollama: 0.32.1
- AppArmor: 4.0.1really4.0.1-0ubuntu0.24.04.7
- Bubblewrap: 0.9.0-1ubuntu0.1, protected by the Ubuntu-provided
  `bwrap-userns-restrict` AppArmor profile

### Installed Ollama models

| Model | Size | Observed compatibility |
| --- | ---: | --- |
| `llama3.1:8b` | 4.9 GB | 131K context; tool request accepted but response behavior was unreliable |
| `hermes-qwen:8b` | 5.2 GB | 40,960-token context, below Hermes minimum |
| `gemma3:12b` | 8.1 GB | 131K context; Ollama reports no tool support |
| `nomic-embed-text:latest` | 274 MB | Embedding model |
| `qwen3:8b` | 5.2 GB | 40,960-token context, below Hermes minimum |

### Listening TCP ports

- `22` and `3000`: listening on all IPv4 and IPv6 interfaces
- `11434`: Ollama, listening on all interfaces
- `53`: local system resolver
- `631`: local loopback printing service
- `37453`: loopback-only ephemeral service at verification time

### Repository

- Root: `/home/len/hermes-agent`
- Branch: `main`
- Repository baseline at this snapshot: `fc5433ce7`
- Upstream relation at this verification: 15 commits ahead of `origin/main`
- `DESIRED.md` is an untracked user file and was left untouched
- Python environment: repository-local `.venv`
- Repository-local generated state: `.tmp`, `.cache`, `.tools`, and `.venv`
  are excluded from Git

## Completed milestones

### Baseline and workflow

- Populated the checkout from the official `NousResearch/hermes-agent`
  repository and verified its source tree and Git integrity.
- Inventoried supported runtimes, test suites, workspaces, and canonical build
  commands.
- Added and maintained `TODO.md` as the execution and evidence log.

### Python core

- Created an isolated repository-local Python environment and installed the
  declared development and integration dependencies.
- Fixed Copilot ACP host-home propagation.
- Made SQLite corruption and WAL recovery coverage portable across protected
  SQLite builds.
- Passed the definitive Python baseline: 46,326 tests across 2,258 files.
- Passed the final post-upgrade regression: 230 tests in 386.05 seconds.
- Passed Ruff; documented the existing whole-tree `ty` baseline separately.

### CLI and configuration

- Verified CLI help, diagnostics, setup, non-interactive entry points, and
  `hermes doctor` using state isolated inside the repository.
- Verified local Ollama transport and validation without changing production
  Hermes configuration.

### JavaScript applications

- Installed root workspace dependencies using the repository lockfile.
- Passed root workspace checks, including type checks, tests, builds, and lint.
- Passed production builds for the web UI, TUI, desktop application, and
  bootstrap installer.

### Gateway and integrations

- Verified gateway lifecycle behavior and platform-independent gateway tests.
- Verified cron, plugin, MCP, messaging-adapter, and ACP contracts.
- Left credential-dependent integrations disabled unless explicitly configured.

### Harness control plane

- Treats inference as a swappable deployment backend; local-model capability
  and provider credentials are not development blockers.
- Added a model-independent managed-node registry backed by
  `control-plane.db`, with stable identity, idempotent enrollment, declared
  role and owner, explicit lifecycle transitions, optimistic concurrency, and
  hash-chained audit history.
- Added the operator-facing `hermes harness nodes` CLI for enrollment,
  inventory, lifecycle transitions, history, and audit verification.

### Packaging, security, and operations

- Verified console scripts, Docker Compose configuration, clean-start behavior,
  updates, and the documented installer/Docker/Nix distribution paths.
- Updated vulnerable Node and Python dependencies.
- Reached zero npm audit vulnerabilities and zero high/critical Hermes OSV
  findings.
- Installed and loaded Ubuntu's scoped `bwrap-userns-restrict` AppArmor
  profile. The isolated user, PID, and network namespace probe passes while
  `kernel.apparmor_restrict_unprivileged_userns=1` remains enabled.
- Scanned tracked files for secret patterns and found no candidates outside
  explicit examples, documentation, fixtures, skills, and tests.
- Added `OPERATOR.md` with verified instructions for this machine.
- Completed the repository verification matrix and committed each coherent
  build slice incrementally.

## Outstanding blockers

- No unresolved Harness control-plane development blocker is known.
- Model selection and provider/account credentials are deferred deployment
  configuration for swappable backends, not control-plane blockers.

## Active priorities

1. Expose managed-node inventory and lifecycle through a versioned
   control-plane API using the same domain registry as the CLI.
2. Add authenticated enrollment and revocable node credentials without storing
   plaintext secrets in control-plane state.
3. Add observed health/capability reports and desired-policy reconciliation.
4. Review the local commit series and decide whether to push it or open a pull
   request against the desired remote branch.
5. Select and configure an inference backend only when deployment work requires
   one, then run its backend contract smoke tests.

## Next recommended task

Add a versioned managed-node API over the model-independent registry, retaining
the CLI and API as consistent views of one authoritative control-plane store.

## References

- `TODO.md`: detailed execution history, evidence, and the open model item
- `OPERATOR.md`: verified local operating instructions
