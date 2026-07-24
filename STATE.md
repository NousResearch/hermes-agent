# Hermes Agent State

Snapshot date: 2026-07-24
Last verification time: 2026-07-24 02:38:00 PDT (UTC-07:00)

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

### Installed Ollama models

| Model | Size | Observed compatibility |
| --- | ---: | --- |
| `llama3.1:8b` | 4.9 GB | 131K context; tool request accepted but response behavior was unreliable |
| `hermes-qwen:8b` | 5.2 GB | 40,960-token context, below Hermes minimum |
| `gemma3:12b` | 8.1 GB | 131K context; Ollama reports no tool support |
| `nomic-embed-text:latest` | 274 MB | Embedding model |
| `qwen3:8b` | 5.2 GB | 40,960-token context, below Hermes minimum |

### Listening TCP ports

- `3000` and `3002`: listening on all IPv4 and IPv6 interfaces
- `11434`: Ollama, listening on all interfaces
- `53`: local system resolver
- `631`: local loopback printing service
- `43791`: loopback-only ephemeral service at snapshot time

### Repository

- Root: `/home/len/hermes-agent`
- Branch: `main`
- Snapshot commit: `623d41908`
- Upstream relation before adding this file: 13 commits ahead of `origin/main`
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

### Packaging, security, and operations

- Verified console scripts, Docker Compose configuration, clean-start behavior,
  updates, and the documented installer/Docker/Nix distribution paths.
- Updated vulnerable Node and Python dependencies.
- Reached zero npm audit vulnerabilities and zero high/critical Hermes OSV
  findings.
- Scanned tracked files for secret patterns and found no candidates outside
  explicit examples, documentation, fixtures, skills, and tests.
- Added `OPERATOR.md` with verified instructions for this machine.
- Completed the repository verification matrix and committed each coherent
  build slice incrementally.

## Outstanding blockers

- A real local-model tool-call smoke test is blocked on selecting or downloading
  a model that provides both a true 64K-or-larger context and reliable
  structured tool calling, or configuring a compatible remote provider.
- No unresolved codebase, build, test, packaging, or high-severity security
  blocker is known at this snapshot.
- Credentials are intentionally absent for provider- and account-dependent
  integrations; those integrations cannot be exercised until the user chooses
  and configures them.

## Active priorities

1. Select a compatible local model or remote provider.
2. Run and record an end-to-end conversation plus structured tool-call smoke
   test using repository-isolated state.
3. Review the local commit series and decide whether to push it or open a pull
   request against the desired remote branch.
4. Configure only the provider credentials and optional integrations that are
   actually needed for deployment.
5. Re-run the focused verification gates after any model, provider, dependency,
   or deployment configuration change.

## Next recommended task

Select a Hermes-compatible model/provider, then run one repository-isolated
end-to-end prompt that requires a structured tool call. Record the selected
model, declared context length, request, tool invocation, response, and command
exit status in `TODO.md`, and update this state snapshot with the outcome.

## References

- `TODO.md`: detailed execution history, evidence, and the open model item
- `OPERATOR.md`: verified local operating instructions
