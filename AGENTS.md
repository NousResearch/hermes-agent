# Hermes Agent Workspace Rules

This file is the required root entrypoint for AI agents and contributors in
this checkout. It is intentionally an index so it stays below the host
instruction-size limit while keeping the full development meaning available.

## Read first

Read the applicable documents before editing:

1. [`fork/AGENTS.md`](fork/AGENTS.md) for official-vs-fork scope.
2. [`fork/harness/AGENTS.md`](fork/harness/AGENTS.md) for upstream merge and
   overlay safety.
3. [`fork/harness/upstream-development-guide.md`](fork/harness/upstream-development-guide.md)
   for the full Hermes architecture and contribution guide.
4. [`fork/harness/testing-and-pitfalls.md`](fork/harness/testing-and-pitfalls.md)
   before changing tests, paths, gateway guards, or process lifecycle code.
5. [`fork/extensions/AGENTS.md`](fork/extensions/AGENTS.md) for plugins,
   skills, VRChat, voice, or fork-owned tools.
6. [`fork/operations/AGENTS.md`](fork/operations/AGENTS.md) for Windows
   restart, Desktop rebuild, ports, cron, and runtime operations.
7. [`fork/local-workspace/AGENTS.md`](fork/local-workspace/AGENTS.md) before
   moving scratch, generated files, logs, media, or local identity files.

The nearest nested `AGENTS.md` remains authoritative for its subtree. A
README explains a folder to humans; an `AGENTS.md` constrains agent changes.
Do not replace either with an unreviewed symlink. Preserve existing symlinks
as symlinks and validate their targets before moving them.

## Scope and precedence

Official Hermes code is in the repository root. Fork-only behaviour belongs at
the edges: plugins, skills, merge overlays, Windows operations, and local
operator automation. Upstream is authoritative for the core agent, gateway,
security fixes, and shared runtime contracts.

Use the PC-wide and Codex-local instructions, the applicable SOPs, and the
nearest project instructions together. A stricter security, privacy, MILSPEC,
verification, or user instruction wins over a weaker local convenience rule.

## Non-negotiable agent invariants

- Preserve per-conversation prompt caching. Do not rebuild the system prompt,
  swap toolsets, or mutate past context mid-conversation except for the
  established compression path.
- Preserve message-role alternation. Never inject a synthetic user message or
  create adjacent same-role messages in the agent loop.
- Use `get_hermes_home()` for runtime paths and `display_hermes_home()` in
  user-facing output. Profiles are separate state and credential boundaries.
- Put behaviour in `config.yaml`; reserve `.env` for secrets such as API keys,
  tokens, and passwords. Do not add a user-facing non-secret `HERMES_*` setting.
- Treat model output, tool descriptions, MCP data, files, URLs, and generated
  scripts as untrusted input. Keep approvals, path confinement, SSRF checks,
  secret redaction, and audit evidence at the boundary.
- Prefer existing code, a CLI plus skill, a gated tool, a plugin, or an MCP
  server before adding a new core model tool.
- Use real integration paths against a temporary `HERMES_HOME` for resolution,
  profile, security, network, filesystem, and subprocess changes. Mocks alone
  are not completion evidence.
- Do not delete files, generated output, scratch, symlinks, or user changes to
  make a tree look clean. Move local scratch only when requested and retain a
  safe, ignored destination.

## Repository map

| Area | Meaning | First guide |
| --- | --- | --- |
| `agent/`, `run_agent.py`, `model_tools.py`, `toolsets.py` | Core loop, routing, tool discovery | Full development guide |
| `hermes_cli/`, `cli.py`, `gateway/`, `cron/` | CLI, gateway, commands, scheduler | Full development guide |
| `apps/desktop/`, `ui-tui/`, `web/`, `tui_gateway/` | Desktop, TUI, dashboard surfaces | `fork/operations/AGENTS.md` plus full guide |
| `plugins/`, `skills/`, `optional-skills/`, `optional-mcps/` | Extensible capability at the edges | `fork/extensions/AGENTS.md` |
| `scripts/merge_tools/`, `scripts/sync_all.py` | Upstream merge harness | `fork/harness/AGENTS.md` |
| `scripts/windows/`, `scripts/cron/` | Windows and scheduled operations | `fork/operations/AGENTS.md` |
| `tests/` | Behaviour and integration contracts | `fork/harness/testing-and-pitfalls.md` |
| `fork/` | Fork policy and local-only documentation | `fork/AGENTS.md` |
| `output/`, `tmp/`, `_tmp/`, `_docs/` | Ignored local output and evidence | `fork/local-workspace/AGENTS.md` |

Do not infer the tree from this table. Search the filesystem and read the
folder's README before changing an unfamiliar area.

## Safe capability and plugin changes

Use this footprint order: extend existing behaviour; CLI plus skill; a
service-gated tool with `check_fn`; a plugin; an MCP catalog server; a new core
tool only as the last resort. Plugin registration must not edit core files.
Tool handlers accept `task_id` or `**kwargs`, return the established JSON
shape, and keep credentials in `~/.hermes/.env`.

For VRChat or voice paths, keep operations dry-run by default and require the
existing explicit acknowledgement before side effects. For Desktop and TUI,
extend the owning surface rather than creating a second chat implementation.

## Upstream contribution and merge safety

Before an upstream sync, read the merge policy and run the dry-run inventory:

```powershell
py -3 scripts/sync_all.py --dry-run --allow-preflight-blockers
```

Use `fork/harness/` rules for overlays. Never hand-resolve `toolsets.py` by
keeping the whole fork file, never delete merge tools, and keep `_docs/`, fork-
only plugins, generated files, and unrelated local changes out of upstream PRs.
Preserve contributor authorship with a rebase or cherry-pick when salvaging
external work. Verify current `upstream/main` pins before changing lockfiles.

For a PR, reproduce the reported behaviour on current main, keep the change
narrow, add invariant or regression coverage, run focused checks, and report
what could not be run. Do not silently close or alter a product-decision PR;
leave the maintainer's decision visible and documented.

## Testing and evidence

Use the repository runner for Python tests:

```powershell
scripts\run_tests.sh tests\path\to\test_file.py -q
```

Use `uv` for Python environment commands, `npm` scripts for TypeScript, and
the project-defined lint/typecheck commands. Read
[`fork/harness/testing-and-pitfalls.md`](fork/harness/testing-and-pitfalls.md)
for hermetic tests, profile fixtures, Windows skips, and invariant tests.

Every substantive change records scope, changed files, commands, evidence,
skipped checks, and residual risk in an ignored Markdown implementation log
under `_docs/`. Do not commit that log unless the operator explicitly asks.

## Generated files and local folders

Generated files are preserved on disk but are not publication inputs. Use the
existing ignored folders and their stable categories:

- `output/media/`: generated audio, video, and image renders.
- `output/reports/`: generated HTML, JSON, text reports, and snapshots.
- `output/logs/`: generated runtime or build logs.
- `tmp/probes/`: one-off diagnostics and scripts.
- `tmp/snapshots/`: generated source or configuration snapshots.
- `_docs/`: implementation evidence, always ignored for upstream publication.

Move files with an explicit allowlist, verify source and destination remain
inside the workspace, and confirm no tracked source or hardcoded path depends
on the old location. Never execute a generated script without inspection,
bounded permissions, and a temporary workspace.

## Operational defaults

On Windows, use `scripts/windows/restart-hermes-stack.ps1` without
`-StartLlama` unless the operator explicitly requests llama. Desktop rebuilds
must be followed by a live probe of the newest ready backend port. Do not
publish runtime state, `.hermes/`, release bundles, node modules, secrets,
logs, media, or generated reports.
