---
name: codex-operations
description: Run Codex agents safely and observably.
version: 0.1.0
author: NousResearch
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: autonomous-ai-agents
    tags: [codex, agents, telemetry, safety, worktrees]
    related_skills: [codex, opencode, repomix-context-packaging, test-driven-development]
---

# Codex Operations

Use this skill when John asks Alfred/Hermes to run Codex agents, adapt OpenCode-style workflows, launch multi-agent coding work, or make a coding task safer and more observable.

## Mission

Run Codex as a disciplined worker under Hermes control:

1. **Bound the workspace.** Use `terminal` to check branch, dirty state, and repo instructions before any write.
2. **Plan the operation.** Split work into lanes: implementation, tests, documentation, and review.
3. **Use safe Codex defaults.** Prefer `codex exec --cd <repo> --sandbox workspace-write`.
4. **Record evidence.** Use `hermes codex-ops status/list/show` after enabling the plugin.
5. **Verify for real.** Run targeted tests and inspect the final diff before reporting success.

## Prerequisites

- Codex CLI installed and authenticated.
- For local telemetry and output hygiene, enable the bundled plugin:

```bash
hermes plugins enable codex-ops
```

Optional config in `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - codex-ops
  entries:
    codex-ops:
      compact_terminal_output: true
      compact_threshold_chars: 18000
      compact_head_lines: 80
      compact_tail_lines: 120
      compact_signal_limit: 80
      record_all_terminal: false
      allow_danger_full_access: false
```

Non-secret settings belong in `config.yaml`; do not put these in `.env`.

## Operating Protocol

### 1. Recon

Use `terminal`, `read_file`, and `search_files` to establish:

- current branch and dirty state;
- repo instruction files such as `AGENTS.md`;
- test command and package manager;
- relevant architecture seams;
- files likely to be touched.

Do not overwrite unrelated user changes. If the repo is dirty, name the dirty files and avoid them unless John explicitly includes them.

### 2. War-room plan

For broad or unfamiliar repositories, package context before launching Codex:

- Use `repomix-context-packaging` to create a narrow, secret-checked bundle when
  file count or token budget would otherwise push Codex into guesswork.
- Treat the bundle as orientation only; Codex must inspect live files before
  editing.
- For feature/bug-fix work, load `test-driven-development` and require a real
  failing test before production changes unless John explicitly chooses a spike.

Write the plan before running agents:

| Lane | Purpose | Output |
|---|---|---|
| Implementation | Build the smallest Hermes-native change. | Patch/diff. |
| Verification | Find realistic tests and edge cases. | Commands and expected evidence. |
| Review | Catch policy, safety, and maintenance risks. | Checklist or requested fixes. |
| Documentation | Capture operator workflow. | Skill/reference update. |

### 3. Codex invocation

Default shape:

```bash
codex exec --cd /path/to/repo --sandbox workspace-write "<self-contained prompt>"
```

Rules:

- Include repo path, branch/dirty-state constraints, and exact deliverables in the prompt.
- Tell Codex not to touch unrelated dirty files.
- Tell Codex to run tests and report real output.
- Avoid `--danger-full-access`, `--dangerously-bypass-approvals-and-sandbox`, `--sandbox danger-full-access`, `-s danger-full-access`, and `--yolo`; the `codex-ops` plugin blocks these by default.
- Use isolated worktrees for parallel agents that may write overlapping files.

### 4. Output hygiene

When `codex-ops` is enabled:

- long Codex/test/dev logs are compacted before entering model context;
- high-confidence secrets are redacted early;
- Codex terminal runs are recorded in a local SQLite ledger under the active Hermes profile;
- full output bodies are not stored by the plugin.

Operator commands:

```bash
hermes codex-ops status
hermes codex-ops list --limit 10
hermes codex-ops show <id>
hermes codex-ops prune --days 30
```

### 5. Verification

Before final response:

- run targeted tests or a syntax/import check;
- inspect `git diff --stat` and the relevant hunks;
- confirm no unrelated dirty files were modified;
- report exact commands and exit codes.

## OpenCode Integration Notes

The first Hermes-native integration wave ports patterns, not runtimes:

- telemetry and local run ledger;
- output compaction/snip behavior;
- log and secret sanitization;
- env/danger guardrails;
- multi-agent worktree discipline;
- skill-based mission-control workflow.

See `references/awesome-opencode-war-room.md` for the council mapping and deferred tracks.
