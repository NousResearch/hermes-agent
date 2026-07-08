---
name: codex-operations
description: Run Codex agents safely and observably with the codex-ops plugin.
version: 0.1.0
author: NousResearch
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: autonomous-ai-agents
    plugin: codex-ops
---

# Codex Operations

Use this skill when running Codex agents from Hermes, especially for multi-agent implementation/review lanes.

This plugin-provided skill mirrors the bundled `autonomous-ai-agents/codex-operations` skill so packaged installs can load `codex-ops:codex-operations` directly after enabling the plugin.

## Enable the plugin

```bash
hermes plugins enable codex-ops
hermes codex-ops status
```

Optional non-secret settings in `config.yaml`:

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

## Operating rules

1. Start with a clean git status or explicitly name unrelated dirty files.
2. Give each Codex lane a bounded, self-contained prompt.
3. Prefer bounded workspace access:

```bash
codex exec --cd /path/to/repo --sandbox workspace-write "<task>"
```

4. Avoid `--danger-full-access`, `--dangerously-bypass-approvals-and-sandbox`, `--sandbox danger-full-access`, `-s danger-full-access`, and `--yolo`; `codex-ops` blocks these by default for Hermes terminal calls.
5. Review diffs and run targeted tests before reporting success.

## Evidence commands

```bash
hermes codex-ops list --limit 10
hermes codex-ops show <id>
hermes codex-ops prune --days 30
```

The ledger stores redacted command summaries, hashes, timings, status, and short signal summaries. It does **not** store raw prompt bodies or full terminal output.
