---
name: grimdall-guard
description: Install and configure Grimdall policy checkpoint plugin.
version: 1.0.0
author: Aniket (@grimdalltech, Grimdall Tech)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [security, policy, guardrails, checkpoint, egress, secrets]
    category: security
---

# Grimdall Guard

Grimdall is a deterministic policy checkpoint that hooks the `pre_tool_call`
lifecycle to block or log destructive shell commands, secret reads, and
non-allowlisted egress before they execute. It ships as a standalone plugin
(`github.com/grimdalltech/hermes-grimdall`); this skill walks through
installing and configuring it.

## When to Use

- The operator wants destructive shell (`rm -rf` of absolute/home/glob paths,
  `DROP TABLE`, fork bombs) blocked before execution.
- The operator wants secret reads (`~/.ssh`, `~/.aws`, `.env`, `*.pem`)
  blocked or logged.
- The operator wants a network egress allowlist enforced on `curl`, `wget`,
  `ssh`, `scp`, `git`, and the `web_extract` / `web_search` tools.
- The operator wants a signed audit receipt for every policy hit.

## Prerequisites

- The plugin is installed either as a pip package (`hermes-grimdall`) or as a
  drop-in directory under `~/.hermes/plugins/grimdall/`.
- The plugin is enabled (`hermes plugins enable grimdall`).
- For enforce-mode enforcement, the operator has configured the egress
  allowlist, since an empty allowlist blocks all non-local network egress.

## How to Run

1. Install the plugin (pip entry point or directory).
2. Enable it: `hermes plugins enable grimdall`.
3. Choose a mode in `~/.hermes/config.yaml`.
4. Restart the Hermes session so the hook loads.

## Quick Reference

| Setting (`plugins.entries.grimdall.settings`) | Purpose | Default |
|---|---|---|
| `mode` | `shadow` (log + pass through) or `enforce` (block) | `shadow` |
| `egress_allowlist` | Hosts (and subdomains) allowed to receive egress | `[]` (loopback always allowed) |
| `signing_key` | Path to the Ed25519 signing key | auto-generated |
| `receipt_log` | JSONL path for signed receipts | `~/.hermes/grimdall/receipts.jsonl` |

Environment overrides (testing): `GRIMDALL_MODE`, `GRIMDALL_ALLOWLIST` (CSV),
`GRIMDALL_SIGNING_KEY`, `GRIMDALL_RECEIPT_LOG`.

## Procedure

### 1. Install

```bash
pip install hermes-grimdall
hermes plugins enable grimdall
```

Or, directory install:

```bash
mkdir -p ~/.hermes/plugins/grimdall
cp -r plugin.yaml hermes_grimdall ~/.hermes/plugins/grimdall/
```

### 2. Configure mode

Add to `~/.hermes/config.yaml`:

```yaml
plugins:
  entries:
    grimdall:
      settings:
        mode: shadow          # or enforce
        egress_allowlist:
          - github.com
          - pypi.org
```

Start with `shadow` to observe what would be blocked before switching to
`enforce`.

### 3. Verify enforcement

Run `hermes plugins show grimdall` to confirm the `pre_tool_call` hook is
registered, then review `receipts.jsonl` for signed hit records.

## Pitfalls

- **Heuristic, not a boundary.** Grimdall scans command strings; it is not an
  OS sandbox. A determined model can evade string matching. Pair it with a
  terminal backend or whole-process sandbox for a real boundary.
- **`enforce` blocks before the approval gate.** Grimdall's block fires at the
  `pre_tool_call` hook regardless of `--yolo` / `approvals.mode: off`.
- **Empty allowlist blocks all egress.** In `enforce` mode, an empty
  `egress_allowlist` denies every external host; configure it before enabling
  enforce, or tooling that needs the network will fail.
- **Shadow mode passes through.** Receipts are logged but nothing is stopped;
  do not read "logged" as "blocked".

## Verification

- `hermes plugins list` shows `grimdall` with `pre_tool_call` hook.
- A destructive command in `enforce` mode returns a `grimdall-receipt` block
  message instead of executing.
- `~/.hermes/grimdall/receipts.jsonl` contains one signed JSON record per hit
  with a non-empty `sig` field.
