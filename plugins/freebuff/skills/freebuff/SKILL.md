---
name: freebuff
description: Freebuff free tier for Hermes AI and interactive TUI coding.
version: 0.2.0
author: zapabob
platforms: [linux, macos, windows]
metadata:
  hermes:
    category: devops
    tags: [coding-agent, freebuff, codebuff, terminal, model-provider]
---

# Freebuff Skill

Hermes bridge for [Codebuff Freebuff](https://freebuff.com/cli) — free
ad-supported models and terminal coding agent from
[CodebuffAI/codebuff](https://github.com/CodebuffAI/codebuff).

## When to Use

- Route **Hermes chat** through Freebuff's free model quota (`hermes freebuff connect`).
- Hand off **multi-file repo work** to Freebuff's dedicated TUI.
- You want DeepSeek / Kimi / MiniMax without a paid Codebuff API key.

## Prerequisites

- Node.js 18+ and npm (TUI path only).
- GitHub login once via `hermes freebuff run` (stores token in `~/.config/manicode/credentials.json`).
- Python 3.13+ for the local OpenAI proxy (`freebuff2api`, installed automatically by `connect`).
- `plugins.enabled` includes `freebuff`; `tools.cli.enabled` includes `freebuff`.

## How to Run

**Hermes AI (OpenAI-compatible proxy):**

```text
hermes freebuff connect --apply-model
hermes freebuff doctor
```

**Interactive TUI:**

```text
hermes freebuff run /path/to/project
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `hermes freebuff connect` | Sync token, start proxy, set `model.provider=freebuff` |
| `hermes freebuff proxy status` | Proxy health at `http://127.0.0.1:8765/v1` |
| `hermes freebuff run .` | Start interactive Freebuff TUI |
| `freebuff_status` / `freebuff_doctor` | Agent tools (after toolset enabled) |

Default Hermes model after connect: `deepseek/deepseek-v4-flash`.

## Procedure

### A. Use Freebuff as Hermes inference backend

1. Run `hermes freebuff connect` (or `/freebuff connect` in gateway).
2. Confirm `hermes freebuff doctor` — upstream token set, proxy running, `model.provider=freebuff`.
3. Restart Hermes if it was already running so the provider profile reloads.
4. Chat normally; requests go to local proxy → Codebuff free tier.

### B. Use Freebuff TUI for coding

1. `freebuff_doctor` → fix Node/npm or `hermes freebuff install`.
2. `freebuff_launch` with `workdir` (keep `dry_run=true` unless in real TTY).
3. Run returned command via `terminal` or `hermes freebuff run`.

## Pitfalls

- The proxy ([freebuff2api](https://github.com/XxxXTeam/freebuff2api)) is **community/unofficial** — not Codebuff official API.
- Respect Codebuff/Freebuff terms; do not abuse shared free quota.
- Freebuff TUI shows **terminal ads** — expected on free tier.
- Windows: run TUI in Windows Terminal (not embedded IDE terminal).
- If proxy fails, read `~/.hermes/logs/freebuff-proxy.log`.

## Verification

```text
hermes freebuff status
hermes freebuff proxy status
```

Expect `upstream_token_set: true`, `proxy.running: true`, `model_provider: freebuff`.
