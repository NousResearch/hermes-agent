# Zapabob Hermes Fork — Layout Guide

This directory documents how **this repository** differs from
[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).
It does **not** replace upstream code. Official Hermes stays in the repo root
(`run_agent.py`, `hermes_cli/`, `gateway/`, `apps/`, …). Fork-only behaviour
lives at the edges: plugins, merge overlays, Windows ops scripts, and local
operator automation.

## Directory map

| Path | Purpose |
|------|---------|
| [`harness/`](harness/README.md) | Upstream merge policy, overlays, and sync entry points (`scripts/merge_tools/`) |
| [`extensions/`](extensions/README.md) | Fork-owned plugins, core tool deltas, and optional skills |
| [`operations/`](operations/README.md) | Windows stack scripts, cron helpers, Tailscale/ngrok, daily automation |
| [`local-workspace/`](local-workspace/README.md) | Gitignored root-level scratch files (kept on disk, never committed) |

## Rules for contributors and agents

1. **Upstream is authoritative** for the agent core, gateway loop, and security fixes.
2. **Never delete** harness files under `scripts/merge_tools/` or vendor pins used by evolution tools.
3. **Prefer plugins + skills** over editing `run_agent.py` / `model_tools.py` when adding capability.
4. **Do not commit** build output, logs, media scratch, secrets, or `_docs/` implementation logs.
5. Read [`AGENTS.md`](AGENTS.md) before changing fork-specific areas.

## Quick commands

```powershell
# Policy dry-run before merging upstream
py -3 scripts\sync_all.py --dry-run --allow-preflight-blockers

# Restart Hermes stack (llama excluded by default)
powershell -ExecutionPolicy Bypass -File scripts\windows\restart-hermes-stack.ps1

# Desktop rebuild
py -3 -m hermes_cli.main desktop --build-only --force-build
```

## Related docs

- Root [`README.md`](../README.md) — fork feature summary for humans
- Root [`AGENTS.md`](../AGENTS.md) — full Hermes development guide (upstream + fork notes)
