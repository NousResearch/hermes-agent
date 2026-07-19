# Agent Harness (Hypura)

Hermes Agent talks to a local **Hypura harness daemon** for scavenging, wisdom,
evolution hooks, VOICEVOX speak, and OSC. This directory is the **human + AI
navigation entry** for that runtime path.

Merge/upstream sync lives under [`../harness/`](../harness/) — different topic.

## Quick start

```powershell
cd "C:\Users\downl\Documents\New project\hermes-agent"
py -3 -m hermes_cli.main harness status
py -3 -m hermes_cli.main harness start
# Enable toolset "harness" for the active platform, then use harness_* tools
```

Default URL: `http://127.0.0.1:18794` — health: `/health`.

## Pointers

| What | Where |
|------|--------|
| Agent rules | [`AGENTS.md`](AGENTS.md) |
| Daemon script | `vendor/openclaw-mirror/extensions/hypura-harness/scripts/harness_daemon.py` |
| OpenClaw extension README | `vendor/openclaw-mirror/extensions/hypura-harness/README.md` |
| CLI | `hermes_cli/harness.py` → `hermes harness …` |
| Model tools | `tools/harness_tools.py` |
| HTTP client | `tools/openclaw/harness_client.py` |
| Merge harness (not this) | [`../harness/README.md`](../harness/README.md) |
| Plugin / VRChat rules | [`../extensions/AGENTS.md`](../extensions/AGENTS.md) |

## Why this folder exists

Runtime code stays in `vendor/` and `tools/` so imports and the fixed daemon
relative path in `hermes_cli/harness.py` keep working. Agents get a clear
fork-local guide without renaming the merge-oriented `fork/harness/` tree.
