---
name: wsl-local-engine-watchdog
description: "WSL2 local engine watchdog — STRICT EXCLUSIVITY arbiter. after-start mmproj prompt: engine loads fast, agent asks in chat, restarts with mmproj if user says yes."
version: 11.0.0
author: Hermes Agent
tags: [wsl, watchdog, vram, provider-switch, local-engine, systemd, arbiter, exclusivity, mmproj, vision]
metadata:
  hermes:
    related_skills: [llama-cpp-wsl-ops]
---

# WSL Local Engine Watchdog (Strict Exclusivity)

A Python watchdog for WSL2 that **auto-starts and auto-stops local inference engines** based on which Hermes provider you select. **Strict exclusivity — only one engine at a time.**

**After-start mmproj prompt:** Engine starts immediately without mmproj (fast path). The agent then asks in the Hermes desktop chat. If you want vision, the watchdog **restarts** the engine with `--mmproj` flags.

---

## How The mmproj Prompt Actually Works

The timing problem: the watchdog detects the provider switch in `agent.log` and starts the engine **before** the agent has loaded (the model is still loading). So the agent can't ask before the engine starts.

The solution is an **after-start prompt**:

```
1. You switch provider in Hermes desktop
              │
              ▼
2. Watchdog starts engine IMMEDIATELY
   (no mmproj, you can use the model)
              │
              ▼
3. Agent loads skill, ASKS IN CHAT:
   "Load vision mmproj? [Yes] [No]"
              │
         ┌────┴────┐
         ▼         ▼
       "Yes"     "No"
         │         │
         ▼         ▼
    Agent sets  Agent does
    WSL flag    nothing
         │         │
         └──┬──────┘
            ▼
4. Watchdog detects flag on next poll
   → RESTARTS engine WITH mmproj
   (takes ~36s but now vision-enabled)

   If "No": engine keeps running
   without mmproj, no restart needed.
```

**Next time you switch to the same model: the agent asks again.** Every single time.

---

## Behavior

- **Start fast, restart on demand** — engine loads without mmproj, restarts if you say yes
- **Agent asks in Hermes desktop chat** every time via `clarify`
- **Works with ANY model** — `clarify` is a built-in tool
- **Strict exclusivity** — only one engine active at a time
- **Dynamic mmproj** via `EnvironmentFile` + `$MMPROJ_ARGS`
- **Auto-stop on app close** — detects Hermes.exe exit, stops all engines
- **Multi-session safe** — checks state.db before unloading
- **Idle timeout (15 min)** — safety-net fallback

## Components

### Engine config: `~/scripts/local_engines.json`

```json
{
  "llama-qwen": {
    "engine": "local",
    "service": "llama-server.service",
    "mmproj": "/home/vibrationall/ai/llama.cpp/models/gemma-4/mmproj-Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf",
    "log": "/home/vibrationall/ai/llama.cpp/server.log"
  }
}
```

### Systemd service

```ini
[Service]
EnvironmentFile=-/home/vibrationall/scripts/mmproj_env.conf
ExecStart=/path/to/llama-server $MMPROJ_ARGS ...
```

## Quick Reference

```bash
# Manual override (set before switching, or after for restart):
wsl touch /tmp/llama-gemma-4_mmproj
wsl sh -c 'echo "/path/mmproj.gguf" > /tmp/llama-gemma-4_mmproj'

# Watchdog logs
wsl journalctl --user -u llama-watchdog.service -n 20 --no-pager
```

## Future / Ideal Implementation

This watchdog exists because Hermes has no `on_provider_switch` plugin hook. **Every feature here is a workaround** for a capability that should live in the Hermes core:

| Problem | Workaround (current watchdog) | Ideal (single core hook) |
|---|---|---|
| **Multiple engines loaded at once** → OOM | External ARBITER polls and stops competing engines | Core enforces one-engine-at-a-time exclusivity natively |
| **Engine stays in VRAM after app close** | WSL polls `tasklist.exe` for `Hermes.exe` every 5s | Core terminates engines on Hermes exit automatically |
| **Slow start on model selection** | Tails `agent.log` every 5s for provider-switch events | Core fires `on_provider_switch` synchronously — no polling delay |
| **mmproj can't be configured before load** | Starts without mmproj, agent asks after, restarts (~36s cold load) | Agent asks BEFORE engine loads — first start is correct |

**On 24 GB consumer GPUs (RTX 4090), every megabyte counts.** A single orphaned engine leaking VRAM, a second model loading while one is already resident, or an unnecessary mmproj file consuming ~2 GB can mean the difference between a working session and a hard OOM freeze. Hermes has the architecture to handle this intelligently — the agent is already in the loop, the skill system can prompt at the right time, and the gateway can manage lifecycle. It just needs the plugin hook to tie it all together.

**No platform handles local inference engines properly on consumer hardware.** LM Studio, Ollama, text-generation-webui — none of them enforce exclusivity, clean up on close, auto-start on selection, or ask about optional VRAM-heavy components at load time. They either always-load (wasting VRAM), never-cleanup (leaking VRAM), or require manual config edits (poor UX).

**A single `on_provider_switch` hook in the Hermes core would put this platform lightyears ahead** of every other local inference frontend for consumer hardware.

## Pitfalls

- Engine starts **without** mmproj first. Agent asks in chat. If yes → engine restarts WITH mmproj (cold load time applies on restart)
- `clarify` works with ANY model — it's a tool, not a model capability
- Flag file is ephemeral — lives in /tmp; clears on WSL restart
- Agent asks EVERY TIME — no skip memory
