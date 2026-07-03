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

## Pitfalls

- Engine starts **without** mmproj first. Agent asks in chat. If yes → engine restarts WITH mmproj (cold load time applies on restart)
- `clarify` works with ANY model — it's a tool, not a model capability
- Flag file is ephemeral — lives in /tmp; clears on WSL restart
- Agent asks EVERY TIME — no skip memory
