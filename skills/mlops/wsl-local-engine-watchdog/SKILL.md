---
name: wsl-local-engine-watchdog
description: "WSL2 local engine watchdog — STRICT EXCLUSIVITY arbiter: only one local LLM engine runs at a time. TRUE OFFLINE mmproj prompt: writes desktop notification, waits indefinitely, works with zero model dependency."
version: 8.0.0
author: Hermes Agent
tags: [wsl, llama.cpp, vllm, watchdog, vram, provider-switch, local-engine, systemd, arbiter, exclusivity, mmproj, vision, multimodal, offline]
metadata:
  hermes:
    related_skills: [llama-cpp-wsl-ops]
---

# WSL Local Engine Watchdog (Strict Exclusivity + True Offline mmproj Prompt)

A Python watchdog for WSL2 that **auto-starts and auto-stops local inference engines** based on which Hermes provider you select. **Enforces strict exclusivity — only one engine service is ever active at a time** to guarantee VRAM safety.

**Dynamic mmproj (vision):** The watchdog writes an `EnvironmentFile` before starting the engine. If `/tmp/<provider>_mmproj` exists, it injects `--mmproj <path>`. **Works for ANY provider, ANY model, no hardcoded names.**

---

## mmproj Prompt: Three Layers (No Failure)

The system tries three approaches in order, so you **always** get asked:

### Layer 1: Agent Prompt (if a capable model is active)
When this skill is loaded and the agent detects a switch to a local provider with mmproj:
- Agent uses `clarify` to ask: *"Load vision mmproj?"*
- Sets/clears the flag **before** the switch
- Fastest path — zero delay

### Layer 2: Desktop Notification + Infinite Wait (offline, no model needed)
If no agent was available to prompt (local-only session), the watchdog:

1. **Writes a file to your Windows desktop:** `mmproj_PENDING_<provider>.txt` with clear instructions
2. **Waits INDEFINITELY** — no 15s timeout, no model dependency, just waits
3. The engine **will not start** until you make a choice
4. You see the file on your desktop, read it, and run one of:

```bash
# Enable vision mmproj (uses default path from config)
wsl touch /tmp/<provider>_mmproj

# Enable with a custom mmproj path
wsl sh -c 'echo "/path/to/mmproj.gguf" > /tmp/<provider>_mmproj'

# Skip mmproj (saves preference — won't ask again for this provider)
wsl sh -c 'echo SKIP > /tmp/<provider>_mmproj'
```

5. The engine starts with your choice, the desktop file is removed
6. Next time: skip preference is remembered, no re-prompt

### Layer 3: Saved Skip Preference (no re-asking)
If you skip mmproj for a provider, a marker file is saved at `~/scripts/mmproj-skip/<provider>`. Next time you switch to that provider, the watchdog respects it silently.

**To reset a skip preference:**
```bash
wsl rm /home/vibrationall/scripts/mmproj-skip/llama-gemma-4
```

---

## Behavior

- **Strict exclusivity** — Only one engine service is ever active
- **Dynamic mmproj** via EnvironmentFile for ANY model
- **Three-layer prompt** — never miss the question
- **Infinite wait** — engine won't start until you decide
- **Skip memory** — won't re-ask for providers you've already declined
- **Auto-stops on app close** (Hermes.exe tasklist check)
- **Multi-session safe** — checks state.db before unloading
- **Idle timeout (15 min)** — safety-net fallback
- **Does NOT manage TTS/STT**

## Architecture

```
  Hermes agent.log ──► "Model switched in-place: ... -> ... (provider)"
                             │
                        every 5s tail
                             ▼
  LOCAL-ENGINE-WATCHDOG (systemd on WSL)
     │
     ├─ wait_for_mmproj_decision(provider, cfg)
     │    ├─ Saved skip exists? → skip silently
     │    ├─ Flag exists? → use it
     │    ├─ Agent set flag? → use it
     │    └─ NONE OF THE ABOVE →
     │         ├─ Write mmproj_PENDING_<provider>.txt → Windows DESKTOP
     │         ├─ Print to journalctl
     │         └─ WAIT FOREVER, polling /tmp/<provider>_mmproj
     │              ├─ touch ..._mmproj → start engine WITH vision
     │              ├─ echo SKIP > ..._mmproj → start without, save skip
     │              └─ (no action) → engine stays stopped, file visible
     │
     ├─ write_mmproj_env(path) → ~/scripts/mmproj_env.conf
     ├─ ARBITER: stop other engines → start target service
     ├─ tasklist.exe → app close
     └─ state.db → multi-session guard
```

## Components

### 1. Engine config: `~/scripts/local_engines.json`

```json
{
  "llama-qwen": {
    "engine": "llama.cpp",
    "service": "llama-server.service",
    "mmproj": "/home/vibrationall/ai/llama.cpp/models/gemma-4/mmproj-Gemma4-26B-A4B-QAT-Uncensored-HauhauCS-Balanced-BF16.gguf",
    "log": "/home/vibrationall/ai/llama.cpp/server.log"
  }
}
```

| Field | Required | Description |
|---|---|---|
| `engine` | Yes | Human-readable name |
| `service` | Yes | systemd service name (with `.service`) |
| `mmproj` | No | Default mmproj GGUF path |
| `log` | No | Server log path for idle timeout |

### 2. Systemd service with dynamic mmproj

```ini
[Service]
EnvironmentFile=-/home/vibrationall/scripts/mmproj_env.conf
ExecStart=/path/to/llama-server $MMPROJ_ARGS ...
```

The `-` prefix means "optional" — if absent, `$MMPROJ_ARGS` is empty.

### 3. Key logic in watchdog

```python
def wait_for_mmproj_decision(provider, cfg):
    """Wait INDEFINITELY. No timeout, no model needed."""
    if has_user_skipped(provider):
        return None  # saved preference
    if flag already exists:
        return extract_path(flag) or cfg["mmproj"]
    if not cfg.get("mmproj"):
        return None  # no mmproj available

    # Show the prompt
    write_desktop_notification(provider, cfg)
    while True:  # NO TIMEOUT
        time.sleep(1)
        if flag exists:
            content = read_flag()
            if content == "SKIP":
                save_skip(provider)
                remove_notification()
                return None
            return content or cfg["mmproj"]
```

## User Reference

```bash
# Enable mmproj for any provider (uses default path from config)
wsl touch /tmp/llama-gemma-4_mmproj

# Custom mmproj path
wsl sh -c 'echo "/home/ai/models/mmproj-MyModel.gguf" > /tmp/llama-gemma-4_mmproj'

# Skip (saves preference)
wsl sh -c 'echo SKIP > /tmp/llama-gemma-4_mmproj'

# Reset a saved skip preference
wsl rm /home/vibrationall/scripts/mmproj-skip/llama-gemma-4

# See what's pending
wsl ls /mnt/c/Users/MaJor/Desktop/mmproj_PENDING_*.txt

# Add mmproj after starting without (requires restart)
wsl touch /tmp/llama-gemma-4_mmproj && wsl systemctl --user restart llama-server.service

# Check watchdog logs
wsl journalctl --user -u llama-watchdog.service -n 30 --no-pager
```

## Pitfalls

- **state.db does NOT track mid-session switches** — use agent.log tailing
- **SQLite over /mnt/c/ 9p** — copy DB to /tmp before querying
- **Cold load time** — 17GB GGUF ~36s; set gateway_timeout >180s
- **LD_LIBRARY_PATH** — must be explicit in each service unit
- **exit.target must be masked** — `systemctl --user mask exit.target --now`
- **No `on_provider_change` hook** — workaround: tail agent.log
- **Auto-detection is case-insensitive** — any provider with "llama", "gemma", "qwen", "heretic", "local", or "wsl"
- **Flag file is ephemeral** — lives in /tmp; cleared on WSL restart
- **Skip preference is persistent** — lives in ~/scripts/mmproj-skip/
- **$MMPROJ_ARGS must be unquoted** in ExecStart
- **Engine won't start until you decide** — the prompt blocks the start
