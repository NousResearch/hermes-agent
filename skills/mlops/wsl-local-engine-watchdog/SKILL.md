---
name: wsl-local-engine-watchdog
description: "WSL2 local engine watchdog — STRICT EXCLUSIVITY arbiter: only one local LLM engine runs at a time. Dynamic mmproj for ANY model. Agent prompts at switch time."
version: 7.0.0
author: Hermes Agent
tags: [wsl, llama.cpp, vllm, watchdog, vram, provider-switch, local-engine, systemd, arbiter, exclusivity, mmproj, vision, multimodal]
metadata:
  hermes:
    related_skills: [llama-cpp-wsl-ops]
---

# WSL Local Engine Watchdog (Strict Exclusivity + Dynamic mmproj)

A Python watchdog for WSL2 that **auto-starts and auto-stops local inference engines** based on which Hermes provider you select. **Enforces strict exclusivity — only one engine service is ever active at a time** to guarantee VRAM safety.

**Dynamic mmproj (vision):** The watchdog writes an `EnvironmentFile` (`~/scripts/mmproj_env.conf`) before starting the engine. If `/tmp/<provider>_mmproj` exists, it includes `--mmproj <path>` in the file; otherwise the file is empty/absent and the engine starts without vision. **This works for ANY provider, ANY model.**

---

## Agent Prompt Workflow

When this skill is loaded and the agent detects the user intends to switch to a local (WSL) provider:

1. **Check if the provider supports mmproj** — look up the provider in `local_engines.json`. If the provider isn't in the config but matches auto-detection patterns ("llama", "gemma", "qwen", "heretic", "local", "wsl"), assume it may have mmproj support.

2. **Use the `clarify` tool** to ask the user:
   > "Provider [name] is a local model. Would you like to load the vision mmproj? This uses ~1-2GB more VRAM."
   > Choices: ["Yes, load vision", "No, save VRAM"]

3. **Based on the answer, set the flag file via WSL:**
   - **Yes:** `wsl touch /tmp/<provider_name>_mmproj`
     - If the user has a custom mmproj path: `wsl sh -c 'echo "<path>" > /tmp/<provider_name>_mmproj'`
   - **No:** `wsl rm -f /tmp/<provider_name>_mmproj`

4. The user then switches providers. The watchdog reads the flag on its next 5s poll cycle.

---

## Behavior

- **Strict exclusivity** — Only one engine service is ever active
- **Dynamic mmproj** — `/tmp/<provider>_mmproj` flag + EnvironmentFile for ANY model
- **Auto-starts/stops** on provider switch (agent.log tail)
- **Auto-stops on app close** (Hermes.exe tasklist check)
- **Multi-session safe** — checks state.db, won't unload if gateway needs it
- **Idle timeout (15 min)** — safety-net fallback
- **Does NOT manage TTS/STT** — independent services

## Architecture

```
  Hermes agent.log ──► "Model switched in-place: ... (X) -> ... (llama-gemma-4)"
                             │
                        every 5s tail
                             ▼
  local-engine-watchdog (systemd on WSL)
     ├─ agent.log tail — THE trigger (ANY provider name)
     ├─ /tmp/<provider>_mmproj flag → writes ~/scripts/mmproj_env.conf
     │    ├─ EXISTS + path → MMPROJ_ARGS=--mmproj <path>
     │    └─ absent       → empty file (no vision, -1~2GB VRAM)
     ├─ ARBITER — stop every other running engine first
     ├─ tasklist.exe — app close → stop all
     ├─ state.db — multi-session guard
     └─ log mtime — idle timeout
             │
             ▼
  Per-engine systemd service via EnvironmentFile
  └── Same service unit, MMPROJ_ARGS injected dynamically
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
| `engine` | Yes | Human-readable engine name |
| `service` | Yes | systemd service unit name (with `.service`) |
| `mmproj` | No | Default mmproj GGUF path for this model |
| `log` | No | Server log path for idle timeout |

Any provider → any engine. The `mmproj` field provides the default path; users can override by writing a custom path into the flag file.

### 2. Systemd service with EnvironmentFile

`~/.config/systemd/user/llama-server.service`:
```ini
[Service]
EnvironmentFile=-/home/vibrationall/scripts/mmproj_env.conf
ExecStart=/path/to/llama-server \
  $MMPROJ_ARGS \
  ...other args...
```

The `-` prefix on `EnvironmentFile` means "optional" — if absent, `$MMPROJ_ARGS` is empty (no mmproj).

### 3. Watchdog script: `~/scripts/local_engine_watchdog.py`

Core mmproj logic:
```python
def write_mmproj_env(provider_name, cfg):
    flag = f"/tmp/{provider_name}_mmproj"
    mmproj_path = None
    if os.path.exists(flag):
        try:
            content = open(flag).read().strip()
            if content: mmproj_path = content
        except: pass
        if not mmproj_path:
            mmproj_path = cfg.get("mmproj", "")
    if mmproj_path:
        with open("~/scripts/mmproj_env.conf", "w") as f:
            f.write(f'MMPROJ_ARGS=--mmproj {mmproj_path} --no-mmproj-offload\n')
    else:
        try: os.remove("~/scripts/mmproj_env.conf")
        except: pass
```

### 4. mmproj toggle script: `~/scripts/toggle_mmproj.sh`

```bash
./toggle_mmproj.sh <provider> on                                       # enable with default path
./toggle_mmproj.sh <provider> off                                      # disable
./toggle_mmproj.sh <provider> /home/.../mmproj-MyModel-BF16.gguf       # enable with custom path
./toggle_mmproj.sh <provider> status                                   # check state
```

## Agent Commands Reference

```bash
# Enable mmproj for a model (ANY provider name)
wsl touch /tmp/llama-gemma-4_mmproj

# Enable with custom mmproj path
wsl sh -c 'echo "/home/vibrationall/ai/models/mmproj-MyModel.gguf" > /tmp/llama-my-model_mmproj'

# Disable mmproj
wsl rm -f /tmp/llama-gemma-4_mmproj

# Check if mmproj flag is set
wsl [ -f /tmp/llama-gemma-4_mmproj ] && echo ON || echo OFF

# View watchdog logs
wsl journalctl --user -u llama-watchdog.service -n 20 --no-pager
```

## Pitfalls

- **state.db does NOT track `/model` switches mid-TUI-session** — use agent.log tailing
- **SQLite over /mnt/c/ 9p** — file locking broken; copy DB to /tmp before querying
- **Cold load time** — 17GB GGUF ~36s; set Hermes gateway_timeout >180s
- **LD_LIBRARY_PATH is critical** — must be set in each service unit
- **exit.target must be masked** — `systemctl --user mask exit.target --now`
- **Hermes has NO `on_provider_change` plugin hook** — workaround: tail agent.log
- **Auto-detection is case-insensitive** — any provider name with "llama", "gemma", "qwen", "heretic", "local", or "wsl"
- **mmproj flag file is ephemeral** — lives in WSL /tmp; cleared on WSL restart. Re-set after reboot.
- **Set the flag BEFORE switching providers** — watchdog reads it on the next 5s cycle
- **$MMPROJ_ARGS must be unquoted in ExecStart** — systemd expands the variable; double-quoting makes it a single arg
