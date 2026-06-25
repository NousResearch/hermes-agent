---
name: vrchat-autonomy
description: Autonomous VRChat chatbox, loop, and movement.
version: 0.1.0
author: Hermes Agent
platforms: [windows, linux, macos]
metadata:
  hermes:
    tags: [vrchat, osc, autonomy]
    category: autonomous-ai-agents
---

# VRChat Autonomy Skill

Drive the `vrchat-autonomy` plugin: autonomous ChatBox speech, queued conversation ticks, and OSC movement.

## When to Use

- Operator wants はくあ (or another persona) to speak in VRChat ChatBox autonomously
- Background loop should consume queued `textBox` / operator observations and reply
- Safe movement pulses (`forward`, `stop`, etc.) via official `/input/*` OSC

## Prerequisites

- VRChat running with OSC enabled (Action Menu)
- `uv pip install 'hermes-agent[vrchat]'` for `python-osc`
- Plugin enabled: `plugins.enabled` includes `vrchat-autonomy`
- Profile at `~/.hermes/config/vrchat-autonomy-profile.json`
- VOICEVOX Engine when `allow_voice=true`

## How to Run

```bash
hermes vrchat-autonomy setup
hermes vrchat-autonomy doctor
hermes vrchat-autonomy setup --arm-live   # または arm-live 単体
hermes vrchat-autonomy start              # background loop
hermes vrchat-autonomy chatbox "こんにちは"
hermes vrchat-autonomy move forward
hermes vrchat-autonomy stop
hermes vrchat-autonomy neuro status
hermes vrchat-autonomy neuro bootstrap --context "bridge ready"
py -3 scripts/vrchat_neuro_bridge.py --profile ~/.hermes/config/vrchat-autonomy-profile.json
```

Neuro API settings in `config.yaml` under `plugins.vrchat-autonomy`:

- `neuro_game` — Neuro API game name (default `Hermes VRChat`)
- `neuro_ws_url` — websocket URL (default `ws://127.0.0.1:8000`)

Enable toolset `vrchat_autonomy` for plugin tools, or use existing core `vrchat` toolset for low-level control.

## Quick Reference

| Command | Purpose |
|---------|---------|
| `setup` | Enable plugin + write profile (dry-run default) |
| `doctor` | Readiness + preflight bundle |
| `tick` | One LLM decision + actuation cycle |
| `start` / `stop` | Background worker |
| `chatbox` / `move` | Direct live actuation (profile gated) |
| `neuro status` | neuro-sdk vendor + action catalog |
| `neuro vendor` | Submodule clone status + init command |
| `neuro bootstrap` | Build Neuro websocket handshake messages |
| `neuro bridge` | Run `scripts/vrchat_neuro_bridge.py` |

Live actuation requires `dry_run=false`, non-observe mode, capability flags, and exact ACK:

`I understand this sends OSC and/or audio to VRChat.`

## Procedure

1. Run `hermes vrchat-autonomy setup` (keeps dry-run safe defaults).
2. Confirm `hermes vrchat-autonomy doctor` shows VRChat + VOICEVOX ready.
3. Queue observations with `vrchat_observation_ingest` or `vrchat_autonomy_plugin_enqueue`.
4. Run `tick` manually or `start` for periodic autonomous conversation.
5. Use `move` only in private/trusted instances with `allow_movement=true`.

## Pitfalls

- Public instances block movement even when enabled in profile.
- Core autonomy schema does not auto-execute LLM `movement` keys — use `hermes vrchat-autonomy move` or avatar actions.
- ChatBox max 144 chars / 9 lines (VRChat limit).

## Verification

```bash
hermes vrchat-autonomy status
scripts/run_tests.sh tests/plugins/test_vrchat_autonomy_plugin.py -q
```
