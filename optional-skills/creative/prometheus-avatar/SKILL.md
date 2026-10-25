---
name: prometheus-avatar
description: Give an agent an animated body with speech and expressions.
version: 1.1.0
author: JC (@jc-myths), Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [avatar, live2d, animation, tts, voice, character, creative, mcp]
    related_skills: [blender-mcp, meme-generation]
    category: creative
---

# Prometheus Avatar Skill

Renders a Live2D character that speaks with lip-sync and shifts facial expression
with emotion, driven through the Prometheus MCP server. Skins, voices, and personas
are equipped from a public marketplace. It does not render 3D characters, and every
avatar operation requires network access to the Prometheus API.

## When to Use

- The user wants the agent to show a face (VTuber, streamer persona, on-screen NPC).
- The user wants spoken output with lip-sync rather than text alone.
- The user wants emotion-driven character reactions.
- The user wants a persistent agent identity: body, voice, and personality together.
- The user says "give my agent a body", "make my agent visual", or "VTuber".

## Prerequisites

This skill depends on the **`prometheus` MCP server**
([`@prometheusavatar/mcp-server`](https://www.npmjs.com/package/@prometheusavatar/mcp-server)).
Register it with `hermes mcp add`, or add it to `config.yaml`:

```yaml
mcp_servers:
  prometheus:
    command: "npx"
    args: ["-y", "@prometheusavatar/mcp-server"]
    env:
      PROMETHEUS_API_KEY: "pak_..."
      GEMINI_API_KEY: ""        # only for generate_asset
```

| Variable | Required for | Notes |
|----------|--------------|-------|
| `PROMETHEUS_API_KEY` | 6 of 10 tools (see Quick Reference) | Agent key (`pak_...`) from https://prometheus.mythslabs.ai/settings/agent-keys — shown once |
| `GEMINI_API_KEY` | `generate_asset` only | Free key at https://ai.google.dev |
| `PROMETHEUS_API_URL` | never | Defaults to `https://prometheus.mythslabs.ai` |

Rendering needs a WebGL-capable surface (browser, Electron, OBS browser source).
Headless use is limited to the key-free tools.

## How to Run

1. Confirm the server is reachable: `hermes mcp test prometheus`.
2. Browse assets with `list_marketplace` (no key needed).
3. Create the avatar with `create_avatar`, then drive it with `speak`.

## Quick Reference

| Tool | Needs `PROMETHEUS_API_KEY` | Description |
|------|:--:|-------------|
| `create_avatar` | yes | Create an avatar instance; returns an avatar ID and embed URL |
| `set_avatar_state` | yes | Set animation state and emotion directly |
| `equip_asset` | yes | Equip or unequip a skin, voice, persona, or effect |
| `get_avatar_status` | yes | Fetch current state and equipped assets |
| `share_avatar` | yes | Return a public share URL and embed code |
| `speak` | yes | Speak text with TTS and lip-sync |
| `list_marketplace` | no | Browse marketplace assets by category |
| `update_asset` | no | Edit a listing you own |
| `generate_image_pro` | no | Generate character art (BYOK key, free quota, or credits) |
| `generate_asset` | no | Generate a new asset from a prompt (needs `GEMINI_API_KEY`) |

Renderer: **Live2D** today, with 9 built-in models. A 3D renderer is on the
roadmap behind the same tool surface; do not promise it to users yet.

## Procedure

### Minimum viable flow

1. Create the avatar. `model` accepts `haru` (default), `koharu`, or a full
   `model3.json` URL. `voice` is a fixed TTS voice name, not a marketplace ID:
   `Kore`, `Aoede`, `Leda`, `Despina`, `Puck`, `Charon`, `Fenrir`, `Zephyr`.

   ```
   create_avatar(name="Aria", model="haru", voice="Kore", persona="<system prompt>")
   ```

   Returns an avatar ID and an embed URL you can hand to the user.

2. Speak on each agent turn:

   ```
   speak(text="<agent reply>", emotion="auto")
   ```

   `emotion="auto"` picks the expression from sentiment. You can also pass
   `happy`, `sad`, `thinking`, `surprised`, or `angry` explicitly.

3. Discover and equip marketplace assets:

   ```
   list_marketplace(category="voices")     # or skins, personas, effects
   equip_asset(asset_id="<id>", action="equip")
   ```

4. Share it:

   ```
   share_avatar()
   ```

   Returns a public URL plus embed code for OBS, Discord, or any iframe surface.

### Example: VTuber reacting to chat

1. `create_avatar(name="Nova", model="koharu", voice="Aoede", persona="upbeat streamer")`
2. On each incoming message, `speak(text, emotion="auto")`
3. `share_avatar()`, then add the URL to OBS as a browser source

### Example: coding agent that shows when it is stuck

1. `create_avatar(name="Pair", model="haru", voice="Charon", persona="pair programmer")`
2. On tool error, `speak(text="Let me try another angle.", emotion="thinking")`
3. On tests passing, `speak(text="Got it.", emotion="happy")`

## Pitfalls

- **Most tools need a key.** Only `list_marketplace`, `update_asset`,
  `generate_image_pro`, and `generate_asset` work without `PROMETHEUS_API_KEY`.
  `create_avatar`, `speak`, and `equip_asset` all require it.
- **`model`, not `skeleton`.** The parameter is `model`, and only `haru`,
  `koharu`, or a `model3.json` URL are accepted.
- **`voice` is an enum, not a marketplace ID.** Marketplace voices are applied
  with `equip_asset`, not through the `create_avatar` `voice` parameter.
- **WebGL required to see anything.** For audio-only output, call `speak` and
  skip `share_avatar`.
- **First load is slow.** Live2D plus WebGL cold start takes roughly 2-5 seconds.
- **Emotion is not motion.** `emotion` drives the face. Full-body motion uses a
  motion ID from the model's motion set.
- **`generate_asset` is opt-in** and calls Google Gemini with your own key.

## Verification

```bash
hermes mcp test prometheus
```

Expect the 10 tools above to be listed. Then, with no key configured:

```
list_marketplace(category="skins", limit=3)
```

This must return assets. If it fails, the server is not registered — check that
`config.yaml` uses `mcp_servers` (snake_case), not `mcpServers`.

With `PROMETHEUS_API_KEY` set, `create_avatar` must return an avatar ID and an
embed URL that loads in a browser.

## Links

- MCP server: [`@prometheusavatar/mcp-server`](https://www.npmjs.com/package/@prometheusavatar/mcp-server)
- SDK: [`@prometheusavatar/core`](https://www.npmjs.com/package/@prometheusavatar/core)
- Live demo: https://prometheus.mythslabs.ai
- Source: https://github.com/myths-labs/prometheus-avatar
