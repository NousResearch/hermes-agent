---
name: prometheus-avatar
description: Give any Hermes agent an animated Avatar body with TTS, emotion-aware expressions, and a public marketplace of skins, voices, and personas. Backed by the Prometheus MCP server. First Avatar skill for Hermes Agent (Live2D engine today, 3D engine on the roadmap).
version: 1.0.0
author: jc-myths
license: MIT
metadata:
  hermes:
    tags: [avatar, live2d, 3d-ready, animation, tts, voice, character, creative, mcp]
    related_skills: [blender-mcp, meme-generation]
    category: creative
---

# Prometheus Avatar

Give your Hermes agent a visible, animated body. Voice comes out, lips move, expressions follow emotion. Users can equip skins, voices, personas, and effects from a public marketplace.

## When to Use

- User wants their agent to **show a face** (VTuber, streamer persona, on-screen NPC)
- User wants **spoken output** with lip-sync, not just text
- User wants emotion-aware character reactions
- User wants a **persistent agent identity** (avatar, voice, and personality bundled)
- User says "make my agent visual", "give my agent a body", "VTuber", "avatar for Hermes"

## Setup

Prometheus ships a public MCP server. Wire it into Hermes config via `stdio`:

```json
{
  "mcpServers": {
    "prometheus": {
      "command": "npx",
      "args": ["-y", "@prometheusavatar/mcp-server"],
      "env": {
        "GEMINI_API_KEY": "optional-for-asset-generation"
      }
    }
  }
}
```

**Environment variables**:

| Variable | Required | Notes |
|----------|:--------:|-------|
| `GEMINI_API_KEY` | Only for `generate_asset` | Free key at https://ai.google.dev |
| `PROMETHEUS_API_URL` | No | Defaults to `https://prometheus.mythslabs.ai` |
| `PROMETHEUS_API_KEY` | No | Required only for authenticated creator operations |

No key is needed for read-only marketplace operations (`list_marketplace`, `equip_asset`, `speak`).

## Available Tools (via MCP)

| Tool | Description |
|------|-------------|
| `create_avatar` | Initialize an avatar instance with skeleton, voice, and persona |
| `list_marketplace` | Browse marketplace assets by category |
| `equip_asset` | Equip a skin, voice, persona, or effect |
| `speak` | Make the avatar speak text with TTS and lip-sync |
| `get_avatar_status` | Fetch current avatar state and equipped assets |
| `share_avatar` | Generate a shareable embed URL |
| `generate_asset` | AI-generate a new asset from a text prompt (requires `GEMINI_API_KEY`) |

## Available Renderers

| Renderer | Status | Notes |
|---------|--------|-------|
| **Live2D** | ✅ Live | 9 built-in skeletons, each with expressions and motions |
| **3D** | 🛠 On roadmap | Same MCP tool surface, swappable at the render layer |

## Procedure

### Minimum viable flow

1. **Create** an avatar:
   ```
   create_avatar(skeleton="haru", voice="<voice_id>", persona="<persona_name>")
   ```
   Returns an embed URL you can hand to the user.

2. **Speak** on each agent turn:
   ```
   speak(text="<agent reply>", emotion="auto")
   ```
   `emotion="auto"` runs sentiment analysis and triggers the matching expression. You can also pass `"happy"`, `"sad"`, `"thinking"`, `"surprised"`, or `"angry"` explicitly.

3. **Discover and equip** assets:
   ```
   list_marketplace(category="voices")       # or "skins", "personas", "effects"
   equip_asset(asset_id="<id>", slot="voice")
   ```

### Share and embed

```
share_avatar()  # returns a public URL + iframe embed code
```

Hand the URL to the user and they can drop it into OBS, Discord, a website, or any iframe-capable surface.

## Examples

**"Make my Hermes agent a VTuber that reacts to Discord chat"**

1. `list_marketplace(category="voices")` → pick a voice_id
2. `create_avatar(skeleton="nito", voice="<voice_id>", persona="streamer_energetic")`
3. On every Discord message, call `speak(text, emotion="auto")`
4. `share_avatar()` → drop the URL into OBS as a browser source

**"Give my coding agent a face that shows when it's stuck"**

1. `create_avatar(skeleton="haru", voice="<voice_id>", persona="pair_programmer")`
2. On `tool_error`, call `speak(text="Hmm, let me try another angle.", emotion="thinking")`
3. On `test_passed`, call `speak(text="Got it!", emotion="happy")`

**"Build a Japanese-speaking tutor NPC"**

1. `list_marketplace(category="voices")` → filter for a Japanese voice
2. `create_avatar(skeleton="haru", voice="<japanese_voice_id>", persona="tutor_gentle")`
3. `speak(text="<japanese text>")`. Pronunciation follows the voice's language config.

> **Tip**: Example voice and persona IDs are placeholders. Always call `list_marketplace` first to get live IDs.

## Pitfalls

- **WebGL required for rendering.** The avatar renders in a browser or Electron surface. For TTS-only output without a visible avatar, call `speak` and skip `share_avatar`.
- **First load is slow.** Live2D model plus WebGL boot takes about 2-5 seconds on cold start. Cache the first render on a hidden iframe if latency matters.
- **Voice immutability.** Each voice has a fixed speaker ID. Swapping voice mid-turn isn't supported. Call `equip_asset` between turns instead.
- **Emotion is not motion.** `emotion` drives facial expression. Full-body motion (wave, bow, dance) uses a `motion_id` from the skeleton's motion set, not `emotion`.
- **China network.** Prometheus auto-routes through a CF Worker relay for China mainland. Other regions hit the origin directly, no client config needed.
- **`generate_asset` is opt-in.** It calls Google Gemini and needs your own `GEMINI_API_KEY`. The other six tools work with zero keys.

## Links

- MCP Server: [`@prometheusavatar/mcp-server`](https://www.npmjs.com/package/@prometheusavatar/mcp-server)
- SDK: [`@prometheusavatar/core`](https://www.npmjs.com/package/@prometheusavatar/core)
- Live demo: https://prometheus.mythslabs.ai
- GitHub: https://github.com/myths-labs/prometheus-avatar
