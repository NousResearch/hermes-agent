# Tool Matrix — Skills + Toolsets per Role

Maps each role archetype to the Hermes skills it should `always_load` and the
toolsets it needs. Only references skills that ship in the public hermes-agent
repository (under `skills/` or `optional-skills/`). External APIs and CLIs are
called from the terminal toolset; they don't appear in `always_load`.

## Hermes skills relevant to video production

### Visual / rendering skills (`hermes-agent/skills/creative/`)

| Skill | What it does | Best fit for |
|-------|--------------|--------------|
| `ascii-video` | Production pipeline for ASCII art video — generative, audio-reactive, video-to-ASCII | Renderer for ASCII / terminal / retro pixel content; cinematographer for ASCII projects |
| `ascii-art` | Static ASCII art generation | Concept artist for ASCII style frames; secondary tool for ASCII renderer |
| `manim-video` | Manim CE animations — math, algorithms, 3Blue1Brown-style explainers | Renderer for math, algorithm walkthroughs, technical concept explainers |
| `p5js` | p5.js sketches — generative art, shaders, interactive, 3D | Renderer for generative art, particle systems, organic motion, web-canvas content |
| `comfyui` | Generate images, video, audio with ComfyUI workflows (image-to-image, image-to-video, etc.) | image-generator, image-to-video-generator, or general renderer for AI-generated content |
| `touchdesigner-mcp` | Control a running TouchDesigner instance — real-time visuals, audio-reactive installation art, VJ | Renderer for real-time/audio-reactive content; installation art; live performance |
| `blender-mcp` *(optional)* | Control Blender 4.3+ via MCP — 3D modeling, animation, rendering | Renderer for 3D scenes, photoreal environments, character animation |
| `pixel-art` | Pixel art with era palettes (NES, Game Boy, PICO-8) | Renderer for retro game aesthetic; concept artist for pixel-style frames |
| `baoyu-comic` | Knowledge-comic generation (educational, biography, tutorial) | Renderer for comic-style narrative; explainer in panel form |
| `baoyu-infographic` | Infographic generation | Renderer for data-driven explainer scenes |
| `meme-generation` *(optional)* | Generate meme images by overlaying text on templates | Generator for satirical/social content; meme-style stills |

### Design / pre-production skills (`hermes-agent/skills/creative/`)

| Skill | What it does | Best fit for |
|-------|--------------|--------------|
| `claude-design` | Design one-off HTML artifacts (landing, deck, prototype) | Concept artist for product video style frames; storyboarder for UI-heavy content |
| `design-md` | Design markdown docs | Concept artist documenting visual specs |
| `popular-web-designs` | Reference patterns for popular web designs | Concept artist; cinematographer when matching a known UI aesthetic |
| `sketch` | Throwaway HTML mockups (2-3 design variants to compare) | Concept artist exploring directions; storyboarder for UI flows |
| `excalidraw` | Excalidraw-style hand-drawn diagrams | Storyboarder; concept artist for sketch-style frames |
| `architecture-diagram` | Software architecture diagrams | Storyboarder for technical content; explainer scenes about systems |
| `concept-diagrams` *(optional)* | Flat, minimal SVG diagrams (educational visual language; physics, chemistry, math, anatomy, etc.) | Renderer / storyboarder for explainer scenes with clean educational diagrams |
| `pretext` | Mathematical/scientific content authoring | Writer / cinematographer for technical-explainer pretexts |
| `creative-ideation` | Constraint-driven project ideation | Director / cinematographer when the brief is wide-open and needs framing |
| `humanizer` | Strip AI-isms from text, add real voice | Writer / copywriter post-process to avoid AI-tells in scripts and VO copy |

### Audio / media skills (`hermes-agent/skills/creative/` + `skills/media/`)

| Skill | What it does | Best fit for |
|-------|--------------|--------------|
| `songwriting-and-ai-music` | Songwriting craft + Suno prompt patterns | Music supervisor when commissioning a track via Suno |
| `heartmula` | Open-source music generation (Apache-2.0, Suno-like) | Music supervisor generating bespoke tracks without external APIs |
| `songsee` | Spectrograms, mel/chroma/MFCC of audio files | Music supervisor analyzing tracks; foley-designer designing to a beat; editor visualizing a mix |
| `spotify` | Spotify control — play, search, queue, manage playlists | Music supervisor sourcing existing tracks; reference research |
| `youtube-content` | Fetch transcripts + transform to chapters/summaries/posts | Documentary cut, content adaptation, research for explainers |
| `gif-search` | Find existing GIFs | Editor / concept artist sourcing references |
| `gifs` | GIF tooling | Masterer producing GIF deliverables |

### Kanban infrastructure (`hermes-agent/skills/devops/`)

| Skill | What it does | When to load |
|-------|--------------|--------------|
| `kanban-orchestrator` | Decomposition playbook + anti-temptation rules for orchestrator profiles | Director only |
| `kanban-worker` | Pitfalls, examples, edge cases for kanban workers (deeper than auto-injected guidance) | Any profile — load when handling tricky multi-step workflows |

The kanban plugin auto-injects baseline orchestration guidance into every
worker's system prompt — the `kanban_create` fan-out pattern, claim/handoff
lifecycle, and the "decompose, don't execute" rule for orchestrators.
`kanban-orchestrator` and `kanban-worker` are deeper playbooks loaded when a
profile needs them.

## External tools (called from terminal toolset)

These are **not** Hermes skills but external CLIs / APIs that profiles invoke.
They don't appear in `always_load`; instead the role's terminal commands hit
them directly.

| Tool | What it does | Profile that uses it |
|------|--------------|----------------------|
| `ffmpeg` | Video / audio encode, splice, mux | renderer, editor, audio-mixer, masterer |
| `ffprobe` | Inspect media | All media-touching profiles |
| Whisper (CLI or API) | Speech-to-text for captions | captioner |
| Text-to-image API (FAL / Replicate / OpenAI / Midjourney) | Stills generation | image-generator (alternative to local `comfyui`) |
| Image-to-video API (Runway / Kling / Luma / Pika) | Animate stills | image-to-video-generator |
| Text-to-speech API (ElevenLabs / OpenAI TTS / etc.) | Voiceover generation | voice-talent |
| Suno API or web | Track composition (paired with `songwriting-and-ai-music`) | music-supervisor |
| Remotion CLI (`npx remotion render`) | React-based motion graphics | renderer-motion-graphics |
| Manim CE (`manim`) | Math animation render (driven by `manim-video` skill's recipes) | renderer-manim |
| Blender (`blender -b`) | 3D rendering (alternative to `blender-mcp`) | renderer-3d |
| Gemini multimodal / Claude vision | AI review of clips | reviewer, cinematographer, editor |

## Standard toolset configurations per role

### director

```yaml
toolsets:
  - kanban
  - terminal
  - file
skills:
  always_load:
    - kanban-orchestrator
```

The director's terminal access is conventional but the SOUL.md rules forbid
execution. Audit logs catch violations.

### writer / copywriter

```yaml
toolsets:
  - kanban
  - file
skills:
  always_load:
    - kanban-worker
    - humanizer            # post-process scripts to strip AI-tells
```

No terminal — writers don't need it.

### concept-artist

```yaml
toolsets:
  - kanban
  - terminal
  - file
skills:
  always_load:
    - kanban-worker
    # plus one or more (style-dependent):
    # - claude-design       (UI / web product video)
    # - sketch              (quick mockup variants)
    # - excalidraw          (hand-drawn frames)
    # - ascii-art           (ASCII style frames)
    # - pixel-art           (retro/game aesthetic)
    # - popular-web-designs (matching known web aesthetic)
    # - design-md           (text-based design docs)
```

### storyboarder

```yaml
toolsets:
  - kanban
  - file
skills:
  always_load:
    - kanban-worker
    # one of:
    # - excalidraw              (sketch storyboards)
    # - architecture-diagram    (technical/system content)
    # - concept-diagrams        (educational / scientific content)
```

### cinematographer

```yaml
toolsets:
  - kanban
  - terminal
  - file
skills:
  always_load:
    - kanban-worker
    # the visual skill that matches the project, e.g.:
    # - ascii-video            (ASCII projects)
    # - manim-video            (math/explainer)
    # - p5js                   (generative)
    # - comfyui                (AI-generated visuals)
    # - blender-mcp            (3D)
    # - touchdesigner-mcp      (real-time/installation)
```

### renderer (specialized variants)

```yaml
toolsets:
  - kanban
  - terminal
  - file
skills:
  always_load:
    - kanban-worker
    # ONE skill per renderer variant (or empty for external-API renderers):
    # - ascii-video               (renderer-ascii)
    # - manim-video               (renderer-manim)
    # - p5js                      (renderer-p5js)
    # - comfyui                   (renderer-comfyui — img/video AI gen)
    # - touchdesigner-mcp         (renderer-touchdesigner)
    # - blender-mcp               (renderer-3d)
    # - pixel-art                 (renderer-pixel)
    # - baoyu-comic               (renderer-comic)
    # - meme-generation           (renderer-meme)
```

For external-API renderers (image-to-video-generator using Runway, voice-talent
using ElevenLabs, renderer-motion-graphics using Remotion), `always_load` only
contains `kanban-worker` — the role's work is API-driven and the API key +
terminal commands suffice.

For multi-skill renderer setups (rare — usually one variant per skill is
cleaner) use `--skill <name>` on individual `kanban_create` calls to override
which skill loads for that specific task.

### image-generator / image-to-video-generator / voice-talent

```yaml
toolsets:
  - kanban
  - terminal
  - file
skills:
  always_load:
    - kanban-worker
    # for image-generator that drives ComfyUI locally:
    # - comfyui
env_required:
  # populate based on the chosen API:
  - FAL_KEY                 # or REPLICATE_API_TOKEN, OPENAI_API_KEY for image-gen
  - RUNWAY_API_KEY          # or KLING_API_KEY, LUMA_API_KEY for image-to-video
  - ELEVENLABS_API_KEY      # or OPENAI_API_KEY for TTS
```

If the user's setup has ComfyUI installed locally, the `comfyui` skill can
replace the external image-gen API entirely (cheaper, more control, supports
custom workflows for image-to-video too).

### music-supervisor

```yaml
toolsets:
  - kanban
  - terminal
  - file
skills:
  always_load:
    - kanban-worker
    - songsee                         # spectrograms / audio analysis
    # plus (depending on what the project needs):
    # - songwriting-and-ai-music      (commissioning Suno tracks)
    # - heartmula                     (commissioning open-source local generation)
    # - spotify                       (sourcing existing tracks)
```

### editor / audio-mixer / captioner / masterer

```yaml
toolsets:
  - kanban
  - terminal
  - file
skills:
  always_load:
    - kanban-worker
```

These are mostly ffmpeg-driven; no special skill needed beyond `kanban-worker`.
For captioner add Whisper invocation patterns to the SOUL.md.

### reviewer / brand-cop

```yaml
toolsets:
  - kanban
  - terminal           # for media inspection
  - file
skills:
  always_load:
    - kanban-worker
env_required:
  - OPENROUTER_API_KEY    # if using Gemini multimodal review
  # or ANTHROPIC_API_KEY if using Claude vision (already required globally)
```

## API key requirements

Track these in the project setup. The setup script should verify each required
key is present in `~/.hermes/.env` (or macOS Keychain) before firing the kanban.

| Service | Env var | Used by |
|---------|---------|---------|
| ElevenLabs | `ELEVENLABS_API_KEY` | voice-talent |
| OpenAI | `OPENAI_API_KEY` | image-generator (DALL-E), voice-talent (TTS) |
| OpenRouter | `OPENROUTER_API_KEY` | reviewer, cinematographer, editor (Gemini multimodal review) |
| FAL | `FAL_KEY` | image-generator (FAL flux models) |
| Replicate | `REPLICATE_API_TOKEN` | image-generator (alternate provider) |
| Runway | `RUNWAY_API_KEY` | image-to-video-generator |
| Kling | `KLING_API_KEY` | image-to-video-generator (alternate) |
| Luma | `LUMA_API_KEY` | image-to-video-generator (alternate) |
| Suno | `SUNO_API_KEY` | music-supervisor (paired with `songwriting-and-ai-music`) |
| Spotify | `SPOTIFY_CLIENT_ID` + `SPOTIFY_CLIENT_SECRET` | music-supervisor (paired with `spotify` skill) |
| Anthropic | `ANTHROPIC_API_KEY` | every Hermes profile (Claude) |

If a key is missing, prompt the user to add it. Storage methods, in order of
preference: macOS Keychain → `~/.hermes/.env` → environment variable.

## Skill version pinning

If a specific skill version is desired, pass it via the per-task
`--skill <name>=<version>` flag. The default is whatever's installed.

## Adding a new skill to the matrix

When a new Hermes-public video skill ships:

1. Add a row to the relevant table at the top of this file
2. If it warrants a specialized renderer variant, add to `role-archetypes.md`
3. Update relevant per-style examples in `examples.md`
