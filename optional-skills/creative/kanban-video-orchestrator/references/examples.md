# Worked Examples

Six concrete pipelines covering different video styles. Each shows the team
composition, task graph, and skill/tool choices the orchestrator would make
for that brief. **These are illustrative, not templates** — adapt to the
actual brief.

## Example 1 — Narrative short film (text-to-image → image-to-video → cut)

**Brief:** A 90-second noir-style short. A detective walks through a rainy
city. Voiceover narration. AI-generated visuals.

**Team:**
- `director` — vision, decomposition, approval
- `writer` — script + voiceover copy (loads `humanizer` for natural voice)
- `storyboarder` — beat-by-beat shot list (loads `excalidraw`)
- `image-generator` — generates each shot's still via local ComfyUI workflows
  (loads `comfyui`)
- `image-to-video-generator` — animates each still (Runway/Kling, OR
  ComfyUI's AnimateDiff/WAN workflows via `comfyui`)
- `voice-talent` — narration via ElevenLabs
- `audio-mixer` — VO + ambient pad
- `editor` — assembly + transitions
- `reviewer` — final QA

**Task graph:**
```
T0  director         decompose
T1  writer           script + voiceover.md                    (parent: T0)
T2  storyboarder     shot list with framing per beat          (parent: T1)
T3  image-generator  one still per shot (~12 shots)           (parent: T2)
T4  image-to-video   animate each still                       (parent: T3)
T5  voice-talent     generate narration audio                 (parent: T1)
T6  audio-mixer      mix VO + ambient                         (parent: T5)
T7  editor           cut + transitions + audio mux            (parents: T4, T6)
T8  reviewer         final QA                                 (parent: T7)
```

**Key choices:**
- Local ComfyUI via `comfyui` skill is preferred over external API for
  cost/control — but external APIs are fine if ComfyUI isn't installed
- `editor` profile is ffmpeg-only, no Hermes skill required beyond
  `kanban-worker`
- Storyboarder produces `storyboard.excalidraw` alongside the markdown

## Example 2 — Product / marketing teaser

**Brief:** A 30-second product teaser for a developer tool. Shows code +
terminal + UI screen recordings, voiceover, CTA at end. Square 1:1.

**Team:**
- `director`
- `copywriter` — taglines, voiceover script, CTA (loads `humanizer`)
- `concept-artist` — style frames (loads `claude-design` for UI mockups)
- `renderer-motion-graphics` — animated UI sequences (Remotion CLI)
- `renderer-ascii` — terminal-style demo scenes (loads `ascii-video`)
- `voice-talent` — VO via ElevenLabs
- `editor` — assembly + brand-color treatment
- `audio-mixer` — VO + light music bed
- `captioner` — burned subtitles for muted-autoplay platforms
- `masterer` — produces 1:1 + 9:16 + 16:9 variants

**Task graph:**
```
T0  director              decompose
T1  copywriter            copy.md + cta + vo script               (parent: T0)
T2  concept-artist        visual-spec.md + style frames           (parent: T1)
T3a renderer-motion-graphics  scene 1: UI sequence                (parent: T2)
T3b renderer-ascii        scene 2: terminal demo                  (parent: T2)
T3c renderer-motion-graphics  scene 3: feature highlight          (parent: T2)
T3d renderer-motion-graphics  scene 4: CTA card                   (parent: T2)
T4  voice-talent          narration                                (parent: T1)
T5  audio-mixer           VO + music bed                          (parent: T4)
T6  editor                cut + transitions                        (parents: T3*, T5)
T7  captioner             SRT + burned subtitles                  (parent: T6)
T8  masterer              1:1, 9:16, 16:9 variants                (parent: T7)
```

**Key choices:**
- Multiple specialized renderers (motion-graphics + ASCII) coexist
- Captioner is included because muted autoplay is the norm on social
- `claude-design` skill for UI mockups maps directly to the product video idiom

## Example 3 — Music video (synced to provided track)

**Brief:** A 3-minute music video for a provided lo-fi hip-hop track. Visuals
should pulse with the beat. Generative + ASCII hybrid. Vertical 9:16.

**Team:**
- `director`
- `music-supervisor` — analyze track, emit `audio/beats.json` (loads `songsee`)
- `storyboarder` — beat-aligned shot list (loads `excalidraw`)
- `renderer-ascii` — ASCII scenes synced to bass kicks (loads `ascii-video`)
- `renderer-p5js` — generative particle scenes synced to highs (loads `p5js`)
- `editor` — beat-cut assembly using `beats.json`
- `reviewer` — sync QA

**Task graph:**
```
T0  director              decompose
T1  music-supervisor      analyze track → beats.json + spectrogram  (parent: T0)
T2  storyboarder          shot list aligned to beats                (parents: T1, T0)
T3a renderer-ascii        scene 1: bass-driven ASCII                (parent: T2)
T3b renderer-p5js         scene 2: high-end particle field          (parent: T2)
... (more scenes)
T4  editor                cut to beats + mux track                  (parents: T3*, T1)
T5  reviewer              sync QA + final approval                  (parent: T4)
```

**Key choices:**
- `music-supervisor` runs FIRST — `beats.json` gates the renderers
- `editor` uses `beats.json` directly to align cuts to bass kicks
- No voice-talent — music is the audio
- Two specialized renderers (`ascii-video` + `p5js`) for visual variety

## Example 4 — Math/algorithm explainer

**Brief:** A 2-minute explainer of an algorithm. 3Blue1Brown-style. Animated
diagrams, equations, narration. Square 1:1.

**Team:**
- `director`
- `writer` — narration script (loads `humanizer`)
- `cinematographer` — visual spec (loads `manim-video`)
- `renderer-manim` — all animated scenes (loads `manim-video`)
- `voice-talent` — narration via ElevenLabs
- `editor` — assembly + audio mux
- `captioner` — burned subtitles

**Task graph:**
```
T0  director           decompose
T1  writer             script + narration                  (parent: T0)
T2  cinematographer    visual spec for all scenes           (parent: T1)
T3a-Tn renderer-manim  scenes 1..N                          (parents: T2)
T4  voice-talent       narration audio                      (parent: T1)
T5  editor             cut + mux                            (parents: T3*, T4)
T6  captioner          SRT + burn                           (parent: T5)
```

**Key choices:**
- `manim-video` skill drives both the cinematographer (visual language) and
  the renderer (actual scene production)
- The `manim-video` skill's reference docs (animation-design-thinking,
  scene-planning, equations) auto-load when needed via the renderer's pinned skill

## Example 5 — ASCII video, music-track-only

**Brief:** A 60-second pure-ASCII video reactive to an existing track. No
voiceover, no other tools. Square 1:1.

**Team:**
- `director`
- `music-supervisor` — track analysis (loads `songsee`)
- `renderer-ascii` — all visuals (loads `ascii-video`)
- `editor` — assembly + audio mux

**Task graph:**
```
T0  director           decompose
T1  music-supervisor   analyze track                       (parent: T0)
T2a renderer-ascii     scene 1                             (parents: T1, T0)
T2b renderer-ascii     scene 2                             (parents: T1, T0)
T2c renderer-ascii     scene 3                             (parents: T1, T0)
T3  editor             stitch + mux audio                  (parents: T2*)
```

**Key choices:**
- Minimal team (4 profiles) for a focused single-tool project
- No reviewer — short experimental piece, director approves directly
- All scenes run through one `renderer-ascii` profile because the `ascii-video`
  skill covers everything

This example illustrates the rule: **don't over-decompose**. Three scenes
through one renderer is fine. Don't spawn three renderer profiles.

## Example 6 — Real-time / installation art

**Brief:** A 2-minute audio-reactive visual for a gallery installation. Driven
by an audio input feed. TouchDesigner-based. 16:9 4K.

**Team:**
- `director`
- `cinematographer` — visual language spec (loads `touchdesigner-mcp`)
- `renderer-touchdesigner` — all visuals + record-to-disk
  (loads `touchdesigner-mcp`)
- `audio-mixer` — final loudness pass on the captured audio (optional if
  pre-mixed source)
- `editor` — assemble final clip from TouchDesigner recording
- `reviewer` — visual QA

**Task graph:**
```
T0  director                decompose
T1  cinematographer         TD operator graph spec           (parent: T0)
T2  renderer-touchdesigner  build TD network + record output (parent: T1)
T3  editor                  trim + audio mux                 (parent: T2)
T4  reviewer                final QA                         (parent: T3)
```

**Key choices:**
- `touchdesigner-mcp` controls a running TouchDesigner instance — the
  cinematographer designs the operator graph, renderer builds it
- Output is a recording from the running TD network, not a render-to-frames
  process; editor mostly just trims

## Pattern recognition

When the user describes a video, look for these signals to map to an example:

- **Plot, characters, scripted dialogue** → Example 1 (narrative)
- **Specific product, CTA, brand colors, voiceover** → Example 2 (marketing)
- **Track file provided, "synced to music"** → Example 3 (music video)
- **"Explain how X works", math/algorithm/concept walkthrough** → Example 4 (manim explainer)
- **Terminal aesthetic, ASCII, retro pixel** → Example 5 (ASCII)
- **"Audio-reactive", "real-time", "installation"** → Example 6 (TouchDesigner)
- **Comic-style narrative** → use `renderer-comic` (`baoyu-comic` skill)
- **Retro game / pixel-art aesthetic** → use `renderer-pixel` (`pixel-art` skill)
- **3D scene, photoreal environment** → use `renderer-3d` (`blender-mcp`)
- **Generative art, particle system, shader** → use `renderer-p5js` (`p5js`)
- **AI-generated photoreal stills + animation** → use `renderer-comfyui`
  (`comfyui`) for both stills and image-to-video
- **"video about how the system works", recursive demo** → composable from
  any of the above; the recursion is a rendering technique, not a style

The actual team should be derived from the specific brief — these examples are
starting points, not endpoints.
