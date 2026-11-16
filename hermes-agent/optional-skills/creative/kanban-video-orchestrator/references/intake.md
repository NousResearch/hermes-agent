# Intake — Discovery Question Banks

The discovery process is **adaptive**. Always start with three baseline
questions to identify the broad style category, then drill into a per-style
question bank. Ask 2-4 questions at a time, listen, then proceed. Make
reasonable assumptions whenever the user implies an answer.

## Tier 0 — Baseline (always ask)

1. **What is the video?** — One-sentence pitch
2. **How long?** — Approximate duration
3. **Aspect ratio + target platform?** — 16:9 / 9:16 / 1:1 / 4:5; X, IG, YouTube, internal, etc.

From these answers, classify the style category and pick the relevant Tier 1
follow-ups. **Do not** continue asking until you have at least these three.

## Style classification

Map the brief to one of these archetypes (or a hybrid):

| Archetype | Tells |
|-----------|-------|
| **Narrative film** | Plot, characters, scenes-with-events, dialogue, location |
| **Product / marketing** | A specific product or feature being shown / sold; CTA at end |
| **Music video** | A specific track exists; visuals sync to music |
| **Explainer / educational** | A concept being taught; voiceover-driven |
| **Tutorial / changelog** | Software demo, terminal-heavy, technical |
| **ASCII / terminal art** | Retro terminal aesthetic explicit, character-grid |
| **Abstract / loop** | Generative, no plot, often perfect-loop |
| **Documentary / interview cut** | Real footage, transcription-driven |
| **Real-time / installation** | Audio-reactive, gallery installation, VJ output |

If ambiguous, **ask** which category fits — don't guess. Hybrids are common
(e.g., a product video with a narrative arc); decompose into the dominant
mode + secondary modifiers.

**Recursive / meta** ("a video that shows its own production") is a
*rendering technique*, not a separate style — compose it from any of the
above by adding a two-pass render step where pass 2 uses pass 1's output as
texture inside the final scene.

## Tier 1 — Per-style follow-ups

### Narrative film

- **Setting / world?** — When and where the story takes place
- **Characters?** — How many, archetypes, who carries dialogue
- **Beat list or full script?** — Has the user written the story or do we draft it
- **Dialogue language?** — Spoken lines, on-screen subs only, silent
- **Visual generation approach?** — Text-to-image (FAL/Midjourney/Imagen) →
  image-to-video (Runway/Kling), 3D animation (Blender), 2D animation,
  procedural, or hybrid
- **Voice approach?** — TTS (which voice), recorded VO, no dialogue
- **Music / score?** — Commissioned (via `songwriting-and-ai-music` Suno
  prompts, or local `heartmula`), licensed track provided, silent

### Product / marketing

- **Product?** — Name, what it does, key feature being shown
- **Target audience?** — Who's watching, what they care about
- **CTA?** — Visit URL, install, sign up, etc.
- **Tone?** — Serious, playful, technical, premium, edgy
- **Brand assets available?** — Logo files, color palette, fonts, existing footage
- **Animation style?** — Motion graphics (Remotion / AE-style), screen recording,
  generative, illustrated
- **Voiceover?** — Yes (which voice / language) or text-only
- **Music?** — Track provided, license-free needed, custom-composed

### Music video

- **Track file?** — Path to the audio (essential — we'll analyze BPM + beats)
- **Track length to use?** — Full song or a section
- **Genre / energy?** — Tells what visual rhythm and density to use
- **Lyric / narrative content?** — Are there lyrics to render on screen,
  or is it purely visual?
- **Visual reference style?** — Existing music videos / artists for reference
- **Performer footage?** — None, has clips, will provide
- **Visual generation approach?** — Per-beat generative, edit-driven cuts of stock
  footage, illustrated, hybrid

### Explainer / educational

- **What concept is being taught?** — One-sentence concept, key takeaway
- **Audience expertise?** — Beginner / intermediate / expert
- **Diagram density?** — Heavy math / formulas / code / abstract concepts
- **Voiceover?** — TTS / recorded / on-screen text only
- **Tool preference?** — `manim-video` (math), `p5js` (generative),
  Remotion (UI motion graphics), `comfyui` (AI-generated visuals),
  `ascii-video` (technical/retro), hybrid
- **Pacing?** — Fast and dense (3Blue1Brown) or slow and contemplative

### Tutorial / changelog / software demo

- **Software being demonstrated?** — Name, what it does
- **Demo script?** — Sequence of commands / screens to show
- **Terminal-only or with GUI?**
- **Voiceover for narration?**
- **Diagram support needed?** — Often these benefit from a diagram skill
  alongside the screen-capture/render step (`excalidraw`,
  `architecture-diagram`, `concept-diagrams`)

### ASCII / terminal art

- **Source material?** — Generative / driven by audio / converting existing
  video / static image starting point
- **Color palette?** — Brand-driven (gold/black/blue), Matrix green, full
  rainbow, monochrome
- **Audio reactivity?** — None / loose mood / tight beat sync / FFT-driven
- **Character set?** — ASCII only / Unicode block-drawing / mystic glyphs
- **Loop or narrative?** — Perfect loop or one-shot

### Abstract / loop

- **Mood / emotion?** — One word that captures the feel
- **Motion type?** — Zoom-into-itself, particle drift, wave, geometric, organic
- **Loop required?** — Perfect loop (Droste-style) or just satisfying ending
- **Audio?** — Silent, ambient pad, beat-synced

### Documentary / interview cut

- **Source footage?** — Provided clips, length per clip
- **Transcript / subtitles?** — Provided or to be generated
- **Story structure?** — Chronological / thematic / arc
- **B-roll approach?** — Generated, stock library, none

### Real-time / installation

- **Output environment?** — Gallery wall, projector, screen, web embed
- **Audio source?** — Live audio input, pre-recorded track, both
- **Reactivity tightness?** — Mood-level (loose) vs. tight beat-sync vs. live
  parameter control
- **Tool preference?** — `touchdesigner-mcp` for full TD operator graphs;
  `p5js` for web-canvas; `comfyui` for generative-AI fed by audio features

## Tier 2 — Always ask near the end

- **Brand assets path?** — Where logo / color palette / fonts / music library lives
- **Output format requirements?** — Codec preference, target file size, accepted
  alternates (vertical cut, GIF, audio-only)
- **Deadline?** — Affects task `max_runtime_seconds` and acceptable scope
- **Quality bar?** — Rough draft for review / polished final / archival
- **Existing footage / assets to reuse?** — Anything that should appear, not just inform

## Reasonable assumption defaults

When the user under-specifies, fill in these defaults rather than asking:

| Question | Default |
|----------|---------|
| Frame rate | 30 fps for X / IG; 60 fps for tutorials/explainers; 24 fps for narrative film |
| Resolution | 1080×1080 for square, 1920×1080 for 16:9, 1080×1920 for 9:16 |
| Codec | H.264 / yuv420p, CRF 18 |
| Audio codec | AAC 192 kbps |
| Voice | Provider's mid-range neutral voice unless brand calls for distinctive timbre |
| Music | Silent (require user to specify if music is wanted) |
| Captions | On for explainer/tutorial; off for narrative/abstract unless requested |
| Quality bar | Polished final unless user says draft |

State the assumption explicitly: *"Assuming 30fps and AAC audio unless you say otherwise — proceed?"*

## Anti-patterns

- **Asking 10 questions at once.** Maximum 4 per turn.
- **Asking for things the brief already implies.** If the user said "music video for my track," do not ask "is there a track?"
- **Failing to classify before drilling in.** Tier-1 questions depend on classification; mixing them up wastes turns.
- **Treating "make a video" as enough to proceed.** Always confirm the three baseline questions.
