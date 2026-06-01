# Role Archetypes

The library of role archetypes for video production. **Compose a team from this
list, don't clone a fixed roster.** Most videos need 4-7 profiles. The director
is always present; everything else is conditional on the brief.

Each role's profile name is by convention `kebab-case` (e.g. `creative-director`,
`image-generator`). Multiple instances of the same role get descriptive suffixes
when they need different focus (e.g., `renderer-ascii`, `renderer-3d`).

For toolset + skill mapping per role, see [tool-matrix.md](tool-matrix.md).

## Always present

### director

The vision-holder. Reads the brief and brand guide, decomposes into a task
graph, comments to steer creative direction, approves the final cut.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-orchestrator`. The kanban plugin auto-injects baseline
  orchestration guidance for free; `kanban-orchestrator` is the deeper
  decomposition playbook. Add `creative-ideation` if the brief is wide-open
  and needs framing help.
- **Personality:** Tied to the brand voice — see `assets/soul.md.tmpl`

The director has the same toolset as everyone else, but its `SOUL.md` rules
**forbid** execution. The "decompose, don't execute" discipline is enforced
by personality + the kanban-orchestrator skill, not by missing tools.

## Pre-production roles

Pick based on what the brief needs.

### writer / screenwriter

Writes scripts, dialogue, voiceover copy, narration. Use for any video with
spoken or written words beyond a tagline.

- **Toolsets:** kanban, file
- **Skills:** `kanban-worker`, `humanizer` (post-process to strip AI-tells)
- **Outputs:** `script.md`, `narration.md`, `dialogue/scene-NN.md`

### copywriter

Like `writer` but specifically for marketing copy: taglines, CTAs, voiceover
scripts for product videos.

- **Toolsets:** kanban, file
- **Skills:** `kanban-worker`, `humanizer`
- **Outputs:** `copy.md`

### concept-artist / visual-designer

Develops the visual identity: mood board, style frames, color palette
rationale, typography choices. Produces a `visual-spec.md` that all generators
follow. Often produces still reference frames using image-generation APIs or
local skills.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker` plus any project-specific design skill —
  `claude-design` (UI/web), `sketch` (quick mockup variants),
  `popular-web-designs` (matching known web aesthetic), `pixel-art` (retro),
  `ascii-art` (terminal/retro), `excalidraw` (hand-drawn frames),
  `design-md` (text-based design docs)
- **Outputs:** `visual-spec.md`, `taste/style-frames/*.png`

### storyboarder

Maps the brief to a beat-by-beat shot list with timing. Critical for narrative
film and music video. Often pairs with a diagramming tool.

- **Toolsets:** kanban, file
- **Skills:** `kanban-worker` plus a diagram skill — `excalidraw` (sketch),
  `architecture-diagram` (technical/system), `concept-diagrams` (educational/
  scientific)
- **Outputs:** `storyboard.md` with one row per scene/shot, optional
  storyboard sketches

### cinematographer / dp

Designs the visual language: framing, color, motion, transitions. Reviews
generator output for visual consistency. Hands off per-scene `VISUAL_SPEC.md`.

- **Toolsets:** kanban, terminal, file, video, vision
- **Skills:** `kanban-worker` plus the visual skill that matches the project
  (e.g., `ascii-video` for ASCII work, `manim-video` for explainers,
  `touchdesigner-mcp` for real-time visuals, etc.)
- **Outputs:** `scenes/scene-NN/VISUAL_SPEC.md`, review comments on renderer
  tasks
- **Reviews via:** `video_analyze` (sends full clip to multimodal LLM for
  native review), `vision_analyze` for spot-checking frames, ffprobe summaries

## Production roles

### renderer (generic)

A worker that produces visual content for one or more scenes. Loaded with
whichever creative skill fits the scene's style. Multiple renderers can run in
parallel, each pinned to a different skill via `always_load` in their profile
or `--skill` on the task.

- **Toolsets:** kanban, terminal, file
- **Skills:** one creative skill (see specialized variants below)
- **Outputs:** `scenes/scene-NN/clip.mp4`

### Specialized renderer variants

When scenes need very different tools, create specialized renderer profiles
instead of overloading one. Each loads a different creative skill.

| Variant | Skill | Best for |
|---------|-------|----------|
| `renderer-ascii` | `ascii-video` | Terminal aesthetic, retro pixel, audio-reactive grid, video-to-ASCII conversion |
| `renderer-manim` | `manim-video` | Math, algorithms, 3Blue1Brown-style explainers, equation derivations |
| `renderer-p5js` | `p5js` | Generative art, particles, shaders, organic motion, web-canvas content |
| `renderer-comfyui` | `comfyui` | AI-generated stills + video using local ComfyUI workflows (img-to-img, img-to-video, etc.) |
| `renderer-touchdesigner` | `touchdesigner-mcp` | Real-time, audio-reactive, installation art, VJ-style content |
| `renderer-3d` | `blender-mcp` *(optional)* | 3D modeling, animation, photoreal environments, character animation |
| `renderer-pixel` | `pixel-art` | Retro game aesthetic with era-correct palettes |
| `renderer-comic` | `baoyu-comic` | Knowledge-comic style narrative scenes |
| `renderer-meme` | `meme-generation` *(optional)* | Meme-style stills for satirical/social content |
| `renderer-procedural` | (none — Python with PIL + ffmpeg directly) | Custom procedural content where no skill fits |
| `renderer-video` | (external image-to-video API: Runway / Kling / Luma) | Animating still images in narrative film |
| `renderer-motion-graphics` | (external — Remotion CLI) | Motion graphics, kinetic typography, UI animations |

For external-API renderers, the profile holds the API client logic; only
`kanban-worker` is loaded, plus the terminal toolset and the API key.

### image-generator

Specifically for text-to-image generation. Often produces stills that go to
`renderer-video` for animation.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`, optionally `comfyui` (drives a local
  ComfyUI install for image generation)
- **External APIs (alternative to local ComfyUI):** FAL, Replicate, OpenAI
  Images, Midjourney
- **Outputs:** `scenes/scene-NN/stills/*.png`

### image-to-video-generator

Takes still images and animates them via Runway/Kling/Luma APIs, or via
ComfyUI's image-to-video workflows locally. Almost always follows
`image-generator` in narrative film pipelines.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`, optionally `comfyui` (for local image-to-video
  workflows like AnimateDiff or WAN)
- **External APIs:** Runway, Kling, Luma, Pika
- **Outputs:** `scenes/scene-NN/clip.mp4`

### music-supervisor

Sources, analyzes, and prepares the music track. For music videos, also
produces a beat/BPM map and key-moment timestamps. Uses `songsee` for
spectrograms when the editor or renderer needs a visual reference of the
audio's energy.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`, `songsee` (audio visualization), plus one of:
  - `songwriting-and-ai-music` — when commissioning lyrics + Suno prompts
  - `heartmula` — when generating music with the open-source local model
  - `spotify` — when sourcing existing tracks
- **Outputs:** `audio/track.mp3`, `audio/beats.json`, optional
  `audio/track-spectrogram.png`

### voice-talent / narrator

Generates voiceover audio. Calls a TTS API directly; no Hermes skill required
beyond `kanban-worker`. The user can also supply pre-recorded VO instead of
generation.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`
- **External APIs:** ElevenLabs, OpenAI TTS, etc.
- **Outputs:** `audio/voiceover/line-NN.mp3`, `audio/voiceover/timeline.mp3`

### foley / sfx-designer

Sound effects and ambient design. Often optional unless the brief calls for
sound design specifically.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`, `songsee` for audio-feature visualization when
  designing to a track
- **Outputs:** `audio/sfx/*.mp3`

## Post-production roles

### editor

Assembles the final cut from clips. Uses ffmpeg for stitching, fades,
transitions. Reviews each clip for pacing and quality before assembly.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`
- **External tools:** ffmpeg, ffprobe
- **Outputs:** `output/final.mp4`, `output/final-noaudio.mp4`

### colorist

Color grading. Usually optional — if the renderers already produce
brand-consistent output and the editor just stitches, the colorist is overkill.
Worth including for narrative film with hero shots.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`
- **Outputs:** `output/final-graded.mp4`

### audio-mixer

Mixes voiceover + music + SFX into a final audio track. Sets levels, ducks
music under VO, normalizes loudness (LUFS).

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`
- **External tools:** ffmpeg with `loudnorm` filter, optional `sox`
- **Outputs:** `audio/final-mix.mp3`

### captioner

Burns subtitles into the video, generates SRT, handles accessibility. Can also
generate captions from audio via Whisper.

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`
- **External tools:** Whisper (CLI or API), ffmpeg subtitle filters
- **Outputs:** `output/captions.srt`, `output/final-captioned.mp4`

### masterer

Final encode + format variants. Produces deliverables for each platform target
(square for IG, vertical for TikTok, full HD for YouTube, etc.).

- **Toolsets:** kanban, terminal, file
- **Skills:** `kanban-worker`
- **Outputs:** `output/final-1080.mp4`, `output/final-9x16.mp4`, etc.

## QA roles

### reviewer

A neutral quality gate. Reads the brief, watches the cut, comments
specifically on what's off (pacing, sync, brand alignment, technical
quality). Distinct from the cinematographer (who reviews visuals during
production) and the editor (who reviews for assembly).

- **Toolsets:** kanban, terminal, file, video, vision
- **Skills:** `kanban-worker`
- **Review tools:** `video_analyze` (native clip review via multimodal LLM),
  `vision_analyze` (frame/thumbnail review), ffprobe
- **Outputs:** `review-notes.md`, comments on tasks

### brand-cop

Reviews specifically for brand compliance — colors, typography, voice. Use
when the brand guidelines are detailed and a generic reviewer might miss
violations.

- **Toolsets:** kanban, file
- **Skills:** `kanban-worker`
- **Outputs:** comments + `brand-review.md`

## Composing teams — heuristics

- **Always:** director + at least one renderer + editor.
- **Add writer** if scripted dialogue / narration / on-screen text exceeds a
  tagline.
- **Add storyboarder** if the brief has more than 5 distinct beats and the
  director hasn't already laid out a beat list.
- **Add cinematographer** if multiple renderer instances need consistent
  visual language. (For a single-tool video, the renderer's own skill spec
  is enough.)
- **Add image-generator + image-to-video-generator pair** for narrative film
  with photorealistic visuals.
- **Add music-supervisor** when music is provided and rhythm matters
  (music videos always; explainers sometimes).
- **Add voice-talent** for any voiceover / narrative dialogue.
- **Add audio-mixer** when there are 2+ audio sources (VO + music, music + SFX).
- **Add captioner** for accessibility-priority projects (explainer, tutorial,
  any platform that defaults to muted playback).
- **Add reviewer** for high-stakes projects. Skip for quick experimental loops.
- **Add masterer** when multiple platform deliverables are needed.

## Anti-patterns

- **One renderer doing everything.** If scenes use very different tools
  (ASCII + 3D + motion graphics), use specialized renderer variants. The
  renderer loads ONE creative skill at a time; mixing styles in a single
  renderer causes thrashing.
- **A separate profile per scene.** No. Profiles are per-role, not per-scene.
  Eight scenes use one or two renderer profiles, not eight.
- **A "general" profile that does everything.** Worse than no specialization.
  The kanban routing breaks down if every task fits every profile.
- **No reviewer for important deliverables.** Saves an hour of pipeline time
  but ships flaws.
