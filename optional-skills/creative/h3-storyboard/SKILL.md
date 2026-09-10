---
name: h3-storyboard
description: MiniMax H3 shot breakdown and facial performance direction.
version: 1.0.0
author: Ray (phileiny), ported by Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [video-generation, minimax-h3, storyboard, directing, creative]
    homepage: https://github.com/phileiny/h3-storyboard-skill
    related_skills: [kanban-video-orchestrator, hyperframes, comfyui]
---

# H3 Storyboard — Shot Breakdown & Performance Direction

Adapted from [h3-storyboard](https://github.com/phileiny/h3-storyboard-skill) by Ray (MIT). Use when breaking a script or scene into MiniMax H3 shots, writing beat timings, directing a character's face or emotion, fixing flat/wooden performance, deciding shot length or how many beats a shot can hold, or writing prompts for episodic narrative work.

Prompt-syntax skills tell you how to format what you already decided. This one covers the step before: how to split a script into shots and turn emotion into motion the model can actually render. Every rule was derived from controlled generations (fixed seed, one variable at a time), and `references/SOURCES.md` marks which rules have a controlled comparison behind them and which are still inference.

## The core findings (read these before writing any H3 prompt)

1. **One shot cannot hold many expression beats.** Cram 9 facial beats into one 7s close-up and the model averages them into a frozen face (PSNR 37-42 dB ≈ freeze frame). Split into 2-3s shots with ONE main beat each and the expressions render (22-23 dB). Splitting the shot is the mechanism; dialogue is only a bonus on top.
2. **Dialogue steals time for its shot.** H3 reallocates frames across a multi-shot video toward the shot carrying `<d>` dialogue — the performance shot gains frames, but background continuity in OTHER shots degrades (side characters vanish, set elements simplify). Want performance → give that shot a line (inner monologue as off-screen voiceover counts). Want background continuity → keep dialogue out of that video, or render both versions and edit.
3. **Emotion goes in the delivery field, never inside `<d>`.** `<d>` holds only the language tag and the verbatim line.
4. **Silent characters: prefer large body motion over facial beats** (exhale + shoulders dropping, hand wiping trousers, head retreating 3 cm) — locomotion is reliable, micro-expressions are risky. Hands: describe contact and motion ("hands press flat against thighs, then curl closed"), never finger geometry — extra-finger risk.
5. **Reference audio is scaffolding, not a voice source.** Wire standalone audio to `ref_audio_0` (NOT `ref_video_audio_0` — miswiring fails silently). It improves mouth/breath rhythm but pitch, range, and accent drift; record or clone the final audio track separately.
6. **The last mile of performance is editing, not prompt revision.** Slow reactions → cut the gap with an insert shot. Too-smooth emotional transitions → pull the two poles together around a short insert. Shoot long, always shoot inserts, capture both emotional poles.
7. **Never let an emotional A→B change happen visibly on screen** — H3 cross-fades it into rubber-face. Hide the change behind a blink, head-dip, or cut, and precede any release with one reverse beat (tighter before looser). Full recipes in the zh reference's occlusion-transition section (§六之一).
8. **Never delegate to H3:** screen contents (UI text garbles), brand assets/logos, precise text, pauses/freezes. Screens get "uniform cold white light" and content is composited in post. ffmpeg alpha-fade over a still image requires `-loop 1` or the overlay is silently fully transparent.

## Workflow

1. Count expression beats in the scene. More than 2-3 in one shot → split.
2. Break into 2-3s shots, one main beat per shot; the cut itself is performance (viewers re-read the character at every cut). Terminology: a generated *video* (5-13s, one frame budget at 24 fps, frame counts on the 17n+5 grid) contains one or more 2-3s *shots* separated by in-prompt cuts.
3. Assign dialogue deliberately (performance shots get lines; continuity shots don't share a video with dialogue).
4. Write beats per the full playbook: `references/h3-storyboard-zh.md` (Traditional Chinese, carried verbatim from upstream — its tables, PSNR data, prompt templates, and ffmpeg recipes are the source of truth; read it fully via terminal `cat` if read_file misdetects the dense-CJK file as binary).
5. Generate via your H3 pipeline — Hermes' `video_generate` tool exposes MiniMax H3 / H3 Max on FAL for hosted t2v/i2v, or ComfyUI (see the `comfyui` skill) for the open-weights Ref2VA path the upstream experiments used. Note: timestamped beat tables and reference-audio wiring apply to the open-weights/ComfyUI path; hosted FAL endpoints take a single prompt, where the shot-splitting and one-beat-per-shot rules still carry.
6. Edit for the last mile (inserts, gap cuts) instead of endlessly revising prompts.

## Pitfalls

- `ref_audio_0` vs `ref_video_audio_0` miswiring: no error, no voice transfer.
- Audio never enters the text encoder — describing the reference audio in the prompt does nothing; only the `<Audio 1>:` tag matters.
- Accent drift is generic to generated speech (ElevenLabs, DashScope too) — the model regresses to majority training accent whenever it has freedom.
- "H3 is great at micro-expressions" claims online usually describe Hailuo 02/2.3 (the hosted API models with a prompt rewriter in front), not open-weights H3.
- Series work: shoot rituals/transitions as reusable assets (no episode-specific props, no protagonist face) — consistency is the brand.

## Verification

- Shot list review: every shot ≤3s or justified, exactly one main beat, dialogue placement intentional.
- After generation: check the emotional peak actually moved (compare frames at the beat timestamp); if frozen, the shot is overloaded — split it, don't add more facial adjectives.
- Composited overlays: confirm the overlay is visible (the `-loop 1` failure is silent).
