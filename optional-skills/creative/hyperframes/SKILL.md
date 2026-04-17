---
name: hyperframes
description: Create HTML-based video compositions, animated title cards, social overlays, captioned talking-head videos, audio-reactive visuals, and shader transitions using HyperFrames. HTML is the source of truth for video. Use when the user wants a rendered MP4/WebM from an HTML composition, wants to animate text/logos/charts over media, needs captions synced to audio, wants TTS narration, or wants to convert a website into a video.
version: 1.0.0
author: heygen-com
license: Apache-2.0
prerequisites:
  commands: [node, ffmpeg, npx]
metadata:
  hermes:
    tags: [creative, video, animation, html, gsap, motion-graphics]
    related_skills: [manim-video, meme-generation]
    category: creative
    requires_toolsets: [terminal]
---

# HyperFrames

HTML is the source of truth for video. A composition is an HTML file with `data-*` attributes for timing, a GSAP timeline for animation, and CSS for appearance. The HyperFrames engine captures the page frame-by-frame and encodes to MP4/WebM with FFmpeg.

**Complement to `manim-video`:** Use `manim-video` for mathematical/geometric explainers (equations, 3B1B-style). Use `hyperframes` for motion-graphics, talking-head with captions, product tours, social overlays, shader transitions, and anything driven by real video/audio media.

## When to Use

- User asks for a rendered video from text, a script, or a website
- Animated title cards, lower thirds, or typographic intros
- Captioned narration video (TTS + captions synced to waveform)
- Audio-reactive visuals (beat sync, spectrum bars, pulsing glow)
- Scene-to-scene transitions (crossfade, wipe, shader warp, flash-through-white)
- Social overlays (Instagram/TikTok/YouTube style)
- Website-to-video pipeline (capture a URL, produce a promo)
- Any HTML/CSS/JS animation that must render deterministically to a video file

Do **not** use this skill for:
- Pure math/equation animation (→ `manim-video`)
- Image generation or memes (→ `meme-generation`, image models)
- Live video conferencing or streaming

## Quick Reference

```bash
npx hyperframes init my-video               # scaffold a project
cd my-video
npx hyperframes lint                        # validate before preview/render
npx hyperframes preview                     # live-reload browser preview (port 3002)
npx hyperframes render --output final.mp4   # render to MP4
npx hyperframes doctor                      # diagnose environment issues
```

Render flags: `--quality draft|standard|high` · `--fps 24|30|60` · `--format mp4|webm` · `--docker` (reproducible) · `--strict`.

Full CLI reference: [references/cli.md](references/cli.md).

## Setup (one-time)

```bash
bash "$(dirname "$(find ~/.hermes/skills -path '*/hyperframes/SKILL.md' 2>/dev/null | head -1)")/scripts/setup.sh"
```

The script:
1. Verifies Node.js >= 22 and FFmpeg are installed (prints fix instructions if not).
2. Installs the `hyperframes` CLI globally (`npm install -g hyperframes@>=0.4.2`).
3. Pre-caches `chrome-headless-shell` via Puppeteer — **required** for best-quality rendering via Chrome's `HeadlessExperimental.beginFrame` capture path.
4. Runs `npx hyperframes doctor` and reports the result.

See [references/troubleshooting.md](references/troubleshooting.md) if setup fails.

## Procedure

### 1. Plan before writing HTML

Before touching code, articulate at a high level:
- **What** — narrative arc, key moments, emotional beats
- **Structure** — compositions, tracks (video/audio/overlays), durations
- **Visual identity** — colors, fonts, motion character (explosive / cinematic / fluid / technical)
- **Hero frame** — for each scene, the moment when the most elements are simultaneously visible. This is the static layout you'll build first.

If the user hasn't specified a visual style, ask three questions before writing HTML: mood, light/dark, any brand colors/fonts/references. Write a short `DESIGN.md` at the project root capturing the answers.

### 2. Scaffold

```bash
npx hyperframes init my-video --non-interactive
```

Templates: `blank`, `warm-grain`, `play-mode`, `swiss-grid`, `vignelli`, `decision-tree`, `kinetic-type`, `product-promo`, `nyt-graph`. Pass `--example <name>` to pick one, `--video clip.mp4` or `--audio track.mp3` to seed with media.

### 3. Layout before animation

Write the static HTML+CSS for the **hero frame first** — no GSAP yet. The `.scene-content` container must fill the scene (`width:100%; height:100%; padding:Npx`) with `display:flex` + `gap`. Use padding to push content inward — never `position: absolute; top: Npx` on a content container (content overflows when taller than the remaining space).

Only after the hero frame looks right, add `gsap.from()` entrances (animate **to** the CSS position) and `gsap.to()` exits (animate **from** it).

See [references/composition.md](references/composition.md) for the full data-attribute schema and composition rules.

### 4. Animate with GSAP

Every composition must:
- Register its timeline: `window.__timelines["<composition-id>"] = tl`
- Start paused: `gsap.timeline({ paused: true })` — the player controls playback
- Use finite `repeat` values (no `repeat: -1` — breaks the capture engine). Calculate: `repeat: Math.ceil(duration / cycleDuration) - 1`.
- Be deterministic — no `Math.random()`, `Date.now()`, or wall-clock logic. Use a seeded PRNG if you need pseudo-randomness.
- Build synchronously — no `async`/`await`, `setTimeout`, or Promises around timeline construction.

See [references/gsap.md](references/gsap.md) for the core GSAP API (tweens, eases, stagger, timelines).

### 5. Transitions between scenes

Multi-scene compositions require transitions. Rules:
1. **Always use a transition between scenes** — no jump cuts.
2. **Always use entrance animations** on every scene element (`gsap.from(...)`).
3. **Never use exit animations** except on the final scene — the transition IS the exit.
4. The final scene may fade out.

Use `npx hyperframes add <transition-name>` to install shader transitions (`flash-through-white`, `liquid-wipe`, etc.). Full list: `npx hyperframes add --list`.

### 6. Audio, captions, TTS

- **Audio:** always a separate `<audio>` element (video is `muted playsinline`).
- **Captions:** run `npx hyperframes transcribe audio.mp3` to get word-level timings, then render with the captions component. See `references/composition.md` for the captions data attributes.
- **TTS:** `npx hyperframes tts "Script text" --voice af_nova --output narration.wav`. List voices with `--list`.

### 7. Lint, preview, render

```bash
npx hyperframes lint              # catches missing data-composition-id, overlapping tracks, unregistered timelines
npx hyperframes preview           # live browser preview
npx hyperframes render --quality draft --output draft.mp4    # fast iteration
npx hyperframes render --quality high --output final.mp4     # final delivery
```

`hyperframes validate` runs a WCAG contrast audit — screenshots at 5 timestamps, samples pixels behind every text element, warns on <4.5:1 ratios.

### 8. Website-to-video (if the user gives a URL)

Use the 7-step capture-to-video workflow in [references/website-to-video.md](references/website-to-video.md): capture → DESIGN.md → SCRIPT.md → storyboard → composition → render → deliver.

## Pitfalls

- **`HeadlessExperimental.beginFrame' wasn't found`** — Chromium 147+ removed this protocol. Ensure you're on `hyperframes@>=0.4.2` (auto-detects and falls back to screenshot mode). Escape hatch: `export PRODUCER_FORCE_SCREENSHOT=true`. See [hyperframes#294](https://github.com/heygen-com/hyperframes/issues/294) and [references/troubleshooting.md](references/troubleshooting.md).
- **System Chrome (not `chrome-headless-shell`)** — renders hang for 120s then timeout. Run `npx puppeteer browsers install chrome-headless-shell` (setup.sh does this). `hyperframes doctor` reports which binary will be used.
- **`repeat: -1` anywhere** — breaks the capture engine. Always compute a finite repeat count.
- **`gsap.set()` on clip elements that enter later** — the element doesn't exist at page load. Use `tl.set(selector, vars, timePosition)` inside the timeline instead, at or after the clip's `data-start`.
- **`<br>` inside content text** — forced breaks don't know the rendered font width, so natural wrap + `<br>` double-breaks. Use `max-width` to let text wrap. Exception: short display titles where each word is deliberately on its own line.
- **Animating `visibility` or `display`** — GSAP can't tween these. Use `autoAlpha` (handles both visibility and opacity).
- **Calling `video.play()` or `audio.play()`** — the framework owns playback. Never call these yourself.
- **Building timelines async** — the capture engine reads `window.__timelines` synchronously after page load. Never wrap timeline construction in `async`, `setTimeout`, or a Promise.
- **Standalone `index.html` wrapped in `<template>`** — hides all content from the browser. Only **sub-compositions** loaded via `data-composition-src` use `<template>`.
- **Using video for audio** — always muted `<video>` + separate `<audio>`.

## Verification

After `render` completes, verify:

1. `final.mp4` exists and has non-zero size: `ls -lh final.mp4`
2. Duration matches the composition's `data-duration`: `ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 final.mp4`
3. Visual check: open with a media player, or extract a mid-composition frame: `ffmpeg -i final.mp4 -ss 00:00:05 -vframes 1 preview.png`
4. Audio present if expected: `ffprobe -v error -show_streams -select_streams a -of default=nw=1:nk=1 final.mp4 | head -1`

If `hyperframes render` fails, run `npx hyperframes doctor` and attach its output when reporting.

## References

- [composition.md](references/composition.md) — data attributes, timeline contract, non-negotiable rules, typography/asset rules
- [cli.md](references/cli.md) — every CLI command (init, lint, preview, render, transcribe, tts, doctor, browser, info, upgrade, benchmark)
- [gsap.md](references/gsap.md) — GSAP core API for HyperFrames (tweens, eases, stagger, timelines, matchMedia)
- [website-to-video.md](references/website-to-video.md) — 7-step capture-to-video workflow
- [troubleshooting.md](references/troubleshooting.md) — OpenClaw fix, env vars, common render errors
