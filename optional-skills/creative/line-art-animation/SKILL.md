---
name: line-art-animation
description: Animated ink-on-paper SVG scenes and videos, upstream-kept.
version: 0.1.0
author: JDhruv14
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [animation, svg, line-art, illustration, typescript, vite, day-night, video, ffmpeg]
    category: creative
    related_skills: [p5js, manim-video, ascii-video, sketch, hyperframes]
    upstream:
      repo: JDhruv14/line-art-animation
      path: .
---

# Line-art animation (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [JDhruv14/line-art-animation](https://github.com/JDhruv14/line-art-animation):
> the repository root is the skill directory (`SKILL.md` + `lib/`,
> `references/`, `scripts/`, `examples/`). `hermes skills install
> official/creative/line-art-animation` pulls the current tree live from that
> repo (quarantined and scanned like any hub install); this directory holds
> only the catalog metadata, so the library never lags upstream.

The skill builds animated line-art components: one thin pen on warm paper,
drawn live in SVG from TypeScript, with people and animals that walk, run,
cycle and react, scenes with wind, lit windows and a day/night cycle, and a
deterministic frame capture for rendering the result to video. It ships four
dependency-free library modules (drawing primitives, a planted-foot walk/run
solver, springs and pendulums, a time-of-day palette), a ~150-line single
figure example and a full canal-street scene to copy from.

## Prerequisites

- Node.js 22.18 or newer (runs the `.ts` scripts directly) and npm; projects
  use Vite + TypeScript (dev dependencies only).
- `ffmpeg` for video export.
- Chrome or Chromium for `scripts/capture.ts` (set `CHROME=/path/to/chrome`
  if it is not found).

## Pitfalls

- Verify motion in `?debug` mode with `window.__step(seconds)`: headless and
  background tabs throttle `requestAnimationFrame`, so free-running captures
  stutter.
- When sampling frames to check a walk, sample faster than half the gait
  period: frames exactly half a stride apart look identical and read as a
  frozen pose.
- On Linux hosts where the session D-Bus is unreachable (cron jobs,
  service accounts), headless Chrome can hang on every network load and
  `capture.ts` never gets past navigation. Run it as
  `env -u DBUS_SESSION_BUS_ADDRESS node scripts/capture.ts ...`.

After install, prefer `p5js` for generative/procedural canvas art and
`manim-video` for maths explainers; use this skill for hand-inked character
scenes and illustration components.

Full documentation: https://github.com/JDhruv14/line-art-animation#readme
