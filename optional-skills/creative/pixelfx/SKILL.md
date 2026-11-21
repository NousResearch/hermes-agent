---
name: pixelfx
description: Pixel-art game VFX and sprite sheets in code, upstream-kept.
version: 0.3.0
author: patch-gg
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [pixel-art, vfx, sprites, sprite-sheet, game-dev, animation, godot, unity, phaser, node]
    category: creative
    related_skills: [pixel-art, p5js]
    upstream:
      repo: patch-gg/pixelfx
      path: skills/pixelfx
---

# pixelfx (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [patch-gg/pixelfx](https://github.com/patch-gg/pixelfx) (MIT). `hermes skills
> install official/creative/pixelfx` pulls the current `skills/pixelfx/SKILL.md`
> live from that repo (quarantined and scanned like any hub install). This
> directory holds only the catalog metadata, so the copy never goes stale.

pixelfx is a library of 244 code-drawn pixel-art effects (fire, explosions,
water, ice, lightning, melee, projectiles, spawn/death, weather, UI, pickups,
transitions…), each one JS file that draws a color buffer and an additive light
buffer. Effects can be recolored, reseeded, mirrored and exported as sprite
sheets for Godot, Unity, Phaser or a canvas game. The skill also teaches a
draft → render → look → refine loop for authoring new effects: render a contact
sheet PNG, read it with the vision tool, adjust the code, repeat.

## Prerequisites

- Node.js 18+ and `git`. No `npm install`: the tools use only Node.
- `ffmpeg` optional (GIF/MP4 previews).
- The installed skill tells the agent to clone the library next to the
  project (`git clone https://github.com/patch-gg/pixelfx`) and run
  `node tools/render.mjs fire/torch` as a smoke test. In Hermes, read the
  rendered `records/<category>/<name>/contact.png` with `vision_analyze`.

Pick `pixel-art` instead to convert an existing image into retro pixel art;
pick pixelfx for animated game effects and sprite sheets.
