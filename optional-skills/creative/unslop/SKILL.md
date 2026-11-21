---
name: unslop
description: Redesign vibecoded, AI-looking UI screens, upstream-kept.
version: 1.0.0
author: Damir Sokolovsky (heydamir)
license: CC-BY-4.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [design, frontend, ui, redesign, anti-slop, landing-page, dashboard, tailwind, react, css]
    category: creative
    related_skills: [impeccable, claude-design, popular-web-designs]
    upstream:
      repo: heydamir/unslop
      path: skills/unslop
---

# unslop (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [heydamir/unslop](https://github.com/heydamir/unslop) (CC BY 4.0, by Damir
> Sokolovsky, [@heydamir](https://x.com/heydamir)). `hermes skills install
> official/creative/unslop` pulls the current `skills/unslop/` bundle live from
> that repo (quarantined and scanned like any hub install). This directory
> holds only the catalog metadata, so the copy never goes stale.

unslop turns a vibecoded screen (purple gradients, glows, gradient text, emoji
labels, "New ✨" badges, uneven spacing) into a designed interface that keeps
the product's features and content. A run audits the screen against a list of
AI-UI tells, writes a five-line direction derived from the product (not from
trends), waits for the user to confirm it, then fixes the screen in nine
ordered passes: spacing, color, icons, copy, type, layout, details, visuals,
motion. Light slop gets a polish pass only; a clean screen is left alone.

## Prerequisites

- Works on HTML/CSS, React and Tailwind code with no extra tooling.
- Optional, for a live URL with no code: Node.js 18+ and Playwright
  (`npm install -D playwright && npx playwright install chromium`) for
  `scripts/snapshot.mjs`; `ffmpeg` too for the before/after story video.
- A landing-page hero can use a generated scene: in Hermes that is the
  `image_generate` tool. Check renders with the browser tools or
  `vision_analyze` on a screenshot.

Pick `impeccable` for a broad design toolbox (23 sub-commands plus a
deterministic anti-pattern detector); pick unslop for one end-to-end redesign
of a specific AI-looking screen with a confirmed direction.
