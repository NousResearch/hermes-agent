---
name: brag-slim
description: Launch video from a project or URL, upstream-maintained.
version: 0.4.0
author: Shunit Haviv Hakimi (shunithaviv)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [video, launch-video, marketing, motion-graphics, share-copy]
    category: creative
    related_skills: [brag, hyperframes]
    upstream:
      repo: latent-spaces/brag
      path: skills/brag-slim
---

# Brag Slim (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [latent-spaces/brag](https://github.com/latent-spaces/brag): the project
> ships `/brag-slim` as a single `SKILL.md` under `skills/brag-slim/`. `hermes
> skills install official/creative/brag-slim` pulls the current file live from
> that repo (quarantined and scanned like any hub install) — this directory
> holds only the catalog metadata, so the vendored copy can never go stale.

`/brag-slim` is the lean `/brag`, written for Claude Opus 5.5: no Hyperframes,
no bundled assets. The model builds the whole launch video itself (story,
visuals, music, sound effects, mix and render) with the tools already on the
machine. It takes the current project directory or a website URL, reuses the
source's real UI, fonts and assets instead of redrawing them, and writes
`brag-output/` with the plan, the rendered video and share copy. Options:
`--tone`, `--format landscape|vertical|square`, `--duration`.

`brag` adds the classic Hyperframes workflow, the bundled soundtrack and SFX,
and `--voice` narration (see its prerequisites). It includes this skill and
hands off to it on Opus 5.5.

## Prerequisites

- No fixed toolchain: the model builds with whatever is installed. In practice
  that means FFmpeg for the audio mix and encode, plus a headless browser
  (Node with Playwright or Puppeteer, or Chrome) to draw frames from HTML and
  to capture websites that render their page with JavaScript.

Full documentation: https://github.com/latent-spaces/brag#readme
