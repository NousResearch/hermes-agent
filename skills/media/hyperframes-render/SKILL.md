---
name: hyperframes-render
description: "Render HTML-defined video compositions to MP4/WebM via HyperFrames (HeyGen's HTML→video framework). Use when the user wants programmatic video creation from HTML/CSS/JS scenes (slides, motion graphics, AI generated reels)."
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [video, rendering, html, motion, hyperframes]
    related_skills: [ffmpeg-media, playwright-browser]
---

# HyperFrames

Installed at `hyperframes` (v0.6+). Renders HTML compositions to
video by driving a headless Chromium and stitching frames with FFmpeg.

## Quick start

```bash
mkdir my-video && cd my-video
hyperframes init                              # scaffolds a new project
hyperframes preview                           # live HTTP preview of the scene
hyperframes render --out my-video.mp4         # render to MP4 (1080p30 default)
hyperframes render --resolution 1920x1080 --fps 30 --duration 8s
```

## When to invoke
- Generating short marketing/reel videos from a script (with `vimax-video`).
- Producing animated KPI dashboards or animated screenshots.
- Pairing with `claude-code-cli` to author the scene HTML and then render.

## Skills bundled by HyperFrames
The HyperFrames repo at `~/paimon/tools/hyperframes/skills/` exposes additional
skill packs (`animejs`, `gsap`, `lottie`, `css-animations`,
`remotion-to-hyperframes`). They can be installed for Claude Code/Codex with:
```bash
hyperframes skills install --target claude
hyperframes skills install --target codex
```

## Dependencies
- Chromium via Playwright (`~/paimon/tools/playwright/browsers`) — already
  installed.
- FFmpeg at `ffmpeg` — already installed.
