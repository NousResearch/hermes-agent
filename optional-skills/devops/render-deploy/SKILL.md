---
name: render-deploy
description: Deploy applications and prepare Render Blueprints.
version: 1.1.0
author: Ojus Save (@ojusave), Render
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [render, cloud, deployment, devops]
    category: devops
    upstream:
      repo: render-oss/skills
      path: skills/render-deploy
---

# Render Deploy (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-deploy` skill](https://github.com/render-oss/skills/tree/main/skills/render-deploy).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Deploy applications and prepare Render Blueprints.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-deploy
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-deploy) for prerequisites and usage.
