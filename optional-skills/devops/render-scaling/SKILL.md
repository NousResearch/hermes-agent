---
name: render-scaling
description: Configure Render instance sizes and autoscaling.
version: 1.0.0
author: Ojus Save (@ojusave), Render
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [render, cloud, deployment, devops]
    category: devops
    upstream:
      repo: render-oss/skills
      path: skills/render-scaling
---

# Render Scaling (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-scaling` skill](https://github.com/render-oss/skills/tree/main/skills/render-scaling).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Configure Render instance sizes and autoscaling.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-scaling
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-scaling) for prerequisites and usage.
