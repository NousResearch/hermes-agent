---
name: render-monitor
description: Check Render service health, metrics and logs.
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
      path: skills/render-monitor
---

# Render Monitor (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-monitor` skill](https://github.com/render-oss/skills/tree/main/skills/render-monitor).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Check Render service health, metrics and logs.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-monitor
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-monitor) for prerequisites and usage.
