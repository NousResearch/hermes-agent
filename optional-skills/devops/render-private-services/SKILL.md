---
name: render-private-services
description: Configure internal services on Render private networking.
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
      path: skills/render-private-services
---

# Render Private Services (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-private-services` skill](https://github.com/render-oss/skills/tree/main/skills/render-private-services).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Configure internal services on Render private networking.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-private-services
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-private-services) for prerequisites and usage.
