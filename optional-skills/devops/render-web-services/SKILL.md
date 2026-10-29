---
name: render-web-services
description: Configure Render web services and health checks.
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
      path: skills/render-web-services
---

# Render Web Services (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-web-services` skill](https://github.com/render-oss/skills/tree/main/skills/render-web-services).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Configure Render web services and health checks.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-web-services
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-web-services) for prerequisites and usage.
