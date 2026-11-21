---
name: render-debug
description: Diagnose failed Render deployments and runtime errors.
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
      path: skills/render-debug
---

# Render Debug (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-debug` skill](https://github.com/render-oss/skills/tree/main/skills/render-debug).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Diagnose failed Render deployments and runtime errors.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-debug
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-debug) for prerequisites and usage.
