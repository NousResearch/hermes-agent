---
name: render-disks
description: Configure persistent disks and snapshots on Render.
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
      path: skills/render-disks
---

# Render Disks (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-disks` skill](https://github.com/render-oss/skills/tree/main/skills/render-disks).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Configure persistent disks and snapshots on Render.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-disks
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-disks) for prerequisites and usage.
