---
name: render-docker
description: Build and deploy Docker containers on Render.
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
      path: skills/render-docker
---

# Render Docker (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-docker` skill](https://github.com/render-oss/skills/tree/main/skills/render-docker).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Build and deploy Docker containers on Render.

The upstream Dockerfile example triggers Hermes's `destructive_root_rm`
scanner rule for removing a temporary `/wheels` directory during a container
build. Review the scan report and the upstream example before installing.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-docker
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-docker) for prerequisites and usage.
