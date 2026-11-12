---
name: render-env-vars
description: Manage environment variables and secrets on Render.
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
      path: skills/render-env-vars
---

# Render Env Vars (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-env-vars` skill](https://github.com/render-oss/skills/tree/main/skills/render-env-vars).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Manage environment variables and secrets on Render.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-env-vars
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-env-vars) for prerequisites and usage.
