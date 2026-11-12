---
name: render-domains
description: Configure custom domains and certificates on Render.
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
      path: skills/render-domains
---

# Render Domains (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-domains` skill](https://github.com/render-oss/skills/tree/main/skills/render-domains).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Configure custom domains and certificates on Render.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-domains
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-domains) for prerequisites and usage.
