---
name: render-migrate-from-heroku
description: Migrate Heroku applications and databases to Render.
version: 1.0.0
author: Ojus Save (@ojusave), Render
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [render, cloud, deployment, devops]
    category: migration
    upstream:
      repo: render-oss/skills
      path: skills/render-migrate-from-heroku
---

# Render Migrate From Heroku (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-migrate-from-heroku` skill](https://github.com/render-oss/skills/tree/main/skills/render-migrate-from-heroku).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Migrate Heroku applications and databases to Render.

The upstream instructions include a command that downloads and runs the
Render CLI installer. Hermes flags it with the `curl_pipe_shell` scanner
rule. Review the scan report and the upstream command before installing.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/migration/render-migrate-from-heroku
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-migrate-from-heroku) for prerequisites and usage.
