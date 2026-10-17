---
name: render-workflows
description: Develop and deploy background tasks with Render Workflows.
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
      path: skills/render-workflows
---

# Render Workflows (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-workflows` skill](https://github.com/render-oss/skills/tree/main/skills/render-workflows).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Develop and deploy background tasks with Render Workflows.

The upstream instructions include a command that downloads and runs the
Render CLI installer. Hermes flags it with the `curl_pipe_shell` scanner
rule. Review the scan report and the upstream command before installing.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/devops/render-workflows
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-workflows) for prerequisites and usage.
