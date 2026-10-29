---
name: render-mcp
description: Set up and troubleshoot the Render MCP connection.
version: 1.0.0
author: Ojus Save (@ojusave), Render
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [render, cloud, deployment, devops]
    category: mcp
    upstream:
      repo: render-oss/skills
      path: skills/render-mcp
---

# Render MCP (upstream-maintained)

> **Catalog entry.** This installs Render's existing
> [`render-mcp` skill](https://github.com/render-oss/skills/tree/main/skills/render-mcp).
> The instructions and supporting files are maintained in
> [render-oss/skills](https://github.com/render-oss/skills).

Set up and troubleshoot the Render MCP connection.

Install from the terminal where Hermes is available:

```bash
hermes skills install official/mcp/render-mcp
```

Hermes downloads the complete upstream skill directory, scans it before
installation, and records the source revision. The installed instructions
come from Render; this catalog description is not installed in their place.

Read the [upstream skill](https://github.com/render-oss/skills/tree/main/skills/render-mcp) for prerequisites and usage.
