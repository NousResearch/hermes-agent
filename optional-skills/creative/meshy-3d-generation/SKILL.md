---
name: meshy-3d-generation
description: 3D models, textures, rigs and animation with Meshy.
version: 0.7.0
author: Xubin Du (Arlieeee), Meshy
license: MIT
platforms: [macos, linux, windows]
prerequisites:
  commands: [node, npm]
metadata:
  hermes:
    tags: [3d, 3d-generation, text-to-3d, image-to-3d, texturing, rigging, animation, meshy]
    category: creative
    requires_toolsets: [terminal]
    related_skills: [meshy-3d-printing]
    upstream:
      repo: meshy-dev/meshy-3d-agent
      path: skills/meshy-3d-generation
---

# Meshy 3D Generation (upstream-maintained)

> **Catalog stub.** This entry is maintained upstream at
> [meshy-dev/meshy-3d-agent](https://github.com/meshy-dev/meshy-3d-agent): the project ships the
> skill directory (`skills/meshy-3d-generation/`) with its setup, pipeline, delivery and
> troubleshooting references, released together with the Meshy CLI version it pins. `hermes
> skills install official/creative/meshy-3d-generation` pulls the current tree live from that
> repo (quarantined and scanned like any hub install) — this directory holds only the catalog
> metadata, so the vendored copy can never go stale.

The skill drives the [Meshy CLI](https://www.npmjs.com/package/meshy-cli) (`meshy`) to turn a
request into a delivered asset: text or image to 3D, retexturing, remeshing and LODs, format
conversion and resizing, UV unwrapping, rigging, animation from Meshy's preset library or a
generated motion, and reference images. It saves the files at the path the user names, shows the
rendered preview, reports the credits each stage consumed, and reuses existing tasks for
follow-ups instead of generating again.

For physical prints (white or multicolour, part splitting, Creative Lab products, slicer
handoff), use `meshy-3d-printing`.

## Prerequisites

- Node.js 22.12+ with npm on `PATH`. The skill uses an installed `meshy` of the pinned version,
  or runs that version temporarily with `npm exec`; a global install is optional.
- A Meshy account. Sign-in is the CLI's browser device flow (`meshy auth login --device`): the
  user approves in the browser and nothing secret is pasted into the chat. An existing
  `MESHY_API_KEY` is honoured instead.
- Generation spends Meshy credits. The skill estimates from the CLI's planner or the published
  price list and asks before any paid stage the user has not agreed to.
- Network: the Meshy API and sign-in page, the media hosts of downloaded assets, and the npm
  registry when the CLI is installed or run temporarily.

Full documentation: https://github.com/meshy-dev/meshy-3d-agent#readme
