---
name: hki
description: "Use native HKI commands to inventory workspace sources, build a manifest, and write a source report."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [HKI, source inventory, workspace, manifest, report]
    category: productivity
    requires_toolsets: [terminal, files]
---

# HKI Source Inventory

HKI — Human Knowledge Infrastructure — is a native project/workspace knowledge subsystem. The current slice inventories workspace sources, creates a stable source manifest, and writes a basic source report.

This is not semantic search, entity indexing, dossier generation, vault/wiki publication, cron refresh, or frontend/UI work yet.

## When To Use

Use this skill when the user asks to:

- assess a repo, project, source corpus, or workspace
- build or update a source inventory
- prepare evidence for later HKI work
- understand what files or sources exist in a workspace
- create a basic source report

## When Not To Use

Do not use this skill for:

- ordinary code edits or simple file reads
- semantic search across conversations
- entity extraction
- dossier generation
- vault/wiki publication
- background cron refresh
- frontend/UI work

## Commands

Run these from any shell, choosing the intended workspace as `--cwd`:

```bash
hermes hki inventory --cwd <path>
hermes hki manifest --cwd <path>
hermes hki report sources --cwd <path>
```

Expected outputs:

```text
<cwd>/.hermes/hki/inventory.json
<cwd>/.hermes/hki/manifest.json
<cwd>/.hermes/hki/reports/sources.md
```

## Recommended Workflow

1. Resolve the intended workspace/cwd with the user or from the active task context.
2. Run `hermes hki inventory --cwd <path>`.
3. Run `hermes hki manifest --cwd <path>`.
4. Run `hermes hki report sources --cwd <path>`.
5. Read the generated report first; read the manifest only as needed.
6. Cite generated paths, `source_id` values, and report sections when summarizing.
7. Avoid loading large manifests or inventories wholesale into prompt context unless the user explicitly needs that detail.

## Safety And Limitations

- Scope is currently cwd-rooted; pass the workspace root explicitly when in doubt.
- Generated files live under `.hermes/hki/`.
- Source IDs are deterministic but path-based, so renames change IDs.
- Reports may be stale if files changed after manifest generation.
- Secret exclusion is currently minimal: `.env` and `.env.*` are excluded, but this is not a full secret scanner.
- Treat HKI output as a source inventory/report, not as a semantic dossier.
