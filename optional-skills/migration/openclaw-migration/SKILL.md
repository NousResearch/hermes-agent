---
name: openclaw-migration
description: Migrate a user's OpenClaw customization footprint into Hermes Agent. Imports Hermes-compatible memories, SOUL.md, command allowlists, user skills, and selected workspace assets from ~/.openclaw, then reports exactly what could not be migrated and why.
version: 1.0.0
author: Hermes Agent (Nous Research)
license: MIT
metadata:
  hermes:
    tags: [Migration, OpenClaw, Hermes, Memory, Persona, Import]
    related_skills: [hermes-agent]
---

# OpenClaw -> Hermes Migration

Use this skill when a user wants to move their OpenClaw setup into Hermes Agent with minimal manual cleanup.

## What this skill does

It uses `scripts/openclaw_to_hermes.py` to:

- import `SOUL.md` into `~/.hermes/SOUL.md`
- transform OpenClaw `MEMORY.md` and `USER.md` into Hermes memory entries
- merge OpenClaw command approval patterns into Hermes `command_allowlist`
- migrate Hermes-compatible messaging settings such as `TELEGRAM_ALLOWED_USERS` and `MESSAGING_CWD`
- copy OpenClaw skills into `~/.hermes/skills/openclaw-imports/`
- optionally copy the OpenClaw workspace `AGENTS.md` into a chosen Hermes workspace
- mirror compatible workspace assets such as `workspace/tts/` into `~/.hermes/tts/`
- archive non-secret docs that do not have a direct Hermes destination
- produce a structured report listing migrated items, conflicts, skipped items, and reasons

With `--migrate-secrets`, it will also import a small allowlisted set of Hermes-compatible secrets, currently:

- `TELEGRAM_BOT_TOKEN`

## Default workflow

1. Inspect first with a dry run.
2. Ask for a target workspace path if `AGENTS.md` should be brought over.
3. Execute the migration.
4. Summarize the results, especially:
   - what was migrated
   - what was archived for manual review
   - what was skipped and why

## Commands

Dry run:

```bash
python3 SKILL_DIR/scripts/openclaw_to_hermes.py --workspace-target "$PWD"
```

Execute:

```bash
python3 SKILL_DIR/scripts/openclaw_to_hermes.py --execute --workspace-target "$PWD"
```

Execute with Hermes-compatible secret migration enabled:

```bash
python3 SKILL_DIR/scripts/openclaw_to_hermes.py --execute --migrate-secrets --workspace-target "$PWD"
```

If the user does not want to import workspace instructions into the current directory, omit `--workspace-target`.

## Important rules

1. Run a dry run before writing unless the user explicitly says to proceed immediately.
2. Do not migrate secrets by default. Tokens, auth blobs, device credentials, and raw gateway config should stay out of Hermes unless the user explicitly asks for secret migration.
3. Do not silently overwrite non-empty Hermes targets unless the user explicitly wants that. The helper script will preserve backups when overwriting is enabled.
4. Always give the user the skipped-items report. That report is part of the migration, not an optional extra.
5. Prefer the primary OpenClaw workspace (`~/.openclaw/workspace/`) over `workspace.default/`. Only use the default workspace as fallback when the primary files are missing.
6. Even in secret-migration mode, only migrate secrets with a clean Hermes destination. Unsupported auth blobs must still be reported as skipped.

## Expected result

After a successful run, the user should have:

- Hermes persona state imported
- Hermes memory files populated with converted OpenClaw knowledge
- OpenClaw skills available under `~/.hermes/skills/openclaw-imports/`
- a migration report showing any conflicts, omissions, or unsupported data
