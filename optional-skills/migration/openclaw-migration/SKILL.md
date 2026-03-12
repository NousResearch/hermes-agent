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

## CLI Command

For a quick, non-interactive migration, use the built-in CLI command:

```bash
hermes claw migrate              # Full interactive migration
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
hermes claw migrate --source /custom/path/.openclaw  # Custom source
```

The CLI command runs the same migration script described below. Use this skill (via the agent) when you want an interactive, guided migration with dry-run previews and per-item conflict resolution.

**First-time setup:** The `hermes setup` wizard automatically detects `~/.openclaw` and offers migration before configuration begins.

## What this skill does

It uses `scripts/openclaw_to_hermes.py` to:

- import `SOUL.md` into the Hermes home directory as `SOUL.md`
- transform OpenClaw `MEMORY.md` and `USER.md` into Hermes memory entries
- merge OpenClaw command approval patterns into Hermes `command_allowlist`
- migrate Hermes-compatible messaging settings such as `TELEGRAM_ALLOWED_USERS` and `MESSAGING_CWD`
- copy OpenClaw skills into `~/.hermes/skills/openclaw-imports/`
- optionally copy the OpenClaw workspace instructions file into a chosen Hermes workspace
- mirror compatible workspace assets such as `workspace/tts/` into `~/.hermes/tts/`
- archive non-secret docs that do not have a direct Hermes destination
- produce a structured report listing migrated items, conflicts, skipped items, and reasons

## Path resolution

The helper script lives in this skill directory at:

- `scripts/openclaw_to_hermes.py`

When this skill is installed from the Skills Hub, the normal location is:

- `~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py`

Do not guess a shorter path like `~/.hermes/skills/openclaw-migration/...`.

Before running the helper:

1. Prefer the installed path under `~/.hermes/skills/migration/openclaw-migration/`.
2. If that path fails, inspect the installed skill directory and resolve the script relative to the installed `SKILL.md`.
3. Only use `find` as a fallback if the installed location is missing or the skill was moved manually.
4. When calling the terminal tool, do not pass `workdir: "~"`. Use an absolute directory such as the user's home directory, or omit `workdir` entirely.

With `--migrate-secrets`, it will also import a small allowlisted set of Hermes-compatible secrets, currently:

- `TELEGRAM_BOT_TOKEN`

## Default workflow

1. Inspect first with a dry run.
2. Present a simple summary of what can be migrated, what cannot be migrated, and what would be archived.
3. If the `clarify` tool is available, use it for user decisions instead of asking for a free-form prose reply.
4. If the dry run finds imported skill directory conflicts, ask how those should be handled before executing.
5. Ask the user to choose between the two supported migration modes before executing.
6. Ask for a target workspace path only if the user wants the workspace instructions file brought over.
7. Execute the migration with the matching preset and flags.
8. Summarize the results, especially:
   - what was migrated
   - what was archived for manual review
   - what was skipped and why

## User interaction protocol

Hermes CLI supports the `clarify` tool for interactive prompts, but it is limited to:

- one choice at a time
- up to 4 predefined choices
- an automatic `Other` free-text option

It does **not** support true multi-select checkboxes in a single prompt.

For every `clarify` call:

- always include a non-empty `question`
- include `choices` only for real selectable prompts
- keep `choices` to 2-4 plain string options
- never emit placeholder or truncated options such as `...`
- never pad or stylize choices with extra whitespace
- never include fake form fields in the question such as `enter directory here`, blank lines to fill in, or underscores like `_____`
- for open-ended path questions, ask only the plain sentence; the user types in the normal CLI prompt below the panel

If a `clarify` call returns an error, inspect the error text, correct the payload, and retry once with a valid `question` and clean choices.

When `clarify` is available and the dry run reveals any required user decision, your **next action must be a `clarify` tool call**.
Do not end the turn with a normal assistant message such as:

- "Let me present the choices"
- "What would you like to do?"
- "Here are the options"

If a user decision is required, collect it via `clarify` before producing more prose.
If multiple unresolved decisions remain, do not insert an explanatory assistant message between them. After one `clarify` response is received, your next action should usually be the next required `clarify` call.

Treat `workspace-agents` as an unresolved decision whenever the dry run reports:

- `kind="workspace-agents"`
- `status="skipped"`
- reason containing `No workspace target was provided`

In that case, you must ask about workspace instructions before execution. Do not silently treat that as a decision to skip.

Because of that limitation, use this simplified decision flow:

1. For `SOUL.md` conflicts, use `clarify` with choices such as:
   - `keep existing`
   - `overwrite with backup`
   - `review first`
2. If the dry run shows one or more `kind="skill"` items with `status="conflict"`, use `clarify` with choices such as:
   - `keep existing skills`
   - `overwrite conflicting skills with backup`
   - `import conflicting skills under renamed folders`
3. For workspace instructions, use `clarify` with choices such as:
   - `skip workspace instructions`
   - `copy to a workspace path`
   - `decide later`
4. If the user chooses to copy workspace instructions, ask a follow-up open-ended `clarify` question requesting an **absolute path**.
5. If the user chooses `skip workspace instructions` or `decide later`, proceed without `--workspace-target`.
5. For migration mode, use `clarify` with these 3 choices:
   - `user-data only`
   - `full compatible migration`
   - `cancel`
6. `user-data only` means: migrate user data and compatible config, but do **not** import allowlisted secrets.
7. `full compatible migration` means: migrate the same compatible user data plus the allowlisted secrets when present.
8. If `clarify` is not available, ask the same question in normal text, but still constrain the answer to `user-data only`, `full compatible migration`, or `cancel`.

Execution gate:

- Do not execute while a `workspace-agents` skip caused by `No workspace target was provided` remains unresolved.
- The only valid ways to resolve it are:
  - user explicitly chooses `skip workspace instructions`
  - user explicitly chooses `decide later`
  - user provides a workspace path after choosing `copy to a workspace path`
- Absence of a workspace target in the dry run is not itself permission to execute.
- Do not execute while any required `clarify` decision remains unresolved.

Use these exact `clarify` payload shapes as the default pattern:

- `{"question":"Your existing SOUL.md conflicts with the imported one. What should I do?","choices":["keep existing","overwrite with backup","review first"]}`
- `{"question":"One or more imported OpenClaw skills already exist in Hermes. How should I handle those skill conflicts?","choices":["keep existing skills","overwrite conflicting skills with backup","import conflicting skills under renamed folders"]}`
- `{"question":"Choose migration mode: migrate only user data, or run the full compatible migration including allowlisted secrets?","choices":["user-data only","full compatible migration","cancel"]}`
- `{"question":"Do you want to copy the OpenClaw workspace instructions file into a Hermes workspace?","choices":["skip workspace instructions","copy to a workspace path","decide later"]}`
- `{"question":"Please provide an absolute path where the workspace instructions should be copied."}`

## Decision-to-command mapping

Map user decisions to command flags exactly:

- If the user chooses `keep existing` for `SOUL.md`, do **not** add `--overwrite`.
- If the user chooses `overwrite with backup`, add `--overwrite`.
- If the user chooses `review first`, stop before execution and review the relevant files.
- If the user chooses `keep existing skills`, add `--skill-conflict skip`.
- If the user chooses `overwrite conflicting skills with backup`, add `--skill-conflict overwrite`.
- If the user chooses `import conflicting skills under renamed folders`, add `--skill-conflict rename`.
- If the user chooses `user-data only`, execute with `--preset user-data` and do **not** add `--migrate-secrets`.
- If the user chooses `full compatible migration`, execute with `--preset full --migrate-secrets`.
- Only add `--workspace-target` if the user explicitly provided an absolute workspace path.
- If the user chooses `skip workspace instructions` or `decide later`, do not add `--workspace-target`.

Before executing, restate the exact command plan in plain language and make sure it matches the user's choices.

## Post-run reporting rules

After execution, treat the script's JSON output as the source of truth.

1. Base all counts on `report.summary`.
2. Only list an item under "Successfully Migrated" if its `status` is exactly `migrated`.
3. Do not claim a conflict was resolved unless the report shows that item as `migrated`.
4. Do not say `SOUL.md` was overwritten unless the report item for `kind="soul"` has `status="migrated"`.
5. If `report.summary.conflict > 0`, include a conflict section instead of silently implying success.
6. If counts and listed items disagree, fix the list to match the report before responding.
7. Include the `output_dir` path from the report when available so the user can inspect `report.json`, `summary.md`, backups, and archived files.
8. For memory or user-profile overflow, do not say the entries were archived unless the report explicitly shows an archive path. If `details.overflow_file` exists, say the full overflow list was exported there.
9. If a skill was imported under a renamed folder, report the final destination and mention `details.renamed_from`.
10. If `report.skill_conflict_mode` is present, use it as the source of truth for the selected imported-skill conflict policy.
11. If an item has `status="skipped"`, do not describe it as overwritten, backed up, migrated, or resolved.
12. If `kind="soul"` has `status="skipped"` with reason `Target already matches source`, say it was left unchanged and do not mention a backup.
13. If a renamed imported skill has an empty `details.backup`, do not imply the existing Hermes skill was renamed or backed up. Say only that the imported copy was placed in the new destination and reference `details.renamed_from` as the pre-existing folder that remained in place.

## Migration presets

Prefer these two presets in normal use:

- `user-data`
- `full`

`user-data` includes:

- `soul`
- `workspace-agents`
- `memory`
- `user-profile`
- `messaging-settings`
- `command-allowlist`
- `skills`
- `tts-assets`
- `archive`

`full` includes everything in `user-data` plus:

- `secret-settings`

The helper script still supports category-level `--include` / `--exclude`, but treat that as an advanced fallback rather than the default UX.

## Commands

Dry run with full discovery:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py
```

When using the terminal tool, prefer an absolute invocation pattern such as:

```json
{"command":"python3 /home/USER/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py","workdir":"/home/USER"}
```

Dry run with the user-data preset:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --preset user-data
```

Execute a user-data migration:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset user-data --skill-conflict skip
```

Execute a full compatible migration:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset full --migrate-secrets --skill-conflict skip
```

Execute with workspace instructions included:

```bash
python3 ~/.hermes/skills/migration/openclaw-migration/scripts/openclaw_to_hermes.py --execute --preset user-data --skill-conflict rename --workspace-target "/absolute/workspace/path"
```

Do not use `$PWD` or the home directory as the workspace target by default. Ask for an explicit workspace path first.

## Important rules

1. Run a dry run before writing unless the user explicitly says to proceed immediately.
2. Do not migrate secrets by default. Tokens, auth blobs, device credentials, and raw gateway config should stay out of Hermes unless the user explicitly asks for secret migration.
3. Do not silently overwrite non-empty Hermes targets unless the user explicitly wants that. The helper script will preserve backups when overwriting is enabled.
4. Always give the user the skipped-items report. That report is part of the migration, not an optional extra.
5. Prefer the primary OpenClaw workspace (`~/.openclaw/workspace/`) over `workspace.default/`. Only use the default workspace as fallback when the primary files are missing.
6. Even in secret-migration mode, only migrate secrets with a clean Hermes destination. Unsupported auth blobs must still be reported as skipped.
7. If the dry run shows a large asset copy, a conflicting `SOUL.md`, or overflowed memory entries, call those out separately before execution.
8. Default to `user-data only` if the user is unsure.
9. Only include `workspace-agents` when the user has explicitly provided a destination workspace path.
10. Treat category-level `--include` / `--exclude` as an advanced escape hatch, not the normal flow.
11. Do not end the dry-run summary with a vague “What would you like to do?” if `clarify` is available. Use structured follow-up prompts instead.
12. Do not use an open-ended `clarify` prompt when a real choice prompt would work. Prefer selectable choices first, then free text only for absolute paths or file review requests.
13. After a dry run, never stop after summarizing if there is still an unresolved decision. Use `clarify` immediately for the highest-priority blocking decision.
14. Priority order for follow-up questions:
    - `SOUL.md` conflict
    - imported skill conflicts
    - migration mode
    - workspace instructions destination
15. Do not promise to present choices later in the same message. Present them by actually calling `clarify`.
16. After the migration-mode answer, explicitly check whether `workspace-agents` is still unresolved. If it is, your next action must be the workspace-instructions `clarify` call.
17. After any `clarify` answer, if another required decision remains, do not narrate what was just decided. Ask the next required question immediately.

## Expected result

After a successful run, the user should have:

- Hermes persona state imported
- Hermes memory files populated with converted OpenClaw knowledge
- OpenClaw skills available under `~/.hermes/skills/openclaw-imports/`
- a migration report showing any conflicts, omissions, or unsupported data
