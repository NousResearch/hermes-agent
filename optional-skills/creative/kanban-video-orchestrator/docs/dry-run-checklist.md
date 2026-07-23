# Dry-Run Checklist

Use this after generating `setup.sh`, `brief.md`, and `TEAM.md`, but before running the real setup.

## 1. Plan validation
- The generator exits cleanly with no validation errors.
- `slug` and `tenant` match the intended project identifier.
- All required `brief_extra` and `taste` fields are present.
- Asset paths point to real local files, not placeholder paths.

## 2. Generated brief.md
- Title, slug, and tenant match the plan.
- Scene table contains the expected number of rows.
- Deliverables table contains every required export, not just the primary one.
- Audio, tone, and aesthetic rules read like the intended contract.

## 3. Generated TEAM.md
- Every profile in the plan appears in the team section.
- The task graph reflects the intended dependencies.
- Multi-renderer projects route scenes to the correct renderer profiles.
- The workspace rule block includes `--workspace dir:<path>` and `--tenant <slug>` in the Hermes CLI child-task command shape.

## 4. Generated setup.sh
- The header shows the correct title, slug, and tenant.
- `check_key` lines only reference APIs actually used by the project.
- The workspace tree includes `taste/style-frames`, `audio/voiceover`, `audio/sfx`, asset subfolders, `scenes/`, `checkpoints/`, `tools/`, and `output/`.
- Profile creation commands exist for every planned profile.
- Profile config commands set `toolsets`, `platform_toolsets.cli`, and `skills.always_load`.
- No line patches `terminal.cwd` or `approvals.mode`.
- The initial kanban task uses `--workspace dir:"$WORKSPACE"` and `--tenant "$TENANT"`.
- The assignee for the initial task is the director profile.

## 5. SOUL.md content
- The director SOUL explicitly says not to execute work directly.
- Specialists describe concrete outputs and workspace conventions.
- Long-running roles mention heartbeats.
- Toolsets and skills match the intended role configuration.

## 6. Assets and prerequisites
- Every file referenced in the asset-copy block exists locally.
- Required API keys are available in Hermes `.env` or Keychain.
- `python3 -c "import yaml"` succeeds in the target environment.
- `hermes profile create`, `hermes kanban create`, and `hermes kanban list` exist in your local Hermes CLI.

## 7. Safe first execution
- Run the generated setup once in a throwaway or test project first.
- Inspect the created workspace before launching a costly production run.
- Confirm the initial director task appears under the correct tenant.
- Use `hermes kanban watch --tenant <slug>` immediately after launch.
