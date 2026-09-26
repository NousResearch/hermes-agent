# Skills and curator

The root guidance applies. Read `website/docs/developer-guide/creating-skills.md` and `website/docs/user-guide/features/skills.md`. Curator behavior is documented in `website/docs/user-guide/features/curator.md`.

## Placement

`skills/` contains bundled default skills. `optional-skills/` contains heavier or niche official skills that users install explicitly. Put a contribution in the smallest appropriate surface. Do not copy an optional skill into the default index.

## Authoring contract

`tests/skills/test_authoring_standards.py` enforces the mechanical rules. For every new or modernized skill:

- `description` is one sentence, at most 60 characters, and ends with a period. State the capability without repeating the name or using marketing language.
- Declare supported `platforms` from actual scripts and imports. Prefer cross-platform `pathlib`, `tempfile`, and process helpers over hardcoded temporary paths, `/proc`, terminal-only modules, shell heredocs, or OS package/service commands.
- Credit the contributing human first in `author`. The agent that helped write the file is not the contributor.
- Use the current section order from `creating-skills.md`. Keep the introduction short and put setup once under prerequisites.
- Refer to native Hermes tools and required MCP servers by their actual names. Use `search_files`, `read_file`, and `patch` instead of instructing the agent to reproduce wrapped shell utilities. Name third-party CLI prerequisites explicitly.
- Put repeatable parsing or non-trivial logic in `scripts/`. Keep supporting detail in `references/` and reusable output in `templates/`. Do not make the model recreate deterministic helpers on every load.
- Instructional loading remains whole-file. Do not add `offset` or `limit` to skill-loading tools.
- Tests live in `tests/skills/test_<skill>_skill.py`, use standard library, pytest, and mocks only, and avoid live network access.
- Add settings to `.env.example` only inside a clearly delimited block owned by the skill.

Supported frontmatter and configuration semantics are canonical in `creating-skills.md` and the loader. Treat the loader and schema tests as authority instead of copying field inventories here.

## Curator safety

`agent/curator.py` and `agent/curator_backup.py` own review and snapshots. `hermes_cli/curator.py` owns commands. `tools/skill_usage.py` owns usage state.

- Automatic LLM review receives only agent-created skills it can read and write. Bundled, hub-installed, disabled, and externally managed skills stay out of that candidate set.
- Deterministic built-in pruning, when enabled, is separate from LLM candidate selection.
- Automatic lifecycle never deletes a skill. Archive is the maximum automatic transition and remains restorable.
- Pinned skills are exempt from automatic transition and LLM review. Delete refuses them, while explicit edits remain possible.
- Curator side-model calls use the auxiliary route and preserve the owning profile scope.

## Verification

Run the focused skill test and `tests/skills/test_authoring_standards.py` through `scripts/run_tests.sh`. Exercise helper scripts on representative local fixtures and check that every referenced skill-relative file exists.