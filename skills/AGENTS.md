# Skills Guide

These rules apply to both `skills/` and `optional-skills/`.

## Skill Surfaces

- `skills/` contains built-in skills shipped and loadable by default.
- `optional-skills/` contains heavier or niche official skills installed via
  `hermes skills install official/<category>/<skill>`.

Heavy dependencies, niche workflows, paid-service integrations, and specialized
skills usually belong in `optional-skills/` or the Skills Hub, not bundled
default skills.

## Frontmatter

Standard fields:

- `name`
- `description`
- `version`
- `author`
- `license`
- `platforms`
- `metadata.hermes.tags`
- `metadata.hermes.category`
- `metadata.hermes.related_skills`
- `metadata.hermes.config`

Top-level `tags:` and `category:` are accepted and mirrored from
`metadata.hermes.*`.

## Hard Standards

Every new or modernized skill must satisfy these before merge.

1. `description` is at most 60 characters, one sentence, and ends with a
   period. Avoid marketing words and do not repeat the skill name.
2. SKILL.md prose should name native Hermes tools or explicit MCP servers in
   backticks. Do not make shell utilities the headline interface when Hermes has
   wrapped tools such as `search_files`, `read_file`, `patch`, or `terminal`.
3. Audit `platforms:` against real scripts/imports. Prefer cross-platform code;
   narrow platforms only for genuine OS dependencies.
4. `author` credits the human contributor first. Do not credit the agent as the
   primary author for external contributions.
5. Use the modern section order: title, short intro, `## When to Use`,
   `## Prerequisites`, `## How to Run`, `## Quick Reference`, `## Procedure`,
   `## Pitfalls`, `## Verification`.
6. Put scripts in `scripts/`, references in `references/`, templates in
   `templates/`, and assets in `assets/`.
7. Tests live at `tests/skills/test_<skill>_skill.py` and use stdlib + pytest +
   `unittest.mock`; no live network calls.
8. `.env.example` additions must be isolated to a clearly delimited block.

For external skill PR salvage, load the `hermes-agent-dev` skill reference
`references/new-skill-pr-salvage.md` before polishing.

## Skills as Progressive Disclosure

Keep `SKILL.md` focused. Move detailed API notes, long examples, parsers,
templates, and scripts into linked files. `skill_view(name)` returns the main
file and a linked-file index; `skill_view(name, file_path)` loads the specific
supporting file on demand.

Avoid bloating the always-visible skill index with content that belongs in
`references/`.

## Curator Interaction

The curator only manages skills with `created_by: "agent"` provenance. Bundled
and hub-installed skills are not auto-archived by curator.

Pinned skills are exempt from automatic transitions.
