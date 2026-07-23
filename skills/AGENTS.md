# Hermes Skill Authoring Guide

These instructions apply to bundled skills under `skills/`. The same standards
apply to `optional-skills/`; niche or dependency-heavy skills belong there
instead of being active by default.

## Frontmatter

Use standard fields: `name`, `description`, `version`, `author`, `license`,
`platforms`, and `metadata.hermes` entries for tags, category, related skills,
and required configuration.

Hard requirements:

- `description` is one sentence, at most 60 characters, and ends with a period.
- State capability, not implementation; avoid marketing words.
- Credit the human contributor first in `author`.
- Audit `platforms` against actual scripts and imports.

POSIX-only primitives, `/proc`, hardcoded `/tmp`, `fcntl`, `termios`,
`os.setsid`, Unix signals, bash-only scripts, `osascript`, `apt`, or `systemctl`
require an appropriate platform declaration unless rewritten portably.

## Interaction surface

Skill prose should name native Hermes tools or explicitly required MCP servers.
Prefer `terminal`, `read_file`, `patch`, `search_files`, `web_extract`,
`vision_analyze`, `browser_navigate`, and `delegate_task` over presenting shell
utilities as the primary interface.

Third-party CLIs and pipelines are fine inside shipped scripts. Document MCP
prerequisites explicitly.

## Structure

Use this body order:

1. `# <Skill> Skill`
2. short introduction including what it does not do;
3. `## When to Use`;
4. `## Prerequisites`;
5. `## How to Run`;
6. `## Quick Reference`;
7. `## Procedure`;
8. `## Pitfalls`;
9. `## Verification`.

Target about 100 lines for a simple skill and 200 for a complex one. Remove
marketing prose and repeated environment-variable explanations.

Place non-trivial logic in `scripts/`, long-form material in `references/`, and
starter artifacts in `templates/`. Do not make the model recreate parsers,
walkers, or setup programs on every invocation.

## Tests

Tests live at `tests/skills/test_<skill>_skill.py`, use stdlib, pytest, and
`unittest.mock`, and make no live network calls.

```bash
scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q
```

Keep `.env.example` edits inside a clearly delimited block for the skill and do
not rewrite unrelated surrounding content.

## External contributions

Load the `hermes-agent-dev` skill and read
`references/new-skill-pr-salvage.md` before modernizing an external skill PR.
Preserve human authorship and salvage useful contributor work rather than
reimplementing it.
