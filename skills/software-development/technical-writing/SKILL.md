---
name: technical-writing
description: "Checklist for Hermes docs and PR prose, not UI copy."
version: 1.0.1
author: Patrick Gibbs (mcpeezy), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [docs, writing, pull-requests, contributing, gardener]
    related_skills: [hermes-agent-skill-authoring, requesting-code-review, github-pr-workflow]
---

# Technical Writing Skill

Checklist for Hermes documentation and PR prose. Covers `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, `website/docs/`, runbooks, PR descriptions, and commit messages. Do not use it for product UI copy.

There is no separate docs-gardener tree in this repo. Load this skill instead.

## When to Use

- Writing or reviewing `README.md`, `CONTRIBUTING.md`, `AGENTS.md`, or `website/docs/`
- Drafting a PR description or commit message for this repo
- Cleaning AI-generated docs that mix tutorial, how-to, and reference in one page
- Checking that cited paths, commands, flags, and counts exist on the current branch

Do not use for product UI strings, marketing copy, or Cursor-only skill imports.

## Prerequisites

- Working tree on the branch you will document (paths must exist here, not on memory of main)
- Ability to open cited files and run cited commands before publishing them
- For skill PRs, also load `hermes-agent-skill-authoring` so frontmatter and section order stay valid

## How to Run

1. Load this skill before drafting or reviewing the doc/PR prose.
2. Pick one document mode (see Quick Reference).
3. Write or edit with one thought per sentence and grounded paths.
4. Run the Verification checklist before you commit or open the PR.

## Quick Reference

| Mode | Job | Typical Hermes targets |
|---|---|---|
| Tutorial | Walk a new reader through one successful path | Getting-started guides |
| How-to | Solve one concrete task | `CONTRIBUTING.md` PR steps, runbooks |
| Reference | List facts, flags, paths, and contracts | CLI flag pages, config tables |
| Explanation | Why the design is this way | `website/docs/developer-guide/architecture.md` |

A PR description is how-to: what changed, why, how to test.

Honest count commands (run them; do not invent the number):

```bash
git grep -l "HERMES_HOME" -- skills | wc -l
rg -c "def should_allow_install" tools/skills_guard.py
```

## Procedure

1. Name the document mode in a heading or the first sentence.
2. State the reader and the outcome.
3. Lead with the action: `Run scripts/run_tests.sh`, not "You may want to consider running the test script."
4. Put each instruction in its own sentence. If you need "and" for a second instruction, start a new sentence.
5. Name the thing. Prefer `hermes_cli/skills_hub.py` over "the skills module" when a path exists.
6. Prefer the imperative in procedures. Prefer the present tense in reference.
7. Remove ambiguous grammar for global readers:
   - Do not use "this" or "it" when two nouns are in play. Repeat the noun.
   - Do not use "should" when you mean must or must not. Say the rule.
   - Do not stack hedges ("might possibly", "fairly unique").
   - Do not use unexplained we/our. Name the actor: the CLI, the gateway, the reviewer.
8. Ground every symbol in this repo:
   - Paths, commands, flags, and env vars must exist on the branch you are editing.
   - Open the file or run the command before you cite it.
   - Do not invent counts. If you need a count, show the command that produced it on this branch.
9. Drop any count you cannot regenerate. Read once for leftover "this/it/should/might".

### Before / after

Bad (mixed mode, hedge, stale path, invented count):

```markdown
You might want to look at the skills hub if things feel off. This has about
12 install paths and they are all documented in docs/skills.md.
```

Good (how-to, one thought, real paths, no count):

```markdown
If `hermes skills install` exits non-zero after a successful write, read the
scan-report print in `hermes_cli/skills_hub.py`. The report is built by
`format_scan_report()` in `tools/skills_guard.py`.
```

Bad (PR description):

```markdown
Improved various docs and fixed some stuff around skills.
```

Good (PR description):

```markdown
Add `skills/software-development/technical-writing/SKILL.md` for docs and PR
prose. Link it from CONTRIBUTING.md under Pull Request Process. No code
paths change. Closes #78355.
```

## Pitfalls

- Do not import Cursor-only skills, slash commands, or runtime names.
- Do not turn this checklist into a style essay. Keep it short.
- Do not "fix" UI strings with this skill.
- Do not cite paths from memory of another branch.
- Do not ship a skill PR that fails the hardline authoring gates this skill's sibling `hermes-agent-skill-authoring` describes.

## Verification

- Every path in the draft exists (`test -e <path>` or open the file).
- Every command in the draft is copied from the repo or from `--help` on this branch.
- A second reader can tell the document mode from the first heading.
- No unexplained "this/it/should/might" remains where a noun or rule should be.
- For this skill itself: description ≤ 60 chars, human author first, modern section headings present, and referenced paths like `tools/skills_guard.py` exist.
