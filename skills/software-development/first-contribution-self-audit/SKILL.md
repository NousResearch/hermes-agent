---
name: first-contribution-self-audit
description: "Use when making an early or first contribution to Hermes Agent or another open-source repo. Forces a pre-commit self-audit for low-signal AI artifacts, scope creep, repo fit, tests, docs, and maintainer-friendly PR quality."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [contributing, open-source, self-audit, pre-commit, quality, anti-slop]
    related_skills: [requesting-code-review, hermes-agent-skill-authoring, writing-plans]
---

# First Contribution Self-Audit

## Overview

Use this skill before committing or opening an early contribution, especially when an AI agent helped write the change. The goal is not to make the contribution bigger; the goal is to make it **easy for maintainers to review, trust, and merge**.

A good first contribution is usually small, boring, verified, and obviously aligned with the repository. A weak AI-assisted contribution often has the same smells: broad rewrites, generic docs, invented conventions, unrelated cleanup, noisy formatting churn, untested behavior, or confident claims not backed by the repo.

This skill is a pre-commit stop: audit the staged and unstaged diff, remove reviewer-hostile noise, verify behavior, and leave a reviewer-friendly explanation.

## When to Use

- User says they want to make their first contribution to Hermes Agent or another open-source project
- Before committing docs, skills, tests, bugfixes, or small feature PRs
- When an agent generated or heavily edited the proposed contribution
- Before opening a PR where maintainer trust matters more than speed
- When the user asks to avoid "AI slop", spammy contributions, or low-signal PRs

**Do not use for:** private scratch work, throwaway experiments, or changes the user explicitly marks as exploratory and not intended for commit.

## Core Rule

**A contribution is not ready because it exists. It is ready when the diff proves it belongs.**

Before commit, answer these questions with evidence from the repository:

1. What exact problem does this solve?
2. Why is this the smallest reasonable change?
3. What existing pattern did this follow?
4. How was it verified?
5. What would a maintainer object to?

If any answer is weak, fix the contribution before committing.

## Audit Pass 1 — Repository Fit

Inspect the current repo instead of relying on generic open-source assumptions.

```bash
git status --short
git ls-files --others --exclude-standard
git diff --cached --stat
git diff --cached --name-only
git diff --stat
git diff --name-only
```

Remember: `git diff` does **not** show untracked file contents. For any `??` file in `git status --short`, either inspect it directly before staging or stage only that intended file and review it with `git diff --cached`.

Then check nearby examples in the same area as the changed files:

```bash
# For Hermes built-in software-development skills
find skills/software-development -maxdepth 2 -name SKILL.md | sort

# For docs
find website/docs -maxdepth 3 -type f | sort | sed -n '1,80p'

# For tests
find tests -maxdepth 3 -type f | sort | sed -n '1,80p'
```

For each touched file, identify the local convention it follows. If you cannot name the precedent, pause and inspect 2–3 closer peers.

## Audit Pass 2 — Low-Signal Diff Review

Read the full staged diff. If nothing is staged yet, read the unstaged diff before deciding what to stage.

If the change includes new untracked files, inspect them explicitly before staging or stage only the intended files first; otherwise `git diff` can make a new-file contribution look empty.

```bash
# Staged pre-commit view
git diff --cached --check
git diff --cached --stat
git diff --cached

# Unstaged working-tree view
git diff --check
git diff --stat
git diff
```

Remove or rewrite anything with these smells:

| Smell | Why it is bad | Fix |
|---|---|---|
| Generic praise or marketing language | Maintainers need specifics | Replace with concrete behavior and constraints |
| Unrequested broad rewrite | Hard to review and likely to conflict | Split into a separate PR or remove |
| Formatting churn in unrelated files | Creates noise | Revert unrelated formatting |
| Invented commands, APIs, or paths | Breaks trust immediately | Verify against source/docs or remove |
| Placeholder text like TODO/lorem/example-only | Looks unfinished | Fill with real content or cut it |
| Overconfident claims without tests | Makes reviewers do your work | Add verification or soften the claim |
| AI-voice paragraphs | Low signal and hard to maintain | Use terse repo-native wording |
| Duplicating existing docs/skills | Increases maintenance burden | Link/extend existing material instead |

## Audit Pass 3 — Scope Control

A first contribution should usually touch one concern.

Run:

```bash
{ git diff --cached --name-only; git diff --name-only; } | sort -u
```

Then classify each changed file:

- **Core:** required to solve the stated problem
- **Support:** tests, docs, or examples directly tied to the core change
- **Noise:** unrelated cleanup, formatting, opportunistic refactor, drive-by edits

Remove every noise file from the diff before commit.

If the diff has multiple independent ideas, split it:

```bash
git restore --staged .
git add path/to/one/logical/change
```

## Audit Pass 4 — Verification

Run the smallest relevant verification first, then broader checks if practical. Do not claim a check passed unless you ran it.

### Hermes Agent skills

For a new or edited `SKILL.md`, first follow `hermes-agent-skill-authoring` for structure and frontmatter conventions. Then validate changed skill files, including staged and unstaged paths:

```bash
python3 - <<'PY'
from pathlib import Path
import re
import subprocess
import yaml

names = set()
for args in (['git', 'diff', '--cached', '--name-only'], ['git', 'diff', '--name-only']):
    out = subprocess.check_output(args, text=True)
    names.update(line for line in out.splitlines() if line.endswith('SKILL.md'))

assert names, 'no changed SKILL.md files found in staged or unstaged diff'
for path in map(Path, sorted(names)):
    text = path.read_text()
    assert text.startswith('---'), f'{path}: frontmatter must start at byte 0'
    m = re.search(r'\n---\s*\n', text[3:])
    assert m, f'{path}: missing closing frontmatter fence'
    fm = yaml.safe_load(text[3:m.start()+3])
    assert isinstance(fm, dict), f'{path}: frontmatter is not a mapping'
    assert fm.get('name'), f'{path}: missing name'
    assert fm.get('description'), f'{path}: missing description'
    assert len(fm['description']) <= 1024, f'{path}: description too long'
    assert text[m.end()+3:].strip(), f'{path}: empty body'
    assert len(text) <= 100_000, f'{path}: too large'
    print(f'OK {path}')
PY
```

### Python/code changes

Use focused tests when possible:

```bash
python3 -m pytest tests/path/to/relevant_test.py -q -o 'addopts='
```

For broader confidence:

```bash
python3 -m pytest tests/ -q -o 'addopts='
```

### Docs changes

Check links/paths manually and ensure commands are copy-pasteable. If the docs site has a build command available, run it; otherwise state that only manual verification was performed.

## Audit Pass 5 — Independent Review

After your own cleanup, use the independent-review pattern from `requesting-code-review`. For docs-only or skill-only changes, do not force the full security/test pipeline if it does not apply; ask the reviewer specifically to check:

- correctness against repo conventions
- duplicated content
- hallucinated commands or paths
- unclear scope
- maintainer objections

The independent review should see the actual diff, not your intentions.

## Commit Readiness Checklist

Do not commit until all boxes are true:

- [ ] The contribution solves one clearly stated problem
- [ ] The diff is the smallest reasonable version of that solution
- [ ] Staged and unstaged changes were both inspected
- [ ] Every changed file is core or directly supportive
- [ ] Nearby repo conventions were inspected and followed
- [ ] No generic AI filler, marketing language, or invented facts remain
- [ ] Commands, paths, APIs, and config keys were verified against the repo
- [ ] Relevant tests or validators were run, or the reason they do not apply is stated
- [ ] Independent review found no blocking issues
- [ ] The final response/PR summary names the verification performed

## Maintainer-Friendly PR Summary

Use this format in the commit/PR body:

```text
Summary:
- <one-sentence problem solved>
- <what changed, in concrete terms>

Why this shape:
- <existing repo pattern followed>
- <why scope is intentionally small>

Verification:
- <command/result>
- <manual check if applicable>

Not included:
- <nearby thing intentionally left out to avoid scope creep>
```

## Common Pitfalls

1. **Reviewing only unstaged changes.** `git diff` misses already-staged files; use `git diff --cached` too.
2. **Creating a contribution that is mostly explanation.** Maintainers merge useful diffs, not enthusiasm. Keep prose short and operational.
3. **Letting the agent touch unrelated files.** Revert noise aggressively.
4. **Claiming tests passed without running them.** Always cite the command actually run.
5. **Inventing project norms.** Inspect nearby files and copy the local style.
6. **Making the first PR too ambitious.** Prefer one narrow, high-quality patch over a sprawling "improvement" branch.
7. **Skipping reviewer empathy.** The PR should lower maintainer workload, not ask maintainers to discover whether the change is safe.

## One-Shot Recipe

For a small Hermes Agent skill contribution:

```bash
git status --short
git ls-files --others --exclude-standard
# inspect any intended untracked files directly, or stage only them before reviewing git diff --cached
{ git diff --cached --name-only; git diff --name-only; } | sort -u
# read 2-3 nearby skills in the same category
git diff --cached --check
git diff --check
# run the SKILL.md validation snippet above
# run a docs/skill-focused independent review using requesting-code-review's pattern
```

Then commit only if the readiness checklist passes.
