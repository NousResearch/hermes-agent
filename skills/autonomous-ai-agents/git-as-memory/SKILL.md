---
name: git-as-memory
description: Use Git refs as durable agent memory.
version: 1.0.0
author: femto (@femto)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [memory, git, agents, workflow, audit]
    related_skills: [hermes-agent, codex, claude-code]
---

# Git as Memory Skill

Use Git as Memory when an agent needs durable, auditable memory that lives in a repository but does not modify the working tree. It stores compact memory entries in a dedicated Git ref so future agents can search, read, update, and audit what was remembered.

This skill does not replace source files, docs, issues, tests, or normal project history. Use it only for stable context that future agents should reuse.

## When to Use

- The user explicitly asks the agent to remember something in the current repo.
- A stable user preference, project convention, workflow, or decision is discovered.
- A recurring bug, workaround, tool behavior, or operational lesson should be retained.
- A session produces a compact conclusion that future agents should reuse.
- The user asks what the agent remembers about a repo, project, convention, or decision.

Do not use this skill for secrets, raw transcripts, noisy logs, one-off task details, or content that belongs in tracked project files.

## Prerequisites

- Run commands through `terminal` from inside a Git repository or pass `--repo /path/to/repo`.
- Prefer the `gam` command if it is installed.
- If `gam` is missing, use one of the package entry points:

```bash
npm install -g git-as-memory
pip install git-as-memory
```

Without a global install, use:

```bash
npx git-as-memory --help
python -m git_as_memory.cli --help
```

Default memory ref:

```bash
refs/git-as-memory/memory/v1
```

## How to Run

Initialize memory once per repo. The command is idempotent.

```bash
gam init --repo /path/to/repo
```

If `--repo` is omitted, `gam` uses the current working repo.

## Quick Reference

| Task | Command |
| --- | --- |
| Search content | `gam search "release workflow" --repo /path/to/repo` |
| Read by key | `gam read entity/user-preference-concise-answers --repo /path/to/repo` |
| Show by id and type | `gam show user-preference-concise-answers --type entity --repo /path/to/repo` |
| List keys | `gam list --repo /path/to/repo` |
| Glob keys | `gam glob "entity/*" --repo /path/to/repo` |
| Inspect history | `gam history user-preference-concise-answers --type entity --repo /path/to/repo` |
| Soft delete | `gam delete user-preference-concise-answers --type entity --repo /path/to/repo` |
| Purge visible files | `gam purge user-preference-concise-answers --type entity --repo /path/to/repo` |

## Procedure

### Start a Task

When repo memory may matter, run a targeted search before acting:

```bash
gam search "<project tool user keywords>" --repo /path/to/repo
```

Read only relevant matches. Do not bulk-load all memory unless the user asks for an audit.

### Decide Whether to Write

Write memory only for stable, reusable information.

Write when:

- The user explicitly says to remember something.
- A stable user preference appears.
- A project convention, workflow, or decision is discovered.
- A recurring bug, workaround, tool behavior, or operational lesson is learned.
- A compact session conclusion will help future agents.

Do not write:

- Secrets, API keys, passwords, tokens, cookies, credentials, or private auth material.
- Large raw transcripts or noisy logs.
- One-off task details with no future value.
- Unverified guesses unless `--type hypothesis`.
- Content that belongs in source files, docs, issues, or tests instead of memory.

### Search Before Writing

Before writing a memory entry, search for related memory:

```bash
gam search "<keywords>" --repo /path/to/repo
```

If a related memory exists, update the same stable id instead of creating a duplicate. If no related memory exists, create a concise new memory.

Always include `--source` explaining where the memory came from.

### Choose a Type

Use a small, consistent type set:

- `entity`: user, project, organization, tool, or other entity preference/profile.
- `semantic`: durable fact, rule, convention, decision, or technical lesson.
- `episodic`: compact summary of a task, session, or event.
- `working`: short-lived task context that may be deleted later.
- `hypothesis`: unverified inference that should be confirmed before strong use.

### Choose an Id

Use stable kebab-case ids that future agents can guess.

Good ids:

```text
user-preference-concise-answers
project-release-workflow
openclaw-memory-policy
git-as-memory-storage-layout
```

Avoid random ids unless there is no stable concept.

### Write Memory

Single-line example:

```bash
gam write "User prefers concise technical answers with concrete commands." \
  --repo /path/to/repo \
  --type entity \
  --id user-preference-concise-answers \
  --tag user \
  --tag preference \
  --source "User corrected the agent for over-explaining and asked for practical output."
```

Multi-line example:

```bash
printf '%s\n' "Decision summary..." | gam write --stdin \
  --repo /path/to/repo \
  --type semantic \
  --id project-decision-example \
  --source "Summarized from the current implementation session."
```

After writing, report the memory key:

```text
Remembered: entity/user-preference-concise-answers
```

If you decide not to write memory, say why only when useful:

```text
Not writing memory: this is a one-off task detail.
```

### Read and Audit Memory

Use `read` when you know the key:

```bash
gam read entity/user-preference-concise-answers --repo /path/to/repo
```

Use `search` for content lookup:

```bash
gam search "release workflow" --repo /path/to/repo
gam search "concise answers" --repo /path/to/repo --json
```

Use `glob` for memory key lookup, not arbitrary working-tree files. It matches `<type>/<id>` and bare ids:

```bash
gam glob "entity/*" --repo /path/to/repo
gam glob "user-*" --repo /path/to/repo
gam glob "*release*" --repo /path/to/repo
```

Use direct Git inspection only when provenance or low-level audit matters:

```bash
git -C /path/to/repo log --oneline refs/git-as-memory/memory/v1
git -C /path/to/repo ls-tree -r --name-only refs/git-as-memory/memory/v1
```

## Pitfalls

- Writing secrets or auth material. Never store credentials in Git as Memory.
- Duplicating entries. Search first and update a stable id when possible.
- Treating hypotheses as facts. Use `--type hypothesis` and confirm before relying on them.
- Bulk-loading memory at task start. Search narrowly and read only relevant entries.
- Using `purge` casually. Prefer `delete`; reserve `purge` for explicit requests to remove current visible files.

## Verification

- Confirm `gam init --repo /path/to/repo` succeeds or the memory ref already exists.
- Confirm a write returns or implies the expected `<type>/<id>` key.
- Confirm `gam read <type>/<id> --repo /path/to/repo` returns the stored content.
- Confirm no secret, transient, or source-file-worthy content was written.
- For audit requests, confirm `gam history` or direct Git inspection shows the expected memory evolution.
