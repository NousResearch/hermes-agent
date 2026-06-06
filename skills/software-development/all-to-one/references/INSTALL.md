# A2O Install / Setup Guide

This guide explains how to make All To One callable in different AI coding environments.

## 1. Hermes Agent

Install as a Hermes skill:

```text
~/.hermes/skills/software-development/all-to-one/SKILL.md
```

Then invoke:

```text
总整理
All To One
A2O deep
A2O handoff
```

Hermes should load the `all-to-one` skill and follow the protocol.

## 2. Codex / Codex CLI

### Per-session use

Paste `templates/PROMPT.md` into Codex, then say:

```text
Run A2O standard for this project. Generate or update docs/all-to-one.md.
```

### Per-repository use

Add the content of `templates/AGENTS_SNIPPET.md` to the repository's `AGENTS.md`.

Then invoke:

```text
总整理
A2O handoff
All To One deep
```

Codex should inspect task-relevant context and write/update `docs/all-to-one.md`.

## 3. Claude Code

Add the content of `templates/CLAUDE_CODE_SNIPPET.md` to the repository's `CLAUDE.md`.

Then invoke in Claude Code:

```text
A2O standard
总整理 deep
A2O handoff，写入 docs/all-to-one.md
```

## 4. Claude Desktop / Claude Projects

Paste `templates/CLAUDE_PROJECT_INSTRUCTIONS.md` into Claude Desktop Project Instructions.

If Claude has repository/project files attached, ask:

```text
Use All To One. Read all available project context and generate a project memory document.
```

If Claude cannot access the repo, provide relevant files/logs manually. It must mark missing evidence as `[blocked]` or `[unverified]`.

## 5. Generic Agent

Use `templates/PROMPT.md`.

Minimal invocation:

```text
Use All To One (A2O). Read all task-relevant context and generate/update docs/all-to-one.md. Use evidence tags. Preserve root causes, failed paths, verification, risks, and 5-10 minute resume path.
```

## Recommended Project Files

```text
AGENTS.md              # Codex / generic agents
CLAUDE.md              # Claude Code
docs/all-to-one.md     # generated durable project memory
```

## Optional Portable Folder

```text
.all-to-one/
├── SKILL.md
├── PROMPT.md
└── templates/
    ├── quick.md
    ├── standard.md
    ├── deep.md
    └── handoff.md
```

## Important Boundary

A2O can read all task-relevant context, but it should not blindly consume:

- dependency folders
- build artifacts
- secrets/credentials
- large binaries
- unrelated files

If context is too large, it should create a context index first: read sources, skipped sources, skip reasons, and possible blind spots.
