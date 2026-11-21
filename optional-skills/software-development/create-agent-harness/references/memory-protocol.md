# Memory Protocol

Reference for `create-agent-harness` Phase 3 — mandatory `.claude/memory/` protocol.

## Overview

Mandatory in every harness. There is no minimal mode without memory. Two memory tiers inside `.claude/memory/`, plus one on-demand protocol reference at `.claude/MEMORY.md` that never stores state or history.

| Tier | File | Lifetime | Content |
| --- | --- | --- | --- |
| **Short-term** | `.claude/memory/memory.md` | Current session, **overwritten**, max 100 lines | Working state: last verified commit, test baseline, active branch and task, work in progress, blockers, next action |
| **Long-term** | `.claude/memory/{YYYYMMDD}-memory.md` | Permanent, **append-only**, one file per day | Durable records: prompt summaries, session summaries, decisions with rationale, lessons learned, technical debt, discoveries, checkpoints |

**Always save everything** — prompts, history and knowledge. Every user prompt or instruction is logged as a one-line summary in today's long-term file, every session ends with a `## Session summary` entry, and reusable knowledge discovered during work is promoted to `.claude/knowledge/{slug}.md`. Nothing may survive only in context.

`memory.md` is **state**, not a log — it answers "where am I right now". The dated files are a **log**, not state — they answer "why is the code like this". Both live under `.claude/memory/` because a `memory/` directory at the repository root is forbidden by the target structure. Both are committed: shared memory is the point.

Distinguish them by glob, never by reading the whole folder:

```bash
ls -1 .claude/memory/[0-9]*-memory.md 2>/dev/null | sort -r | head -3
```

## memory.md template

Rewrite it in full on every update; never append.

```markdown
# memory.md — short-term memory

Working state of the current session. Overwritten on every update.
Durable records belong in .claude/memory/{YYYYMMDD}-memory.md.

- Last verified commit: {hash} ({subject})
- Test baseline: {X/Y passing} ({pre-existing failures and where})
- Active branch: {feature/...}
- Active work item: {task, issue or change name}
- In progress: {what is partially done, OR "nothing — clean boundary"}
- Uncommitted files: {paths, OR "none"}
- Blockers / out-of-scope findings:
  - {item discovered but deliberately not fixed}
- Next action: {the first thing the next session must do}
- Last updated: {YYYYMMDD}
```

## Long-term template

One file per calendar day, appended at the end. Each entry is at most 5 lines — terse facts, never prose dumps. Omit sections with no entries; never create the file empty.

```markdown
# {YYYY-MM-DD} — long-term memory

## Prompts
- {one-line summary of the user prompt or instruction — never the raw prompt}

## Session summary
- {what the session did, the outcome, and where work stopped — written before compaction, context reset or any possible end of session}

## Decisions
- **{decision}** — rationale: {why}. Alternatives discarded: {what and why not}.

## Lessons learned
- **{context}** — mistake: {what went wrong}. Avoid by: {rule}.

## Technical debt
- **{item}** — impact: {what it costs}. Priority: {high/medium/low}.

## Discoveries
- **{fact about the codebase}** — source: {file path}.

## Checkpoints
- `{commit hash}` — {group or task completed}. Verification: {command} passed.
```

## Read protocol

At session start: read `memory.md` always, then the 3 most recent dated files descending by filename, then stop. Never read the whole folder. Treat long-term memory as a **hint, not truth** — verify a remembered fact just-in-time against the current code before acting on it. If the code contradicts the memory, the code wins and the entry must be corrected.

## Write triggers

Writing is mandatory at each trigger, not "when useful".

| Trigger | Write to | What |
| --- | --- | --- |
| Session start | — | Read only |
| User prompt or instruction received | Long-term | Append a one-line summary to `## Prompts` — **every prompt, always; never the raw prompt** |
| Session boundary — task done, before compaction or context reset | Both | Append a `## Session summary` entry; promote durable entries; reset `memory.md` |
| Verified checkpoint or commit | Both | Update `memory.md`; append a `## Checkpoints` entry to today's file |
| Decision taken | Long-term | Append to `## Decisions` with the rationale |
| Mistake corrected | Long-term | Append to `## Lessons learned` |
| Reusable knowledge discovered | `.claude/knowledge/` | Promote to `{slug}.md`; append a `## Discoveries` entry linking it |
| Out-of-scope problem found | Short-term | Add to blockers; **do not fix it now** |

There is no reliable "end of session" event. Treat **every checkpoint commit or completed task as a possible end of session** — write the `## Session summary` and update memory as if the context is about to be lost.

## Promotion and rotation

Promote when a checkpoint was committed or `memory.md` exceeds 100 lines:

1. Move every durable entry out of `memory.md` into today's `{YYYYMMDD}-memory.md` under the matching section.
2. Rewrite `memory.md` from the clean template, keeping only current state.
3. Create today's file from the template if it does not exist yet.

Blockers stay in `memory.md` until resolved or promoted to technical debt.

## Retention and correction

Long-term files are append-only — never delete or edit an existing entry. Correct a wrong or obsolete entry by appending a new one prefixed with `SUPERSEDED:` stating the new fact and the reason. Entries tied to a deleted branch or reverted work are superseded, not removed. `memory.md` may be freely rewritten; it holds no history.

## Security

Zero secrets, tokens, passwords, connection strings or private keys. Zero PII: no customer names, documents, account numbers or identifiers. Reference identifiers instead of values — "auth token read from the `AUTH_TOKEN` environment variable", never the token itself.

## Wiring

The protocol only works when the rest of the harness points at it.

| Artifact | Required wiring |
| --- | --- |
| `CLAUDE.md` | A `## Memory Protocol` section with the read protocol, the write triggers and the always-save rule — mandatory on generation; added verbatim (merge, never overwrite) when completing an existing `CLAUDE.md` during migration |
| `.claude/rules/global-rules.md` | Reading `.claude/memory/` is part of the mandatory pre-execution ritual |
| `.claude/MEMORY.md` | On-demand reference: documents the protocol, stores no state or history |
| Agent loop step 4 | Read `memory.md` and the 3 most recent dated files |
| Agent loop step 12 | Update memory before the checkpoint commit |
| `.claude/commands/dod.md` | Definition of Done includes updating memory |
| `.claude/settings.json` | `permissions.deny` must **not** block `Read(.claude/memory/**)` |

## Migration from legacy memory files

| Legacy file | Action |
| --- | --- |
| `.agents/MEMORY.md` | Copy durable content into `{YYYYMMDD}-memory.md`, then remove the legacy file |
| `.claude/memory/MEMORY.md` | Split: current state into `memory.md`, history into `{YYYYMMDD}-memory.md`, then remove |
| Root `.claude/MEMORY.md` (legacy with state/history) | Copy durable records into `{YYYYMMDD}-memory.md`, working state into `memory.md`, then rewrite `MEMORY.md` as on-demand protocol documentation |
| `SESSION_STATE.md` | Copy current state into `memory.md`, then remove |
| Root `memory/*` | Move into `.claude/memory/`, renaming to the `{YYYYMMDD}-memory.md` pattern |

Never keep two files describing the same state. One concept, one location.
