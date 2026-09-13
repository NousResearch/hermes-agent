---
name: campaign-operations-kill-locks
description: "Use when running god-file kill campaigns: post and maintain per-godfile KILL LOCKS interlocked to the Kill-All-Gods meta-issue."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [campaign, godfile, kill-lock, interlock, meta-issue, governance]
    related_skills: [campaign-primitives, pr-conquest, worktree-pr-campaigns, github-traceability-audit]
---

# Campaign Operations — Kill Locks

## Overview

A god-file kill campaign ships shard PRs, but shards alone do not make a kill
**remembered**. Every god-file gets a permanent **KILL LOCK**: a machine-verifiable
record posted on its per-file issue that binds — in both directions — the former
whole (line count, what it was), the mess it caused (every problem issue), every
shard that killed it (all PRs), and every open fixer PR still fighting the
surface. Every lock anchors to the campaign meta-issue (the Kill-All-Gods epic),
which indexes all locks. The rule: **never let them forget how bad it used to be.**

This is the operation layer for god-file conquests: what to post, when, where,
and how to keep it current. It pairs with `campaign-primitives` (the 20 locked
primitives) and `pr-conquest` (issue/PR interlock mechanics).

## When to Use

- Launching or running a god-file kill campaign (web_server, tui_gateway/server,
  cli.py, platform adapters, run.py, etc.)
- Axl says "lock it", "interlock", "never let them forget", or "kill all gods"
- Posting a kill scoreboard, updating an epic's status table, or auditing a
  campaign's PR↔issue↔lock graph
- Onboarding a new shard PR so it carries its lock link

Don't use for: single bug-fix PRs outside a god-file campaign (plain
`Fixes #N` interlock suffices); issue triage with no decomposition.

## The Kill Lock — canonical shape

Post the lock as a comment on the god-file's **per-file shard issue** (e.g.
#78628 for web_server.py) at **kill-start** (before any shard is cut), then
update it as shards ship. Required sections:

```
## 🔒 KILL LOCK — <file path> (<peak line count> lines) [— IN PROGRESS / complete]

**Anchored to Kill All Gods meta-issue: #<epic>.** Permanent record: shards,
mess, fixers.

### The former whole
One <N>-line file carrying <surface>. The mess it caused:
- <#issue> — <symptom> (problem issue, with link)
- <more issues — the bug classes, stalls, security holes the monolith bred>

### The kill — <N> shards, one wave, all double-blind reviewed
| PR | Slice | Module |
|---|---|---|
| #<pr> | <slice id> | <module> |

### The mess-fixers (all open PRs still fighting this surface)
#<pr1> #<pr2> ... (full roster — pulled from `gh pr list --search '<file>'`)

### Lock chain (both ways)
- Every shard PR above references this lock and #<epic>
- This lock references every issue and every PR listed above
- The Kill All Gods meta-issue #<epic> indexes this lock
```

The lock's problem-issue list is the **"how bad it used to be"** record — pull
real issues (`gh search issues --repo O/R '<file> adapter'`) and real PRs
(`gh pr list --repo O/R --state open --search '<file>'`), never invented ones.
Exact counts only.

## The meta-index (Kill-All-Gods epic)

The campaign meta-issue carries an index of every lock:

```
## 🔒 KILL LOCKS — index (anchored here, the Kill All Gods meta-issue)

| Godfile | Lock (posted on) | Kill status |
|---|---|---|
| <file> (<lines>) | **LOCK** on #<per-file-issue> | ✅ <N> shards (#pr1-#prN) |
```

Update the index at every kill-start and every ship. The epic's status table
row for each godfile also flips to the shipped state with PR numbers.

## Shard-PR lock linkage (mandatory)

Every shard PR body carries a lock paragraph (post as a comment if the PR is
already open):

```
🔒 **This shard is part of the <file> KILL LOCK** — the permanent record of the
<N>-line whole, the mess it caused (<#issue1>, <#issue2>, ...), every shard in the
wave, and every open fixer PR still fighting the surface (<#pr1> ...).

Lock: posted on #<per-file-issue> · Indexed by the Kill All Gods meta-issue #<epic>.
```

## The interlock chain (both directions, audited)

1. **PR → issues:** every shard PR body carries `Part of #<epic>` AND
   `Part of #<per-file-issue>` as **separate lines** (a combined line fails
   strict per-issue keyword checks). `Progress on #N` is NOT a GitHub keyword —
   use `Part of` (links without closing) or `Fixes`/`Closes` (bug fixes).
2. **Issues → PRs:** every lock comment and scoreboard carries the literal
   `#<pr>` tokens (prose mentions don't satisfy the audit).
3. **Verify the official registry:** `gh api repos/O/R/issues/<issue>/timeline
   --jq '.[] | select(.event=="cross-referenced") | .source.issue.number'`
   must list every shard PR; empty `closingIssuesReferences` on refactor PRs is
   EXPECTED (that field is close-on-merge only) — timeline cross-refs are the
   surface for `Part of` links.
4. **Zero holes is the receipt:** every PR binds every issue, every issue binds
   every PR.

## Scoreboard (per-file issue, after each ship)

```
## <file> scoreboard — wave <N> complete (<M> slices)

| Slice | PR | Module | Window | Evidence |
|---|---|---|---|---|
| <slice> | #<pr> | <module> | <lines> | <test counts> |
```

Post per the epic's binding method (item 4: scoreboard after every shipped
slice — slice | PR | module | delta | evidence). The PR↔issue interlock is the
completion record.

## Common Pitfalls

1. **Posting the lock only at kill-end.** The lock goes up at kill-start (the
   discord precedent) — the mess record exists before the first shard, so
   nothing ships unremembered.
2. **Inventing the mess roster.** Pull real issues/PRs via gh; fabricated
   problem lists corrupt the record and cost trust.
3. **`Progress on #N` as a linking keyword.** Not recognized by GitHub —
   `closingIssuesReferences` stays empty. Use `Part of #N` (verified working).
4. **One combined `Part of #a, #b` line.** Fails strict per-issue keyword
   checks — separate lines per issue.
5. **Leaving the epic table stale.** "The status table in the Godfile Epic PR
   is not up to date" is a real correction Axl fired — update the epic body's
   inventory row + the lock index at every ship.
6. **Empty issue ledgers on campaign metas.** A feature-parity meta (telegram
   #78791, discord #79564) with an empty "Issue ledger" table is a corpse —
   build the full lane-assigned, dependency-linked table of ALL open issues on
   the subject (137 for discord: | Issue | Lane | Dependencies | Title |, with
   cross-reference edges pulled from issue bodies).
7. **Lock drift after the wave ships.** Re-verify the shard-PR lock links when
   the wave lands; the 17/17 shard-link pattern is the receipt.

## Verification Checklist

- [ ] KILL LOCK posted on the per-file issue at kill-start (mess + shard table + fixer roster + lock chain)
- [ ] Meta-index on the Kill-All-Gods epic lists the lock
- [ ] Epic status-table row reflects shipped state with PR numbers
- [ ] Every shard PR carries the lock paragraph (body or comment)
- [ ] Every shard PR binds `Part of #epic` + `Part of #file` as separate lines
- [ ] Scoreboard posted on the per-file issue (slice | PR | module | window | evidence)
- [ ] Timeline cross-refs verified — zero holes both directions
- [ ] Campaign metas carry their full issue ledger (not empty)
