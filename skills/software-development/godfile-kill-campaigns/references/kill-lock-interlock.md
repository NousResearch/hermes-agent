# KILL LOCK — the per-godfile interlock posting sequence

Standing doctrine (Axl, 2026-08-05): **every godfile gets its own interlock that
decisively links all shards to the former whole and all issues and existing PRs
related to the mess of problems caused by that god — and every lock must be
linked to the Kill All Gods meta-issue.** "Never let them forget how bad it
used to be."

## When to post

- **At kill-start** (immediately after Wave-1 dispatch): the lock goes up with
  TBD shard rows and the mess already catalogued. Discord (#78634) and slack
  (#78638) both posted at kill-start per this doctrine.
- **At every ship**: scoreboard comment (slice | PR | module | window | evidence)
  on the per-file issue, per the epic's binding method item 4; the epic status
  table row updated.

## Lock file shape (C:/tmp/tg-campaign/locks/LOCK-<god>.md, posted verbatim)

```markdown
## 🔒 KILL LOCK — <path> (<line count> lines) [— IN PROGRESS]

**Anchored to Kill All Gods meta-issue: #78647.** Permanent record: shards, mess, fixers.

### The former whole
One <N>-line file: <what it carried>. The mess it caused:
- **#<issue>** — <problem 1> (link)
- **#<issue>** — <problem 2>
- ...

### The kill — <N> shards, all double-blind reviewed
| PR | Slice | Module | Window | Evidence |
|---|---|---|---|---|
| #<pr> | <slice> | <module> | <lines> | <test counts> |
| — | <blocked slice> | — | — | **BLOCKED** on #<pr> (land-order coordinated) |

### The mess-fixers (all open PRs still fighting this surface)
#<pr> #<pr> ... (full roster from `gh pr list --state open --search "<godfile>"`)

### Lock chain (both ways)
- Every shard PR references this lock and #78647
- This lock references every issue and every PR listed above
- The Kill All Gods meta-issue #78647 indexes this lock
```

## Posting sequence (proven 2026-08-05)

1. **Gather the mess**: `gh issue view <per-file-issue>` + `gh search issues "<godfile>"` +
   `gh pr list --state open --search "<godfile>"` → problem issues (the bugs the
   monolith caused) + open fixer PR roster.
2. **Write + post the lock**: `gh issue comment <per-file-issue> --body-file LOCK-<god>.md`.
3. **Update the meta-index**: a comment on the Kill All Gods issue listing the new
   lock (godfile | lock location | kill status), so the meta-issue is the single
   index of every god's death and debt.
4. **On every shard PR**, post the lock-link comment (mess list, wave table, fixer
   roster, chain to the lock + meta-issue) — 17/17 shard PRs carried it this wave.
5. **Scoreboard** on the per-file issue at each ship: `slice | PR | module | window | evidence`.
6. **Epic status table**: update the godfile's row (e.g. "7 slices shipped (#79123-#79129)").
7. **Coordination on colliding PRs** (both-ways interlock — "all related PRs and
   issues, not just the ones I wrote"): name the lock, state the land-order
   proposal (their PR first, extraction rebases — or vice versa), note DIRTY
   merge states that need rebase. Posted on #75735, #12355, #77547, #78393,
   #78413, #79612 during this campaign.

## Why it works

Any maintainer opening a shard PR, the per-file issue, or the meta-issue sees:
the whole mess the monolith caused (stall bugs, security issues, tracking chaos),
the complete shard map, the fixers still fighting, and the chain to the meta-issue.
The interlock is machine-verifiable and permanent — not a PR-body flourish.
