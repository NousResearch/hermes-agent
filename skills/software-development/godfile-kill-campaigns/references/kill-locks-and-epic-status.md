# KILL LOCKS + epic status rows (interlock doctrine, session 2026-08-05)

Two Axl corrections fired repeatedly during the Kill-All-Gods campaign
(epic #78647, 20-godfile inventory). Both are binding for any god-file
kill campaign.

## 1. Per-godfile KILL LOCK (posted at kill-start)

Every god-file gets a permanent KILL LOCK posted on its per-file shard
issue BEFORE the first shard is cut. Canonical sections:

```
## 🔒 KILL LOCK — <file> (<peak lines> lines)
**Anchored to Kill All Gods meta-issue: #<epic>.**
### The former whole   (line count + the mess: problem issues w/ links)
### The kill — N shards (PR table: | PR | Slice | Module |)
### The mess-fixers    (all open PRs touching the surface, individually)
### Lock chain (both ways)
```

- The epic (#78647) carries a meta-index of every lock.
- Every shard PR body/comment carries the lock paragraph.
- Pull the mess roster from live gh (issues + PRs), never invent it.
- The lock goes up at kill-START (discord precedent), fills as shards ship.

## 2. Epic status rows — SHIPPED standard, no ranges

- An OPEN PR with all shards done + individually linked IS SHIPPED work
  for the epic. Never "merge pending", never "open" for a done shard set.
- Canonical row shape: `**SHIPPED** — N slices done, all individually
  linked: #a #b #c` — EVERY PR its own token, NEVER `#a-#c` ranges.
- Audit: `grep -cE '#[0-9]+-#[0-9]+'` after every table edit = 0.
- When the epic says a godfile is "sharding DONE / in progress" and the
  user says it should be done — VERIFY the actual shard PRs exist and are
  linked before editing the row (run.py was the case: the "24 PRs" claim
  was a poisoned precedent citation copied into 17 issue bodies; the truth
  was 38 shards).

## 3. Poisoned precedent citations (class fix)

The run.py "26,877 → ~2,300 across 24 PRs" claim was false (run.py is
26,986 lines; 38 shards exist, all open). It had been copied as boilerplate
into 17 shard issues. Fix the CLASS: script the same replacement across all
N issues (CCC — destroy the category). Verify precedent claims against live
data before they propagate.

## 4. Scoreboard format (per-file issue, after each ship)

```
## <file> scoreboard — wave N complete (M slices)
| Slice | PR | Module | Window | Evidence |
```

Posted per the epic's binding method item 4. The PR↔issue interlock is the
completion record.
