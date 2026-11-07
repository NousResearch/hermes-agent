# whats-next mode

Answer "what do we do now" with the next move, not a map. The user does not
need the dependency graph - they need what got finished, what to take next,
and why, in plain words. This skill decides; `step-by-step` mode walks the user
through their part; `done-or-not` mode reports what came of it.

## Language

Write in the language the user speaks in this session, detected from the
conversation. Commands, file names and identifiers stay English.

## When to use

Triggers: the description above, plus a session just ended wanting a next move.

## Step 1 - read the state, quickest first

Stop as soon as you can answer.

1. **The last report** - `${HERMES_HOME:-$HOME/.hermes}/open-steps/reports/<project>/latest.md`:
   a handover written for exactly this moment.
2. **Local state** - uncommitted changes, unpushed commits, current branch.
3. **Open pull requests** - one call:
   `gh pr list --json number,title,mergeStateStatus,reviewDecision,isDraft`.
4. **The backlog - always.** The issue tracker when one is already connected
   (never authenticate or install one), otherwise task files in the repo:
   `BIG-PICTURE.md`, `PLAN.md`, `TODO.md`, `docs/plan*`. Next work comes from
   the backlog, not from imagination. No backlog anywhere → say so.

   Where `big-picture` mode keeps that file, **only the "What is next" section
   is the backlog.** The feature table above it is an inventory, and a row
   that has not changed in months is a finished feature, not a task. Reading
   work out of it is inventing work, which rule 4 below forbids. A row under
   "Worth retiring" is a real candidate, but it is a decision to put to the
   user, never a task to start.

   That file carries its own age: a measured date at the top and a date beside
   every stage. Never take a number out of it. The two columns that come from
   git - when a part was last worked on, and whether anything still reaches it
   - you measure yourself, here, before using them; it is two commands and it
   is the difference between a current answer and a confident old one. The
   stages you cannot measure: use the dates beside them, and say the age out
   loud whenever it is months behind the newest commit. Dates that stopped
   moving mean the reports stopped, not that the work did.

Say which sources you did not read: an unread source is not an empty source.

## Step 2 - finish what is finished

A pull request with green checks and an approval is not a decision - it is
unfinished business. Verify it through `check-work` mode's accept rules in
this same pass and report it as **ready, proof attached**. The merge itself
waits for the user's word (given in this ask or in a standing instruction they
wrote); a failed claim stops it even then. Mechanical unblocking - updating the
branch, restarting a stuck check - is yours without asking.

## Step 3 - sort what remains into two lists

| List | Belongs there when |
|---|---|
| **I can do this alone** | everything needed is at hand: no decision, no secret, no approval, no device |
| **Needs you** | a decision, an approval, a secret, a purchase, or a device only the user has |

Blocked work gets no section of its own. Fold it into the reasoning, in plain
words - "X waits on an outside check; I watch it" - the user trusts the
recommendation, not the graph.

## The shape - ten lines, like every report in this pack

```
<Lead: one sentence on where things stand - including what this pass merged.>

**I can do alone:** <up to three items, five words of why each>
**Needs you:** <up to three items, one line each - or drop the list>

**Next I take: <the one task> - <plain words: what it closes or unblocks>.**
<One line: what was not checked.>
```

When a quick small win and a big item are both real candidates, offer the
choice with a numbered option list (or the platform's option picker where one exists) - two to four options, the recommended one first
and marked; where the picker is not available, one plain sentence. On the
pick, prepare the launch: a prompt complete enough to paste or a command
complete enough to run, one line saying what comes out - and never run it
yourself.

## How many at once

Before offering to start several, prove they will not collide - all three:

| Check | They collide when |
|---|---|
| Same files | both touch the same files, module, or migration sequence |
| Same shared resource | one working copy, branch, database, container project, port |
| One feeds the other | the second needs the first one's output |

Any check failing → one at a time, saying which failed. All passing → say so.
Never claim parallel safety you did not verify - "I did not check" is honest;
a collision discovered mid-run is not. Where the project isolates parallel
work - a working copy per task, separate container projects or ports - name
that as the precondition instead of assuming it.

## Hard rules

1. **Three items per list, maximum** - more → say how many were left out and
   on what basis you chose.
2. **Every item names its source** - the report, a pull request, a backlog
   entry, a failing check. Your own idea is marked a suggestion, and lists are
   never padded: two real items beat five with filler.
3. **One recommendation, always** - even when offering the small-versus-big
   choice, one option carries the mark and one line of plain-words reasoning.
4. **Verify, then prepare - never start.** Verifying a ready pull request
   is finishing your part; the merge waits for the word. New work is prepared
   as a ready-to-run launch and waits for the pick.
5. **Say what you did not check** - especially the backlog. Silence reads as
   "nothing there".
6. **Plain words** - no engineering identifiers except where they name an
   action.

## Known gotchas

- Deferred-until-Monday is not a task on Saturday: do not re-propose it early.
- A stale tracker is worse than none - say when you read it. The same is true
  of the map, and it hides it better: its measured columns refresh themselves
  while the stages behind them age.
- Draft pull requests are yours to finish, not the user's to merge.
- A quiet feature in the map is not a task. It is quiet because it is
  finished; the skill that wrote it already checked that something still uses
  it.
- A needs-you pick goes to `step-by-step` mode, never explained inline.
- "Ready to merge" is still a claim: the verify step is what makes it true -
  skipping it to move faster is how wrong work lands.
