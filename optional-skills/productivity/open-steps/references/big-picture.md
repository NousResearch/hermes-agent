# big-picture mode

One file, `BIG-PICTURE.md`, where `whats-next` mode already looks for a backlog.
It answers "where are we now" - not a plan, nothing in it a promise. Write it
in the user's language; paths stay English.

**When to use it.** The user asks where the project stands, or there is no
`BIG-PICTURE.md` yet → the full pass: describe, census, signals, backlog. A
session report was just written → the fold-in: only the rows that session
touched. Both re-measure, because a session that moves no row is the one after
which the dates go quietly stale.

## Step 1 - find the file, and never clobber it

`BIG-PICTURE.md` in the project root, and it may already be somebody's own
work. Everything this skill writes lives between `<!-- open-steps:begin -->`
and `<!-- open-steps:end -->`; no markers → append at the end; no file → create
it, and add it to this clone's ignore list, never to `.gitignore`:

```bash
f=BIG-PICTURE.md; e="$(git rev-parse --git-dir)/info/exclude"
git check-ignore -q "$f" || git ls-files --error-unmatch "$f" >/dev/null 2>&1 \
  || grep -qxF "$f" "$e" 2>/dev/null || printf '%s\n' "$f" >> "$e"
```

Each of the three is a reason to leave it alone. Say in one line that you did
it; `git add -f BIG-PICTURE.md` shares it instead. Never read `Last worked on`
or `Signal` back out of the file - only `Stage` and the queue carry forward.

## Step 2 - the census, measured by the script

```bash
bash "${HERMES_SKILL_DIR}/scripts/census.sh" .
```

An `AGE` line, then one measured `PART` row per part: path, last worked on,
commits in six months, whether anything outside reaches it, the signal.

## Step 3 - reading what it printed

| Signal | What it means | What to do |
|---|---|---|
| `active` | Being worked on | Nothing |
| `stable` | Quiet, and something still reaches it | **Leave it alone** - finished and in use |
| `unused - N months` | Quiet, and nothing reaches it | The only real retire candidate |

**When the `AGE` line says `young`**, write one line directly above the table,
before the header row: *"This project is N days old. Nothing here can be quiet
for six months yet, so the Signal column will only start to mean something from
the date the script gave."* Every `active` is then a fact about the
calendar, not the code. On an `ok` line, write no such note.

## Step 4 - the backlog, sourced and never invented

Three sources, each item naming its own: session reports in
`${HERMES_HOME:-$HOME/.hermes}/open-steps/reports/<project>/` (⏳ deferred rows, "Anything needed
from you", recorded debt); Step 3 (each `unused` part, one retire candidate);
the user, by hand. Nothing else - a gap you noticed while reading the code is
not a task, so say it in the chat and let the user decide.

## Step 5 - the tracker, if there is one

An offer, never a quiet action, and only for the "What is next" rows.

1. **Find it, never install it.** `gh repo view --json hasIssuesEnabled -q
   .hasIssuesEnabled`, then any tracker already connected. Nothing connected
   is a normal answer: say so under "What is next", then Step 6.
2. **Search before you propose:** `gh issue list --search '<a few words>'
   --state all --json number,title,state`. A closed ticket is still a match;
   where the search did not run, say so.
3. **Ask, then create.** Plain lines through `ask-simple` mode; on a yes create
   those and nothing else, and only where this pass measured what the item
   rests on. Write each number back into its row, same pass.

## Step 6 - plain words, before it is written

Put every sentence in the block through the `say-simple` mode rules first: one
sentence one idea, active voice, one word for one thing. That covers the prose,
the "What it does" column and the backlog items - never a measured cell.

## The shape

```
<!-- open-steps:begin -->
_Measured <date>. Stages are dated where they stand; anything undated in this
file is not measured._

## What this is

<Three or four sentences. What the product does, who runs it, what it runs
on. No jargon - a reader who has never seen the code.>

## What is in it

| Feature | What it does | Stage | Last worked on | Signal |
|---|---|---|---|---|
| <plain name> | <one line> | live (1 Sep) | 1 Sep | active |
| <plain name> | <one line> | not checked | 28 May | stable |

## Worth retiring

<One line each: what it is, how long unused, what would break. Empty is a
result - write "nothing found" and keep the heading.>

## What is next

- <item> - <source>
- <item> - <source> - #<ticket number, once one exists>

_Age and wiring measured <date>, fresh this pass. Stage comes from session
reports and is only as current as the date beside it._
<!-- open-steps:end -->
```

`Stage` is one of `building`, `built - not shipped`, `live`, `retired` or `not
checked`, and **it carries the date it came from, always** - `live (12 Jul)`,
never bare `live`, and `not checked` never dated. When the newest date in that
column is months behind the newest commit, the reports have stopped: say so.

## Hard rules - these rules *are* the skill

1. **Measured and assumed never mix.** `Last worked on` and `Signal` are
   script output every time. `Stage` has no measurement.
2. **Enumerate, never guess.** The parts come from the script.
3. **Quiet is not dead.** No retire candidate on age alone: quiet plus wired
   in is `stable`, which is what stops the map recommending you delete a
   working product.
4. **Never invent work.** Every backlog item names its source.
5. **Never clobber.** Only between the markers. Their file stays theirs.
6. **Recommend, never act.** No deleting code, moving files, opening a pull
   request. "Worth retiring" is a sentence, not an action.
7. **Plain words.** A feature is named as the user says it, not as the folder.
8. **One screen per section.** Twelve features is a wall - group and say so.
9. **A ticket only after an explicit yes.** Rule 6 is the code, this is
   everyone else's inbox.
10. **The plain-words pass never touches a measured cell.**
11. **Re-measure, never quote.** A number read out of the file is a number
    about the past wearing today's date.
12. **Every claim carries its date or says it has none.** A map that cannot go
    visibly stale will go invisibly stale.

Why the rules are there, the traps the script cannot catch, and a worked
example: `references/big-picture-rationale.md` and `references/sessions/big-picture-quiet-is-not-dead.md`.
