# Why the rules are there, and the traps the script cannot catch

`SKILL.md` holds the rules. This file holds the reasons, and the failures each
rule was written after. Read it when a rule looks like ceremony, or when the
map you are about to write disagrees with one.

## Why the file exists at all

Every other skill in the pack is one session wide. `whats-next` mode is told to
read the backlog **always**, and until this skill landed nothing in the pack
ever wrote one. The map is the standing half: the part that outlives the
session that produced it, and the part an owner can open without an agent
present.

That is also why it is a file and not a query. A census generated on demand
would be fresher, and useless to the person this pack is for - they would need
an agent running to see what they own.

## Why quiet code is not dead code (rules 1 and 3)

Age alone says nothing. Finished code is quiet; so is abandoned code. The two
read identically in `git log`, which is why a map built on dates alone
recommends deleting a working product - see
[`01-quiet-is-not-dead.md`](01-quiet-is-not-dead.md), where six of eight
"stale" modules were load-bearing.

**`stable` is the row that protects the user.** Most quiet code is still
imported by dozens of files. Only quiet *and* unreached is a retire candidate,
and even then it is a sentence for the user, never an action (rule 6).

## Why nothing measured is ever read back out of the file (rule 11)

A map that already exists was true on the day it was written and has been
decaying since. `Last worked on` and `Signal` cost seconds to measure, so
there is no case where quoting the stored value is worth being wrong. The only
cells carried forward are the ones nothing can measure: `Stage`, and the
queue.

The same reason drives the fold-in rule. A session that fixed a bug in a
finished feature moves no row - and it is exactly the session after which the
dates would silently be a day older than they claim. Measuring needs no reason
and no permission, so the fold-in measures whether or not it has news.

## Why `Stage` carries a date (rule 12)

`Stage` is the one column nothing can measure. It comes from the session
reports, from the user, or it says `not checked`.

The failure this file is most likely to have is being **half fresh, and
therefore trusted whole**: the measured columns keep refreshing themselves and
look healthy while `Stage` quietly ages behind them. The date beside each
stage is the cure, because it puts the decay in the row rather than in a
footnote nobody reads. A `Stage` column that is all `not checked` is itself a
finding - it means the reports have stopped being written, not that the
product is unknowable. Say so.

## Why the map stays out of git

The map is a working note about the project, not part of the product, and a
`git add -A` should never sweep it into somebody's commit. So the skill puts
it in that clone's own `.git/info/exclude`, never in the shared `.gitignore`,
which belongs to the whole team.

Three conditions stop it, and each is somebody's decision to respect: the file
is already ignored, it is tracked because somebody chose to commit it, or the
line is already there. **A tracked map stays tracked** - never remove it from
git, and say once that its edits will show up in diffs.

## Why the tracker step is an offer (rule 9)

A team that runs on a tracker will not read the map, so the queue has to reach
them where they already look. But rule 6 protects the code and rule 9 protects
everyone else's inbox: searching a tracker is free, filling one is not.

- **The same item proposed twice** is how this step fails. A ticket whose
  number never made it back into its row will be offered again next session,
  and the team gets two of it.
- **A closed ticket is still a match.** Search `--state all`. An item somebody
  already did should leave the map, not reopen as a new ticket.
- **A stale row must never become a ticket.** A retire candidate needs the
  wiring check run today, not the one in the file. The tracker reaches people
  who never open this map, and that is exactly who a stale row would mislead.
- **No tracker is not a fault.** Plenty of projects run on the map alone. Say
  it in one line and never turn it into a recommendation to adopt one.

## Traps the script cannot catch

`scripts/census.sh` measures dates and wiring. These four it cannot see, and
they are yours:

- **A renamed folder looks new.** A whole subsystem that moved reads as three
  months old on the day it moved. `git log --follow` on a single file inside
  it when a date looks wrong.
- **A monorepo is not one product.** Several deployables → group the map by
  deployable, or "What this is" becomes a lie of averages.
- **git says last *touched*, not last *used*.** A part with no commits may run
  every day. That is why wiring is a separate signal and not a tiebreaker.
- **Vendored and generated code reads as abandoned.** The script drops the
  usual folder names, but it cannot know yours. A copied-in dependency has one
  commit, years old, and is not yours to retire.

## The young repository

A project younger than the quiet window has no quiet code yet - by definition,
not by luck. Every part comes back `active`, and the map would report a
sweeping clean bill of health it did not earn.

Measured on a real five-month-old project with 1245 commits: 51 parts, all
`active`, nothing to retire. Not one of those was a finding. So the script
prints the repository's age, and on a young one the map says in a line above
the table that the liveness column cannot mean anything yet, and from when it
will. The reader then knows the column is a fact about the calendar.

## Plain words are not fewer facts (rule 10)

The plain-words pass rewrites sentences, never facts. A shorter "Worth
retiring" line that drops what would break is worse than the long one, and
`say-simple` mode's own hard rules carry over whole: add nothing, drop no bad
news, numbers and dates stay exact, a claim from a report stays a claim.

## Options that were considered and rejected

So a later session does not re-run the choice:

| Rejected | Why |
|---|---|
| A hook that warns on a stale map | A threshold to tune, and it nags every session |
| Store only the non-perishable half, generate the census on demand | Kills the reason the file exists: an owner opening it without an agent |
| The tracker as the source of truth | No truth at all for a solo owner with no tracker |
| Dropping `Stage` | It is the only column that says how far a feature got |
| Keeping the map in the agent's home dir | The team could never see it |
