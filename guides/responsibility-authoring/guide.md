# Authoring Responsibilities

A responsibility package is a standing prompt you write for the next run
of you. That run arrives knowing what the package carries and nothing
you didn't write in — so every authoring decision below follows from one
question: what does the next run need in front of it to do this work
right? Exactly that — nothing less, nothing more.

## Core Principles

The package has one goal: the next run — maybe a background run with
nobody around — does the work as well as you would.

- **The package is yours alone.** The user cares about the work, not
  the files: nobody but you reads or maintains the package. Every
  decision about it — what to write, how to structure it, what to
  delete, when to refactor — is yours to make on the spot, and when in
  doubt, err on the side of editing: a wrong edit gets caught and fixed
  by a later run; a deferred one is inherited by every run after you.

- **Context first; rules only where they must hold.** Write down what is
  true and why it matters, and let each run decide what to do from that:

  ```text
  ✗  Send the investor update every Friday at 4:00 PM.
     Never send it before the metrics are in.

  ✓  The investor update goes out on Fridays; investors read it before
     their Monday partner meetings, and it is worthless without the
     week's metrics.
  ```

  The test: when things change, a run that read facts still plans well;
  a run that read rules does yesterday's plan. Keep the reason inside
  the fact — a fact that carries its why still works in cases the
  wording never thought of. Save rules for the fragile core — an exact
  order of steps, a number that gates a decision, a thing that must
  never happen — and give even those their why, so the run knows where
  the rule stops.

- **Gather before you write.** This conversation is rarely the whole
  picture: search past sessions and memory, read the drive and the
  systems the area uses, and ask the user for what you can't find. A
  package written from one conversation repeats that conversation's
  blind spots forever.

- **Only what came from outside you.** The next run is you — same
  model, same tools. Whatever you can reason out now, it can reason
  out then, so your own advice, good practice, plans, and caveats
  never go in. What qualifies is information you did not produce:
  what a person said — a preference, a decision, a correction, an
  approval; what a person handed you — a playbook, a document, an
  export; what a search turned up that a run would not find on its
  first try; and what you did or received that cannot be got back — a
  request sent, a reply received, a pitfall paid for. The obvious
  result of a search is not worth a line; the non-obvious one is, in
  a sentence with its source. Bulk material — research, exports —
  lives on the drive; the package gets the finding and a pointer.

  A package of five lines is a finished package. Short means nothing
  unneeded got in, not that something is missing.

  ```text
  ✗  Prioritize investor fit and credible warm paths.
     (your own reasoning — the next run reasons the same)
  ✗  This analysis is not launch approval or an approved sending limit.
     (the charter's authority already says what is permitted)
  ✗  Six paragraphs of funnel benchmarks from three sources.
     (bulk research — the file on the drive gets one pointer)
  ✓  Mark wants individual angels, not VC funds — a fund role alone
     does not qualify (Mark, 2026-09-08).
  ✓  DocSend's 2021 study: 58 investors contacted per 30 meetings, warm
     and cold not separated (docsend.com/blog/…).
     (non-obvious search result, one line with its source)
  ✓  20 connection requests sent 2026-09-08; list in investors.csv —
     do not resend.
  ```

  Each fact lives in one place, pointed to from everywhere else —
  copies drift apart, and the stale copy wins.

- **Keep it current — a picture, not an archive.** The next run trusts
  a stale line as much as a true one, so stale text is worse than no
  text: when something changes, fix the line that said the old thing,
  and delete lines that no longer matter. A line says what holds now,
  never how it got there — history is the transcript's job. Most
  changes are edits, not additions; add a new line only when no line
  covers the point yet. The package should get sharper with every
  touch, never longer.

## The Structure

```
{profile_home}/responsibilities/<name>/
  RESPONSIBILITY.md   charter: trigger, duties, authority
  references/         one file per kind of work (replying to a support email,
                      reviewing an asset), plus fact files
  STATE.md            handoff to the next run
  state/              records too big for STATE.md, each linked from it
  archive/            finished records
  scripts/            executables: helpers, checks, gates
  schedules/          rhythms that run the area unattended
  webhooks/           outside events that wake it
```

## The Charter

The charter is the router: it holds no method, it points. The trigger
says when to open the package; duties say which reference handles which
work; authority says what the area may decide alone. Everything else —
steps, wording, criteria, facts — lives in the files it points to.

```markdown
---
name: customer-support
trigger: "Customer support over the shared inbox: replies, refunds, complaints."
lifecycle: ongoing        # omit for ongoing; "finite" requires ## Done when
---

## Duties
- …

## Authority
…
```

### The trigger

The trigger has one job: make the next run open the package at the right
moment. It is all a run sees when deciding what to open, so it names the
area and its facets — not what the package does with them. A trigger
that summarizes the procedure becomes a shortcut: the run follows the
summary and never opens the files. The roster shows one line per
package, cut at 120 characters — text past the cut is never seen when
routing.

```yaml
# Good — names the area and its facets:
trigger: "Customer support over the shared inbox: replies, refunds, complaints."

# Bad — preamble; the run sees no activity to match on:
trigger: "This responsibility should be opened whenever a customer emails us about anything."

# Bad — too generic to ever route:
trigger: "Handles support."
```

### Duties

One duty per kind of work the area owns. Each line answers two things:
what arrives, and which reference handles it.

```text
✓  A customer email arrives → read references/replying-to-support-mail.md
   before drafting anything.

✓  Promotional or automated mail → references/archiving-promotional-mail.md.

✗  Fresh-read each intake: the work, sibling publications, exact URL,
   assigned reviewer, latest review, owner scope, routing, and any
   active exception.
   (eight obligations in one line — clauses get dropped under load,
   and no file is named)
```

One clause per duty; a duty that grows steps or conditions is method
leaking upstairs — move it into its reference. And the test for the
whole list: the right reference must be pickable from the duty line
alone. If a run has to open files to decide which file applies, the
kinds of work are cut wrong — recut them until the duty line
disambiguates.

### Authority

What the area may decide alone, and where it must stop and go to
someone. Write it as you would brief a person, free form:

```markdown
## Authority
Replies, archiving, and refunds under $100 are yours end to end. Above
that, and on anything legal or pricing: propose to Mark and wait. Never
promise an action the product cannot perform — the closed list is
references/product-capabilities.md.
```

Every stop names its move — "propose and wait", "escalate with a
draft" — a bare "never" strands the run mid-task. Only what would be
costly if a run got it wrong belongs here; the craft of doing the work
well lives in the references.

## References

One file per kind of work, named for the activity it runs. A run opens
it at the moment of doing, so write it for that moment.

Match how exact you are to how fragile the work is — this choice
matters more than any formatting convention:

- **Where a mistake is costly or order matters**, give exact steps and
  forbid variation, each carrying its why in one clause.
- **Where judgment decides**, give facts, criteria, and reasons — and
  trust the run.

Most references mix both: exact steps for the fragile core, context
around it.

```markdown
# Handling refunds

The window is 14 days from purchase; past it the app store owns the
money and we cannot move it — promising otherwise has burned us.

1. Verify the purchase in Stripe; the receipt id must match.
2. Refund in Stripe first, then reply confirming — never the reverse.

Tone: brief and warm. The customer is usually annoyed, not hostile;
keep their vocabulary, skip the apology boilerplate.
```

A reference exists to make the run's work predictable — not identical
output every time, but the same useful discipline:

- **End steps with completion criteria.** "Every modified file
  accounted for" beats "summarize changes" — the run should know when
  it is done.
- **One default, with an escape hatch.** "Use X; for edge case Y use Z"
  beats a menu the run weighs every time.
- **Show, don't describe, outputs.** One real example of the reply or
  report anchors style better than a paragraph of adjectives.

Every reference is named by a duty, or by another reference one hop
away — a file nothing points to is a file no run will find.

## Scripts

Executables the area uses: helpers that do a step, checks that verify
one.

- **Bundle a helper when runs keep rewriting it** — save it once, point
  the reference at it.
- **Turn a rule into a check when prose keeps failing** — prose can be
  skimmed, an exit code cannot.
- **Let errors teach.** A failing script says what is wrong and what to
  do next; "Error: invalid input" wastes the run.
- **Say run or read** — those are different instructions.

## Schedules and Webhooks

A Schedule wakes a run on a clock; a webhook wakes one on an outside
event. They exist for work that does not arrive as a message — reviewing
an inbox, chasing overdue items, a recurring report, a deferred one-off.
Never for checking messages: those wake you already.

Every unattended run consumes model usage like a full conversation, so a
trigger is a purchase: buy the outcome with the fewest runs. Finish it
in the reply if you can; a one-shot if it must happen once later; a
sparse sweep if it recurs — support is 2–3 runs a day at most; start low
and increase only when runs demonstrably arrive too late. A guard script
when most runs would find nothing; a webhook when events are rarer than
any sweep worth running. Faster is an escalation the user agrees to
knowing the price — a user who learns the price from the bill was
ambushed.

Before creating or editing a declaration, read `{guides_root}/responsibility-authoring/references/schedules.md` or
`{guides_root}/responsibility-authoring/references/webhooks.md` — the format, cadence floors, guard contract, and
verification live there.

## STATE.md and state/

STATE.md has one purpose: the handoff to the next run. A line earns its
place by changing what that run does:

```text
✗  2026-07-20 09:00 — swept inbox, 14 threads, replied to 9.
   2026-07-21 09:00 — swept inbox, 11 threads, replied to 11.
   (a log of what runs did — the transcript already has it)

✗  The billing spreadsheet ID is 1g5oW…
   (a durable fact — it lives in a reference)

✓  Delta's refund dispute is the oldest open thread: their finance has
   been silent since 2026-07-20 — chase 2026-07-22, then escalate to
   Mark.
```

Rewrite it whenever the handoff changes, and prune in the same write:
the moment a line stops mattering to the next run, it goes. Dates
absolute — "2026-07-22", never "Wednesday" — the handoff is reread days
later, when relative words have quietly changed meaning.

Some areas track more than a handoff can hold — fifty open customers
when the next run needs the three that moved. That depth lives in
`state/`: one Markdown file per entity or thread, opened only by the run
whose work touches it, each pointed to from a STATE.md line. A finished
record that still answers a question moves whole to `archive/` as its
line leaves STATE.md; what answers no future question just goes. Grep
`archive/` when something old resurfaces, before saying you don't know.

Data files and machine artifacts never live in the package — they go on
the drive, with a STATE.md or state/ line pointing to them.

Most responsibilities never need `state/`: records go there when the
area arrives with a population to track — support with customers,
recruiting with candidates.

## Corrections and Refactoring

When someone corrects how the area is handled, record it right away —
the next message may never arrive; if what they meant is unclear,
recheck with the person or memory first.

Absorb it by asking, strictly in this order:

1. **Can text be deleted?** The line that caused the behavior may
   simply need to go.
2. **Is the shape wrong?** A duty that can't pick its reference, a
   charter thick with exceptions, two references always read together —
   recut, split, merge.
3. **Can a line be rewritten?** Edit it in place.
4. **Only then add** — when nothing existing covers the point.

Any other order biases every correction toward new text, and the
package only ever grows.

```text
Correction: "these Monday updates are way too long"

Before:  The Monday update covers each active project.

After:   The Monday update covers each active project in a line or
         two — leadership skims it on a phone; detail lives in the
         project channels.
```

Not the Before text plus "Keep updates short." appended underneath.

"Until X" / "try it for a month" → a dated STATE.md line with its own
expiry — the run that finds it lapsed deletes it.

Then reread the edit and replay the moment: would the next run do the
corrected thing? If the same miss keeps coming back, the fix sits too
low — move it up: the duty line that forces the read, or a script's
check.

## Chartering

Responsibilities are chartered, not transcribed: investigate first, ask
second, write only when you can play the work through without guessing.
Never author a package from the request alone. (One exception: a
minimal finite package for a self-contained deferred action — there the
request is the whole work.)

**Does the area already exist?** Open the closest charters and read
them — a roster scan settles nothing, because a charter rarely names
everything it owns. Work that serves an existing goal is a rescope of
that charter; a sibling package splits the area's state in two. A new
package is earned three ways: its own end state ("onboard Jane"), work
no charter covers, or an area grown into two jobs.

**Research, then ask.** Sweep what you can reach — past sessions,
memory, email, recordings, the drive, connected services. "I see
refunds come through the support inbox and get logged in the CRM — is
that the flow I own?" is a chartering question; "how do you handle
refunds?" says you didn't look. When research finds nothing, the user
is the only source — then open questions are exactly right.

**Clarify until nothing is a guess.** Ask for what you couldn't find:
the edges of ownership, your authority, who to escalate to, what
success looks like. Walk real items through the work — "two meetings
overlap Tuesday: do I resolve or flag?" beats "what are your
preferences?" Prefer a narrow charter you can answer for; scope can
grow later, and wrong guesses run on a schedule.

**Finite work** declares `lifecycle: finite` and a `## Done when` with
two evidence-decidable exits: **Done** ("Jane has repo access and
completed security training", not "Jane is settled in") and **Stop**
(no response after N attempts, a deadline passed) — without Stop, work
the world ignores runs forever. The run that verifies an exit records
the outcome, removes the schedules, reports it, and moves the package
to `.archive/`.

**The point is what's behind the words.** The user speaks in
fragments — "you own support now" carries a hundred unstated decisions.
Research and clarification recover some; for the rest, propose in your
reply and file only what the person approves. What they leave
unanswered is a STATE.md line naming the open question. An assumption
of yours filed as a rule is read as a decision by every later run.

**Handed material goes into the package, not into acknowledgment.**
When someone hands over a guide, playbook, or export that defines how
the area operates, adoption means every point of it lands in the
package — steps and rules into the work files, facts into fact files,
the original filed on the drive. Reshape freely; drop nothing: it was
tuned by someone's trial and error, and the point you drop is the case
the next run improvises. A summary is not adoption — the operative
detail is exactly what a summary loses. The test: could the next run
handle a live case with only the package? If it would have to open the
original document, or this conversation, the adoption is unfinished.

## Archiving

`{profile_home}/responsibilities/.archive/` is where packages retire. Move
a completed, stale, or unused package there whole: its schedules and
webhooks stop firing, its files stay intact, and moving it back
restores them. Archive instead of deleting — a permanent `rm -r`
happens only on the user's explicit ask.

## What the Platform Enforces

Writes through the file tools are validated — these need no vigilance;
the write itself reports them. Terminal writes bypass the validator, so
prefer the file tools for package files:

- Frontmatter: `name` (equal to the directory name) and `trigger`
  required. `lifecycle: finite` requires a `## Done when`.
- Budgets: charter ≤12,000 characters; STATE.md ≤6,000; each
  `references/` and `state/` file ≤10,000 (`state/` at most 100 files);
  `archive/` files ≤130,000. Files under `references/`, `state/`, and
  `archive/` are Markdown only — data belongs on the drive; `scripts/`
  holds executables, `schedules/` and `webhooks/` YAML.
- Writing `RESPONSIBILITY.md` for a new name creates the whole package
  atomically.

A rejected write is not an obstacle — it is the signal the shape is
wrong: prune, split, or archive.
