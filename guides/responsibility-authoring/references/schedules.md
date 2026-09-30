# Schedules

Every agent Schedule — recurring or one-shot — belongs to a responsibility.
Read that package before creating or retuning its rhythms.

You wake automatically when anyone messages you on the platform. Schedules
exist for the work that does **not** arrive as a message — reviewing an
inbox, chasing overdue items, producing a recurring report, firing a
deferred one-off. Never create a Schedule to check for messages: direct
messages wake you already, and a Schedule doing the same work is pure cost.

## Cadence

The instinct to schedule frequently is wrong: agents wake far too often,
and every unneeded run costs real money for nothing. Choose the fewest
runs that still catch the work in time. When the user asks for more than
the work needs, say what the extra runs buy and what they cost, recommend
your cadence, and arm the faster one only when they confirm knowing both.
Anchors:

- Email or customer support: 2–3 runs per day, maximum.
- Managing freelancers or one-off workers: 1 run per day is enough.
- Managing full-time people: 2–3 runs per day is enough.

Start at the low end; increase only when runs demonstrably arrive too late.

## One-Shots and "Make Sure"

A one-shot discharges an ask that one touch completes: "remind me
Tuesday" is done when the reminder is sent. "Make sure X happens" /
"ensure it's done" is a different ask — it is discharged by the outcome
observed, never by the reminder sent. That work follows the first touch
with checks that chase until the outcome is confirmed or handed back: a
finite responsibility with Done and Stop exits, not a single fire.

## One Schedule, or a Partition

One Schedule per responsibility is the default, and its scope must say so
explicitly: the run carries the full responsibility. Add more Schedules
only when the responsibility has grown large enough that one run cannot
cover it — then partition, don't duplicate: each Schedule's scope states
exactly which slice of the responsibility its runs perform, and the slices
together cover the whole with no work claimed twice. A Schedule whose
scope is left implicit either redoes another rhythm's work or assumes
someone else has it.

Schedules of one responsibility must never run at the same time: they share
`STATE.md`, and a run that fires while another is still working rewrites
the same handoff — one run's record silently wins. Set fire times far
enough apart that one run finishes before the next begins.

## Declaration Format

A Schedule is one YAML file under the package's `schedules/` directory; the
filename is its name:

```yaml
# schedules/<name>.yaml
schedule: "0 9 * * 1-5"     # required — when to fire
scope: |                     # required — the slice this run carries
  This run carries the full responsibility.
report: "slack:C0123ABC"   # required — the run's reporting line
repeat: 12                   # optional — total runs; omit for unlimited
script: scripts/check.sh     # optional — guard: fires instead of the agent,
                             #   which wakes only when the script prints output
```

Four accepted forms for `schedule`: `every 30m` / `every 2h` (recurring
interval), `0 9 * * 1` (five-field cron, recurring), `30m` / `2h` / `1d`
(one-shot, that far from now), `2026-08-01T09:00` (one-shot at that time).
Times and cron fields are the workspace's timezone — when someone says
"9am", that's their 9am. An optional `timezone: Europe/Berlin` (IANA name)
pins one Schedule to a different zone; omit it unless the rhythm genuinely
belongs elsewhere. `report` is required and deliberate: one exact
target returned by `send_message(action="list")` — the current
conversation is always listed first — or `muted`. Report to a channel
or group, not a thread, unless the user explicitly asks for the
thread; a thread target is the channel target plus `:<thread id>`.
`report` is the run's reporting line: the final response posts there
automatically, and so do the platform's own notices about the Schedule —
failure pauses, limit skips. Point it where oversight of this work
lives — usually the conversation that commissioned it, or the team's
channel. The people the work manages or serves are never the
reporting line: reaching them is the work itself, done in-run with
`send_message` and judgment about whether and what to send. A run that
leaves nothing for the reporting line replies exactly `[SILENT]`.
Older declarations with `report: origin` keep running unchanged, but
any edit must replace it with an exact target.

The scope states the run's share and nothing else: the full
responsibility, or exactly the slice this rhythm carries. The work itself
lives in the package — every run receives the whole charter and STATE.md
and reads the references its duties name — so a scope that restates
tasks, policy, or approval rules plants a second copy that keeps firing
after the package is corrected. When an
existing declaration — Schedule or webhook — still carries task or
policy text (older files name the field `prompt`), move that text into
the package (a charter duty, a reference) in the same edit that renames
the field and cuts it to scope; never just delete it. There is no
immediate-run action: to run something now, write a one-shot declaration
with a near-future time; editing a completed one-shot's `schedule` to a
new time re-arms it.

## Lifecycle

Writing the file arms the Schedule; deleting it removes it — removal is
the stop, there is no manual pause. Every run opens the responsibility's
current package. A failed run is recorded and skipped, never retried;
after three consecutive failures the platform stops arming the Schedule
and notifies its reporting line, and editing the file re-arms it.

## Guard Scripts

A Schedule that exists to watch something — an external batch, a queue, a
long-running process — should not spend an agent run to discover that
nothing changed. Give it a guard script: the script fires on the schedule
instead of the agent, and the agent runs only when the script reports
something worth acting on.

- `script` names a file inside the package (`scripts/check.sh`); `.sh`
  and `.bash` run via bash, `.py` via python, from the script's
  directory, with a bounded timeout and no model cost. A declaration
  whose script file is missing is not armed.
- The script's stdout is the whole gate. Print nothing, or exactly
  `{"wakeAgent": false}`, and the tick ends there — no agent run. Print
  anything else and a run fires carrying the Schedule's scope plus the
  script's output as context.
- Only stdout gates the wake; stderr is discarded on a healthy tick. Put
  diagnostics on stderr so a quiet tick stays quiet.
- A script that exits non-zero, times out, or fails to start wakes a run
  carrying the failure text — a broken guard fails loud, never silent.
  Don't trap errors into silence.
- Stdout is injected into the woken run after secret redaction: print
  the facts the run needs, not a bare signal — and never credentials.
- The script decides whether to wake a run, never what to decide:
  checking state belongs in the script, judgment in the scope.
- Editing a script takes effect on the next tick — no reconciliation, no
  YAML rewrite needed.

A guard for a checkpointed long job:

```bash
#!/bin/bash
# scripts/check-crunch.sh — prints nothing while healthy
cd /path/to/crunch || { echo "repo missing"; exit 1; }
if [ -f out/DONE ]; then
  echo "crunch finished; results in repos/crunch/out/"
  exit 0
fi
if ! pgrep -f "python crunch.py" >/dev/null; then
  echo "crunch.py not running; last checkpoint: $(tail -1 out/checkpoint.log)"
fi
```

Cadence floors, enforced at reconcile — a declaration below them is not
armed, and the write's warning says what to change:

- Without a guard script, recurring intervals under 15 minutes are not
  armed.
- With one, the floor is 5 minutes — and any recurring interval under 15
  minutes must also declare `repeat`: watching is a bounded activity,
  not a permanent state. If the watch has no natural end, prefer a
  webhook that pushes to you, or widen the interval.
- Ticks tighter than about 15 minutes keep the workspace computer awake
  between them — cheap in runs, not free. Choose the widest interval
  that still catches the change in time.
- One-shots are unaffected.

The same pattern revives long heavy jobs. A background process dies with
machine replacement (routine, and possible at any time), so a job that
may outlive the machine writes checkpoints to files and carries a guard
that checks it: alive and progressing prints nothing; dead or finished wakes
a run to resume from the checkpoint or wrap up. The run that finds the
job complete deletes the Schedule's file as part of finishing.

## Verification

Confirm each declaration armed — each write's own `schedule_reconciliation`
shows its next fire — before reporting done. A write that changes nothing
confirms as `verified_unchanged` with the same armed state; never edit a
declaration's content merely to force a receipt.

- [ ] Cadence is at the low end of the anchors, and any faster cadence
      was armed only on the user's informed confirmation
- [ ] Every scope states the run's share — the full responsibility, or
      exactly which slice of it
- [ ] No scope carries instructions, policy, or approval wording
- [ ] Every `report` points where oversight lives — never at people the
      work manages or serves
- [ ] Fire times of a responsibility's Schedules never overlap
- [ ] No Schedule exists to check for messages
- [ ] Anything that watches an external system or a long job is guarded
      by a script, not polled by bare agent runs
- [ ] Every recurring interval under 15 minutes carries both `script` and
      `repeat`
- [ ] Each write's `schedule_reconciliation` shows it armed with its next
      fire
