# claude-selfimprove

This pipeline reads Claude Code transcripts. It finds lessons in them. It
writes safe, approved lessons into Claude Code's global configuration
files. It runs on a schedule. A human does not review each change before
it lands. Conservative rules control what qualifies.

## What this pipeline does

1. It scans transcript files under `~/.claude/projects/` and
   `~/.claude-hermes/projects/`. It reads only new content since the last
   scan.
2. It finds short pieces of text that look like a lesson: a correction, a
   direct instruction, a confirmed fix, or a confirmed procedure.
3. It sends only those short pieces of text to a model. It asks the model
   to confirm the lesson and to write it in plain language.
4. It stores each lesson as a candidate. A candidate tracks how much
   evidence supports it.
5. Once a candidate has enough evidence, and does not conflict with an
   existing lesson, the pipeline writes it to one of three places:
   - the managed section of `~/.claude/CLAUDE.md`
   - a rule file under `~/.claude/rules/`
   - a skill file under `~/.claude/skills/`
6. The pipeline sends a short Slack message for each applied change,
   blocked conflict, or failure. It never sends the raw transcript text.

## What this pipeline does not do

- It does not touch project-specific files. A lesson that applies to one
  repository only stays out of the global configuration.
- It does not run any instruction found inside a transcript. Transcript
  text is data. The pipeline reads it. It never obeys it.
- It does not edit content it did not create. It checks an ownership
  marker before it changes any rule file or skill file. It refuses the
  write if the marker is missing.
- It does not grow `CLAUDE.md` without bound. The managed section has a
  strict character limit.

## First activation: seed before you scan

The transcript corpus already on disk predates this pipeline by months.
The first scan must not read all of it and score it as if it happened
last night. Run `seed` once, before the first `scan`:

```
python -m claude_selfimprove.cli seed
```

`seed` marks every transcript file that exists right now as already
read. It mines no candidates from that history. After `seed`, the nightly
scan only ever looks at content written after this point. `seed` refuses
to run a second time unless you pass `--force` — a safety check against
losing the place of a profile that has already been scanning for a
while.

## The two passes

### Scan (nightly)

The scan pass reads new transcript turns. It flags candidates. It
classifies them with a model call. It records evidence in a local
database. **The scan pass never writes to a Claude Code target file.**

Run it by hand:

```
python -m claude_selfimprove.cli scan
python -m claude_selfimprove.cli scan --dry-run
```

### Consolidate (weekly)

The consolidate pass checks every pending candidate against its evidence
threshold. It checks the candidates that pass against everything already
applied, to catch a direct conflict. It writes only the candidates that
clear both checks.

Run it by hand:

```
python -m claude_selfimprove.cli consolidate
python -m claude_selfimprove.cli consolidate --dry-run
```

Add `--dry-run` to see what a pass would do, with no file changes and no
database changes.

## Evidence thresholds

A candidate needs enough proof before it can apply. The proof needed
depends on why the candidate exists.

| Kind | Threshold |
|---|---|
| An explicit instruction from George | One confirmed message |
| A rule, or a CLAUDE.md entry | Three sessions and two tasks |
| A skill | Three confirmed uses |

Every candidate also needs a confidence score of 0.6 or higher from the
model that classified it.

A conflicting candidate never applies. This check has no threshold that
overrides it.

## Safety rules

- Every write makes a backup first. Roll back any write with the
  `rollback` command below.
- Every write happens through a temp file and an atomic rename. A crash
  mid-write cannot leave a broken file.
- Every write is checked again right after it lands. A mismatch triggers
  an automatic rollback.
- Every write is scanned for credential-shaped text first. The pipeline
  refuses the write if it finds one.
- Every mutation is recorded in an append-only log at
  `~/.hermes/state/claude-selfimprove/audit.log.jsonl`.

## Operator commands

```
python -m claude_selfimprove.cli install [--dry-run]
python -m claude_selfimprove.cli status
python -m claude_selfimprove.cli seed [--force] [--dry-run]
python -m claude_selfimprove.cli scan [--dry-run] [--model M] [--provider P]
python -m claude_selfimprove.cli consolidate [--dry-run] [--model M] [--provider P]
python -m claude_selfimprove.cli rollback --list
python -m claude_selfimprove.cli rollback --target /path/to/file
python -m claude_selfimprove.cli rollback --backup-id /path/to/backup.json
```

`install` deploys the package and the two entry scripts into
`$HERMES_HOME/scripts/`, then registers the nightly-scan and
weekly-consolidate cron jobs — but only the ones that do not already
exist by name. Running `install` again after a code change redeploys the
latest code and touches no cron state, because both jobs are already
there. This is the only supported way to activate or update this
pipeline; nothing about it is a manual step.

`status` shows: candidate counts by state, the lock state, the backup
count, and the last ten audit log entries.

`rollback --list` shows every backup, most recent first.

`rollback --target PATH` restores the most recent backup for that one
file.

## Where state lives

All state lives under `~/.hermes/state/claude-selfimprove/`:

- `checkpoints.json` — the scan watermark for each transcript file
- `candidates.db` — the candidate database (SQLite)
- `audit.log.jsonl` — the append-only audit log
- `backups/` — one JSON file per write, holding the prior file content
- `run.log`, `.lock` — run bookkeeping

No raw transcript text lives in any of these files. A candidate stores a
hash of its source location, not the source text.

## Notifications

The pipeline does not send Slack messages on its own. It writes events
into `~/.hermes/state/self-improvement-notify/queue.jsonl` — the same
queue George's skill curator and memory review already use. The existing
`self_improvement_watch.sh` job reads that queue, groups the events, and
posts one message. This keeps one delivery path and one place that
prevents a duplicate message.

## Model calls

The pipeline never imports the Hermes agent runtime. It calls the
`hermes chat` command instead, with the `session_search` toolset only —
a read-only toolset. This stops a hostile instruction hidden in a
transcript from reaching a tool that could change a file, run a command,
or send a message. See `llm.py` for the full explanation.

Override the model with the `CLAUDE_SELFIMPROVE_MODEL` and
`CLAUDE_SELFIMPROVE_PROVIDER` environment variables. With neither set, the
pipeline uses the Hermes profile's own default model.

## Scheduled jobs

`python -m claude_selfimprove.cli install` registers two Hermes cron jobs:

- `claude-selfimprove-nightly-scan` — schedule `0 3 * * *`, runs the scan
  pass every night.
- `claude-selfimprove-weekly-consolidate` — schedule `0 4 * * 0`, runs the
  consolidate pass once a week.

Both run `--no-agent` (script stdout only, no model call for the run
itself) and `--deliver local` (Slack visibility comes from the shared
notification queue described above, not from cron's own delivery).

Check their state with `hermes cron list` or `hermes cron status <job-id>`.

## Recovering from a bad change

1. Run `python -m claude_selfimprove.cli rollback --list` to find the
   backup.
2. Run `python -m claude_selfimprove.cli rollback --target <path>` to
   restore the file's content from right before the pipeline's last write
   to it.
3. Check the candidate's row in `candidates.db` if you also want to stop
   the pipeline from reapplying it. Update its `status` column to
   `rejected` directly with `sqlite3`, or wait for a future `reject`
   CLI command.

## Known limits

- Conflict detection depends on a model call. When that call fails, the
  pipeline treats every candidate under review as conflicting. It blocks
  every one of them rather than guessing wrong. Run `consolidate` again
  once the model is available.
- The heuristic pass that flags candidates uses fixed word patterns. It
  will miss a lesson phrased in an unusual way, and it will occasionally
  flag ordinary conversation. The model classification step that follows
  drops the false flags.
- A "task" is one matched candidate occurrence. A "session" is the
  transcript session it came from. These two counts approximate real
  independent confirmation. They are not a perfect measure of it.
- **The pipeline cannot tell George's own words from pasted text.** A
  Claude Code session often includes text George pasted in: a web page, a
  log file, a file from a repository other sessions also read. The
  heuristic pass reads that pasted text the same way it reads a message
  George typed himself. A phrase planted in content George pastes across
  three separate sessions — for example, a comment buried in a shared
  script — can clear the same-session-count threshold, pass the
  confidence floor, pass the conflict check, and land in `CLAUDE.md`, a
  file every future Claude Code conversation loads. The confidence floor
  and the conflict check both reduce this risk. Neither one closes it.
  Read `status` output after a consolidation run, especially early on,
  and use `rollback` if a written lesson looks wrong.
