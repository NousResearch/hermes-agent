# Monitoring — Watch the Pipeline + Intervene

After `setup.sh` fires the kanban, the work runs autonomously. The role of
this skill in the execution phase is to help the user (and the AI overseeing
the session) detect problems early and intervene effectively.

## Live monitoring commands

```bash
# Live event stream — task spawns, status changes, heartbeats, completions
hermes kanban watch --tenant <project-slug>

# Snapshot of the board
hermes kanban list --tenant <project-slug>
hermes kanban list --tenant <project-slug> --json     # machine-readable

# Per-status counts + oldest-ready age
hermes kanban stats --tenant <project-slug>

# Visual dashboard (browser)
hermes dashboard

# Inspect a specific task (includes comments + events)
hermes kanban show <task-id>

# Follow a single task's event stream
hermes kanban tail <task-id>
```

Verify available subcommands with `hermes kanban --help` — the kanban CLI
ships with `init / create / list / show / assign / link / unlink / claim /
comment / complete / block / unblock / archive / tail / dispatch / watch /
stats / heartbeat / log / runs / context / gc`.

The companion `scripts/monitor.py` polls the kanban via the CLI and surfaces
common issues (stuck tasks, missing heartbeats, repeated retries, dependency
deadlocks).

## What to watch for

### Healthy pipeline indicators

- Tasks transition `READY → RUNNING → DONE` in roughly the expected order
- Renderers emit periodic `kanban_heartbeat` events with progress (e.g. "frame
  240/720")
- Each task's runtime is well under its `max_runtime_seconds` cap
- No task accumulates more than 1 retry
- Dependency arrows resolve (children unblock as parents complete)

### Warning signs

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Task RUNNING but no heartbeat in 2+ min | Worker stuck, infinite loop, blocked on input | `hermes kanban show <id>` — read the worker's last events. The dispatcher SIGTERMs tasks that exceed their `max-runtime`; if you need to stop one earlier, `hermes kanban block <id>` then `hermes kanban archive <id>`, and create a re-run task. |
| Same task retried 2+ times | Reproducible failure (missing key, bad spec, broken tool) | `hermes kanban show <id>` to read failure events. Fix root cause before re-running. |
| RUNNING longer than max_runtime | Task is slow but progressing OR genuinely stuck | Check heartbeats with `hermes kanban tail <id>`. If progressing, the dispatcher will SIGTERM eventually anyway — raise `max-runtime` on a re-created task. |
| Child task READY but parents still RUNNING for >2× expected | Cascade slow, dependency miswired | Check the dependency graph. Inspect the parent: sometimes it completed but its handoff fields (summary, metadata) were empty so the child has nothing to consume. |
| New tasks not appearing | Director is hung in decomposition | Inspect director task with `kanban show`. Often a malformed `kanban_create` call. |
| Specialist tasks completing instantly | Decomposition created tasks without bodies | Director didn't pass enough context. Re-create with explicit body content. |
| Tasks created but never picked up | Profile not running, or tenant mismatch, or dispatcher not running | Check `hermes profile list` (profile exists?), `hermes status` (gateway/dispatcher up?), and verify tenant. |
| Specific renderer task fails → review note → renderer redoes → fails again | Brief is asking for the impossible | Pivot the brief, not the renderer. |

## Intervention recipes

### Rejecting bad output

When a renderer ships a clip that doesn't pass review:

```bash
# 1. Comment on the renderer's task with specific feedback
hermes kanban comment <renderer-task-id> "Scene 3 looks too sparse \
— increase visual density. Tighten color palette to brand spec."

# 2. Create a re-render task with the original as parent
hermes kanban create "Scene 3 — re-render with feedback" \
    --assignee renderer-ascii \
    --parent <renderer-task-id> \
    --workspace dir:"$HOME/projects/video-pipeline/<slug>" \
    --tenant <slug> \
    --skill ascii-video \
    --max-runtime 30m
```

### Adding a new dependency mid-flight

When the editor needs an asset that wasn't originally planned (e.g., a captions
file):

```bash
# 1. Create the new task and capture its id
NEW_TASK_ID=$(hermes kanban create "Generate SRT captions from voiceover" \
    --assignee captioner \
    --workspace dir:"$HOME/projects/video-pipeline/<slug>" \
    --tenant <slug> \
    --json | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")

# 2. Wire it as a parent of the editor's task with `kanban link`
hermes kanban link "$NEW_TASK_ID" <editor-task-id>
```

`kanban link` takes `parent_id child_id` (parent first). Use `kanban unlink`
to remove a dependency.

### Stopping a worker that's stuck

The kanban dispatcher will SIGTERM (then SIGKILL) any task that exceeds its
`--max-runtime` automatically. To stop one sooner:

```bash
# Mark blocked so the dispatcher leaves it alone, then archive
hermes kanban block <task-id>
hermes kanban archive <task-id>

# Diagnose what happened
hermes kanban show <task-id>      # task body, comments, recent events
hermes kanban tail <task-id>      # follow the live event stream
hermes kanban log <task-id>       # worker process log
```

After stopping, decide: fix root cause + re-create the task, or skip and
adjust dependent tasks.

### Pivoting the brief

If during execution the user wants something fundamentally different:

1. Cancel the active director task and all RUNNING children
2. Edit `brief.md` and `TEAM.md`
3. Re-fire the initial `hermes kanban create` for the director

Don't try to "edit while running" — the kanban's audit trail makes a clean
pivot more legible than mid-stream changes.

## Periodic check-in script

A simple polling pattern for hands-off monitoring:

```bash
while true; do
    clear
    hermes kanban list --tenant <slug>
    echo "---"
    hermes kanban stats --tenant <slug>
    sleep 30
done
```

For a live event feed, run `hermes kanban watch --tenant <slug>` in a
separate terminal — it streams task lifecycle events as they happen.

For automated intervention (auto-restart stuck tasks, auto-create re-render on
review failure), see the `scripts/monitor.py` patterns.

## When to call it done

The pipeline is finished when:

1. All RENDER tasks complete and pass review
2. The editor's `output/final.mp4` exists and `ffprobe` confirms expected
   duration + streams
3. The reviewer (if present) has approved
4. Optional masterer variants exist

At this point, present the final.mp4 path to the user along with any review
notes. Do NOT delete the workspace — the user may want to iterate on a single
scene without re-running the whole pipeline.

## Common gotchas

- **Tenant mismatches.** A task created with the wrong tenant won't appear in
  monitoring. Always pass `--tenant <slug>` consistently.
- **Profile process not running.** Tasks queue indefinitely in READY if no
  worker for that profile is online. Check `hermes profile list` and start
  any missing profiles.
- **Workspace permissions.** All profiles need read+write to the workspace
  directory. `chmod -R u+rw <workspace>` if any worker reports permission
  errors.
- **Audio/visual sync.** The editor's clip stitching must match the
  renderer's actual output durations. Don't hardcode scene durations in
  the editor — read from the renderer's handoff metadata.
