# delegation-router

`delegate_task` normally spawns an **in-process clone** of the calling agent (same model, same
toolset). This plugin makes it route the work into a **profile pool** instead: each kind of task
goes to the Hermes profile that owns it, running as its own session with that profile's skills,
tools, memory and project knowledge.

```
you → default profile (command centre)
        └── delegate_task(routing="coding", project="acme")
              └── runs `hermes -p acme-dev chat` → real profile session
                    └── result re-enters your conversation as one message
```

## Install

This plugin ships in-tree; it still needs a per-profile opt-in, because it takes over a
built-in tool:

```bash
hermes plugins enable delegation-router
```

then copy the template and point it at profiles you actually have:

```bash
cp plugins/delegation_router/routing.yaml.example "$HERMES_HOME/plugins/delegation-router/routing.yaml"
hermes profile create coder --no-alias      # one profile per role you want to route to
```

Bucket → candidate names must match real profile names (`hermes profile list`). A bucket whose
candidates do not exist falls back to stock delegation, so a half-filled table is safe.

## What the model sees

The plugin replaces the `delegate_task` schema, so the calling model gets two extra arguments:

| arg | meaning |
|---|---|
| `routing` | bucket name (`coding`, `qa`, `design`, `ads`, …) — see `routing.yaml` |
| `project` | business/project key, so a bucket with a `role:` picks `<project>-<role>` |
| `profile` | explicit profile name, overriding both |

An unresolvable task (no bucket, no profile, unknown name, or `action=list/steer/stop`) is handed
to the stock delegation path unchanged — routing never silently swallows a task.

## Results

A routed run is a subprocess, so:

- the tool returns a dispatch handle immediately and the consolidated result re-enters the
  conversation as one message when the last task finishes (stock background-batch behaviour);
- each task reports `status`, `profile`, `bucket`, `persona`, `session_id`, `summary`,
  `skills_preloaded`, `skills_ambiguous_not_preloaded`, `skills_unavailable` and a
  `resume_hint` (`hermes -p <profile> --resume <session_id>`);
- runs appear in `delegate_task(action="list")` and the TUI/desktop agent view, and
  `action="stop"` with a `subagent_id` ends one early;
- every run writes a full transcript to `$HERMES_HOME/delegation-router/runs/<run>.log` and an
  audit line to `runs.jsonl`;
- a routed child may route once more when `max_depth: 2` — its own children run stock.

## Role charters (personas)

A bucket may name a `persona:` — a role charter under `$HERMES_HOME/personas/<name>/` that is
injected into the child's prompt and whose member skills are preloaded into the run. See
`website/docs/user-guide/features/profile-pool-delegation.md` for the format, and
`personas/` in that doc for an example.

## Compatibility

The plugin prefers the core hook — `AIAgent._dispatch_delegate_task` asking
`registry.plugin_handler("delegate_task")` who owns the tool. On a build without that hook it
falls back to wrapping `AIAgent._dispatch_delegate_task` at load time, so the same plugin file
works on stock Hermes too. The log line says which path is live:

```
delegation-router loaded (dispatch=core-hook, 12 buckets, 1 personas, host persona=researcher)
```

## Troubleshooting

- **Nothing routes**: `routing.yaml` missing (the plugin logs "no usable routing.yaml") or
  `default_profile: ""` with no matching bucket. Check `delegation_router_status`.
- **Skill refuses to load in a child**: the skill name resolves in two different roots for that
  profile. Hermes refuses an ambiguous skill; name one owner or archive the other copy.
- **A batch died with `status=interrupted`**: the owning session was interrupted or shut down —
  routed children end with their session, exactly like stock background subagents.
