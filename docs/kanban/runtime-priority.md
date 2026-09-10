# Kanban runtime priority guard

Hermes can reserve host capacity for a strict application runtime without
starting, stopping, or signalling that runtime. The guard compares process
argument vectors and working directories with exact configured project roots
and entrypoints. When linked-worktree matching is enabled, only worktrees that
share the configured repository's Git common directory are admitted.

```yaml
kanban:
  max_in_progress: 8
  priority_runtime_guard:
    enabled: true
    project_roots:
      - /absolute/path/to/project
    entrypoints:
      - main.py
    include_linked_worktrees: true
    normal_max_in_progress: 8
    max_in_progress: 3
```

The explicit `kanban.max_in_progress` is the ordinary performance cap. While
the protected entrypoint is running, the effective cap is the lower of that
value and `priority_runtime_guard.max_in_progress`. An incomplete process scan
fails safe to the protected cap. Unrelated files named `main.py`, descendant
directories, and repositories with a different Git common directory do not
match.

Local PR CI audits use a separate exact-head lifecycle receipt. The audit
claims its repository, PR, base, head, and lane-manifest identity before
bootstrapping, records the real supervisor PID, and fences a second audit while
that supervisor remains alive. This lets deterministic long-running lanes use
a larger task runtime envelope without creating duplicate restart loops.
