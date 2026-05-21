# Hermes Git Projects Architecture

## Data model

A project is a local Git clone plus scanned metadata. The canonical key is a stable SHA-1 derived from the local repo path. This avoids leaking remote URLs into IDs and keeps IDs stable if the dashboard rescans.

An issue is a local operational note that becomes a Kanban task. The issue record keeps the task id, selected skills, workspace path, recommended branch name, labels, severity, and optional parent task ids.

Suggested skills are stored as editable JSON records:

```json
{
  "skills": [
    {
      "name": "systematic-debugging",
      "label": "Systematic Debugging",
      "reason": "Use for bugs, regressions, failures, and unclear root causes.",
      "default": true,
      "triggers": ["bug", "error", "failure", "regression"]
    }
  ]
}
```

## Boundaries

- The plugin does not create remote GitHub issues automatically.
- The plugin does not store credentials.
- Destructive Git actions are intentionally omitted from the dashboard API.
- Pull uses `git pull --ff-only` so dashboard actions do not create merge commits.
- Push only pushes the current local branch to origin with upstream set.

## Kanban integration

Issue save attempts to import `hermes_cli.kanban_db` lazily. If Kanban is unavailable, the issue is still stored locally with a `todo_error` field so the user does not lose the report.
