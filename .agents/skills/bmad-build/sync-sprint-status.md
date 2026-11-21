Set `development_status[{story_key}]` to `{target_status}` in `{{.implementation_artifacts}}/sprint-status.yaml`.
If `{story_key}` is missing, warn once and stop.
If the story is already at `{target_status}` or later, stop.
When `{target_status}` is `in-progress`, set parent epic (e.g. `3-2-foo` → `epic-3`) from `backlog` to `in-progress` if present.
Update `last_updated`. Preserve comments and structure.
