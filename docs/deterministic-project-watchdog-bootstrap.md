# Deterministic downstream project watchdog bootstrap

Status: downstream-only Hermes Maintenance artifact. Do not upstream to NousResearch/hermes-agent.

Every active downstream project index entry must be sufficient to derive one project-owned watchdog without LLM guesswork. The watchdog belongs to the coordinator/project layer, not the PA/global layer.

Required active project index fields:

- project_tag
- status: active, working, or running
- board_slug
- coordinator_profile
- matrix.room_alias
- matrix.space_alias
- matrix.encryption with an explicit encrypted/unencrypted project decision
- github.repository
- repository.path and repository.worktree_root when the project has a delivery repo
- worker_profiles for expected project workers
- metadata.dispatch_owner: exactly one owner, equal to the coordinator profile
- metadata.watchdog_owner: equal to the coordinator profile
- metadata.pa_audit_owner: PA/global audit owner only, not runtime owner

Backfill output for each active project:

- projects/<project>/watchdog/policy.yaml
- projects/<project>/watchdog/runtime_config.yaml
- projects/<project>/watchdog/state/
- projects/<project>/watchdog/latest-report/
- projects/<project>/watchdog/recipes/project-watchdog.service
- projects/<project>/watchdog/recipes/project-watchdog.crontab
- projects/<project>/watchdog/recipes/hermes-cron.yaml
- projects/<project>/watchdog/validation_gates.yaml

Validation gates require:

- Kanban board binding exists in the index.
- Matrix route binding exists in the index.
- GitHub repository binding exists in the index.
- Exactly one dispatch owner exists.
- Scheduler ownership is coordinator-owned.

Reusable templates live in docs/templates/project_watchdog/ and must stay sanitized. They may contain placeholders such as {{ project_tag }} but must not contain live Matrix room/user IDs, API tokens, GitHub tokens, or host-specific absolute paths.

Use hermes_cli.project_watchdog_bootstrap.backfill_project_watchdogs(...) to emit artifacts and hermes_cli.project_watchdog_bootstrap.validate_project_watchdog_index(...) to verify every active index entry.
