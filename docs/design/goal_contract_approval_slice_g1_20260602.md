# Goal Contract + Approval Slice G1

Date: 2026-06-02

## Scope

G1 defines static vocabulary for future goal contracts and approval slices. This
slice is specification-only: it adds names and values that later work can
reference, without changing command parsing, runtime behavior, persistence,
gateway handling, dashboard UI, or enforcement hooks.

The Python source of truth is `hermes_cli/goal_contract_spec.py`.

## Goal Contract Control Field

`stop_after_status_report` is the only G1 control field. It is a declarative
flag for a future contract that says the agent should stop after reporting
state. G1 does not interpret or enforce the flag.

## Approval State Values

The G1 approval state vocabulary is:

- `none`
- `active`
- `expired`
- `revoked`
- `completed`
- `blocked`

These values describe the lifecycle of an approval slice. G1 does not map them
onto existing approval inbox statuses or existing goal states.

## Approval Slice Source

The G1 source fields are:

- `created_by`
- `created_from`
- `raw_user_approval`
- `created_at`

`created_from` may be:

- `manual_command`
- `ui_form`
- `template`

These fields are provenance metadata only in G1.

## Presets

The G1 preset names are:

- `discussion-only`
- `inspection-only`
- `implement-slice`
- `commit-only`
- `local-smoke-test`
- `stop-state-only`

Presets are named vocabulary only. G1 does not add preset expansion, command
syntax, persistence, or enforcement.

`stop-state-only` maps to `stop_after_status_report`.

## Action Categories

The G1 action category vocabulary is:

- `discuss`
- `plan`
- `inspect_repo`
- `read_files`
- `search_files`
- `edit_files`
- `run_focused_tests`
- `run_broad_tests`
- `run_build`
- `run_lint`
- `run_dev_server`
- `browser_qa`
- `install_dependencies`
- `change_config`
- `touch_secrets`
- `commit`
- `push`
- `open_pr`
- `deploy`
- `restart_service`
- `public_bind`
- `oauth_connector`
- `external_network`
- `destructive_git`

These categories give later slices a shared language for approvals and contracts
without coupling this spec to current runtime modules.

## Checkpoints

The G1 checkpoint vocabulary is:

- `stop_after_status_report`
- `stop_after_plan`
- `stop_after_inspection_report`
- `stop_after_implementation_report`
- `stop_after_validation_report`
- `stop_after_local_commit_report`
- `stop_on_scope_expansion`
- `stop_on_validation_failure`
- `stop_on_unrelated_dirty_files`
- `stop_on_dependency_change_needed`
- `stop_on_restart_or_deploy_needed`
- `stop_on_user_message_conflict`

These names are not connected to the existing filesystem checkpoint manager in
G1. They are event-based continuation checkpoint labels for future design work.

## Policy Values

The G1 policy vocabulary for commit, restart, deploy, and push is:

- `forbidden`
- `allowed`
- `requires_approval`

The constants are duplicated by policy surface in the spec module so later
schema work can reference a named set for each side-effect class.

## Non-Goals

G1 does not change:

- runtime goal or approval behavior
- command parsing
- CLI, gateway, TUI, or dashboard behavior
- persistence, state shape, or migrations
- enforcement hooks
- existing goal or approval modules

Any behavior that consumes these constants belongs to a later slice.
