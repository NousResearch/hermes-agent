# Kanban Project Policy

Hermes Kanban supports per-board project policy files so project-specific workflow discipline stays out of core Hermes source code.

A policy file declares the canonical project checkout, allowed task worktree root, denied control-plane/scratch roots, concurrency limits, and cleanup behavior for a board.

## Location

Policies live under the shared Kanban home:

```text
<hermes-root>/kanban/policies/<board>.json
```

For a standard install this is usually:

```text
~/.hermes/kanban/policies/<board>.json
```

Use:

```bash
hermes kanban --board <board> policy show --json
hermes kanban --board <board> policy validate --json
```

## Example

```json
{
  "project_root": "/path/to/projects/example-app",
  "base_branch": "main",
  "worktree_root": "/path/to/projects/example-app/.worktrees",
  "denied_workspace_roots": [
    "~/.hermes/kanban",
    "~/.hermes/scripts",
    "/tmp"
  ],
  "shared_project_root_writable": false,
  "scratch_repo_operations_allowed": false,
  "max_active_issue_pipelines": 1,
  "cleanup_after_merge": true,
  "forbid_dirty_completion": true
}
```

Do not put secrets, tokens, API keys, connection strings, or credentials in policy files.

## Fields

- `project_root`: canonical project checkout. Workers may inspect it, but it should usually be read-only.
- `base_branch`: expected base branch for new work.
- `worktree_root`: allowed root for task-specific repo-editing worktrees.
- `denied_workspace_roots`: roots where repo clone/build/install work is forbidden.
- `shared_project_root_writable`: when false, repo-touching cards targeting `project_root` fail validation.
- `scratch_repo_operations_allowed`: documents whether scratch/control-plane folders can run repo clone/build/install operations. Default false.
- `max_active_issue_pipelines`: optional board-level throttle for implementation/review/QA/security/release pipelines.
- `cleanup_after_merge`: documents that merged clean secondary worktrees should be eligible for cleanup.
- `forbid_dirty_completion`: repo-touching cards should complete clean or block with dirty-file evidence.

## Validation behavior

`hermes kanban validate` loads the active board policy and emits errors when repo-touching cards:

- target the shared project root while it is read-only,
- use a worktree outside the configured `worktree_root`,
- use a denied control-plane/scratch root,
- omit an explicit worktree path on a policy-controlled board.

## Cleanup vs teardown

Conservative cleanup:

```bash
hermes kanban --board <board> cleanup --dry-run --json
hermes kanban --board <board> cleanup --json
```

Cleanup only removes safe inactive clean secondary worktrees. Dirty, active, main, or unclear worktrees are blocked and reported.

Destructive teardown:

```bash
hermes kanban --board <board> teardown --remove-all-worktrees --delete-board --yes --json
```

Teardown is intentionally explicit. It stops relevant processes, force-removes non-main registered worktrees when requested, deletes the board directory when requested, and verifies without recreating the board.

## Updateability guidance

Prefer policy config over source patches. If a rule is specific to one project, put it in `<board>.json`, skills, profiles, or operator prompts. Only change Hermes source for generic behavior that any board can use.
