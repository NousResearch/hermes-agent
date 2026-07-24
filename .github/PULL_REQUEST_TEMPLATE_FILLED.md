## What does this PR do?

Add structured key naming to the Kanban swarm blackboard.

The blackboard already lets workers write `post_blackboard_update(conn, root, key="anything", value=...)`. But currently keys are ad-hoc strings — `"worker3_status"`, `"w3-result"`, `"sources"`. A coordinator scanning the merged blackboard can't tell which worker owns which key, or even whether a key IS a worker key.

This PR adds `kv_helpers.py` — **builders and parsers that enforce a naming convention without changing the storage layer**. If callers write `worker_key(3, "status")` instead of `"worker3_status"`, the coordinator can later `parse_worker_key(k)` → `(3, "status")` and machine-read the board. No schema changes, no new tables, no new protocol — it's a convention library, not a persistence layer.

The same naming convention was independently arrived at in CC_Pure's blackboard (`src/blackboard/kvHelpers.ts`) — an existence proof in another codebase that it works well for multi-worker blackboards.

## Related Issue

N/A — new capability, not a bug fix. Happy to open an issue first if preferred.

Fixes #

## Type of Change

- [x] ✨ New feature (non-breaking change that adds functionality)
- [x] ✅ Tests (adding or improving test coverage)

## Changes Made

- **New**: `hermes_cli/kv_helpers.py` — key builders (`worker_key`, `team_key`, `coordinator_key`), parsers (`parse_worker_key`, `parse_namespaced_key`), read-back utilities (`worker_fields`, `get_worker_field`, `worker_prefix`), and well-known constants (`TOPOLOGY_KEY`, `GOAL_KEY`, field names like `FIELD_STATUS`)
- **Modified**: `hermes_cli/kanban_swarm.py` — imports and re-exports all helpers; replaces bare `"topology"` string with `TOPOLOGY_KEY` constant
- **Modified**: `tests/hermes_cli/test_kanban_swarm.py` — 10 new tests (builder, parser, namespace, integration with real blackboard)

## How to Test

1. Run the swarm-specific tests:
   ```
   pytest tests/hermes_cli/test_kanban_swarm.py -v
   ```
2. Verify no regressions in the broader kanban suite:
   ```
   pytest tests/hermes_cli/test_kanban_*.py tests/tools/test_kanban_tools.py -q
   ```
3. (Optional) Exercise the helpers manually:
   ```python
   from hermes_cli.kv_helpers import worker_key, parse_worker_key, worker_fields
   assert worker_key(3, "status") == "worker:3:status"
   assert parse_worker_key("worker:3:status") == (3, "status")
   ```

## Checklist

### Code
- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md)
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`fix(scope):`, `feat(scope):`, etc.)
- [x] I searched for [existing PRs](https://github.com/NousResearch/hermes-agent/pulls) to make sure this isn't a duplicate
- [x] My PR contains **only** changes related to this fix/feature (no unrelated commits)
- [x] I've run `pytest tests/ -q` and all tests pass
- [x] I've added tests for my changes (10 new tests, 13 total including existing)
- [x] I've tested on my platform: Ubuntu 24.04 (Linux arm64, DGX Spark)

### Documentation & Housekeeping
- [x] I've updated relevant documentation (README, `docs/`, docstrings) — docstrings in `kv_helpers.py` serve as API reference; module docstring explains the convention
- [x] I've updated `cli-config.yaml.example` if I added/changed config keys — N/A
- [x] I've updated `CONTRIBUTING.md` or `AGENTS.md` if I changed architecture or workflows — N/A
- [x] I've considered cross-platform impact (Windows, macOS) per the [compatibility guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md#cross-platform-compatibility) — N/A (pure Python, no platform-specific code)
- [x] I've updated tool descriptions/schemas if I changed tool behavior — N/A (no new tools; these are library functions)
