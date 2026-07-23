# Changelog

## 1.0.3 - 2026-07-11

### Fixed
- Asset-copy step in the generated `setup.sh` quoted generator-controlled destination paths (`$WORKSPACE/...`) with `shell_single_quote`, which prevented Bash variable expansion and broke every `cp` against a placeholder path. The copy now uses a `shell_double_quote_expand_vars` helper for those targets; user-supplied asset paths keep strict single-quote escaping.
- `setup.sh` only wrote `toolsets` and `skills.always_load` to each profile's `config.yaml`. Kanban workers spawned today read their tool surface from `platform_toolsets.cli`, so dispatcher-spawned workers ended up with the wrong tool set. The profile patch now writes `platform_toolsets.cli` alongside the legacy keys and asserts on both.
- Workspace guidance in the director's opening body and in every generated `SOUL.md` used a Python-shaped example (`workspace_kind="dir"`) instead of the actual Hermes CLI flags. Replaced with concrete `hermes kanban create --workspace dir:<path> --tenant <slug>` examples.
- `scripts/monitor.py` only inspected fields the JSON from `hermes kanban list` does not return (`heartbeat_at`, `max_runtime_s`, `retries`), and its timestamp parsing assumed ISO strings while Hermes actually returns Unix epoch seconds. The monitor now resolves run state through `hermes kanban show --json` (`task.runs[-1]`), parses both epoch and ISO formats as UTC, and counts retries from `len(runs)-1`. STUCK / OVERTIME / FLAPPING detection now actually fires.

### Added
- `create_profile` shell function in `setup.sh` that marks every created profile directory with `.kanban-video-orchestrator-owner` containing the project slug. Re-runs against an existing profile are idempotent; collisions with foreign profiles abort with a clear message before overwriting anything.
- `profile_description()` generator function and a `hermes profile describe --text` call so every pipeline profile gets a non-empty description, which is what the kanban decomposer routes on (no description → blind decompose).
- Concrete heartbeat invocation (`hermes kanban heartbeat <task-id> --note "..."`) in the common rules section of every generated `SOUL.md`.
- README and `docs/dry-run-checklist.md` updated to match the fixed Hermes CLI shape (no more `hermes kanban stats --tenant`, no more `workspace_kind=` pseudocode).

### Notes
- Placeholder asset paths in the example plans must still be replaced before a real setup run.
- Smoke test: ran the new release against a fake-Hermes sandbox with dummy assets — `setup.sh` exits 0, all 10 profiles created with owner markers, all asset copies land in the right subdirectories, initial director task is fired.
- This release does **not** modify `terminal.cwd` or approval settings in any Hermes profile.