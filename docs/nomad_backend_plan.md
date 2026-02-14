# PR3 Plan: Optional Nomad SlotPool Backend (Draft)

Goal: Reintroduce the old `atropos-agent` Nomad SlotPool sandbox backend **without competing** with the current default terminal/Modal code path.

## Non-goals
- Do not change the model-facing tool schema.
- Do not change the default behavior.
- Do not require Nomad unless explicitly enabled.

## Desired UX

Default (today):
- `TERMINAL_ENV=local|docker|singularity|ssh|modal`

Add (PR3):
- `TERMINAL_ENV=nomad` (or `TERMINAL_ENV=sandbox`)
  - Uses slot-based multiplexing: tasks acquire a slot, execute commands inside it, release.
  - Driver can be `docker` or `singularity` (Apptainer).

Minimal switching friction:
- same `terminal_tool(command, task_id=...)` interface
- only env vars change

## Proposed integration approach

Because the full Nomad/SlotPool implementation lives in `/Users/shannon/Workspace/Nous/atropos-agent`, we should not duplicate it in Hermes-Agent.

Instead:

1) Add an **optional dependency** on `atropos-agent` (package name TBD) behind an extra, e.g.
   - `pip install hermes-agent[nomad]`

2) Add a small adapter module in Hermes-Agent:
   - `tools/nomad_pool.py` (adapter)

3) In `tools/terminal_tool.py` extend `_create_environment()`:
   - accept `task_id` (already added in PR2 for modal pooling)
   - add a new env_type:
     - `env_type == "nomad"` → return `NomadPooledTaskEnvironment`

4) `NomadPooledTaskEnvironment` mirrors `ModalPooledTaskEnvironment`:
   - owns a slot lease
   - executes command within slot workspace
   - releases slot on cleanup

## Configuration env vars

- `TERMINAL_ENV=nomad`
- `TERMINAL_NOMAD_ADDRESS=http://localhost:4646`
- `TERMINAL_NOMAD_DRIVER=docker|singularity`
- `TERMINAL_NOMAD_IMAGE=...`
- `TERMINAL_NOMAD_SLOTS=10`
- `TERMINAL_NOMAD_MIN=1`
- `TERMINAL_NOMAD_MAX=10`

Optionally:
- `TERMINAL_NOMAD_AUTOSTART=1` (start `nomad agent -dev ...` locally if not running)

## Testing plan (when compute available)

- Sanity: `TERMINAL_ENV=nomad` run `terminal_tool("echo hello")`
- Concurrency: run N parallel tasks, ensure slot leases are unique and isolated
- Cleanup: ensure slot release happens even on exceptions

## Current status

- Hermes-Agent `main` does **not** contain the Nomad backend. The full implementation is in `atropos-agent`.
- PR3 should be an adapter + optional extra, not a rewrite.
