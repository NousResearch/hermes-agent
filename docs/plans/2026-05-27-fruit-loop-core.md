# Fruit-Loop Core Implementation Plan

## Goal

Make `/loop` a Hermes-core project controller that works from CLI and gateway-backed UIs, with simple file-backed state and one-story-at-a-time execution.

## Current truth

Branch/worktree:

- Worktree: `/Users/codyclawford/projects/hermes-loop-core`
- Branch: `feature/loop-core`
- Current commits on the lane:
  - `826f95415 feat: add core loop command state`
  - `746932a55 feat: support loop gateway dispatch and close`
  - `10cc77633 feat: add loop story outcomes`

Implemented:

- `hermes_cli/loops.py` owns deterministic file-backed loop state under `.hermes/loops/<slug>/`.
- `/loop init`, `/loop status`, `/loop run`, `/loop complete`, `/loop block`, and `/loop close` are supported by the core helper.
- CLI dispatch calls the core helper.
- Gateway dispatch calls the core helper.
- `/loop run` selects the first pending story, marks it running, writes derived status, and returns the one-story execution prompt.
- `/loop close` updates state and archives `prd.json`, `progress.md`, and `status.md`.
- Story outcome helpers update PRD state, append progress, and refresh status.

Latest slice:

- Gateway `/loop run <slug>` queues the generated story execution prompt as the next normal `MessageEvent` when adapter/session state is available, and falls back to returning the prompt when it is not.
- CLI `/loop run <slug>` queues the generated story execution prompt on `_pending_input` when available, and falls back to printing the prompt when it is not.

Latest hardening slice:

- `/loop` command metadata and usage now include `complete` and `block`.
- TUI pending-input routing includes `loop`, so `/loop run` does not get stranded in the slash-worker subprocess.
- Closed loops now refuse `run`, `complete`, and `block` mutations and render as `Status: closed`.
- Close archives use microsecond timestamps to avoid same-second archive collisions.

Known gap:

- The core state model still uses the minimal `prd.json` shape from the first slice. It should eventually converge with the richer PRD/story manifest shape from the project-loop skill, but not before the V1 command spine is reviewed.

## Non-negotiables

- Keep `/loop` in Hermes core, not Herm2/OpenTUI-specific code.
- Keep UX simple: `/loop init|run|status|complete|block|close <slug>`.
- Keep Kanban/workers backstage and later; do not add a board or project-OS surface now.
- Keep `hermes_cli/loops.py` deterministic and file-backed. Runtime dispatch can sit in CLI/gateway wrappers.
- Stop before destructive changes, credential requirements, push/merge, or broad refactors outside the `/loop` lane.
- Do not touch `/Users/codyclawford/.hermes/hermes-agent` from this lane except reading reference docs/skills.

## Next slice

Prepare the branch for PR/review:

1. Inspect the full branch diff against `origin/main` as a single product slice.
2. Decide whether to keep the minimal `prd.json` shape for the first PR, or migrate toward `prd.md` + `stories.json` before review.
3. If keeping the minimal shape for this PR, add a short docs note/command help example so users know how to seed `userStories`.
4. Run the focused verification suite and a broader command/TUI smoke if the diff still touches `tui_gateway/server.py`.

Completed hardening slice:

1. Command metadata:
   - Include `complete` and `block` in `/loop` args/subcommands and usage output.
2. TUI routing:
   - Add `loop` to `_PENDING_INPUT_COMMANDS` so slash.exec rejects it and the TUI routes through command.dispatch.
3. Closed-state guard:
   - Refuse run/complete/block mutation after `/loop close`.
4. Tests:
   - Add guards for TUI pending-input routing, closed-loop mutation, and usage string.

Completed slice — safe execution handoff for `/loop run`:

1. Gateway:
   - When `/loop run <slug>` succeeds and returns a story execution prompt, enqueue that prompt as the next normal `MessageEvent` for the same session using the existing FIFO queue helper.
   - Return a short acknowledgement instead of dumping the full prompt back to the user.
   - If adapter/session state is unavailable, fall back to returning the prompt so no work is silently lost.

2. CLI:
   - When `/loop run <slug>` succeeds and returns a story execution prompt, put that prompt on `_pending_input` so the current CLI session executes it as the next turn.
   - Print a short acknowledgement instead of dumping the full prompt.
   - If `_pending_input` is unavailable, fall back to printing the prompt.

3. Tests:
   - Add gateway test proving `/loop run` enqueues a normal text event with the story prompt and returns a queued acknowledgement.
   - Add/adjust CLI-level test if an existing low-friction command-dispatch test harness exists; otherwise keep the CLI change small and cover shared behavior through gateway/core tests.

## Verification

Run before committing:

```bash
TERM=xterm-256color uv run python -m pytest tests/gateway/test_loop_command.py tests/hermes_cli/test_loops.py tests/hermes_cli/test_commands.py -o 'addopts=' -q
TERM=xterm-256color uv run ruff check hermes_cli/loops.py cli.py gateway/run.py tests/hermes_cli/test_loops.py tests/gateway/test_loop_command.py
TERM=xterm-256color uv run python -m py_compile hermes_cli/loops.py cli.py gateway/run.py
git diff --check
git status --short --branch
```

Expected result:

- Focused tests pass.
- Ruff passes.
- Py compile passes.
- No whitespace errors.
- Worktree contains only this lane's intended files before commit.

## Stop conditions

Stop and report if:

- Existing gateway queue semantics require a broader adapter/session refactor.
- Tests reveal `/loop run` would interrupt an active agent instead of queueing safely.
- Command dispatch changes collide with unrelated dirty files or another active lane.
- The implementation needs credentials, external services, push/merge, or destructive repo operations.

## Default decision if evidence is weak

Prefer safe handoff over automatic execution. If direct submission risks interrupting an active run, enqueue through the existing FIFO path or fall back to returning the prompt with clear text. Do not add Kanban/worker routing in this slice.
