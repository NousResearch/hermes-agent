# NJ Statutes post-wave closeout dispatch generation proposal

## Goal

Append a mandatory post-wave closeout step for NJ Statutes bounded waves: after a wave is otherwise complete, the worker must audit control-plane issues, handle any issues through a formal proposal/review/implementation loop, identify the next executable wave from the ledger/contract, and write a full local dispatch markdown file suitable for Benjamin to hand back to Galt in a fresh Discord thread.

## Current implementation evidence

- `hermes_cli/control_worker.py::build_agent_prompt` is the central prompt constructor for statute-worker agent dispatches. It currently injects generic control-result rules and the dispatch payload JSON.
- `hermes_cli/control_worker.py::run_agent_dispatch` trusts a successful `CONTROL_RESULT_JSON` from the worker: if status is `completed` / `completed_with_warnings`, it records artifacts and marks the child dispatch `completed`.
- `hermes_cli/control_contracts.py::validate_statute_dispatch_v1` preserves arbitrary `constraints` fields and validates strict safety constraints. Existing wave payloads include `constraints.wave`, `constraints.sprint_ids`, `constraints.hard_stop_before`, `constraints.waveReviewBaseSha`, and CI metadata.
- `hermes_cli/statutepm_flow.py` creates one child worker payload via `make_child_payload(...)`; it does not synthesize sprint-specific post-wave instructions.
- Tests already cover worker prompt creation and agent runner behavior through `tests/hermes_cli/test_control_worker.py`.

## Root cause / gap

The execution contract for a bounded wave currently depends on the human-authored dispatch payload. There is no durable invariant that every bounded wave closes by:

1. auditing control-plane issues,
2. forcing a formal issue-handling loop if issues exist,
3. identifying the next wave from the ledger/contract after closeout, and
4. producing a new full dispatch markdown prompt for the next thread.

A prompt-only instruction is insufficient: the worker could return success without the next-dispatch artifact and the control plane would accept it.

## Final implementation plan

### 1. Define bounded-wave detection

Modify `hermes_cli/control_worker.py`:

- Add `_is_bounded_wave_payload(payload: dict[str, Any]) -> bool`.
- Return true when:
  - `constraints.wave` is truthy, or
  - `constraints.sprint_ids` is a list with more than one sprint.
- Return false for no wave fields or a single-sprint `sprint_ids` list.

This avoids imposing post-wave work on ordinary single-sprint dispatches while still catching the current F.1-F.5 payload (`constraints.wave: "F.1-F.5"`).

### 2. Add prompt-level wave closeout instructions

Add `_post_wave_closeout_instructions(payload: dict[str, Any]) -> str` and append it in `build_agent_prompt(...)` only for bounded waves.

The instruction block must require:

- Execute the dispatched wave first; do not start the next wave from inside the current wave.
- After the wave is otherwise complete, inspect control-plane state for issues: parent/child dispatch rows, result rows, messages to `default`, open blockers, agent-run artifacts, handoff files, and relevant logs.
- If issues exist, follow this loop: research -> diagnose -> plan -> write executable proposal -> looped oppositional review scoped only to the issue/proposal -> finalize proposal -> implement proposal -> review implementation -> fix any implementation errors found.
- Identify the next wave using the executable ledger and contract, starting from the ready sprint, specifically:

```bash
/Users/johngalt/.hermes/hermes-agent/venv/bin/python /Users/johngalt/.hermes/profiles/nj-statutes-pm/scripts/autonomous_contract.py ready --db .contract-ledger/state.sqlite
```

- Determine the wave boundary from the executable contract/ledger: include the ready sprint plus sequential dependent sprints that belong in the same bounded wave, respect stop conditions/gates/parallelSafety/scope, and hard-stop before the following wave.
- Do not infer the next wave from the just-completed `sprint_ids` list.
- Do not stop at a one-sprint prompt unless the next executable wave is genuinely one sprint.
- Write the full next-wave dispatch markdown file under `docs/dispatches/`.
- Include that dispatch markdown file as a `control_result_v1.artifacts[]` entry.
- If the next dispatch file cannot be written because there is no next ready wave, permissions are insufficient, or the state is blocked, return `action_required` with blocker kind `next_dispatch_prompt_missing`.

### 3. Enforce the closeout artifact in control-plane code

Add `_enforce_bounded_wave_postcondition(item: DispatchWorkItem, result: dict[str, Any]) -> dict[str, Any]`.

When `result.status` is successful and the payload is a bounded wave:

- Require `task_permissions` to include `write`.
- Require at least one successful artifact path that:
  - is a `.md` file,
  - exists on disk,
  - is under `repo_root/docs/dispatches/`, and
  - is within one of the dispatch `allowed_paths`.
- If missing, convert the result to:
  - `schema: control_result_v1`
  - `status: action_required`
  - `summary: bounded wave completed without required next-dispatch markdown artifact`
  - original artifacts/tests preserved
  - blocker kind `next_dispatch_prompt_missing` with precise reason metadata.

This makes the step mandatory without trying to parse the markdown content or mutate the live NJ control DB.

### 4. Regression tests

Modify `tests/hermes_cli/test_control_worker.py`:

- Import `build_agent_prompt`.
- Add prompt-gating tests:
  - bounded wave includes closeout block, ledger command, `docs/dispatches/`, issue loop, and `next_dispatch_prompt_missing`.
  - non-wave and single-sprint payloads do not include the closeout block.
- Add enforcement tests around `run_agent_dispatch()` fake runners:
  - bounded wave + successful result + no dispatch artifact -> action_required / child dispatch blocked / blocker kind `next_dispatch_prompt_missing`.
  - bounded wave + successful result + valid `docs/dispatches/*.md` artifact -> completed.
  - bounded wave + successful result + valid file but no write permission -> action_required.
  - non-wave + successful result + no dispatch artifact -> completed unchanged.
  - bounded wave + worker returns `action_required` -> preserved; enforcement does not overwrite original blocker.

### 5. Verification

Run:

```bash
./venv/bin/python -m pytest -o addopts='' tests/hermes_cli/test_control_worker.py -q
./venv/bin/python -m pytest -o addopts='' tests/hermes_cli/test_control_db.py tests/hermes_cli/test_control_worker.py tests/hermes_cli/test_statutepm_flow.py tests/hermes_cli/test_control_smoke.py tests/hermes_cli/test_control_cli.py -q
./venv/bin/python -m ruff check hermes_cli/control_worker.py tests/hermes_cli/test_control_worker.py
```

## Non-goals

- Do not dispatch or run a new NJ Statutes wave now.
- Do not mutate the live NJ Statutes control-plane DB.
- Do not commit or push.
- Do not generate the next-wave dispatch file in this Hermes-agent implementation task; future completed bounded NJ waves must generate it.

## Oppositional review disposition

The first oppositional review blocked the prompt-only plan. This final plan incorporates the required enforcement layer, permission/allowed-path checks, precise bounded-wave detection, concrete ledger command, and behavior-level tests.
