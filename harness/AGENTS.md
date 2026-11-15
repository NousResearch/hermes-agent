# harness/ — Dynamic Agent Harness (execution governance)

Applies on top of the root `AGENTS.md` (narrow waist, facade + siblings,
behavior contracts, real-path tests with temp `HERMES_HOME`).

## What lives here

The harness controls execution around the Hermes agent loop; it never
replaces the loop. One harness iteration maps to one agent turn
(`AIAgent.chat`), so prompt caching and role alternation are preserved.

- `state.py` — Task / FeatureState / ExecutionState / budgets / lock. Pure data.
- `store.py` — persistence via `SessionDB` meta KV (`harness:` prefix). No new
  files, no schema change. Never `Path.home()/.hermes`; the store takes a
  `SessionDB` (profile-aware, temp-home testable).
- `loop.py` — `HarnessRunner`: create / step / run / pause / cancel / resume /
  status plus every LOOP gate. Owns all state mutation.
- `verify.py` — evidence hierarchy (tests > typecheck > runtime > static >
  config > claim) + real check runners (`pytest_check`, `command_check`).
- `recovery.py` — failure classification + strategy table + identical-failure rule.
- `budget.py` — multi-dimensional hard-limit counters.
- `knowledge.py` — durability gate for observations → knowledge.
- `adapter.py` — thin `chat_step(agent)` bridge. No model imports at module level.

## Rules

- No imports from `run_agent`, `model_tools`, or `tools/` at module level in
  `harness/` (keeps unit tests key-free and avoids import cycles; late-import
  inside the adapter factory only).
- No new model tools, hooks, env vars, or dependencies (stdlib only).
- No `if/elif` ladders on names/kinds — tables (`_FIRST_SEEN`, strength order).
- Outcomes are runtime-owned: agent hints (`StepStatus`) are advisory; gates decide.
- Tests live in `tests/harness/`, assert contracts, and use a real `SessionDB`
  on `tmp_path` — never mocks of the store, never source reading.
