# Delegation auth + context continuity plan

> For Hermes: use subagent-driven-development and double-TDD-parallel-suites. Do not merge without both spec and invariants green.

Goal: fix real subagent delegation end-to-end and add architecture so model switches preserve conversation continuity under a dedicated context owner/controller.

Architecture:
- Part A: restore delegate_task test baseline, then fix child credential refresh so subagents do not inherit stale auth.
- Part B: introduce a context-controller layer that owns context/compression continuity across model/provider switches without resetting task state.

Tech stack:
- Python
- pytest
- existing AIAgent / ContextCompressor / ContextEngine abstractions

---

## Evidence already verified

1. Current routing-v2 tests are green: 213 passed.
2. delegate_task still has unresolved auth risk.
3. `tests/tools/test_delegate.py` is currently red for a more basic reason:
   - ImportError: `build_environment_hints` missing from `agent.prompt_builder`
   - command verified:
     `/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/tools/test_delegate.py -q -o 'addopts='`
   - result verified: 28 failed, 39 passed
4. Context continuity today is partial but not fully owned by one component:
   - run_agent keeps messages in-memory across model switches
   - context_compressor survives via `update_model(...)`
   - responsibility is scattered; tool schema registration and runtime snapshots are brittle

---

## Phase 0: restore test baseline for delegate path

### Task 0.1: reproduce delegate suite failure exactly
Objective: keep a recorded RED baseline for delegate tests.

Files:
- Test: `tests/tools/test_delegate.py`
- Likely implementation: `agent/prompt_builder.py`, `run_agent.py`

Step 1: Run failing suite
Run:
`cd /home/jh/.hermes/worktrees/routing-lab && /home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/tools/test_delegate.py -q -o 'addopts='`
Expected: FAIL with `ImportError: cannot import name 'build_environment_hints'`

Step 2: Save exact failure summary in commit/notes
Expected: 28 failed, 39 passed (current known baseline)

---

### Task 0.2: restore `build_environment_hints` contract
Objective: make delegate tests importable again before touching auth logic.

Files:
- Modify: `agent/prompt_builder.py`
- Verify: `tests/agent/test_prompt_builder.py`
- Verify: `tests/tools/test_delegate.py`

Step 1: Write/confirm failing prompt-builder tests first
Run:
`cd /home/jh/.hermes/worktrees/routing-lab && /home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/agent/test_prompt_builder.py -q -o 'addopts='`
Expected: RED if function is missing/broken

Step 2: Implement minimal `build_environment_hints()` matching tests
Requirements inferred from tests:
- WSL-aware hint when on WSL
- empty/no-WSL-safe behavior otherwise
- exported from `agent.prompt_builder`

Step 3: Verify prompt-builder tests
Run:
`/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/agent/test_prompt_builder.py -q -o 'addopts='`
Expected: PASS

Step 4: Re-run delegate suite baseline
Run:
`/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/tools/test_delegate.py -q -o 'addopts='`
Expected: import error gone; remaining failures now reflect actual delegate logic/auth behavior

---

## Phase 1: strict auth fix for delegate_task

### Root cause hypothesis already grounded
- Child agents can inherit stale runtime credentials from parent in-memory state.
- If device-flow reauth updates auth.json/credential pool after parent instantiation, the child may still use stale `parent.api_key` instead of fresh pool/runtime credentials.

### Task 1.1: write RED spec tests for stale-parent-credential scenario
Objective: lock the auth bug with exact tests before implementation.

Files:
- Modify/create: `tests/tools/test_delegate.py`
- Maybe create: `tests/tools/test_delegate_auth_refresh_spec.py`

Spec cases:
1. parent has stale `api_key`, no in-memory pool, load_pool(provider) returns fresh pool -> child gets pool
2. parent has stale `api_key`, provider requires runtime resolution -> child uses resolved runtime credentials
3. when shared provider pool exists on disk but not in parent memory, child must not inherit stale parent key
4. logging/trace states which credential source was used

Step 1: write failing tests
Step 2: run targeted tests
Run:
`/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/tools/test_delegate.py -q -o 'addopts=' -k 'credential or auth or delegation'`
Expected: RED on new tests

---

### Task 1.2: write invariants suite for credential refresh behavior
Objective: ensure fix is robust, not just one happy path.

Files:
- Create: `tests/tools/test_delegate_auth_refresh_invariants.py`

Laws:
- if parent pool missing and disk pool available, child pool is non-null
- no provider mismatch when inheriting same provider
- explicit override still wins over inherited/pool creds
- child never uses blank api_key when resolvable runtime exists
- lease/release symmetry preserved
- deterministic source precedence: override > disk/runtime refresh > inherited parent key

Run:
`/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/tools/test_delegate_auth_refresh_invariants.py -q -o 'addopts='`
Expected: RED first

---

### Task 1.3: minimal implementation for child credential refresh
Objective: fix root cause with smallest safe patch.

Files:
- Modify: `tools/delegate_tool.py`
- Possibly read-only support from: `agent/credential_pool.py`, `hermes_cli/runtime_provider.py`

Implementation intent:
- in `_build_child_agent(...)`, before final child creation:
  - if `parent._credential_pool` is missing or `parent_api_key` is missing/stale-looking, try `load_pool(effective_provider)`
  - if provider/runtime resolution can return fresh credentials, use resolved runtime instead of stale inherited key
- preserve existing precedence:
  1. explicit override credentials
  2. refreshed disk pool/runtime credentials
  3. inherited parent credentials
- add lightweight debug logging for credential source selection

Verification:
- targeted RED tests go green
- full `tests/tools/test_delegate.py` green

---

### Task 1.4: real E2E subagent verification
Objective: verify actual delegate_task behavior, not only unit tests.

Files:
- No production change required
- optional transient diagnostics only

Steps:
1. ensure auth source is valid
2. run minimal delegate_task call
3. verify child returns summary successfully
4. if auth still fails, capture exact error/log path and only then iterate

Success criteria:
- one real delegate_task invocation succeeds end-to-end
- no AuthenticateToken failure

---

## Phase 2: context continuity controller for model switches

### Requirements from user
- if conversation changes model, new model must not start from zero
- context continuity must survive model/provider switch
- one component/agent must own context + compression

### Task 2.1: RED spec suite for context continuity
Objective: express exact continuity contract first.

Files:
- Create: `tests/agent/test_context_controller_spec.py`

Spec cases:
1. switch_model keeps prior conversation messages visible
2. switch_model keeps prior compression summary visible
3. fallback restore keeps same active task continuity
4. switching provider invalidates system prompt cache but does not erase message continuity
5. controller can return current context snapshot before and after switch, with same semantic content
6. controller can re-register engine tools if runtime switch requires it

Run:
`/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/agent/test_context_controller_spec.py -q -o 'addopts='`
Expected: RED (module missing)

---

### Task 2.2: RED invariants suite for context continuity
Objective: verify laws over many runtime switch combinations.

Files:
- Create: `tests/agent/test_context_controller_invariants.py`

Laws:
- context snapshot not empty after runtime switch if it was non-empty before
- previous summary preserved unless explicit reset
- reset clears controller state, switch does not
- tool schema refresh idempotent
- repeated switch A->B->A preserves monotonic message history length
- controller output deterministic under same inputs

Run:
`/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/agent/test_context_controller_invariants.py -q -o 'addopts='`
Expected: RED first

---

### Task 2.3: minimal controller implementation
Objective: introduce thin wrapper, not massive rewrite.

Files:
- Create: `agent/context_controller.py`
- Modify: `run_agent.py`

Minimal API:
- `on_session_start(...)`
- `on_session_reset()`
- `on_session_end(messages)`
- `get_tool_schemas()`
- `get_messages_for_model(...)`
- `update_from_response(...)`
- `switch_runtime(...)`
- optional `persist_state()` / `load_state()`

Rules:
- controller wraps existing `ContextEngine` / `ContextCompressor`
- model switch must call `switch_runtime(...)`, not reset
- summary/compression state must remain owned by controller
- run_agent should stop directly poking compressor internals where practical

Verification:
- spec suite green
- invariants suite green
- no regression in existing context/fallback tests

---

### Task 2.4: integration verification
Objective: prove model switch continuity end-to-end.

Files:
- existing tests touching fallback/compression
- maybe add integration test under `tests/cli/` or `tests/integration/`

Checks:
1. model switch in session keeps semantic context
2. fallback -> restore keeps summary/task continuity
3. compression still triggers safely on smaller context models
4. tool schemas remain valid after switch

---

## Quality gates

Do not proceed to next phase unless previous gate is green.

Gate A:
- `tests/agent/test_prompt_builder.py` green
- `tests/tools/test_delegate.py` importable

Gate B:
- delegate auth spec + invariants green
- full `tests/tools/test_delegate.py` green
- real delegate_task E2E passes

Gate C:
- context-controller spec + invariants green
- existing compression/fallback tests green
- no routing-v2 regressions

Final command set:
- `/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/tools/test_delegate.py -q -o 'addopts='`
- `/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/agent/test_context_controller_spec.py tests/agent/test_context_controller_invariants.py -q -o 'addopts='`
- `/home/jh/.hermes/hermes-agent/venv/bin/python -m pytest tests/routing/v2 -q -o 'addopts='`

---

## Suggested commit structure

1. `fix: restore build_environment_hints contract for delegate imports`
2. `test: add delegate auth refresh spec and invariants`
3. `fix: refresh child delegation credentials from pool/runtime`
4. `feat: add context controller for model-switch continuity`
5. `test: add context-controller spec and invariants`

---

## Next recommended execution step

Start with Phase 0 Task 0.2.

Reason:
- current delegate suite is red for a more basic import issue
- fixing auth before restoring that baseline would mix two bugs and break strict debugging discipline
