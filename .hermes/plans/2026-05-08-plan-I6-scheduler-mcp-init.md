# Plan I6: Scheduler MCP Init Stub — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Document and plan the verified scheduler MCP initialization feature (cron/scheduler.py lines 1373-1385) and its hermetic test stub strategy, ensuring future maintainers can reproduce, extend, or port the pattern.

**Architecture:** The cron scheduler calls `discover_mcp_tools()` before constructing `AIAgent()` so that MCP-registered tools are available in the agent's tool registry at job runtime. A try/except wrapper makes MCP failures non-fatal. Hermetic CI tests stub `resolve_runtime_provider` (via `_stub_runtime_provider` autouse fixture copied from test_scheduler.py:1871) so the code path is actually reached without real provider credentials.

**Tech Stack:** Python 3.13, pytest, unittest.mock.patch, hermes-agent cron/scheduler.py

---

## Verified State (Do Not Re-implement)

The following is already correct and verified (parent task t_2845b333, run 897). Treat as ground truth.

### Implementation: cron/scheduler.py lines 1373-1385

```python
# Initialize MCP servers so configured mcp_servers are available to
# the agent's tool registry before AIAgent is constructed. Without
# this, cron jobs never saw any MCP tools — only the gateway / CLI
# paths called discover_mcp_tools() at startup. Idempotent: subsequent
# ticks short-circuit on already-connected servers inside
# register_mcp_servers(). Non-fatal on failure: a broken MCP server
# shouldn't kill an otherwise-working cron job. See #4219.
try:
    from tools.mcp_tool import discover_mcp_tools
    _mcp_tools = discover_mcp_tools()
    if _mcp_tools:
        logger.info(
            "Job '%s': %d MCP tool(s) available",
            job_id, len(_mcp_tools),
        )
except Exception as _mcp_exc:
    logger.warning(
        "Job '%s': MCP initialization failed (non-fatal): %s",
        job_id, _mcp_exc,
    )
```

### Test Stub: tests/cron/test_scheduler_mcp_init.py

The test file contains an **autouse fixture** (pattern copied from `tests/cron/test_scheduler.py` line 1871) that stubs `hermes_cli.runtime_provider.resolve_runtime_provider` so the MCP discovery block is reached in hermetic CI:

```python
@pytest.fixture(autouse=True)
def _stub_runtime_provider(self):
    """Stub resolve_runtime_provider for MCP-init tests.
    run_job resolves the runtime provider BEFORE constructing AIAgent,
    so these tests must mock resolve_runtime_provider in addition to
    AIAgent — otherwise in a hermetic CI env (no API keys), the resolver
    raises and the test fails before the patched AIAgent is ever reached.
    """
    fake_runtime = {
        "provider": "openrouter",
        "api_mode": "chat_completions",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": "test-key",
        "source": "stub",
        "requested_provider": None,
    }
    with patch(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        return_value=fake_runtime,
    ):
        yield
```

### Verified Behaviors (3 tests pass)

1. **discover_mcp_tools called before AIAgent construction** — `test_run_job_calls_discover_mcp_tools_before_agent_construction`
2. **discover_mcp_tools failure tolerated (non-fatal)** — `test_run_job_tolerates_discover_mcp_tools_failure`
3. **no_agent jobs skip MCP init entirely** — `test_no_agent_cron_job_does_not_initialize_mcp`

---

## Task 1: Save This Plan to the Project Plans Directory

**Objective:** Persist the plan so downstream workers and humans can find it.

**Files:**
- Create: `.hermes/plans/2026-05-08-plan-I6-scheduler-mcp-init.md`

**Step 1: Copy plan to project**

```bash
cp /home/d/.hermes/kanban/workspaces/t_b41578ba/PLAN-I6-scheduler-mcp-init.md \
   /home/d/Desktop/agenda/hermes-agent/.hermes/plans/2026-05-08-plan-I6-scheduler-mcp-init.md
```

**Step 2: Verify copy**

```bash
md5sum /home/d/.hermes/kanban/workspaces/t_b41578ba/PLAN-I6-scheduler-mcp-init.md \
       /home/d/Desktop/agenda/hermes-agent/.hermes/plans/2026-05-08-plan-I6-scheduler-mcp-init.md
```

Expected: both hashes match.

**Step 3: Stage (do not commit)**

```bash
cd /home/d/Desktop/agenda/hermes-agent
git add .hermes/plans/2026-05-08-plan-I6-scheduler-mcp-init.md
```

---

## Task 2: Verify Tests Still Pass

**Objective:** Confirm the verified state is still valid before declaring the plan complete.

**Files:**
- Read-only: `tests/cron/test_scheduler_mcp_init.py`

**Step 1: Run the 3 tests**

```bash
cd /home/d/Desktop/agenda/hermes-agent
source .venv/bin/activate
pytest tests/cron/test_scheduler_mcp_init.py -v --tb=short
```

**Expected output:**
```
tests/cron/test_scheduler_mcp_init.py::TestSchedulerMCPInit::test_run_job_calls_discover_mcp_tools_before_agent_construction PASSED
tests/cron/test_scheduler_mcp_init.py::TestSchedulerMCPInit::test_run_job_tolerates_discover_mcp_tools_failure PASSED
tests/cron/test_scheduler_mcp_init.py::TestSchedulerMCPInit::test_no_agent_cron_job_does_not_initialize_mcp PASSED

3 passed in X.XXs
```

**Step 2: If any test fails**

Do not proceed. The verified state has drifted. Run `git diff` to check if scheduler.py or the test file changed since parent task t_2845b333. Report findings in a kanban_comment and block the task.

---

## Task 3: (Optional) Extend Test Coverage for Additional Edge Cases

**Objective:** If requested by a downstream worker or human, add tests for edge cases not yet covered.

**Potential additions (YAGNI — only implement if explicitly asked):**

- **Empty MCP tool list:** `discover_mcp_tools()` returns `[]` — verify no crash, AIAgent still constructed.
- **Multiple MCP servers:** Verify `len(_mcp_tools)` logged correctly when >1 tools returned.
- **MCP init timing:** Ensure `discover_mcp_tools()` is called exactly once per job tick (idempotency via `register_mcp_servers()` internal cache).
- **Custom provider runtime:** Verify MCP init works with non-openrouter provider stubs.

**Files:**
- Modify: `tests/cron/test_scheduler_mcp_init.py`

---

## Notes for Downstream Implementers

### Why the `_stub_runtime_provider` fixture is necessary

In a hermetic CI environment (no API keys, no provider config), `resolve_runtime_provider()` raises `AuthError` before the MCP discovery block is reached. Without the stub, the test patches for `discover_mcp_tools` and `AIAgent` are never triggered because `run_job()` exits early via the `except Exception as e:` handler at line ~1554.

### Why `no_agent` test doesn't need the stub

`run_job()` returns early at line 1023 when `job["no_agent"] == True`, entirely bypassing provider resolution and MCP discovery. The test only asserts `discover_mcp_tools` is *not* called, which holds trivially.

### Portability of the stub pattern

Any future cron test that patches `AIAgent` or code after provider resolution in `run_job()` MUST also stub `resolve_runtime_provider`. Copy the `_stub_runtime_provider` fixture exactly (only the class name in the docstring needs to change). The fixture is intentionally an `autouse` so it cannot be forgotten.

---

## Files Referenced

| File | Role | Verified Lines |
|---|---|---|
| `cron/scheduler.py` | Implementation | 1373-1385 |
| `tests/cron/test_scheduler_mcp_init.py` | Test file | ~140 lines total |
| `tests/cron/test_scheduler.py` | Pattern source | 1871-1892 (`_stub_runtime_provider`) |
| `hermes_cli/runtime_provider.py` | Stub target | `resolve_runtime_provider()` |
| `tools/mcp_tool.py` | MCP discovery | `discover_mcp_tools()` at ~3166 |
