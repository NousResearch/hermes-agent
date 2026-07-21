# Preemptive Task Cancellation Implementation Plan

> **For Hermes:** Implement task-by-task with TDD. Feature flag default-off.

**Goal:** Enable out-of-band cancellation of running agent tasks via Discord STOP commands.

**Architecture:** Add a JobManager that assigns unique job_ids to each agent turn, tracks state machine (queued -> running -> cancel_requested -> cancelling -> cancelled), maintains per-job CancellationToken, and propagates cancellation to LLM streaming, HTTP, sleep, retry loops, and shell child processes. The Discord command listener processes STOP commands independently of the agent execution loop.

**Tech Stack:** Python 3.11, asyncio, psutil (process tree kill), config.yaml feature flag

---

## Background: Current Architecture

### Existing Interrupt Mechanism
- `AIAgent._interrupt_requested` (bool) - set by `interrupt()` method
- Checked at iteration boundaries in `agent/conversation_loop.py` (5 places) and `agent/tool_executor.py` (6 places)
- Gateway `_handle_stop_command()` calls `agent.interrupt("Stop requested")`
- `_set_interrupt(thread_id)` - per-thread tool interrupt signal
- **Limitation:** cooperative only - can't interrupt mid-tool-call

### Key Files
- `run_agent.py` - AIAgent class, `interrupt()`, `_interrupt_requested`
- `agent/conversation_loop.py` - main conversation loop, checks `_interrupt_requested`
- `agent/tool_executor.py` - tool dispatch, checks `_interrupt_requested`
- `gateway/run.py` - gateway, `_handle_stop_command()`, `_interrupt_running_agents()`
- `plugins/platforms/discord/adapter.py` - Discord adapter, `/stop` slash command
- `agent/chat_completion_helpers.py` - `interruptible_api_call()`, `interruptible_streaming_api_call()`

---

## Task 1: CancellationToken + JobState + JobManager

Create `agent/cancellation.py` with:
- `JobState` enum: QUEUED, RUNNING, CANCEL_REQUESTED, CANCELLING, CANCELLED
- `CancellationToken`: thread-safe, `is_cancelled`, `request_cancel()`, `throw_if_cancelled()`, `register()`, `sleep()`
- `JobManager`: `create_job()`, `get_token()`, `get_state()`, `cancel_job()`, `cancel_all()`, `unregister_job()`
- `CancellationResult`: job_id, state, last_completed_step, cancelled_step, remaining_processes
- `get_job_manager()` singleton
- `is_preemptive_cancellation_enabled()` feature flag check

## Task 2: Integrate into AIAgent

Modify `run_agent.py`:
- Import JobManager
- In `run_conversation()`: create job_id, register in JobManager
- After conversation: unregister job
- In `interrupt()`: if preemptive cancellation enabled, also call `job_manager.cancel_job()`
- Store `job_id` on the agent instance for STOP <job_id> targeting

## Task 3: Propagate CancellationToken to tools

Modify `agent/tool_executor.py`:
- Before each tool call: check `token.is_cancelled` and skip if cancelled
- After each tool call: check again
- Pass token to tool handlers that support it (terminal, file, HTTP)
- Track current step for cancellation metadata

Modify `agent/conversation_loop.py`:
- Check token at loop boundaries (before/after API call, before/after tool batch)
- If token is cancelled, break out of loop and return cancelled result

## Task 4: LLM streaming cancellation

Modify `agent/chat_completion_helpers.py`:
- In `interruptible_streaming_api_call()`: check CancellationToken in the streaming loop
- If cancelled: abort the HTTP request (close connection)
- Distinguish cancelled (user request) from failed (network error)

## Task 5: Shell command process tree termination

Create `agent/process_killer.py`:
- `kill_process_tree(pid)`: kill process and all children using psutil
- `graceful_then_force_kill(pid, grace_seconds=5)`: try SIGTERM, wait 5s, then SIGKILL

Modify `tools/terminal_tool.py` (or equivalent):
- Track spawned process PIDs per job_id
- On cancellation: call `kill_process_tree()` for all tracked PIDs
- 5 second graceful -> hard-stop

## Task 6: Safe cancellation gates

Create `agent/cancellation_gates.py`:
- `guard_file_write(token, path)`: raise if cancelled
- `guard_commit(token)`: raise if cancelled  
- `guard_push(token)`: raise if cancelled
- `guard_pr(token)`: raise if cancelled
- `guard_external_send(token)`: raise if cancelled
- `guard_db_migration(token)`: check safe point + rollback/postcondition
- `guard_deploy(token)`: check safe point

Integrate gates into:
- `tools/` tool implementations (file, terminal, etc.)
- Before each external side-effect operation

## Task 7: Gateway out-of-band STOP

Modify `gateway/run.py`:
- Parse `STOP <job_id>` and `STOP ALL` from messages
- When STOP received: immediately call `job_manager.cancel_job(job_id)` (out-of-band)
- Don't wait for current tool to finish
- Return cancellation result to user

Modify `plugins/platforms/discord/adapter.py`:
- Add STOP <job_id> and STOP ALL as recognized commands
- Process independently of agent execution loop

## Task 8: Cancelled vs Failed

Modify `run_agent.py` and `agent/conversation_loop.py`:
- Return dict with `status: 'completed' | 'cancelled' | 'failed'`
- When interrupted by user: status = 'cancelled'
- When failed by error: status = 'failed'
- Include cancellation metadata: job_id, last_completed_step, cancelled_step, remaining_processes

## Task 9: Feature flag

Add to `hermes_cli/config.py` DEFAULT_CONFIG:
- `agent.preemptive_cancellation: false`

In `agent/cancellation.py`:
- `is_preemptive_cancellation_enabled()` reads from config + env var
- When off: existing `_interrupt_requested` behavior preserved
- When on: new preemptive cancellation used

## Task 10: Tests

- `tests/agent/test_cancellation.py`: CancellationToken, JobManager unit tests
- `tests/agent/test_process_killer.py`: process tree kill tests
- `tests/agent/test_cancellation_integration.py`: integration with AIAgent loop
- `tests/gateway/test_stop_command.py`: STOP <job_id>, STOP ALL tests
- `tests/agent/test_cancellation_smoke.py`: smoke test - start long task, cancel mid-execution, verify process tree killed
