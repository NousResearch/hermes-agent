# Zero-Content Truncation Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent empty Ollama length completions from entering Hermes's four-continuation loop, and keep WhatsApp sessions within a reliable context window.

**Architecture:** Add a turn-scoped, one-shot recovery branch to the existing conversation loop: empty `finish_reason="length"` responses compress once and retry, while non-empty partial responses keep the existing continuation path. Bound the live runtime to an explicit 65,536-token Ollama context with 2,048 output tokens and rotate gateway sessions daily or after 24 hours idle.

**Tech Stack:** Python 3.11, pytest, Hermes conversation loop and gateway configuration, YAML, systemd user service, Ollama OpenAI-compatible API.

## Global Constraints

- Delegation, specialist routing, emergency escalation, ordinary tool use, and provider fallback remain unchanged.
- Empty-length recovery performs at most one compression retry per user turn.
- Empty-length recovery never persists a synthetic continuation prompt.
- Non-empty length completions and truncated tool calls retain existing behavior.
- Live runtime settings are exactly `context_length: 65536`, `ollama_num_ctx: 65536`, and `max_tokens: 2048`.
- Session reset mode is `both`, with `at_hour: 4` and `idle_minutes: 1440`.
- Do not expose secrets, household identifiers, or conversation content in commits.
- Restart `hermes-gateway.service` exactly once, after code and configuration validation.

---

### Task 1: Add Empty-Length Recovery Regression Coverage

**Files:**
- Modify: `tests/run_agent/test_run_agent.py`
- Reference: `agent/conversation_loop.py:1758-1962`

**Interfaces:**
- Consumes: `AIAgent.run_conversation(prompt: str, conversation_history: list | None = None) -> dict` and existing `_mock_response` test helper.
- Produces: regression contracts for compression retry, terminal `/new` recovery, partial-content continuation, and tool-call isolation.

- [ ] **Step 1: Replace the obsolete empty-length continuation expectation with a failing compression-recovery test**

Change `test_length_empty_content_without_think_tags_retries_normally` into a test that supplies an empty length response followed by a normal response, stubs `_compress_context` to return coherent compressed messages, and asserts:

```python
assert result["completed"] is True
assert result["api_calls"] == 2
assert result["final_response"] == "Recovered after compression"
mock_compress.assert_called_once()
second_messages = agent.client.chat.completions.create.call_args_list[1].kwargs["messages"]
assert all("previous response was truncated" not in str(m.get("content", "")).lower()
           for m in second_messages)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
venv/bin/pytest -q tests/run_agent/test_run_agent.py::TestRunConversation::test_length_empty_content_compresses_once_then_recovers
```

Expected: FAIL because the current implementation makes four continuation requests and does not call `_compress_context`.

- [ ] **Step 3: Add a failing exhaustion test**

Supply two empty length responses, stub `_compress_context`, and assert:

```python
assert result["completed"] is False
assert result["api_calls"] == 2
assert "/new" in result["final_response"]
assert "truncated by the output length limit" not in str(result["messages"])
mock_compress.assert_called_once()
```

- [ ] **Step 4: Run the exhaustion test and verify RED**

Run:

```bash
venv/bin/pytest -q tests/run_agent/test_run_agent.py::TestRunConversation::test_length_empty_content_after_compression_requests_new_session
```

Expected: FAIL because the current implementation performs four continuation attempts and returns the generic truncation error.

- [ ] **Step 5: Confirm existing control tests before implementation**

Run:

```bash
venv/bin/pytest -q \
  tests/run_agent/test_run_agent.py::TestRunConversation::test_length_finish_reason_requests_continuation \
  tests/run_agent/test_run_agent.py::TestRunConversation::test_length_with_tool_calls_returns_partial_without_executing_tools
```

Expected: PASS, establishing the behaviors the patch must preserve.

### Task 2: Implement One-Shot Compression Recovery

**Files:**
- Modify: `agent/conversation_loop.py:600-620`
- Modify: `agent/conversation_loop.py:1758-1962`
- Test: `tests/run_agent/test_run_agent.py`

**Interfaces:**
- Consumes: `agent._compress_context(messages, system_message, approx_tokens, task_id)`, `conversation_history_after_compression(agent, messages)`, and the existing response normalization fields.
- Produces: a turn-local `empty_length_compression_attempted` guard and terminal actionable recovery result.

- [ ] **Step 1: Add the minimal turn-scoped recovery guard**

Near `length_continue_retries = 0`, add:

```python
empty_length_compression_attempted = False
```

The guard must live outside the inner API retry state so it survives the single compressed retry but resets for the next user turn.

- [ ] **Step 2: Add the empty-length branch before generic continuation handling**

When `finish_reason == "length"`, assistant content is empty after think-block stripping, and there are no tool calls:

```python
if not empty_length_compression_attempted and agent.compression_enabled:
    empty_length_compression_attempted = True
    messages, active_system_prompt = agent._compress_context(
        messages,
        system_message,
        approx_tokens=request_pressure_tokens,
        task_id=effective_task_id,
    )
    conversation_history = conversation_history_after_compression(agent, messages)
    _retry.restart_with_compressed_messages = True
    break
```

If compression is disabled, fails to produce a retry, or the compressed retry is also empty, persist coherent state and return:

```python
recovery_text = (
    "Hermes could not produce a response because this conversation has "
    "exhausted the model context. Send /new to start a clean session."
)
```

Return `completed=False`, `partial=True`, and `error=recovery_text`. Do not append the empty assistant response or a continuation user message.

- [ ] **Step 3: Run both new tests and verify GREEN**

Run:

```bash
venv/bin/pytest -q \
  tests/run_agent/test_run_agent.py::TestRunConversation::test_length_empty_content_compresses_once_then_recovers \
  tests/run_agent/test_run_agent.py::TestRunConversation::test_length_empty_content_after_compression_requests_new_session
```

Expected: 2 passed.

- [ ] **Step 4: Run the focused truncation regression group**

Run:

```bash
venv/bin/pytest -q \
  tests/run_agent/test_run_agent.py -k 'length or truncated_tool_call' \
  tests/run_agent/test_partial_stream_finish_reason.py \
  tests/run_agent/test_anthropic_truncation_continuation.py \
  tests/agent/test_turn_retry_state.py
```

Expected: all selected tests pass with no new warnings.

- [ ] **Step 5: Review the diff for scope and commit**

Run:

```bash
git diff --check
git diff -- agent/conversation_loop.py tests/run_agent/test_run_agent.py
git add agent/conversation_loop.py tests/run_agent/test_run_agent.py
git commit -m "fix: recover empty length completions with compression"
```

Verify the diff does not touch delegation or provider routing.

### Task 3: Validate Session-Rotation Configuration

**Files:**
- Modify: `tests/gateway/test_config.py`
- Reference: `gateway/config.py:358-407`

**Interfaces:**
- Consumes: `SessionResetPolicy.from_dict(data: dict) -> SessionResetPolicy`.
- Produces: a regression contract proving the approved `both` policy round-trips unchanged.

- [ ] **Step 1: Add a configuration contract test**

Add:

```python
def test_both_mode_daily_and_idle_policy(self):
    policy = SessionResetPolicy.from_dict(
        {"mode": "both", "at_hour": 4, "idle_minutes": 1440}
    )
    assert policy.mode == "both"
    assert policy.at_hour == 4
    assert policy.idle_minutes == 1440
```

- [ ] **Step 2: Run the test**

Run:

```bash
venv/bin/pytest -q tests/gateway/test_config.py::TestSessionResetPolicy::test_both_mode_daily_and_idle_policy
```

Expected: PASS because the supported policy already exists; this is a configuration deployment contract, not new production behavior.

- [ ] **Step 3: Run the relevant session policy suite and commit**

Run:

```bash
venv/bin/pytest -q \
  tests/gateway/test_config.py::TestSessionResetPolicy \
  tests/gateway/test_session_reset_notify.py
git diff --check
git add tests/gateway/test_config.py
git commit -m "test: lock daily and idle session rotation policy"
```

Expected: all selected tests pass.

### Task 4: Apply and Validate the Approved Live Configuration

**Files:**
- Modify: `/home/hermes/.hermes/config.yaml`
- Inspect: `/home/hermes/home-ai-orchestrator/docs/hermes-config.yaml`

**Interfaces:**
- Consumes: Hermes YAML configuration loader and `SessionResetPolicy`.
- Produces: live model context/output limits and gateway reset policy.

- [ ] **Step 1: Check the live diff target and operational projection**

Run read-only checks:

```bash
sed -n '1,16p' /home/hermes/.hermes/config.yaml
sed -n '/^session_reset:/,+5p' /home/hermes/.hermes/config.yaml
sed -n '1,16p' /home/hermes/home-ai-orchestrator/docs/hermes-config.yaml
sed -n '/^session_reset:/,+5p' /home/hermes/home-ai-orchestrator/docs/hermes-config.yaml
```

If the operational example already projects 65,536/2,048 and the approved reset policy, leave that repository untouched. Otherwise report the mismatch before expanding into a separate repository change.

- [ ] **Step 2: Patch only the approved live keys**

Apply:

```yaml
model:
  context_length: 65536
  ollama_num_ctx: 65536
  max_tokens: 2048

session_reset:
  mode: both
  idle_minutes: 1440
  at_hour: 4
```

Preserve every unrelated key and all delegation configuration byte-for-byte.

- [ ] **Step 3: Validate YAML and resolved gateway policy before restart**

Run:

```bash
/home/hermes/.hermes/hermes-agent/venv/bin/python - <<'PY'
from pathlib import Path
import yaml
from gateway.config import SessionResetPolicy

cfg = yaml.safe_load(Path("/home/hermes/.hermes/config.yaml").read_text())
assert cfg["model"]["context_length"] == 65536
assert cfg["model"]["ollama_num_ctx"] == 65536
assert cfg["model"]["max_tokens"] == 2048
policy = SessionResetPolicy.from_dict(cfg["session_reset"])
assert (policy.mode, policy.at_hour, policy.idle_minutes) == ("both", 4, 1440)
print("validated live model and session-reset configuration")
PY
```

Expected: exit 0 and the validation message. Do not restart on failure.

### Task 5: Full Verification and Single Restart

**Files:**
- Verify: branch worktree and `/home/hermes/.hermes/config.yaml`
- Service: `hermes-gateway.service`

**Interfaces:**
- Consumes: committed patch, validated live configuration, systemd user service.
- Produces: running gateway with WhatsApp connected and corrected recovery behavior available.

- [ ] **Step 1: Run the full relevant test suite**

Run:

```bash
venv/bin/pytest -q \
  tests/run_agent/test_run_agent.py \
  tests/run_agent/test_partial_stream_finish_reason.py \
  tests/run_agent/test_anthropic_truncation_continuation.py \
  tests/agent/test_turn_retry_state.py \
  tests/gateway/test_config.py \
  tests/gateway/test_session_reset_notify.py
```

Expected: exit 0 with zero failures.

- [ ] **Step 2: Run repository hygiene checks**

Run:

```bash
git diff --check
git status --short
git log --oneline --decorate -4
rg -n 'delegation|specialist|auto_escalation' agent/conversation_loop.py tests/run_agent/test_run_agent.py tests/gateway/test_config.py
```

Expected: clean tracked worktree after commits; the final search shows no newly added delegation changes.

- [ ] **Step 3: Make the tested branch code active without touching main**

Because the user service executes `/home/hermes/.hermes/hermes-agent`, do not overwrite the dirty main checkout. Stop and report the deployment boundary if the service cannot safely execute the tested worktree path without a unit-file edit. A unit-file path change is outside the approved actions and requires separate approval.

If the existing deployment mechanism can activate the tested branch without overwriting unrelated user changes, use it and record the exact reversible operation.

- [ ] **Step 4: Restart the gateway exactly once**

Run:

```bash
systemctl --user restart hermes-gateway.service
```

Do not issue another restart during this deployment.

- [ ] **Step 5: Verify service, WhatsApp, and resolved runtime**

Run:

```bash
systemctl --user is-active hermes-gateway.service
systemctl --user status hermes-gateway.service --no-pager -l
journalctl --user -u hermes-gateway.service --since "5 minutes ago" --no-pager \
  | rg 'WhatsApp|whatsapp|Ollama num_ctx|ERROR|Traceback|failed'
curl -sS --max-time 10 http://10.0.30.56:11434/api/ps \
  | jq '.models[] | select(.name == "qwen3.6:27b") | {name, context_length}'
```

Expected: service is `active`; WhatsApp bridge connects; Hermes logs that it will request 65,536 tokens; no new traceback; after the first model request, Ollama reports a 65,536-token context.

- [ ] **Step 6: Final requirements audit**

Confirm from fresh evidence:

- zero-content length uses one compression retry;
- the second zero-content result instructs `/new`;
- non-empty continuation and truncated tool calls still pass tests;
- live model values are 65,536/65,536/2,048;
- reset policy is both/04:00/1,440 minutes;
- delegation remains unchanged;
- the gateway was restarted once; and
- unrelated main-checkout changes remain untouched.
