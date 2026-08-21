---
id: agent-loop
kind: process
universe: runtime
name: Agent Loop
summary: >
  Core conversation orchestration in `AIAgent.run_conversation`:
  build messages, call the model, execute tool calls, compress if needed, persist.
aliases: []
tags: [agent, tool-call, conversation]
shape: process
steps:
  - id: step.1
    summary: >
      Acquire durable session turn lease when a durable `SessionDB` row exists
      for `session_id`; reload latest transcript if the lease had to wait.
      On timeout or interrupt, return early without mutating caller history.
  - id: step.2
    summary: >
      Append the user message to conversation history, generate `task_id` if missing,
      and build or reuse the cached system prompt via prompt builder.
  - id: step.3
    summary: >
      Resolve provider/api mode, build API messages, apply preflight compression
      if needed, and make an interruptible API call.
  - id: step.4
    summary: >
      Parse response: if tool calls, execute sequentially or concurrently and loop;
      if text, persist messages, flush memory if needed, and return.
entrypoints: [step.1]
produces: [repo:run_agent.py]
consumes: [repo:model_tools.py, repo:agent/conversation_loop.py]
---

# Agent Loop

1. Acquire durable session turn lease when a durable `SessionDB` row already exists for `session_id` (`run_agent.py:8493`). Wait up to 1800s or abort on interrupt. If wait times out, return `{"final_response": timeout_msg, "completed": False, "failed": True}`. If interrupted, return `interrupted: True`. On admission, reload the latest transcript and resolve resume session id if the lease had to wait (`run_agent.py:8581-8590`).
2. Append the user message to `conversation_history` and generate `task_id` if missing (`run_agent.py:8368-8374`).
3. Build or reuse the cached system prompt and resolve provider/api mode (`run_agent.py` forwards to `agent.agent_init.init_agent`; mode resolution documented at `website/docs/developer-guide/agent-loop.md:53-57`).
4. Check preflight compression threshold before the API call (`website/docs/developer-guide/agent-loop.md:203-204`).
5. Build API messages in the selected mode's format: `chat_completions`, `codex_responses`, or `anthropic_messages` (`website/docs/developer-guide/agent-loop.md:69-72`).
6. Make an interruptible API call in a background thread with interrupt/timeout monitoring (`website/docs/developer-guide/agent-loop.md:108-119`).
7. Parse the response:
   - If `tool_calls`: execute sequentially or concurrently (`ThreadPoolExecutor`), reinsert `{"role": "tool"}` results in call order, loop back to step 5 (`website/docs/developer-guide/agent-loop.md:129-148`).
   - If text: persist messages to `SessionDB`, flush memory if needed, return result dict (`website/docs/developer-guide/agent-loop.md:214-219`).
8. Honor `IterationBudget` and `fallback_model` chains on provider/transport errors (`website/docs/developer-guide/agent-loop.md:180-197`).

## Human check

Confirm `AIAgent.run_conversation` still starts with `task_id or uuid4()` and acquires a session turn lease before loading history. Confirm the result dict still contains `final_response`, `messages`, `api_calls`, `completed`, `interrupted`.

## Deterministic validation

```bash
grep -n "def run_conversation" run_agent.py
grep -n "acquire_session_turn_lease" run_agent.py
grep -n "return interrupt_result" run_agent.py
grep -n '"final_response"' run_agent.py | head -20
```

Expected: `run_conversation` at line 8339, `acquire_session_turn_lease` around 8493, and both return shapes still present.
