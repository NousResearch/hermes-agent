# Tool Adapter Wiring Plan for `run_agent.py`

This document describes the exact changes needed to wire the tool adapter
(text-based tool calling for models without native function calling) into
`/Users/andy/hermes-agent/run_agent.py`.

---

## 1. Import Section (after line 99)

**Location:** After the existing `agent.*` imports (lines 78-99).

**Add after line 99:**

```python
from agent.model_capabilities import needs_tool_adapter, detect_capabilities
from agent.tool_prompt_injector import inject_tools_into_system_prompt, format_tool_response
from agent.tool_response_adapter import adapt_response, should_adapt, format_tool_results_as_content
```

This places the new imports alongside the other `agent.*` imports, maintaining
the existing grouping convention.

---

## 2. Capability Caching in `__init__` (after line 254)

**Location:** Inside `__init__`, after line 254 where `self._use_prompt_caching`
is set. This is a natural place for model-capability flags.

**Current code (lines 252-255):**

```python
        is_openrouter = "openrouter" in self.base_url.lower()
        is_claude = "claude" in self.model.lower()
        self._use_prompt_caching = is_openrouter and is_claude
        self._cache_ttl = "5m"  # Default 5-minute TTL (1.25x write cost)
```

**Add after line 255:**

```python
        # Tool adapter: for models without native function calling (e.g. local
        # Qwen via llama.cpp), inject tool definitions into the system prompt
        # and parse XML tool_call blocks from the response text.
        self._needs_tool_adapter = needs_tool_adapter(self.model, self.base_url)
```

---

## 3. System Prompt Injection (lines 3058-3064)

**Location:** In `run_conversation`, where `effective_system` is assembled from
`active_system_prompt` and `ephemeral_system_prompt`, just before it is
prepended to `api_messages`.

**Current code (lines 3055-3064):**

```python
            # Build the final system message: cached prompt + ephemeral system prompt.
            # The ephemeral part is appended here (not baked into the cached prompt)
            # so it stays out of the session DB and logs.
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if self._honcho_context:
                effective_system = (effective_system + "\n\n" + self._honcho_context).strip()
            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
```

**Replace with:**

```python
            # Build the final system message: cached prompt + ephemeral system prompt.
            # The ephemeral part is appended here (not baked into the cached prompt)
            # so it stays out of the session DB and logs.
            effective_system = active_system_prompt or ""
            if self.ephemeral_system_prompt:
                effective_system = (effective_system + "\n\n" + self.ephemeral_system_prompt).strip()
            if self._honcho_context:
                effective_system = (effective_system + "\n\n" + self._honcho_context).strip()

            # Tool adapter: inject tool definitions into the system prompt for
            # models that don't support native function calling.  The model will
            # emit <tool_call> XML blocks in its response text instead.
            if self._needs_tool_adapter and self.tools:
                effective_system = inject_tools_into_system_prompt(effective_system, self.tools)

            if effective_system:
                api_messages = [{"role": "system", "content": effective_system}] + api_messages
```

---

## 4. Suppress `tools` Param in API Call (lines 2142-2145)

**Location:** In `_build_api_kwargs`, where the chat-completions `api_kwargs`
dict is constructed.

**Current code (lines 2142-2146):**

```python
        api_kwargs = {
            "model": self.model,
            "messages": api_messages,
            "tools": self.tools if self.tools else None,
            "timeout": 900.0,
        }
```

**Replace with:**

```python
        # When the tool adapter is active, do NOT send `tools` to the API --
        # tool definitions are injected into the system prompt instead.
        # Some local backends error on the `tools` parameter entirely.
        effective_tools = None if self._needs_tool_adapter else (self.tools if self.tools else None)

        api_kwargs = {
            "model": self.model,
            "messages": api_messages,
            "tools": effective_tools,
            "timeout": 900.0,
        }
```

---

## 5. Response Adaptation (after line 3539, before line 3641)

**Location:** After `assistant_message` is extracted from the response
(lines 3536-3539), but BEFORE the `if assistant_message.tool_calls:` check
at line 3641. The best insertion point is after line 3596 (the reset of
`_incomplete_scratchpad_retries`), since scratchpad handling should still
run on the raw response.

**Current code (lines 3594-3597):**

```python
                # Reset incomplete scratchpad counter on clean response
                if hasattr(self, '_incomplete_scratchpad_retries'):
                    self._incomplete_scratchpad_retries = 0
```

**Add after line 3596:**

```python
                # Tool adapter: parse <tool_call> XML blocks from the response
                # text into structured tool_calls so the rest of the loop
                # (validation, execution, message building) works unchanged.
                if self._needs_tool_adapter and should_adapt(response, self.model):
                    assistant_message = adapt_response(response, self.tools)
```

**Why this location:** The adapt step must happen:
- AFTER scratchpad/think-block handling (lines 3563-3596) so those checks
  see the original text.
- BEFORE the tool_calls check (line 3641) so parsed tool calls are visible.
- BEFORE invalid-tool-name validation (line 3650) so it validates the
  parsed calls.

---

## 6. Tool Result Injection in `_execute_tool_calls` (lines 2664-2670)

**Location:** At the end of `_execute_tool_calls`, where each tool result
message is appended to `messages`.

**Current code (lines 2664-2670):**

```python
            tool_msg = {
                "role": "tool",
                "content": function_result,
                "tool_call_id": tool_call.id
            }
            messages.append(tool_msg)
            self._log_msg_to_db(tool_msg)
```

**Replace with:**

```python
            if self._needs_tool_adapter:
                # For adapted models, tool results go as user messages with
                # XML-formatted content, since the model expects text-based
                # <tool_response> blocks, not structured tool role messages.
                tool_msg = {
                    "role": "user",
                    "content": format_tool_response(tool_call.id, function_name, function_result),
                }
            else:
                tool_msg = {
                    "role": "tool",
                    "content": function_result,
                    "tool_call_id": tool_call.id,
                }
            messages.append(tool_msg)
            self._log_msg_to_db(tool_msg)
```

**Also update the skip messages** at lines 2474-2480 and 2680-2686 (the
interrupt skip paths) with the same pattern:

Lines 2474-2480 (first interrupt check):

```python
                for skipped_tc in remaining_calls:
                    if self._needs_tool_adapter:
                        skip_msg = {
                            "role": "user",
                            "content": format_tool_response(
                                skipped_tc.id, skipped_tc.function.name,
                                "[Tool execution cancelled - user interrupted]",
                            ),
                        }
                    else:
                        skip_msg = {
                            "role": "tool",
                            "content": "[Tool execution cancelled - user interrupted]",
                            "tool_call_id": skipped_tc.id,
                        }
                    messages.append(skip_msg)
                    self._log_msg_to_db(skip_msg)
```

Lines 2679-2686 (second interrupt check):

```python
                for skipped_tc in assistant_message.tool_calls[i:]:
                    if self._needs_tool_adapter:
                        skip_msg = {
                            "role": "user",
                            "content": format_tool_response(
                                skipped_tc.id, skipped_tc.function.name,
                                "[Tool execution skipped - user sent a new message]",
                            ),
                        }
                    else:
                        skip_msg = {
                            "role": "tool",
                            "content": "[Tool execution skipped - user sent a new message]",
                            "tool_call_id": skipped_tc.id,
                        }
                    messages.append(skip_msg)
                    self._log_msg_to_db(skip_msg)
```

---

## 7. Summary of All Changes

| # | File Location | Line(s) | Change Type |
|---|---------------|---------|-------------|
| 1 | Import section | after 99 | Add 3 import lines |
| 2 | `__init__` | after 255 | Add `self._needs_tool_adapter` |
| 3 | `run_conversation` (system prompt) | 3058-3064 | Insert tool injection block |
| 4 | `_build_api_kwargs` | 2142-2147 | Suppress `tools` param when adapted |
| 5 | `run_conversation` (response) | after 3596 | Insert `adapt_response` call |
| 6a | `_execute_tool_calls` (tool result) | 2664-2670 | Conditional `role: user` vs `role: tool` |
| 6b | `_execute_tool_calls` (skip msg 1) | 2474-2480 | Same conditional for interrupt skip |
| 6c | `_execute_tool_calls` (skip msg 2) | 2679-2686 | Same conditional for interrupt skip |

---

## 8. Testing Checklist

1. **Adapter off (default):** Verify existing Claude/GPT models work exactly
   as before -- `self._needs_tool_adapter` should be `False`, no code paths
   change.

2. **Adapter on (local model):** Test with a local Qwen or LLaMA model via
   llama.cpp/Ollama. Verify:
   - Tool definitions appear in the system prompt.
   - `tools` param is NOT sent in the API request.
   - `<tool_call>` blocks in the response text are parsed into structured
     `tool_calls` on `assistant_message`.
   - Tool results are sent as `role: user` with `<tool_response>` XML.
   - The agent loop completes normally.

3. **Edge cases:**
   - Model outputs no `<tool_call>` blocks: should behave as a normal text
     response (no tool calls).
   - Model outputs malformed XML: `adapt_response` should handle gracefully
     (return no tool calls, preserve original text).
   - Interrupt during adapted tool execution: skip messages use correct format.
