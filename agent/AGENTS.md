# agent/ — AIAgent, turn loop, prompt, compression

Applies on top of the root `AGENTS.md` (prompt-caching invariant, facade + siblings rules).

## Shape

`run_agent.py` is the public facade: `AIAgent` is assembled from mixins (`agent/turn_facade.py`,
`client_lifecycle.py`, `stream_delivery.py`, `session_persistence.py`, `compression_facade.py`, ...).
Construction runs `agent/agent_init.py::init_agent`; a turn is
`agent/conversation_loop.py::run_conversation`, which `AIAgent.run_conversation` forwards to after
taking the session turn lease (`turn_facade_lease.py`). `AIAgent.__init__` takes ~60 parameters
(credentials, routing, callbacks, session context, budget, credential pool, ...) — read
`run_agent.py` for the list; the subset you usually touch: `base_url`, `api_key`, `provider`,
`api_mode` (`"chat_completions" | "codex_responses" | ...`), `model` (empty → resolved from
config/provider later), `max_iterations` (default 500, shared with subagents),
`enabled_toolsets`/`disabled_toolsets`, `quiet_mode`, `save_trajectories`, `platform`
(`"cli"`, `"telegram"`, ...), `session_id`, `skip_context_files`, `skip_memory`, `credential_pool`.
`chat(message) -> str` is the simple interface; `run_conversation(user_message, system_message=None,
conversation_history=None, task_id=None) -> dict` returns `final_response` + `messages`.

## Agent loop (`agent/conversation_loop.py` + `agent/turn_*.py`)

Entirely synchronous, with interrupt checks, budget tracking, and a one-turn grace call:

```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) \
        or self._budget_grace_call:
    if self._interrupt_requested: break
    response = client.chat.completions.create(model=model, messages=messages, tools=tool_schemas)
    if response.tool_calls:
        for tc in response.tool_calls:
            messages.append(tool_result_message(handle_function_call(tc.name, tc.args, task_id)))
        api_call_count += 1
    else:
        return response.content
```

Each phase of an iteration is its own sibling, so a change to (say) overflow handling touches one
~600-line file: `turn_preflight*`, `turn_iteration_prep`, `turn_request_assembly`/`turn_api_request`,
`turn_api_call`, `turn_api_error`, `turn_response_intake`/`turn_response_check`,
`turn_empty_response`, `turn_tool_round`/`turn_tool_validation`, `turn_overflow`,
`turn_truncation`, `turn_context_compaction`, `turn_recovery`, `turn_retry_state`,
`turn_stop_gates`, `turn_liveness`, `turn_usage`, `turn_final_response`, `turn_finalizer`,
`turn_summary`. Find the phase with `grep -rn "def X" agent/turn_*.py`.

Messages use OpenAI format `{"role": "system|user|assistant|tool", ...}`; reasoning content is stored
in `assistant_msg["reasoning"]`.

**Agent-level tools** (`todo`, `memory`, ...) are intercepted by `agent/tool_executor.py` through the
`INLINE_TOOL_EXECUTORS` table in `agent/inline_tool_executors.py` before `handle_function_call()`.
Adding one: register in that table (no `if name == ...` chain); `tools/todo_tool.py` is the pattern.

## Message-flow invariants (every change is reviewed against these)

- **Prompt caching must not break.** Never alter past context, change toolsets, reload memories,
  or rebuild the system prompt mid-conversation. The system prompt is byte-stable for the life of
  a conversation; the ONLY context mutation is compression. Anything that must inject content
  mid-conversation rides a **user message or tool result**, never the system prompt: skill slash
  commands (`agent/skill_commands.py`) inject as a user message; subdirectory `AGENTS.md` hints
  (`agent/subdirectory_hints.py`) append to the tool result (head+tail truncated past `_MAX_HINT_CHARS = 32_000`, with a warning).
- **Strict role alternation.** Never two same-role messages in a row; never a synthetic user
  message injected mid-loop. Cron deliveries live in their own session for this reason.
- **Context files** (`agent/prompt_builder.py`) load from the CWD only at startup and are capped
  (`CONTEXT_FILE_MAX_CHARS` / dynamic cap from the context window / `context_file_max_chars`).
  Never load an install-tree `AGENTS.md` as project context (PR #64611); subdirectory hints reject
  paths outside the working dir so `~/.codex/AGENTS.md` / `~/.claude/CLAUDE.md` never mix in.
- **`_last_resolved_tool_names` is a process-global in `model_tools.py`.** `_run_single_child()`
  in `tools/delegate_tool.py` saves/restores it around subagent execution; code reading it may see
  a temporarily stale value during child runs.

## Compression (`agent/compression_facade.py`, `conversation_compression.py`, `turn_context_compaction.py`)

Two layers: gateway session hygiene (85% threshold) and the agent `ContextCompressor` (50%,
configurable; per-model overrides; failure cooldown after provider-proven overflow). The algorithm
prunes old tool results first (no LLM call), then picks boundaries, then generates a structured
summary with the `auxiliary` compression model. In-place compaction keeps a single stable session
id; native Responses/Codex compaction paths are provider-specific. Compression is the sanctioned
cache break — keep it the only one. Full detail:
`website/docs/developer-guide/context-compression-and-caching.md`.

## Model and provider resolution

- Runtime provider/model resolution and its precedence: `website/docs/developer-guide/provider-runtime.md`.
  Provider profiles are plugins (`plugins/model-providers/<name>/`, see `plugins/AGENTS.md`);
  `agent/model_metadata.py` holds context lengths and capabilities.
- **Auxiliary (side-LLM) work** — curator, vision, embedding, title generation, session_search,
  compression — resolves through `agent/auxiliary_client.py::_resolve_auto_route`; each task can pin
  its own `provider/model/base_url/max_tokens/reasoning_effort` under `auxiliary:` in config.yaml.
- Fallback models and credential pools are resolution-chain code: E2E them with real imports
  against a temp `HERMES_HOME`, not mocks (root rubric).

## Memory, context engines, curator

`agent/memory_provider.py` (ABC) + `agent/memory_manager.py` (orchestrator) drive memory-provider
plugins; `agent/context_engine.py` drives context-engine plugins; `agent/image_gen_provider.py`
image-gen plugins (all in `plugins/AGENTS.md`). `agent/curator.py` + `curator_backup.py` implement
the skill curator (`skills/AGENTS.md`). Cron sessions pass `skip_memory=True` by default — memory
providers intentionally do not run during cron.

## Tests

Loop/phase tests go in `tests/agent/`; patch the binding the phase actually reads (siblings often
`from run_agent import X` inside the function — root "patch where production reads"). Assert
message-shape invariants (alternation, byte-stable system prompt) rather than snapshotting prompt
text.

Long-form: `website/docs/developer-guide/agent-loop.md`, `prompt-assembly.md`,
`context-compression-and-caching.md`, `provider-runtime.md`, `session-storage.md`,
`subagent-lifecycle-api.md`.
