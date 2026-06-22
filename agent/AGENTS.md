# Agent Module — Development Guide

Module-level extension of the root `AGENTS.md`. Read this when working inside
`agent/`, `run_agent.py`, or `tools/delegate_tool.py`.

---

## AIAgent Class (run_agent.py)

The real `AIAgent.__init__` takes ~60 parameters (credentials, routing, callbacks,
session context, budget, credential pool, etc.). The signature below is the
minimum subset you'll usually touch — read `run_agent.py` for the full list.

```python
class AIAgent:
    def __init__(self,
        base_url: str = None,
        api_key: str = None,
        provider: str = None,
        api_mode: str = None,              # "chat_completions" | "codex_responses" | ...
        model: str = "",                   # empty → resolved from config/provider later
        max_iterations: int = 90,          # tool-calling iterations (shared with subagents)
        enabled_toolsets: list = None,
        disabled_toolsets: list = None,
        quiet_mode: bool = False,
        save_trajectories: bool = False,
        platform: str = None,              # "cli", "telegram", etc.
        session_id: str = None,
        skip_context_files: bool = False,
        skip_memory: bool = False,
        credential_pool=None,
        # ... plus callbacks, thread/user/chat IDs, iteration_budget, fallback_model,
        # checkpoints config, prefill_messages, service_tier, reasoning_config, etc.
    ): ...

    def chat(self, message: str) -> str:
        """Simple interface — returns final response string."""

    def run_conversation(self, user_message: str, system_message: str = None,
                         conversation_history: list = None, task_id: str = None) -> dict:
        """Full interface — returns dict with final_response + messages."""
```

### Agent Loop

The core loop is inside `run_conversation()` — entirely synchronous, with
interrupt checks, budget tracking, and a one-turn grace call:

```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) \
        or self._budget_grace_call:
    if self._interrupt_requested: break
    response = client.chat.completions.create(model=model, messages=messages, tools=tool_schemas)
    if response.tool_calls:
        for tool_call in response.tool_calls:
            result = handle_function_call(tool_call.name, tool_call.args, task_id)
            messages.append(tool_result_message(result))
        api_call_count += 1
    else:
        return response.content
```

Messages follow OpenAI format: `{"role": "system/user/assistant/tool", ...}`.
Reasoning content is stored in `assistant_msg["reasoning"]`.

---

## Delegation (`delegate_task`)

`tools/delegate_tool.py` spawns a subagent with an isolated
context + terminal session. By default the parent waits for the
child's summary before continuing its own loop. With `background=true`,
Hermes returns a delegation id immediately and the result re-enters the
conversation later through the async-delegation completion queue.

Two shapes:

- **Single:** pass `goal` (+ optional `context`, `toolsets`).
- **Batch (parallel):** pass `tasks: [...]` — each gets its own subagent
  running concurrently. Concurrency is capped by
  `delegation.max_concurrent_children` (default 3).

Roles:

- `role="leaf"` (default) — focused worker. Cannot call `delegate_task`,
  `clarify`, `memory`, `send_message`, `execute_code`.
- `role="orchestrator"` — retains `delegate_task` so it can spawn its
  own workers. Gated by `delegation.orchestrator_enabled` (default true)
  and bounded by `delegation.max_spawn_depth` (default 2).

Key config knobs (under `delegation:` in `config.yaml`):
`max_concurrent_children`, `max_spawn_depth`, `child_timeout_seconds`,
`orchestrator_enabled`, `subagent_auto_approve`, `inherit_mcp_toolsets`,
`max_iterations`.

Durability rule: background `delegate_task` is detached from the current
turn but still process-local. For work that must survive process restart, use
`cronjob` or `terminal(background=True, notify_on_complete=True)` instead.

---

## Curator (skill lifecycle)

Background skill-maintenance system that tracks usage on agent-created
skills and auto-archives stale ones. Users never lose skills; archives
go to `~/.hermes/skills/.archive/` and are restorable.

- **Core:** `agent/curator.py` (review loop, auto-transitions, LLM review
  prompt) + `agent/curator_backup.py` (pre-run tar.gz snapshots).
- **CLI:** `hermes_cli/curator.py` wires `hermes curator <verb>` where
  verbs are: `status`, `run`, `pause`, `resume`, `pin`, `unpin`,
  `archive`, `restore`, `prune`, `backup`, `rollback`.
- **Telemetry:** `tools/skill_usage.py` owns the sidecar
  `~/.hermes/skills/.usage.json` — per-skill `use_count`, `view_count`,
  `patch_count`, `last_activity_at`, `state` (active / stale /
  archived), `pinned`.

Invariants:
- Curator only touches skills with `created_by: "agent"` provenance —
  bundled + hub-installed skills are off-limits.
- Never deletes; max destructive action is archive.
- Pinned skills are exempt from every auto-transition and from the
  LLM review pass.
- `skill_manage(action="delete")` refuses pinned skills; patch/edit/
  write_file/remove_file go through so the agent can keep improving
  pinned skills.

Config section (`curator:` in `config.yaml`):
`enabled`, `interval_hours`, `min_idle_hours`, `stale_after_days`,
`archive_after_days`, `backup.*`.

Full user-facing docs: `website/docs/user-guide/features/curator.md`.
