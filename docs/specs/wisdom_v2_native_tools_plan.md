# Hermes Wisdom v2 Native Tools Plan

## Objective

Expose the existing Hermes Wisdom Kernel v1 as first-class Hermes model tools so the normal Hermes/Codex agent can use Wisdom during natural conversation. v2 keeps the v1 kernel intact: exact originals stay authoritative, SQLite writes remain inside `wisdom/`, `/wisdom` commands remain available, and no external task/reminder system is used.

## Repo Findings

### 1. How Hermes Tools Are Registered

Hermes built-in tools are registered by Python modules under `tools/`. A tool module is discovered when it has a top-level `registry.register(...)` call. `tools/registry.py::discover_builtin_tools()` imports those modules, and each module registers:

- a stable tool name
- a toolset key
- an OpenAI-style function schema
- a handler returning a JSON string
- an optional availability check

`model_tools.py` imports and discovers built-in tools at module load, then exposes schemas through `get_tool_definitions(...)` and execution through `handle_function_call(...)`.

### 2. How Model-Facing Schemas Are Generated

Each tool file owns its schema dictionary. `tools.registry.ToolRegistry.get_definitions()` wraps each schema as:

```python
{"type": "function", "function": schema}
```

`model_tools.get_tool_definitions()` resolves requested toolsets through `toolsets.py`, filters registered schemas, and sanitizes them for the active provider/backend. Descriptions in these schema dictionaries are the correct place for model-facing Wisdom guidance.

### 3. Where Wisdom Tools Should Live

Add `tools/wisdom_tool.py`. It should contain only thin wrappers and schemas for:

- `wisdom_status`
- `wisdom_capture`
- `wisdom_search`
- `wisdom_original`
- `wisdom_interpret`
- `wisdom_apply`
- `wisdom_review`
- `wisdom_archive`
- optionally `wisdom_inbox`
- optionally `wisdom_set_enabled`

The wrappers must call the `wisdom/` package and must not contain SQL.

### 4. Active/Default Toolset Inclusion

Add a `wisdom` toolset in `toolsets.py`, and include the Wisdom tool names in `_HERMES_CORE_TOOLS`. This makes Wisdom available to the default CLI and messaging toolsets, including `hermes-telegram`, without additional gateway routing.

Add `wisdom` to `hermes_cli/tools_config.py::CONFIGURABLE_TOOLSETS`, but do not add it to `_DEFAULT_OFF_TOOLSETS`. Users can disable it through the existing tool configuration path, while new/default Hermes sessions can see it.

### 5. How the Model Will Discover and Call Wisdom Tools

For normal Telegram chat, the gateway creates an `AIAgent` with the Telegram platform toolset. That toolset resolves through `toolsets.py`; because Wisdom is in `_HERMES_CORE_TOOLS`, the model receives the Wisdom schemas alongside existing core tools. When the model chooses a Wisdom function call, `model_tools.handle_function_call()` dispatches it through `tools.registry`.

The natural-language product path is therefore:

authorized Telegram message -> AIAgent conversation -> model sees Wisdom tool schemas -> model calls Wisdom tool -> `wisdom/` package writes/reads `wisdom.db` -> model replies naturally from the tool result.

### 6. Source-Backed Skill Usefulness

Hermes has built-in and user skills, but the inspected skill path is not an always-loaded source-backed instruction layer for every running gateway conversation. Skills are primarily listed/viewed/managed through tools or invoked through slash-command style skill loading. For v2, a Wisdom skill would not be a reliable way to make the model use Wisdom.

Do not mutate `~/.hermes/skills`. Put the required natural-language guidance in tool descriptions and update docs instead.

### 7. Shared Core Service Layer

v1 command handling currently calls `wisdom.capture`, `wisdom.retrieve`, `wisdom.interpret`, `wisdom.apply`, and `wisdom.db` directly. For v2, add `wisdom/service.py` as the canonical high-level API for commands and tools:

- `status`
- `capture`
- `search`
- `original`
- `interpret`
- `apply`
- `review`
- `archive`
- `inbox`
- `set_enabled`

Then migrate `/wisdom` command handlers to call the service where practical. This keeps command and tool behavior aligned while still preserving v1 behavior.

### 8. Avoiding Duplicated DB Logic

All SQL remains in `wisdom/db.py`. The service layer may compose existing DB methods and existing modules, but `tools/wisdom_tool.py` must not perform direct DB operations or model-generated SQL. Tests should inspect behavior and tool registration rather than duplicate schema assumptions.

### 9. Minimal Gateway Edits

No gateway edits are needed for the native tool path. v1 already added `/wisdom` command handling and a small post-auth explicit capture intercept. v2 should rely on the existing AIAgent tool loop instead of adding another Telegram router.

### 10. Fail-Open Behavior

Tool handlers should catch exceptions and return concise JSON errors. The registry and `model_tools.handle_function_call()` already catch tool exceptions, so a Wisdom failure should not crash the agent loop. Existing gateway fail-open behavior for v1 command/natural capture integration must remain unchanged.

For explicit Wisdom requests, the model can report the tool error. For ordinary chat, no Wisdom tool should be called unless the model decides the user asked for durable memory behavior.

### 11. Tests Proving Tool Availability

Add tests that prove:

- all Wisdom tool names are registered
- schemas have required inputs and non-empty model-facing descriptions
- descriptions include natural trigger language for capture, search, original, apply, review, and archive
- `resolve_toolset("wisdom")` returns the Wisdom tools
- default CLI/Telegram toolsets include Wisdom through `_HERMES_CORE_TOOLS`
- `model_tools.get_tool_definitions(enabled_toolsets=["wisdom"])` exposes the schemas
- behavior tests can call the tools through `model_tools.handle_function_call()` with temp Wisdom DB paths

### 12. Natural-Language Coverage Without Live Model Calls

Tests should not depend on real model calls. Instead, they should prove that descriptors and schemas expose the natural-language affordances the model needs:

- "Find that idea about peace of mind." maps to search language.
- "Show me exactly what I wrote." maps to exact-original language.
- "Turn that into client language." maps to application language.
- "What have I captured recently?" maps to review/inbox language.

### 13. Intentionally Not Included in v2

v2 does not add:

- a large regex/NL router
- automatic capture of all chat
- external task/reminder execution
- old productivity DB migration or writes
- voice transcription
- scheduled reviews
- embeddings/vector search
- cloud sync or dashboard/export integrations
- challenge-mode tables or new Wisdom OS data models
- live model calls in tests

### 14. Repo-Specific Risks or Deviations

- Adding tools to `_HERMES_CORE_TOOLS` increases the default tool schema set. Mitigation: keep descriptions concise but specific and make the `wisdom` toolset configurable.
- The currently running launchd gateway will not load new tool modules until restarted. Mitigation: do not restart in this run; document the safe restart command.
- Tool descriptions are the primary product guidance. Mitigation: test for required trigger language and keep `/wisdom` commands as fallback/debug controls.
- Optional category/source-type overrides must not weaken exact-original preservation. Mitigation: only override structured metadata, never `original_text`.
- Secret-like captures must be blocked, not redacted into modified originals. Mitigation: reuse v1 secret detection in the service path and test it through the native tool.

## Implementation Notes / Deviations

- Use native Hermes tools, not MCP. The repo convention for model-callable local functions is `tools/*.py` plus `tools.registry`.
- Do not add a bundled or user skill in v2. The repo does not show a clearly always-loaded source-backed skill mechanism for gateway conversations.
- No new gateway hook should be added. Natural-language Wisdom use comes from the model tool loop.
- `period` and `context` parameters may be accepted by tool schemas for model ergonomics, but v2 should keep deterministic storage behavior. Rich wording can be produced by the model in the final response after grounded tool calls.
