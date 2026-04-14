# Matrix Reimplementation Plan for Hermes v0.9.0

> Status: ACTIVE WORK PLAN
> Repo: `~/.hermes/hermes-agent`
> Base: `v2026.4.13` / `v0.9.0`
> Constraint: DO NOT restart the gateway without Chris's approval.

## Goal
Reintroduce Whyland's missing Matrix functionality on top of the current mautrix-based Hermes Agent implementation without regressing existing Matrix behavior or non-Matrix platforms.

## Upstream baseline already present
- `gateway/platforms/matrix.py` is mautrix-based
- `gateway/run.py` already supports generic adapter hooks:
  - `send_exec_approval(...)`
  - `send_model_picker(...)`
- Matrix adapter already has public methods for:
  - `redact_message`
  - `fetch_room_history`
  - `create_room`
  - `invite_user`
  - `set_presence`
  - `send_read_receipt`
- Matrix adapter already has internal reaction plumbing:
  - `_send_reaction`
  - `_on_reaction`

## Still missing upstream
- `tools/matrix_tools.py`
- Matrix tool registration in `model_tools.py`
- Matrix toolset wiring in `toolsets.py`
- Matrix platform hint in `agent/prompt_builder.py`
- Matrix approval UI implementation
- Matrix model picker implementation
- Matrix thinking/acting pane implementation

## Core design rule: event-role registry / interaction routing
Matrix interaction behavior MUST be driven by explicit runtime event roles, not string matching or ad hoc skip logic.

### Event roles
- `conversation_output`
- `processing_lifecycle_target`
- `interactive_control`
- `thinking_pane`
- `acting_pane`
- `system_status` (optional)

### Routing rules
- Processing lifecycle reactions ONLY apply to `processing_lifecycle_target`
- Approval/model picker control events are registered as `interactive_control`
- Thinking/Acting pane events are registered as `thinking_pane` / `acting_pane`
- `_on_reaction(...)` routes based on target event role metadata

### Control state requirements
State MUST be keyed by authoritative event identity, including:
- `event_id`
- `room_id`
- thread/root context
- session key
- authorized actor/user context
- subtype (`approval`, `model_picker`)

This avoids cross-resolution and makes teardown safe.

---

# Branch A — Matrix tools + platform hint

## Branch name
`feat/matrix-tools-v090`

## Status
IMPLEMENTED LOCALLY

## Scope
Restore Matrix agent-callable tools and platform hint with minimal adapter surface changes.

## Files
### Create
- `tools/matrix_tools.py`
- `tests/gateway/test_matrix_tools.py`

### Modify
- `gateway/platforms/matrix.py`
- `model_tools.py`
- `toolsets.py`
- `agent/prompt_builder.py`

## Rules
- REUSE existing public `MatrixAdapter` methods
- DO NOT duplicate raw Matrix SDK/API logic in `tools/matrix_tools.py`
- DO NOT add Matrix tools to `_HERMES_CORE_TOOLS`
- Add Matrix tools only via Matrix-specific toolset wiring
- Tool availability requires a live connected Matrix adapter
- Preserve optional import/discovery behavior in `model_tools._discover_tools()`

## Tool list
- `matrix_send_reaction`
- `matrix_redact_message`
- `matrix_create_room`
- `matrix_invite_user`
- `matrix_fetch_history`
- `matrix_set_presence`

## Public wrapper decision
A minimal public `send_reaction(...)` wrapper was added so tools DO NOT call private adapter internals.

## Acceptance criteria
- Matrix tools are discoverable only in Matrix-specific tool context
- Tools require a live adapter instance
- Disconnected adapter path fails cleanly
- No import-time failures if Matrix gateway is absent/inactive
- Matrix platform hint accurately reflects tool availability
- No runtime change when Matrix tools are unused

## DO NOT break
- Shared core toolsets
- Existing Matrix gateway runtime behavior
- Current mautrix lifecycle
- Optional tool discovery behavior

## Tests
- `tests/gateway/test_matrix_tools.py`
- `tests/gateway/test_matrix.py`
- `tests/test_toolsets.py`
- `tests/test_model_tools.py`
- `tests/agent/test_prompt_builder.py`

## Local validation result
Command run:
```bash
venv/bin/python -m pytest \
  tests/gateway/test_matrix_tools.py \
  tests/gateway/test_matrix.py \
  tests/test_toolsets.py \
  tests/test_model_tools.py \
  tests/agent/test_prompt_builder.py \
  -o 'addopts=' -q
```

Result:
- `272 passed`

## Local commit
- `dc1a275e` — `feat(matrix): add Matrix tool wrappers and platform hint`

---

# Branch B — Matrix approval UI + model picker

## Branch name
`feat/matrix-interactive-v090`

## Status
IMPLEMENTED LOCALLY

## Scope
Implement Matrix-native interactive approval and model selection using existing upstream gateway hooks.

## Files
### Create
- `tests/gateway/test_matrix_interactive.py`

### Modify
- `gateway/platforms/matrix.py`

## Rules
- Implement:
  - `send_exec_approval(...)`
  - `send_model_picker(...)`
- REUSE `_on_reaction(...)`
- Use event-role registry and interaction routing
- Control state must be keyed by authoritative control event identity
- Preserve gateway fallback behavior if interactive send fails
- Consume current metadata shape from `gateway/run.py`, especially thread metadata

## Acceptance criteria
- Approval and model picker render correctly
- Selection resolves exactly once
- Duplicate reactions/events do not double-resolve
- Self-reactions are ignored
- Wrong-user / unauthorized reactions are ignored
- Malformed / irrelevant reactions are ignored
- Timeout/cancel/stale-state cleanup works
- Disconnect/restart cleans pending state and resolves/cancels waiters exactly once
- Controls render in correct Matrix thread/root context
- If interactive send fails, existing text fallback still works
- Processing lifecycle reactions never target interactive control events
- No stray bot emoji reaction on approval/model-picker messages
- Multiple concurrent controls in the same room/thread do not cross-resolve

## DO NOT break
- Existing Matrix processing lifecycle reactions
- Existing text fallback approval flow
- Current `/model` persistence behavior in gateway
- Non-Matrix approval/model-picker behavior
- Existing `_on_reaction()` self-ignore / dedup behavior

## Tests
- `tests/gateway/test_matrix_interactive.py`
- `tests/gateway/test_matrix.py`
- `tests/gateway/test_model_switch_persistence.py`

## Local validation result
Command run:
```bash
venv/bin/python -m pytest \
  tests/gateway/test_matrix_interactive.py \
  tests/gateway/test_matrix.py \
  tests/gateway/test_model_switch_persistence.py \
  -o 'addopts=' -q
```

Result:
- `133 passed`

## Local commit
- `4aee3621` — `feat(matrix): add interactive approval and model picker`

---

# Branch C — Matrix thinking/acting panes

## Planned branch name
`feat/matrix-thinking-v090`

## Status
IMPLEMENTED LOCALLY

## Scope
Add Matrix-only panes:
- `Agent Thinking: <AgentName> via <ModelName>`
- `Agent Acting:`

while preserving current Hermes action formatting.

## Files
### Create
- `gateway/platforms/matrix_thinking.py`
- `tests/gateway/test_matrix_thinking.py`
- `tests/gateway/test_matrix_run_regression.py`
- optional: `tests/gateway/test_matrix_thinking_delta_merge.py`

### Modify
- `gateway/platforms/matrix.py`
- `gateway/run.py`

## Rules
- REUSE existing gateway callback outputs
- DO NOT invent a new acting formatter
- DO NOT add provider-specific reasoning parsing
- `Agent Acting` MUST preserve existing Hermes:
  - emoji
  - tool/action labels
  - quoted args/labels
  - ordering
- Matrix-only suppression applies only to pane/interim progress traffic
- Suppression MUST NOT suppress final assistant response
- Keep `gateway/run.py` patch surface minimal
- Explicitly define behavior for:
  - no reasoning available
  - non-streaming turn
  - final-only reasoning
  - tool-only turn

## Acceptance criteria
- Thinking heading format:
  - `Agent Thinking: <AgentName> via <ModelName>`
- Acting heading:
  - `Agent Acting:`
- Acting preserves current Hermes action formatting semantics
- No reasoning/tool event loss under rapid updates
- Repeated edits remain stable under bursts
- Edit failure falls back safely with bounded replacement behavior
- No duplicate Matrix timeline spam from pane traffic
- Final assistant response still delivered exactly once
- If `tool_progress` is off, Acting pane is off
- When panes are disabled, behavior matches current upstream
- No regressions for non-Matrix platforms
- Finalize on success, abort on timeout, cleanup on restart/disconnect all work
- Pane edits preserve Matrix thread/root context

## DO NOT break
- Existing tool progress behavior on other platforms
- Existing stream/reasoning behavior outside Matrix
- Existing Matrix behavior when panes are disabled
- Current approval/model/session-note behavior in `gateway/run.py`
- Current Matrix batching/split-message aggregation
- Final response delivery

## Planned tests
- `tests/gateway/test_matrix_thinking.py`
- `tests/gateway/test_matrix_run_regression.py`
- optional `tests/gateway/test_matrix_thinking_delta_merge.py`
- `tests/gateway/test_matrix.py`
- `tests/gateway/test_model_switch_persistence.py`

## Local validation result
Command run:
```bash
venv/bin/python -m pytest \
  tests/gateway/test_matrix_interactive.py \
  tests/gateway/test_matrix.py \
  tests/gateway/test_model_switch_persistence.py \
  tests/gateway/test_matrix_thinking.py \
  tests/gateway/test_matrix_run_regression.py \
  -o 'addopts=' -q
```

Result:
- `141 passed`

## Local branch state
- Branch C branch: `feat/matrix-thinking-v090`
- Branch C includes:
  - `gateway/platforms/matrix_thinking.py`
  - `tests/gateway/test_matrix_thinking.py`
  - `tests/gateway/test_matrix_run_regression.py`

---

# Restart / validation rule
Before any gateway restart for validation, STOP and get Chris's approval first.

## Current next step
- Branch B needs restart-based live validation before moving to Branch C.

## Current repo state snapshot
- Branch A commit: `dc1a275e`
- Branch B commit: `4aee3621`
- Branch C: not started
