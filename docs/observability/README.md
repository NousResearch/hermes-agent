# Hermes Observer Hooks

Hermes observer hooks are the read-only telemetry contract for plugins that
need to reconstruct agent execution without changing runtime behavior. This
contract supports trace, metrics, audit, replay, and export integrations such
as Langfuse, OpenTelemetry-style collectors, and NeMo Relay.

Observer hooks are intentionally backend-neutral. They expose stable lifecycle
events, correlation IDs, sanitized payloads, timing, status, and error fields.
They do not replace Hermes' planner, model providers, memory, tool registry,
approval UX, CLI, gateway behavior, or execution semantics.

Behavior-changing request or execution wrappers are outside this observer
contract. Observer hooks should report what happened; they should not replace
provider requests, tool arguments, or execution callbacks.

## Contract

Plugins register observer callbacks from `register(ctx)`:

```python
def register(ctx):
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
```

Every hook callback receives keyword arguments. Plugins should accept
`**kwargs` so additive fields remain backward-compatible:

```python
def on_post_tool_call(**kwargs):
    tool_name = kwargs.get("tool_name")
    status = kwargs.get("status")
    result = kwargs.get("result")
```

The plugin manager injects this field into every hook payload:

```text
telemetry_schema_version = "hermes.observer.v1"
```

Hook callbacks are fail-open. Hermes catches callback exceptions, logs a
warning, and keeps the agent loop running.

Most observer hook return values are ignored. The exceptions are older
behavior-affecting hooks:

| Hook | Return behavior |
| --- | --- |
| `pre_llm_call` | May return a string or `{"context": "..."}` to inject ephemeral context into the current user message. |
| `pre_tool_call` | May return `{"action": "block", "message": "..."}` to block a tool before execution. |
| `transform_tool_result` | May return a replacement tool result string after `post_tool_call`. |
| `transform_llm_output` | May return a replacement final assistant text string. Legacy post-loop compatibility hook; it cannot retry the model or execute tools in the same turn. |

Telemetry plugins should treat these behavior-affecting returns as optional
compatibility features, not as observability requirements. New behavior-changing
final-draft validation should use `assistant_response` middleware instead of an
observer hook when it needs structured `pass`, `rewrite`, `retry_with_feedback`,
`require_tool`, or `block` control flow.

## Correlation IDs

Observer payloads use stable IDs so plugins can join events without relying on
callback order alone.

| Field | Meaning |
| --- | --- |
| `session_id` | Conversation/session identity. |
| `task_id` | Task identity, especially useful for subagents and isolated execution. |
| `turn_id` | User-turn identity shared by API attempts and tool calls in a turn. |
| `api_request_id` | Opaque provider-attempt identity. Do not parse its string format. |
| `api_call_count` | Numeric API attempt count within the agent loop. |
| `tool_call_id` | Provider-supplied tool call ID when available. |
| `parent_session_id` / `child_session_id` | Session link for delegated subagents. |
| `parent_subagent_id` / `child_subagent_id` | Subagent link when available. |
| `parent_turn_id` | Parent turn that spawned delegated work. |

Consumers should prefer explicit fields over parsing compound IDs. In
particular, `api_request_id` is an opaque correlation value.

## Event Families

### Session Lifecycle

Session hooks describe conversation boundaries and resets:

| Hook | When it fires |
| --- | --- |
| `on_session_start` | A brand-new session starts after the system prompt is built. |
| `on_session_end` | A `run_conversation` call ends, including interrupted or incomplete turns. |
| `on_session_finalize` | CLI or gateway tears down an active session identity. |
| `on_session_reset` | CLI or gateway moves from an old session identity to a new one. |

Common fields include `session_id`, `completed`, `interrupted`, `reason`,
`old_session_id`, and `new_session_id` where available.

`on_session_end` is turn/run scoped. It is not necessarily the final lifetime
boundary for a chat identity. Use `on_session_finalize` and `on_session_reset`
for lifecycle cleanup that must happen once per session identity.

### Turn-Scoped LLM Hooks

These hooks frame the user turn, not individual provider API attempts:

| Hook | When it fires |
| --- | --- |
| `pre_llm_call` | Before the tool loop begins for a user turn. |
| `post_llm_call` | After the turn completes with final assistant output. |

Common `pre_llm_call` fields include `session_id`, `turn_id`,
`user_message`, `conversation_history`, `is_first_turn`, `model`, `platform`,
and `sender_id`.

Common `post_llm_call` fields include `session_id`, `turn_id`,
`user_message`, `assistant_response`, `conversation_history`, `model`, and
`platform`.

Use request-scoped API hooks for LLM span telemetry. Use `pre_llm_call` and
`post_llm_call` for turn-level context, compatibility, and final turn summary.

### Request-Scoped API Hooks

API hooks describe provider attempts inside the agent loop:

| Hook | When it fires |
| --- | --- |
| `pre_api_request` | Immediately before a provider API request. |
| `post_api_request` | After a successful provider response. |
| `api_request_error` | After a failed provider request or retryable error path. |

`pre_api_request` includes:

- identity: `session_id`, `task_id`, `turn_id`, `api_request_id`
- runtime: `platform`, `model`, `provider`, `base_url`, `api_mode`
- attempt metadata: `api_call_count`, `message_count`, `tool_count`,
  `approx_input_tokens`, `request_char_count`, `max_tokens`
- timing: `started_at`
- sanitized request payload: `request`

`post_api_request` includes the same identity/runtime fields plus:

- `api_duration`, `started_at`, `ended_at`
- `finish_reason`, `message_count`, `response_model`
- `usage`
- `assistant_content_chars`, `assistant_tool_call_count`
- sanitized response payload: `response`
- compatibility object: `assistant_message`

`api_request_error` includes the same identity/runtime fields plus:

- `api_duration`, `started_at`, `ended_at`
- `status_code`, `retry_count`, `max_retries`, `retryable`, `reason`
- structured `error = {"type": ..., "message": ...}`
- sanitized failed request payload: `request`

The sanitized `request`, `response`, and `error` fields are the canonical
observer inputs for new consumers.

### Tool Lifecycle

Tool hooks describe individual tool calls:

| Hook | When it fires |
| --- | --- |
| `pre_tool_call` | Before guardrail-approved tool dispatch. |
| `post_tool_call` | After tool dispatch, cancellation, block, or error completion. |
| `transform_tool_result` | After `post_tool_call`, before the result is appended to model context. |

`pre_tool_call` includes `tool_name`, `args`, `task_id`, `session_id`,
`tool_call_id`, `turn_id`, and `api_request_id`.

`post_tool_call` includes the same identity fields plus `result`,
`duration_ms`, `status`, `error_type`, and `error_message`.

`status` is the observer-grade lifecycle outcome. Common values include:

| Status | Meaning |
| --- | --- |
| `ok` | Tool completed normally. |
| `error` | Tool ran and returned or raised an error outcome. |
| `blocked` | A `pre_tool_call` hook blocked execution. |
| `cancelled` | Execution was cancelled before normal completion. |

`post_tool_call` is emitted for blocked and cancelled paths so telemetry
plugins can close spans cleanly.

### Approval Lifecycle

Approval hooks describe dangerous-command approval prompts:

| Hook | When it fires |
| --- | --- |
| `pre_approval_request` | Before the approval request is shown or sent. |
| `post_approval_response` | After the user responds or the request times out. |

Common fields include `command`, `description`, `pattern_key`,
`pattern_keys`, `session_key`, and `surface`.

`post_approval_response` also includes `choice`, with values such as `once`,
`session`, `always`, `deny`, and `timeout`.

Approval hooks are observer-only. Plugins cannot pre-answer or veto approvals
from these hooks. To prevent a tool from reaching approval, use
`pre_tool_call` blocking.

### Subagent Lifecycle

Subagent hooks describe delegated child-agent work:

| Hook | When it fires |
| --- | --- |
| `subagent_start` | A delegated child agent is created. |
| `subagent_stop` | A delegated child agent returns or fails. |

`subagent_start` fields include `parent_session_id`, `parent_turn_id`,
`parent_subagent_id`, `child_session_id`, `child_subagent_id`, `child_role`,
and `child_goal`.

`subagent_stop` fields include parent/child session IDs, role/status fields,
`child_summary`, and `duration_ms`.

Observers can use these hooks to model nested trajectories while keeping child
agent execution linked to the parent turn that spawned it.

## Payload Safety

Observer payloads are designed for telemetry consumers, not raw object access.
New consumers should use the sanitized API payloads:

- `pre_api_request.request`
- `post_api_request.response`
- `api_request_error.request`
- `api_request_error.error`

Sanitization converts provider objects to JSON-compatible structures, bounds
large payloads, redacts sensitive keys, and avoids exposing raw response
objects in sanitized fields.

Legacy compatibility fields such as `request_messages`, `conversation_history`,
and `assistant_message` may still be present for existing plugins. New
observability consumers should prefer the sanitized payloads.

## Performance

The default uninstrumented path should stay cheap. Expensive request/response
payload construction is gated behind `has_hook(...)`, so Hermes only builds
sanitized API telemetry payloads when at least one plugin registered the
relevant hook.

Plugin authors should preserve this property:

- Register only hooks the plugin actually consumes.
- Avoid deep-copying or re-sanitizing already sanitized payloads.
- Keep hook callbacks fast and fail-open.
- Offload network export or batch writes when practical.

## Writing An Observer Plugin

Minimal observer plugin:

```python
def register(ctx):
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)


def on_pre_api_request(**kwargs):
    start_llm_span(
        request_id=kwargs.get("api_request_id"),
        turn_id=kwargs.get("turn_id"),
        request=kwargs.get("request"),
        model=kwargs.get("model"),
    )


def on_post_api_request(**kwargs):
    finish_llm_span(
        request_id=kwargs.get("api_request_id"),
        response=kwargs.get("response"),
        usage=kwargs.get("usage"),
        duration=kwargs.get("api_duration"),
    )


def on_pre_tool_call(**kwargs):
    start_tool_span(
        call_id=kwargs.get("tool_call_id"),
        name=kwargs.get("tool_name"),
        args=kwargs.get("args"),
    )


def on_post_tool_call(**kwargs):
    finish_tool_span(
        call_id=kwargs.get("tool_call_id"),
        result=kwargs.get("result"),
        status=kwargs.get("status"),
        duration_ms=kwargs.get("duration_ms"),
    )
```

Use `session_id`, `turn_id`, `api_request_id`, and `tool_call_id` for span
correlation. Use subagent and approval hooks when the export format supports
nested agent work or security lifecycle events.

## Existing Consumers

The bundled Langfuse plugin demonstrates direct hook-based observability for
turns, provider requests, and tool calls.

The bundled NeMo Relay plugin maps the same generic observer contract to NeMo
Relay scopes, LLM spans, tool spans, marks, ATOF streams, and ATIF exports.
NeMo Relay-specific configuration and examples live in
[`plugins/observability/nemo_relay/README.md`](../../plugins/observability/nemo_relay/README.md).
