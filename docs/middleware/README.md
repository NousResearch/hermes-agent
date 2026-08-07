# Hermes Middleware

Hermes middleware is the behavior-changing companion to observer hooks.
Observer hooks report what happened. Middleware can change what happens by
rewriting a request before execution or by wrapping the execution callback
itself.

This contract is intentionally backend-neutral. A plugin can use it for local
policy, request shaping, tracing, adaptive routing, cache control, sandbox
selection, or handoff to runtimes such as NeMo Relay without changing Hermes'
planner, model provider adapters, tool registry, memory, or CLI UX.

With middleware enabled, plugins can:

- Rewrite LLM provider request kwargs before Hermes calls the provider.
- Rewrite tool arguments before guardrails, approval checks, hooks, and tool
  execution see them.
- Wrap the actual LLM execution callback while preserving Hermes retry,
  streaming, interrupt, and hook behavior.
- Wrap the actual tool execution callback while preserving Hermes guardrails,
  approval, post-tool hooks, and tool-result transformation.

## Contract

Plugins register middleware from `register(ctx)`:

```python
def register(ctx):
    ctx.register_middleware("llm_request", on_llm_request)
    ctx.register_middleware("llm_execution", on_llm_execution)
    ctx.register_middleware("tool_request", on_tool_request)
    ctx.register_middleware("tool_execution", on_tool_execution)
```

Every middleware callback receives:

- `telemetry_schema_version`: currently `hermes.observer.v1`
- `middleware_schema_version`: currently `hermes.middleware.v1`
- Runtime context such as `session_id`, `task_id`, `turn_id`,
  `api_request_id`, `provider`, `model`, `api_mode`, `tool_name`, and
  `tool_call_id` when applicable.

Supported middleware kinds:

| Kind | Payload | Return shape | Purpose |
| --- | --- | --- | --- |
| `llm_request` | `request`, `original_request` | `{"request": {...}}` | Replace effective provider kwargs before provider execution. |
| `tool_request` | `tool_name`, `args`, `original_args` | `{"args": {...}}` | Replace effective tool args before hooks, guardrails, approvals, and execution. |
| `llm_execution` | `request`, `original_request`, `next_call` | Any provider response | Wrap or replace the actual provider call. |
| `tool_execution` | `tool_name`, `args`, `original_args`, `next_call` | Any tool result | Wrap or replace the actual tool call. |

Request middleware can return optional trace fields:

```python
return {
    "request": updated_request,
    "source": "my-plugin",
    "reason": "selected fallback model",
}
```

Hermes stores those trace entries in later observer hook payloads as
`middleware_trace`.

Execution middleware receives a `next_call` callback. Call it to continue the
chain:

```python
def on_tool_execution(**kwargs):
    result = kwargs["next_call"](kwargs["args"])
    return result
```

If multiple plugins register the same execution middleware kind, Hermes runs
them as a nested chain in registration order. Middleware failures are fail-open:
Hermes logs a warning and continues with the next middleware or the base
runtime path.

## Execution Order

### LLM Calls

For each provider request, Hermes applies middleware in this order:

1. Build provider kwargs from the current conversation.
2. Apply `llm_request` middleware.
3. Emit `pre_api_request` observer hooks with the effective request.
4. Run provider execution through `llm_execution` middleware.
5. Emit `post_api_request` or `api_request_error` observer hooks.

Request middleware sees the full provider kwargs, including `messages` or
Responses API `input`, model settings, tool definitions, stream options, and
provider-specific options. Execution middleware receives the same effective
request plus `next_call`.

### Tool Calls

For each tool call, Hermes applies middleware in this order:

1. Parse and coerce model-provided tool arguments.
2. Apply `tool_request` middleware.
3. Run the normal Hermes pre-execution path against the effective arguments:
   tool availability checks, observer block directives, guardrails, and
   approval checks.
4. Run tool execution through `tool_execution` middleware.
5. Emit `post_tool_call` observer hooks.
6. Apply `transform_tool_result` hooks before the result is appended back into
   conversation context.

Tool request middleware runs before approval checks. Use it carefully: a
rewritten path, command, or URL is the value downstream policy will evaluate.

## Enablement

Middleware only runs for enabled plugins. For a bundled plugin:

```bash
hermes plugins enable <plugin-name>
```

For isolated local testing, use one `HERMES_HOME` for plugin enablement and the
agent run:

```bash
export HERMES_HOME=/tmp/hermes-middleware-test
mkdir -p "$HERMES_HOME"
hermes plugins enable <plugin-name>
hermes chat --query 'Reply exactly ok'
```

For source checkouts, prefer the source command so the runtime sees plugins and
middleware from the working tree:

```bash
uv sync
uv run hermes plugins enable <plugin-name>
uv run hermes chat --query 'Reply exactly ok'
```

## Generic Plugin Examples

The examples below are intentionally small. They show the middleware contract
shape without depending on NeMo Relay.

### LLM Request Middleware

This plugin tags provider requests and records a middleware trace entry:

```python
def register(ctx):
    ctx.register_middleware("llm_request", tag_llm_request)


def tag_llm_request(**kwargs):
    request = dict(kwargs["request"])
    extra_body = dict(request.get("extra_body") or {})
    extra_body.setdefault("metadata", {})["hermes_middleware_demo"] = True
    request["extra_body"] = extra_body
    return {
        "request": request,
        "source": "middleware-demo",
        "reason": "tagged provider request",
    }
```

The effective request is passed to `pre_api_request`, provider execution, and
`post_api_request`.

### Tool Request Middleware

This plugin constrains `terminal` calls to a known working directory:

```python
def register(ctx):
    ctx.register_middleware("tool_request", normalize_terminal_workdir)


def normalize_terminal_workdir(**kwargs):
    if kwargs.get("tool_name") != "terminal":
        return None
    args = dict(kwargs["args"])
    args.setdefault("workdir", "/tmp/hermes-middleware-demo")
    return {
        "args": args,
        "source": "middleware-demo",
        "reason": "defaulted terminal workdir",
    }
```

Because this runs before hooks and approvals, downstream telemetry and policy
observe the rewritten `workdir`.

### LLM Execution Middleware

This plugin wraps the provider call and preserves the raw provider response:

```python
import time


def register(ctx):
    ctx.register_middleware("llm_execution", time_llm_execution)


def time_llm_execution(**kwargs):
    started = time.monotonic()
    response = kwargs["next_call"](kwargs["request"])
    elapsed_ms = int((time.monotonic() - started) * 1000)
    print(f"llm_execution elapsed_ms={elapsed_ms}")
    return response
```

Return the same response shape Hermes expects from the provider adapter. Do not
wrap the response in a plugin-specific envelope unless the rest of the runtime
expects that envelope.

### Tool Execution Middleware

This plugin wraps tool execution while preserving the tool result:

```python
def register(ctx):
    ctx.register_middleware("tool_execution", annotate_tool_execution)


def annotate_tool_execution(**kwargs):
    result = kwargs["next_call"](kwargs["args"])
    # Metrics, logging, or external routing can happen here.
    return result
```

Execution middleware may call `next_call(modified_args)` to pass a changed
payload to later middleware and the base tool dispatcher.

Plugin-specific examples should live with the plugin that owns the behavior.
For NeMo Relay adaptive execution middleware, see
[`plugins/observability/nemo_relay/README.md`](../../plugins/observability/nemo_relay/README.md).

## Safety Notes

- Middleware should be deterministic for the same input unless it is explicitly
  routing to a dynamic external system.
- Request middleware should return complete replacement payloads, not partial
  patches.
- Execution middleware should call `next_call(...)` exactly once unless it is
  intentionally short-circuiting execution.
- If execution middleware raises before calling `next_call(...)`, Hermes treats
  that as middleware failure and continues with the remaining middleware chain
  and base execution.
- If execution middleware calls `next_call(...)` successfully and then raises
  during post-processing, Hermes preserves the downstream result and does not
  run the provider or tool a second time.
- If downstream provider or tool execution fails, middleware may let that error
  propagate or translate it deliberately. Hermes does not convert downstream
  failure into a successful `None` result.
- Tool request middleware runs before approvals. If it mutates file paths,
  commands, URLs, or arguments, the mutated values are what guardrails and
  approvals evaluate.
- Observer hooks remain the right place for read-only telemetry. Use middleware
  only when a plugin needs to alter or wrap behavior.
