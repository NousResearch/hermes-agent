# live-time

Inject the **live current time** into every LLM call so the agent always
knows "now" — even in conversations that span multiple days.

## Why

Hermes stamps `Conversation started: ...` once at session creation and never
refreshes it. In long or cross-day conversations the model has no reliable
sense of the current date, weekday, or time of day unless it burns a tool
call to find out. This plugin closes that gap
([#10421](https://github.com/NousResearch/hermes-agent/issues/10421)).

## What it does

On every LLM call the plugin injects a small context block on the
user-message side of the request:

```
[LIVE-TIME] Now: 2026-08-13 16:30:00 (Weekday 4/7, 四), TZ Asia/Shanghai.
Injected by live-time plugin at THIS LLM call's moment. Use THIS as the
authoritative current time for any today/now/elapsed/date judgment. ...
```

Design properties:

- **Ephemeral** — the block is appended to the request, never written into
  the cached system prompt, so prompt caching is unaffected.
- **Timezone aware** — resolves the timezone in this order:
  1. `HERMES_TIMEZONE` environment variable
  2. `timezone` key in `<HERMES_HOME | ~/.hermes>/config.yaml`
  3. local system timezone (reported as `TZ UTC±H`)
- **Zero dependencies** — stdlib only, no internal Hermes imports (so the
  plugin survives upstream refactors).

## Install

Copy the `live_time` directory into your Hermes `plugins/` directory
(alongside the built-in plugins) and restart Hermes. Verify by asking the
agent "what time is it right now?" — it should answer accurately without
calling any tool.

## How it works

```python
def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
```

`_on_pre_llm_call` returns `{"context": "..."}`; the runtime appends that
context to the user message before the LLM call (see
`agent/turn_context.py`).

## Tests

```bash
pytest tests/plugins/live_time/test_live_time.py -v
```
