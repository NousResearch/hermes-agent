---
name: langfuse-sdk-upgrade
description: Upgrade Langfuse SDKs from older versions to the latest. Use when migrating Python SDK v2/v3 to v4, or JS/TS SDK v3/v4 to v5.
---

# Langfuse SDK Upgrade Guide

Assist users in upgrading their Langfuse SDK to the latest version. The Python and JS/TS SDKs share the same architectural changes but differ in syntax.

## When to Use

- User asks to upgrade/migrate their Langfuse SDK
- User is on an older SDK version and encounters deprecated APIs
- User wants to adopt the latest Langfuse features

## Migration Docs

Always fetch the latest migration guide before starting — these pages are the source of truth:

- **Python (v3 → v4):** https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4
- **JS/TS (v4 → v5):** https://langfuse.com/docs/observability/sdk/upgrade-path/js-v4-to-v5

Fetch the relevant page as markdown before implementing any changes:

```bash
curl -s "https://langfuse.com/docs/observability/sdk/upgrade-path/python-v3-to-v4.md"
curl -s "https://langfuse.com/docs/observability/sdk/upgrade-path/js-v4-to-v5.md"
```

## Upgrade Checklist

Work through each item in order. Skip items that don't apply to the user's codebase.

### Both SDKs

- [ ] **Update the SDK package** to the latest version
- [ ] **Audit span filtering**: Non-LLM spans (HTTP, DB, queues) no longer export by default. If the user relied on these, configure a custom `should_export_span` / `shouldExportSpan` filter
- [ ] **Replace `update_current_trace()` / `updateActiveTrace()`**: Split into three calls:
  - `propagate_attributes()` / `propagateAttributes()` for correlating attributes (`user_id`, `session_id`, `tags`, `metadata`, `trace_name`)
  - `set_current_trace_io()` / `setActiveTraceIO()` for input/output (deprecated — prefer setting I/O on root observation directly)
  - `set_current_trace_as_public()` / `setActiveTraceAsPublic()` for public flag
- [ ] **Replace `.update_trace()` / `.updateTrace()`** on observation objects (same decomposition as above)
- [ ] **Update API namespace references**: `observations_v_2` / `observationsV2` → `observations`, `score_v_2` / `scoreV2` → `scores`, `metrics_v_2` / `metricsV2` → `metrics`. Legacy v1 APIs moved to `api.legacy.*`
- [ ] **Validate metadata format**: Must be `dict[str, str]` / `Record<string, string>` with values ≤200 characters
- [ ] **Move `release` and `environment`** from code parameters to environment variables (`LANGFUSE_RELEASE`, `LANGFUSE_TRACING_ENVIRONMENT`)
- [ ] **Enable debug logging** during migration to catch issues (`debug=True` in Python, `LANGFUSE_DEBUG="true"` in JS/TS)
- [ ] **Test trace hierarchies** to verify no spans are unexpectedly dropped

### Python-specific

- [ ] **Replace `start_span()` / `start_generation()`** with `start_observation()` (use `as_type="generation"` for generations)
- [ ] **Replace `start_as_current_span()` / `start_as_current_generation()`** with `start_as_current_observation()`
- [ ] **Replace dataset `item.run()`** with `dataset.run_experiment(name=..., task=...)`
- [ ] **Remove `CallbackHandler(update_trace=...)`** parameter — use `propagate_attributes()` wrapper instead
- [ ] **Upgrade to Pydantic v2** — the SDK now requires it. Use `pydantic.v1` compatibility shim if migrating gradually
- [ ] **Update removed types**: `TraceMetadata`, `ObservationParams` removed from `langfuse.types`. Import `MapValue`, `ModelUsage`, `PromptClient` from `langfuse.model`

### JS/TS-specific

- [ ] **Update LangChain `CallbackHandler`** — `traceMetadata` now requires string values; internal behavior uses `propagateAttributes()` instead of direct trace updates
- [ ] **Update OpenAI integration** — `traceMethod` wrapper now uses `propagateAttributes()` internally; wrap entire execution in `propagateAttributes()` if relying on parent attribute inheritance

## Key API Changes Reference

### Correlating attributes (both SDKs)

**Before:**
```python
# Python
langfuse.update_current_trace(name="trace-name", user_id="user-123", session_id="session-abc", tags=["tag1"])
```
```typescript
// JS/TS
updateActiveTrace({ name: "trace-name", userId: "user-123", sessionId: "session-456", tags: ["prod"] });
```

**After:**
```python
# Python
from langfuse import propagate_attributes

with propagate_attributes(trace_name="trace-name", user_id="user-123", session_id="session-abc", tags=["tag1"]):
    result = call_llm("hello")
```
```typescript
// JS/TS
import { propagateAttributes } from "langfuse";

await propagateAttributes(
  { traceName: "trace-name", userId: "user-123", sessionId: "session-456", tags: ["prod"] },
  async () => { /* traced code */ }
);
```

### Span/Generation creation (Python)

**Before:**
```python
langfuse.start_span(name="x")
langfuse.start_generation(name="x", model="gpt-4")
```

**After:**
```python
langfuse.start_observation(name="x")
langfuse.start_observation(name="x", as_type="generation", model="gpt-4")
```

### Dataset experiments (Python)

**Before:**
```python
for item in dataset.items:
    with item.run(run_name="my-run") as span:
        result = my_llm(item.input)
        span.update(output=result)
```

**After:**
```python
def my_task(*, item, **kwargs):
    return my_llm(item.input)

dataset.run_experiment(name="my-run", task=my_task)
```

### Span filtering (both SDKs)

To restore pre-upgrade "export all" behavior:

```python
# Python
langfuse = Langfuse(should_export_span=lambda span: True)
```
```typescript
// JS/TS
const spanProcessor = new LangfuseSpanProcessor({ shouldExportSpan: () => true });
```

To extend defaults with custom scopes:

```python
# Python
from langfuse.span_filter import is_default_export_span

langfuse = Langfuse(
    should_export_span=lambda span: (
        is_default_export_span(span)
        or span.instrumentation_scope.name.startswith("my_framework")
    )
)
```
```typescript
// JS/TS
import { isDefaultExportSpan } from "@langfuse/otel";

shouldExportSpan: ({ otelSpan }) =>
  isDefaultExportSpan(otelSpan) || otelSpan.instrumentationScope.name.startsWith("my_framework")
```

## Common Pitfalls

| Pitfall | Impact | Fix |
| --- | --- | --- |
| Dropping intermediate spans via filtering | Breaks trace trees — child spans become orphaned | Use `is_default_export_span` as base and only add/remove specific scopes |
| Metadata with non-string values | Values silently coerced or dropped | Ensure all metadata values are strings ≤200 characters |
| Setting attributes outside `propagate_attributes()` callback | Attributes don't attach to observations | Wrap all traced code inside the callback |
| Using deprecated `set_current_trace_io()` for new code | Will be removed in future versions | Set input/output directly on the root observation |
| Forgetting Pydantic v2 upgrade (Python) | Import errors or runtime failures | Upgrade Pydantic or use `pydantic.v1` shim |
| `release`/`environment` still passed as parameters | Silently ignored | Use `LANGFUSE_RELEASE` and `LANGFUSE_TRACING_ENVIRONMENT` env vars |
| LangChain/OpenAI attribute propagation direction changed | Attributes propagate downward only, not upward to parent traces | Wrap outer call in `propagate_attributes()` |

## Best Practices

1. **Always fetch the migration docs first** — they are the canonical source and may have been updated since this guide was written
2. **Enable debug logging during migration** to surface dropped spans and trace hierarchy issues
3. **Use `propagate_attributes()` as the primary mechanism** for setting trace-level correlating attributes
4. **Set input/output on root observations directly** rather than using deprecated trace-level setters
5. **Compose custom span filters** with `is_default_export_span` / `isDefaultExportSpan` to extend defaults rather than replacing them entirely
6. **Test thoroughly** — run the application with debug logging, check the Langfuse UI for missing or orphaned spans, verify metadata appears correctly
7. **Migrate incrementally** — upgrade the SDK first, fix breaking changes, then adopt new patterns
