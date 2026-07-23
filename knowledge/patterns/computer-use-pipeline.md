# Computer-Use Pipeline — Evidence & Architecture

## Summary

The `computer_use` tool in the Hermes codebase is a universal desktop-control tool that works with any tool-capable model (Anthropic, OpenAI, etc.) via standard OpenAI function-calling schema. It is implemented in `tools/computer_use/` and registered in `tools/computer_use_tool.py`.

## Pipeline Smoke Test

**Script:** `scripts/test_computer_use_pipeline.py`
**Workflow:** `.github/workflows/computer-use-pipeline.yml` (PR #59456 on `feat/computer-use-pipeline`)

### Run Result: 5/5 PASS (Linux, Jul 2026)

| Step              | Status | Detail                                                          |
|-------------------|--------|-----------------------------------------------------------------|
| import_hermes     | PASS   | tools package imported OK                                       |
| registered        | PASS   | Tool "computer_use" registered in registry, toolset "computer_use" |
| schema_valid      | PASS   | 20 params, 13 actions (capture, click, type, key, scroll, drag, …) |
| requirements_gate | PASS   | Linux → returns False (no cua-driver) — expected on non-macOS     |
| dispatch_graceful | PASS   | Handler ran, returned "backend unavailable" — graceful error path  |

**Evidence file:** `logs/computer-use-pipeline-run-001.log`

## Tool Invocation Architecture

The `computer_use` tool is invoked through the Hermes tool registry dispatch:

1. **Registration** — `tools/computer_use_tool.py` registers the tool with:
   - `registry.register(name="computer_use", handler=handle_computer_use, check_fn=check_computer_use_requirements, ...)`
2. **Dispatch** — `tools/registry.py::Registry.dispatch(name, args)`:
   - Looks up the entry by name
   - Calls `entry.handler(args, **kwargs)` — this is `handle_computer_use()`
   - Catches exceptions and returns `{"error": sanitized}`
3. **Handler** — `tools/computer_use/tool.py::handle_computer_use(args)`:
   - Parses `action` from args
   - Validates destructive actions (type blocking, key combo blocking, approval gate)
   - Acquires backend (`_get_backend()`) — returns cua-driver instance
   - Dispatches to `_dispatch(backend, action, args)` which calls `backend.xxx()` methods
4. **Backend** — `ComputerUseBackend` wrapper:
   - Methods: `capture()`, `click()`, `type_text()`, `key()`, `scroll()`, `drag()`, `focus_app()`, `set_value()`, `wait()`, `list_apps()`
   - On non-macOS: `_get_backend()` raises, caught by handler which returns graceful error

## Tool Call Invocation Pattern (Agent Runtime)

When an agent model requests the `computer_use` tool:

1. Agent emits a function/tool call with `name="computer_use"` and `arguments={...}`
2. `run_agent.py` routes the tool call through `registry.dispatch("computer_use", args)`
3. `handle_computer_use()` processes the action and returns either:
   - Text-only JSON string (for non-capture actions)
   - Multi-modal dict with `_multimodal: True` containing base64 screenshot image + text summary (for captures)
4. The result is shaped back into the model's message format (OpenAI `content` array for chat, `tool_result` block for Anthropic)

## Key Conclusions

1. **Tool is actively used** — Full dispatch chain exists from agent runtime → registry → handler → backend.
2. **Graceful degradation on Linux** — Backend acquisition failure returns `{"error": "computer_use backend unavailable: ..."}` with a hint to install cua-driver. No crash, no silent skip.
3. **Safety measures** — Blocked type patterns (shell commands), blocked key combos (destructive system shortcuts), approval gate for destructive actions (type, key, click with modifiers, drag).
4. **Multi-modal support** — Capture results include a base64 screenshot + generated SOM (Set-of-Mark) overlay index for the model to reference elements.
5. **13 supported actions**: capture, click, double_click, right_click, middle_click, drag, scroll, type, key, wait, list_apps, focus_app, set_value.

## Next Steps / Considerations

- The pipeline currently verifies tool registration and graceful failure only (no real macOS backend on CI).
- To test actual backend invocations, a macOS self-hosted runner or a mock backend fixture would be needed.
- Consider adding a mock-backend test mode to exercise the full `_dispatch()` code paths in CI.
