# Ideas tools via Cursor `hermes-tools` MCP

Condensed from a session where `ideas_boards()` worked but `ideas_list` always returned the empty default board, and the agent fell back to `hermes ideas …` shell without explaining why.

## Symptom

| Call | Expected | Broken MCP passthrough |
|------|----------|-------------------------|
| `ideas_boards()` | Boards + counts | Works (no args) |
| `ideas_list(all_boards=true)` | All boards with ideas | Only **default**, 0 ideas |
| `ideas_list(board="roguelike-td")` | That board's ideas | Same empty default |

User data is fine; the handler never receives `all_boards` / `board`.

## Root cause

`agent/transports/hermes_tools_mcp_server.py` registers tools with `def _dispatch(**kwargs)` and `mcp.add_tool(...)` **without** passing Hermes' JSON `parameters` schema. FastMCP infers a single required field `kwargs`. Clients then send:

```json
{"kwargs": {"all_boards": true}}
```

`handle_function_call` receives `{"kwargs": {"all_boards": true}}` instead of `{"all_boards": true}`, so `ideas_list` ignores filters.

## Agent behavior (user-corrected)

1. Prefer `ideas_*` MCP/native tools over `hermes ideas …` CLI.
2. If results look wrong, **do not silently shell** — state that MCP args may be nested wrong, retry flat top-level args if the client schema allows, then point at the server fix.
3. `ideas_boards()` + empty `ideas_list` is a strong signal for this bug, not "user has no ideas."

## Server fix (fork maintainers)

In `_make_handler` / `_dispatch`:

- Pass `params_schema` from `get_tool_definitions()` into `mcp.add_tool(..., input_schema=...)` (or `parameters_schema` on older MCP SDK).
- Unwrap before dispatch: if `kwargs` has exactly one key `"kwargs"` whose value is a dict, use that inner dict as the tool arguments.

Regression tests: `tests/agent/transports/test_cursor_mcp_servers.py` (extend when schema/passthrough is fixed).

## Correct invocations (after fix)

```json
{"all_boards": true}
{"board": "roguelike-td"}
{"idea_id": "i_50bf4ffcb7984f03"}
```

Same pattern affects other `hermes-tools` exports (`skill_view`, `skills_list`, browser tools) until schemas are wired.
