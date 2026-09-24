# Remote tool workflows

## Generation

1. Verify that Hermes has exactly one `Plugin-Hermes-kling-ai` connection at `https://kling.ai/mcp/plugin`, then read the current `tools/list`.
2. Create a UUIDv7 `taskTraceId`, call `who_am_i`, and use only the models, arguments, enums, defaults, and inputs declared for the selected live tool.
3. When local media is present, upload it with `file_upload` and pass the returned URL under the live input name. Never pass a local path directly to the remote generation tool.
4. Show the final billable settings and wait for explicit user confirmation.
5. Call the selected generation tool exactly once with `{model, arguments[], inputs[], rationale, taskTraceId}`, and preserve its `generationId`.
6. When Hermes mounts the generation MCP App, let that one App call headless `query_tasks` internally and update in place; do not query the same submission from the model.
7. If no App mounts, use headless `query_tasks` at the provider-permitted interval until terminal, cancelled, or the turn cannot continue; then return the current state, task number, text fallback, and at most one primary result link.

## Read-only operations

- Account and credits: call `query_membership_and_credits` once.
- Task status: call `query_tasks` once; a direct status request does not start long-running polling.
- Model capabilities: call `who_am_i` without creating a task.

## State changes

- Call `element_delete` only after the user explicitly confirms the deletion target.
- Call `logout` or switch accounts only when the user explicitly requests it. Immediately re-enter the native Hermes OAuth flow after logout, and stop other Kling calls until authorization completes.

Never retry a generation automatically.
