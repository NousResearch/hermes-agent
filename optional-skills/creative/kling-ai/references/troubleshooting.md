# Hermes troubleshooting

- Run `hermes mcp list` and `hermes mcp test kling-ai` to inspect the native connection.
- Run `hermes mcp login kling-ai` for first authorization or intentional re-authentication.
- Run `/reload-mcp` after changing `~/.hermes/config.yaml`.
- On SSH or a remote gateway, use Hermes' documented redirect-URL paste flow or port forwarding. Never paste access or refresh tokens into chat.

## Upload failure

- Refresh the live schema and verify the upload tool and response fields.
- Reuse the upload result exactly in generation inputs and keep the same `taskTraceId`.
- Do not pass local paths, expired signed URLs, or undeclared input names to a remote generation tool.

## Task is still running

When the generation MCP App is mounted, let that App refresh the task internally and do not call `query_tasks` for the same submission from the model. If no App mounts, use headless `query_tasks` at the provider-permitted interval; if the user cancels or the turn cannot continue, return the current state and task number. A later explicit status request can call `query_tasks` once.

## Submission timeout or lost response

Do not call the generation tool again. If a `generationId` is known, query it once. Otherwise report that creation state is unknown; the current MCP cannot list account history or recover a task by `taskTraceId`. Obtain fresh confirmation before any new submission.

## Result URL expired

Query the preserved task number for a fresh URL. An expired URL does not mean that the generated work was deleted.
