# Hermes MCP workflow

Hermes registers the server's tools as `mcp__Plugin_Hermes_kling_ai__<tool>` after sanitizing the server key.

1. Inspect the active MCP tools and their current schemas; do not infer model names or enums from this file.
2. For media inputs, call the discovered upload tool first and preserve its returned reference.
3. After the user confirms final billable settings, call the selected generation tool exactly once.
4. Preserve `generationId` and `taskTraceId`. On ambiguity, call the discovered `query_tasks` tool rather than resubmitting.
5. Do not enable parallel calls for this server: generation submission and related reads may share account/task state.
