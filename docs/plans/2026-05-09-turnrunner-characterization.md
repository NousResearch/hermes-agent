# TurnRunner Characterization Seam

## Current seam

`run_agent.TurnRunner` is the smallest runtime interface currently implicit in
`AIAgent`: one foreground user turn enters through `run_conversation(...)` and
returns the existing result mapping. The seam deliberately does not include
agent construction, provider configuration, persistence ownership,
interrupt/steer control, or transport-specific streaming. Those concerns still
belong to the delivery surfaces until their behavior is characterized.

The characterized one-turn path is:

1. Copy prior `conversation_history`.
2. Append the current user message.
3. Call the configured model until a final assistant response is available.
4. Append the assistant response to the returned `messages`.
5. Return the existing result dict, including `final_response`, `messages`,
   `api_calls`, `completed`, `interrupted`, and token/cost fields.

## Future adapters

CLI adapter:
Wrap the prompt loop in `cli.py` and submit each accepted user input to a
`TurnRunner`. The CLI should keep owning keyboard input, slash commands,
display, local cwd, and session lifecycle. `AIAgent.chat(...)` remains a
convenience wrapper over the same one-turn seam.

Gateway adapter:
Wrap platform events in `gateway/run.py` into `user_message`,
`conversation_history`, `task_id`, and optional `persist_user_message` before
calling a `TurnRunner`. Platform delivery, batching, auth, thread/session
routing, and progress callbacks stay in the gateway layer.

TUI adapter:
Map JSON-RPC `prompt.submit` requests in `tui_gateway/server.py` onto the
`TurnRunner` call. The TUI gateway should continue owning session records,
JSON-RPC event emission, approval prompts, resize/display state, and
stream/progress callback translation.

ACP adapter:
Map ACP session prompts in `acp_adapter/session.py` and `acp_adapter/server.py`
onto the `TurnRunner` call. ACP protocol state, editor-facing permissions,
request cancellation, and event formatting remain in the ACP layer.

## Stop line

Do not make delivery surfaces depend on this protocol until each adapter path
has its own characterization tests. If the next slice requires changing CLI,
gateway, TUI, and ACP in one patch, stop and split the extraction into smaller
adapter-specific work.
