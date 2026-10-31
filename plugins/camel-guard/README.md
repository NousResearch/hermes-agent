# CaMeL Guard for current Hermes

This is the current-Hermes port of the CaMeL-style runtime guard from
NousResearch/hermes-agent PR #3987. It is now a plugin-only integration: it
does not patch `run_agent.py`, `agent/conversation_loop.py`,
`agent/tool_executor.py`, the CLI, configuration defaults, or Hermes's tool
message construction.

The plugin uses the generic lifecycle contract already present in Hermes:

- `pre_llm_call` captures only the current trusted user message.
- `post_tool_call` records that tool output entered the turn as untrusted data.
- `pre_tool_call` classifies trusted intent only when a later side effect needs
  authorization.
- Hermes still routes sequential and concurrent execution through its native
  executor and builds tool results with `make_tool_result_message()`.

## Enable it explicitly

The plugin and its traces are both off unless the operator opts in:

```yaml
plugins:
  enabled:
    - camel-guard
  entries:
    camel-guard:
      settings:
        mode: monitor       # off | monitor | enforce
        trace_enabled: false
```

`monitor` records what would be blocked but lets execution continue.
`enforce` returns a native `pre_tool_call` block directive. Invalid or missing
mode values resolve to `off`.

## Security behavior

The auxiliary classifier receives one JSON field: `trusted_user_message`. It
never receives tool output, conversation retrieval, memory, file contents, or
the untrusted source payload. If classification fails, the derived plan is
read-only: monitor reports the decision and enforce blocks the side effect.

The capability table covers current native mutation surfaces, including file
and command execution, background-process control, browser/computer use,
desktop UI controls, project and kanban state, messaging connectors, scheduled
jobs, delegation, skills, memory, and conservative unknown MCP operations.
Known read-only variants such as process polling, computer capture, kanban
listing, and connector reads remain ungated.

Trace persistence is separately opt-in and is ignored while mode is `off`.
Trace events are bounded and omit user text, tool arguments, and tool results;
they contain only the tool/capability decision, source tool names, and
classifier status.

## Maintainer-review mapping

This port addresses Teknium's July 12, 2026 review on PR #3987:

1. Trace persistence defaults off and cannot run with the guard off.
2. No custom tool-result path exists. Current executor hooks run before native
   dispatch, and Hermes retains its current untrusted-data wrapper and
   `name`/`tool_name` message contract.
3. The focused tests use real plugin discovery under a temporary
   `HERMES_HOME`, exercise direct plus sequential/concurrent dispatch, parse a
   structured host-LLM response, and cover classifier failure.

The executable information-flow invariants and their explicit non-claims are
documented in `docs/camel-guard-information-flow.md`.

The reproducible Codex-subscription benchmark is
`scripts/camel_guard_live_benchmark.py`; its current generated report is
`docs/camel-guard-live-benchmark.md` with machine-readable results under
`benchmarks/camel_guard/results/`.
