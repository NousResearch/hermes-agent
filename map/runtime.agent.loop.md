---
id: runtime.agent.loop
kind: process
universe: runtime
name: Agent Loop
summary: Core conversation loop in run_agent.py.
aliases: [agent loop, conversation loop]
tags: [runtime, core]
shape: process
steps:
  - id: step.model.call
    summary: Call client.chat.completions.create with messages and tool_schemas.
  - id: step.tool.dispatch
    summary: For each tool call, handle_function_call and append result.
  - id: step.budget.check
    summary: Check api_call_count and iteration_budget. Grace call if allowed.
  - id: step.response.return
    summary: Return content if no tool calls remain.
entrypoints: [step.model.call]
produces: []
consumes: []
---

# Agent Loop

Core conversation loop in run_agent.py.

## Steps

1. **step.model.call**: Call client.chat.completions.create with messages and tool_schemas.
2. **step.tool.dispatch**: For each tool call, handle_function_call and append result.
3. **step.budget.check**: Check api_call_count and iteration_budget. Grace call if allowed.
4. **step.response.return**: Return content if no tool calls remain.

## Entrypoints

- `step.model.call`

## Artifacts

None produced.
