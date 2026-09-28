# Delegation

`delegate_task` spawns subagents to work in parallel or in the background. Use it to split large work; don't use it for anything that needs to talk to people, remember, or recur.

## Calling it

- One task: `goal` (self-contained). Batch: `tasks=[{goal, context?, role?}, ...]` — when `tasks` is present, top-level `goal`/`context`/`role` are ignored; don't mix modes.
- Native delegation is synchronous unless you request `background=true`. Background results arrive through the native completion queue; follow the tool result and handoff instructions.

## What a subagent is

- **A blank conversation, not a blank identity.** It carries the workspace personality and knows the responsibilities roster, but it has no access to this conversation, your memory contents, or anything you didn't put in `goal`/`context`. Write the goal as if briefing a stranger: paths, error text, constraints, expected output.
- **Same computer as you.** Subagents share your files, ports, and processes — parallel subagents can collide on the same file, and so can you. Partition the work by file or directory. If a result carries a note that a subagent modified files you'd already read, re-read them before editing.
- **Narrower tools.** Every subagent loses: asking the user anything, `memory`, `send_message`, and `schedule`. A `leaf` (default) also loses `delegate_task` itself, but retains `execute_code` for bounded programmatic tool calling.
- **Nesting is off by default.** An `orchestrator` child can only spawn its own workers if the workspace raised the depth limit; otherwise it silently behaves as a leaf. And an orchestrator's own `delegate_task` calls run synchronously — only your top-level calls background.

## Batches and results

Native delegation controls concurrency, nested delegation, completion grouping and interruption. Read the actual tool schema for the active limits. Background delegation is process-local and does not survive restart. Partition files between workers; they share your computer. Verify side effects using checkable paths, URLs or other evidence before reporting them as done.

## When not to delegate

- The task needs to ask someone a question, message a channel, save a memory, or set up recurring work — subagents can't; do it yourself.
- The work must survive beyond this turn on a rhythm — that's a Schedule.
- A turn has a finite tool-call budget. Split large work into delegation, a Schedule, or bounded code execution before exhausting it.
