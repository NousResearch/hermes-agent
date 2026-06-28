---
name: hermes-agent-internals
description: "Use when onboarding or pairing on the agent heartbeat — explain one user message through run_conversation (prologue, finish_reason loop, epilogue) with line anchors in a local clone."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, run_agent, conversation_loop, architecture, onboarding]
    related_skills: [hermes-agent, git-worktrees-hermes]
---

# Hermes agent core (internals tour)

## Overview

Hermes turns each user message into one **turn** inside `run_conversation()`. The implementation is large, but the product shape is three acts: **prologue once**, **main loop until the model stops calling tools**, **epilogue once**. This skill keeps explanations at that conceptual level and points to verified line anchors in `agent/conversation_loop.py` and related modules.

## When to Use

- User asks where the loop lives, what `finish_reason` does, or how tools relate to the reply on screen.
- Pair-programming on `conversation_loop.py`, `turn_context.py`, or `turn_finalizer.py`.
- Module-style exercises (e.g. "walk one turn like a product builder").

**Don't use for:** CLI setup, config keys, or tool authoring — load `hermes-agent` instead.

## Plumbing above the loop

```
hermes (CLI) → HermesCLI (cli.py) → AIAgent (run_agent.py)
              → run_conversation() in agent/conversation_loop.py
```

`AIAgent.run_conversation` in `run_agent.py` forwards into `conversation_loop.run_conversation`; the heartbeat body lives there.

## One turn = three acts

| Act | Responsibility | Primary symbol |
|-----|----------------|----------------|
| Prologue | Sanitize user message, build/restore system prompt **once**, turn bookkeeping | `build_turn_context()` in `agent/turn_context.py` |
| Main loop | `messages` → `api_messages`, model call, branch on `finish_reason`, append trajectory | `while` in `conversation_loop.py` |
| Epilogue | Trajectory save, SQLite persist, cleanup, background memory/skill review | `finalize_turn()` in `agent/turn_finalizer.py` |

**Done when:** you can name the three acts, the trajectory roles (`user` / `assistant`+`tool_calls` / `tool`), and which `finish_reason` path shows the user a final answer.

## Trajectory vs API payload

- **`messages`** — durable transcript the loop appends to each lap.
- **`api_messages`** — per-lap provider view (system + copies; ephemeral injections on the current user turn only — not persisted).

## finish_reason (cheat sheet)

| Signal | User sees final reply now? | Loop |
|--------|---------------------------|------|
| `tool_calls` | No | `_execute_tool_calls` → append `role: tool` → `continue` |
| `stop` (no `tool_calls`) | Yes | Final text → `break` |
| `length` | Often no explicit "we compressed" | Compress/retry/continuation paths |

Tools: `run_agent.py` `_execute_tool_calls` (~5258) → `agent/tool_executor.py` (sequential vs concurrent when safe).

## Guided tour workflow

1. Resolve clone path (typical: `~/.hermes/hermes-agent/`).
2. Load `references/conversation-loop-one-turn.md` for line anchors.
3. Re-verify anchors with `search_files` on `def run_conversation`, `build_turn_context`, `finalize_turn` — **line numbers drift** on `main`.
4. Explain conceptually first; offer click-through ranges only if the user wants them.

**Done when:** the user can answer: "Model returned `tool_calls` — do I see a reply?" (No — next lap after tools.)

## Common Pitfalls

1. **Citing stale line numbers** — grep symbols before docs/PRs.
2. **Patching bundled `hermes-agent` SKILL.md** — extend this skill or `references/` instead (protected skill).
3. **Assuming epilogue is inline** — post-loop logic is `turn_finalizer.py` (handoff from `conversation_loop.py` ~4877+).
4. **Confusing `api_messages` with `messages`** — only `messages` is the trajectory SQLite replays.

## Verification Checklist

- [ ] Clone path confirmed; `agent/conversation_loop.py` exists
- [ ] Prologue, loop branch, and epilogue each tied to a file/function
- [ ] `tool_calls` vs `stop` behavior stated correctly for product builders
- [ ] Line anchors re-grepped if the user will open the editor

## Reference

- Line map: `references/conversation-loop-one-turn.md`