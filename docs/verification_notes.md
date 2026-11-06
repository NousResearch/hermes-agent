# Week 1 verification (B1)

Pinned commit: <paste from docs\pinned_commit.txt>

## 1. Trajectory module
- Save/load file: `agent/trajectory.py` (2,573 bytes)
- Compressor: `trajectory_compressor.py` at repo ROOT (46,468 bytes)
- README says both are under `agent/` — README is wrong about `trajectory_compressor.py`.

## 2. Review/nudge trigger
- Trigger file: `agent/turn_context.py`
- Trigger function: `_tick_memory_nudge(agent)` — returns True when review should fire.
- Call site: `should_review_memory = _tick_memory_nudge(agent)` in turn_context.
- Nudge intervals set in: `agent/agent_init.py`
  - `_memory_nudge_interval = 10`
  - `_skill_nudge_interval = 10`

## 3. What the trigger calls
- `agent/background_review.py` — `spawn_background_review_thread()`
  - Forks a new AIAgent, runs the review prompt, reports usage to parent.
  - Runs a second full agent turn in the background.
  - Not the chat_completions path.
- `agent/review_engine.py` — `/review` command engine (separate from automatic review).
- `agent/review_idle_queue.py` — defers reviews when the local runtime is busy.

## 4. Does memory_manager.py call review directly?
- No. `findstr "review nudge trigger"` on `agent/memory_manager.py` and `agent/memory_provider.py` → zero matches.
- The memory layer is invoked BY the review fork, not the other way around.

## 5. Subagents: shared process or separate?
- Not yet verified. Needs a separate probe (later).

## 6. jcode retrieval/extraction
- Not yet read. Pending (Team A territory, B1 not blocked).

## 7. AVO paper
- B2 owns.

## Open questions for B2/Team A
- Does `background_review.py` call `memory_manager` on the fork's behalf? (Likely yes via memory tools whitelist.)
- Confirm `_tick_memory_nudge` fires per user turn or per assistant turn.