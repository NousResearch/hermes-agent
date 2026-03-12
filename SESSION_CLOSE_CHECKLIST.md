# Session Close Checklist (Hermes Evolution Work)

Purpose: leave a clean, low-friction handoff for the next session.

## 1) Stabilize environment state

1. Ensure venv is active for any final test command:
   - `source .venv/bin/activate`
2. If running broader tests, set file descriptor limit:
   - `ulimit -n 1024`

## 2) Validate current checkpoint

Run at least targeted checks for touched areas:
- `python -m pytest tests/agent/test_self_correction.py -q`
- `python -m pytest tests/test_run_agent.py -q`
- `python -m pytest tests/test_model_tools.py tests/tools/test_mcp_tool.py -q`

If full-suite confidence is needed:
- `python -m pytest tests/ -q`

## 3) Preserve handoff clarity

1. Confirm `PHASE3_HANDOFF_NEXT_AGENT.md` reflects:
   - what is done
   - what remains
   - exact next deliverable
2. Keep scope boundaries explicit (do not mix unrelated diffs into handoff notes).

## 4) Record tree state

Capture and review:
- `git status --short`

If there are many unrelated changes, note this explicitly in handoff docs and isolate the file list that matters for the next objective.

## 5) Commit boundary readiness (if committing)

Preferred split:
- Commit A: feature/report code + tests
- Commit B: docs/handoff updates

If not committing yet, keep docs updated so next agent can proceed without reconstruction.

## 6) Next-session launch pointer

Use:
- `START_HERE_NEXT_SESSION.md`

And kickoff prompt:
- "Continue from PHASE3_HANDOFF_NEXT_AGENT.md and execute it exactly."
