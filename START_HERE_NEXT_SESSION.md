# Start Here (Next Session Quickstart)

Run from repo root.

## Commands

```bash
source .venv/bin/activate
ulimit -n 1024
python -m pytest tests/agent/test_self_correction.py -q
python -m pytest tests/test_run_agent.py -q
python -m pytest tests/test_model_tools.py tests/tools/test_mcp_tool.py -q
```

Optional full-suite confidence pass:

```bash
python -m pytest tests/ -q
```

## Primary handoff doc

- `PHASE3_HANDOFF_NEXT_AGENT.md`

## Session close checklist

- `SESSION_CLOSE_CHECKLIST.md`

## Ready-to-paste kickoff prompt

Continue from `PHASE3_HANDOFF_NEXT_AGENT.md`. Treat Phase 3 core as implemented. Your goal is to deliver Retry Outcome Report v1 and get to a commit-ready checkpoint for Phase 4. Preserve prompt-cache invariants, keep retries bounded/flagged, and add tests proving report accuracy and high-risk retry compliance. Use `.venv`, set `ulimit -n 1024` before full-suite runs, and finish with explicit pass/fail against Phase 3 exit criteria.
