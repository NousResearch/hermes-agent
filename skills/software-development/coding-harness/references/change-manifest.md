# Change Manifest — schema and worked example

Every increment is a **falsifiable hypothesis**: it states up front what should improve and
what might break, and the next verification renders a verdict. This is the agent-facing form of
the agentic-harness-engineering Change Manifest (`references/HARNESS.md`). The
`harness_state.py` helper persists this structure to `.hermes/coding-harness/state.json`; this
file documents the shape so you can read/extend it directly when needed.

## State file schema

```json
{
  "manifest_version": "1.0",
  "goal": "Migrate auth from sessions to JWT; all auth tests green and login still works",
  "created_at": "<iso8601, stamped by the helper>",
  "increments": [
    {
      "change_id": "ch_001",
      "summary": "Add JWT issue/verify helpers in auth/jwt.py",
      "predicted_impact": {
        "expected": "new unit tests in tests/auth/test_jwt.py pass",
        "at_risk": "tests/auth/test_session.py may break if it shares config"
      },
      "verification": {
        "status": "pass",
        "note": "pytest tests/auth -q -> 14 passed, 0 failed",
        "verdict": "keep"
      }
    }
  ],
  "log": [
    "<iso8601> init: goal set",
    "<iso8601> add-increment ch_001: Add JWT issue/verify helpers",
    "<iso8601> verify ch_001: pass (keep)"
  ]
}
```

## Field meanings

| Field | Required | Meaning |
|-------|----------|---------|
| `goal` | ✅ | The falsifiable done-definition from SCOPE. |
| `increments[].change_id` | ✅ | Auto-assigned `ch_NNN`. Use it in `record-verification`. |
| `increments[].summary` | ✅ | What this increment changes (and where). |
| `predicted_impact.expected` | ✅ | **What should get better** — the success signal. |
| `predicted_impact.at_risk` | recommended | **What might break** — the regression to watch. |
| `verification.status` | set at VERIFY | `pending` / `pass` / `fail` / `partial`. |
| `verification.note` | recommended | The **actual command + outcome** (external proof). |
| `verification.verdict` | set at ATTRIBUTE | `keep` / `revert` / `partial`. |

## The three verdicts

- **keep** — prediction held (or the change is sound and verified). Advance to the next
  increment.
- **revert** — ineffective or caused a regression. Undo the change so the tree returns to
  known-good, and write the lesson into the log so you don't retry the same thing.
- **partial** — partially worked. Refine the change and re-verify; do not advance on a partial.

## Why falsifiable

Forcing a prediction before implementing, then checking it, does three things:
1. Surfaces regressions the moment they happen instead of at the end.
2. Stops the agent accreting changes it can't account for ("why is this here?").
3. Produces an audit trail (the `log`) that a fresh post-compaction context — or a human — can
   replay to understand exactly what was tried and what reality said about it.

A change you can't state a prediction for is a change you don't understand well enough to make.
Break it down further.
