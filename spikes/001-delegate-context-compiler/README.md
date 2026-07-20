# Delegation Context Compiler Spike

## Verdict: VALIDATED

A deterministic, opt-in recent projection can reduce naive parent-history context, retain useful recent constraints, exclude disallowed message categories, and obey a hard character budget without an LLM call.

This verdict supports proceeding to B1 **only as a default-off operator configuration**. It does not support replacing explicit context or automatically copying parent history. The counterexample proves that facts outside the recent window are lost.

## Run

```bash
python3 spikes/001-delegate-context-compiler/probe.py
```

The probe is pure Python standard library. It imports no Hermes production implementation, calls no model or network, and writes no production state.

Verification used:

```bash
python3 -m py_compile spikes/001-delegate-context-compiler/probe.py
python3 spikes/001-delegate-context-compiler/probe.py > /tmp/delegate-context-spike-1.json
python3 spikes/001-delegate-context-compiler/probe.py > /tmp/delegate-context-spike-2.json
cmp /tmp/delegate-context-spike-1.json /tmp/delegate-context-spike-2.json
```

Both runs were byte-identical.

## Results

| Fixture | Naive chars | Explicit chars | Projection chars | Reduction | Required recall | Improved vs explicit | Forbidden leaks |
|---|---:|---:|---:|---:|---:|---|---:|
| role filtering + recent constraints | 267 | 46 | 186 | 30.34% | 2/2 | yes | 0 |
| forbidden roles/non-text payload | 227 | 44 | 102 | 55.07% | 1/1 | yes | 0 |
| explicit context remains default | 187 | 94 | 94 | 49.73% | 1/1 | no (already explicit) | 0 |
| 20k+ history and hard budget | 22,117 | 51 | 1,200 | 94.57% | 2/2 | yes | 0 |
| exact OOB wrapper extraction | 206 | 34 | 92 | 55.34% | 2/2 | yes | 0 |
| old relevant fact outside window | 255 | 54 | 115 | 54.90% | 0/1 | no | 0 |
| explicit context budget priority | 1,810 | 375 | 600 | 66.85% | 3/3 | yes | 0 |

Aggregate criteria from the probe:

```json
{
  "all_budget_task_final_deterministic": true,
  "all_projection_fixtures_reduce_vs_naive": true,
  "harmful_losses_vs_explicit": 0,
  "useful_recall_improvements": 5,
  "zero_forbidden_leaks": true
}
```

All seven outputs stayed within budget, ended with the exact task block, and were deterministic. The long-history fixture hit the 1,200-character limit exactly while retaining both head and tail markers.

## What was validated

1. Only recent textual `user` and `assistant` messages enter projection.
2. `system`, `developer`, `tool`, and non-text attachment payloads are excluded.
3. Explicit mode does not read parent history and remains the default behavior.
4. The final task block is reserved first and remains last under truncation.
5. Exact Hermes OOB wrappers can retain the user body with an `[OOB user]` marker without copying wrapper instructions.
6. Explicit context receives budget before projection; the mixed pressure fixture retained both explicit markers and added the projected marker.
7. Projection adds useful task constraints in five fixtures compared with explicit-only context.

## Limits and recommendation

- Recent projection loses relevant facts older than its window (`OLD_REQUIRED` recall was 0/1). It must remain opt-in and must not be described as complete parent context.
- Static fixtures do not prove security. The probe intentionally does not attempt heuristic secret detection; safety comes from excluding privileged/tool/non-text categories. Callers remain responsible for explicit context content.
- Character count is a deterministic proxy, not provider token count.
- The spike does not test model answer quality; required-marker recall is only a deterministic feasibility signal.

**Recommendation:** proceed to B1 with `context_mode: explicit` as the unchanged default and `recent_projection` enabled only by operator configuration. Preserve explicit context priority, role filtering, deterministic budget, exact task-last framing, and the documented recent-window limitation.

No production files were imported or modified. The spike remains uncommitted pending user review.
