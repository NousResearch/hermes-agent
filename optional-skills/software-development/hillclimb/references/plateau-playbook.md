---
name: hillclimb
description: "Optimize one metric via frozen harness and decision log."
version: 1.0.0
author: "Emmanuel Ketcha (@ketchalegend) + Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [optimization, metrics, experiments, decision-log, benchmarking, iteration]
    category: software-development
    related_skills: [systematic-debugging, spike, test-driven-development, requesting-code-review, subagent-driven-development]
---

# Plateau Playbook

Protocol for when `decision_log.py stats` reports a plateau. A plateau is a signal that the current hypothesis class is exhausted; either pivot categories or document a proven dead end.

## Temporary Stall vs Genuine Dead End
- **Temporary stall**: recent attempts still improving but sub-threshold; pivot categories not exhausted -> pivot and continue.
- **Genuine dead end**: floor is provable and outside the loop (algorithmic lower bound, minimum network round trips, fixed external API latency). Write the proof in the final log row and stop.

## Worked Examples

### (a) Test-Suite Wall Clock (minimize, seconds)
- **Metric**: CI wall-clock time for full test suite.
- **Stalled hypothesis class**: per-test micro-optimizations (fixture scope tweaks, import hoisting).
- **Log showed**: two prior attempts each <1% improvement; combined they reached ~1.8% but no further gains.
- **Pivot taken**: change *how many tests run concurrently* instead of how fast one runs. Split slowest files into their own lane, run them in parallel with the rest.
- **Why pivot was right**: micro-optimizations hit diminishing returns; concurrency addresses the fundamental bottleneck of sequential execution. The pivot category change yields a measurable win.

### (b) Bundle Size (minimize, KB)
- **Metric**: final bundle size after webpack build.
- **Stalled hypothesis class**: bundler tree-shaking config tweaks.
- **Log showed**: plateau at 1.2 MB reduction; analyzer revealed the largest chunk was a dependency never touched by config changes.
- **Pivot taken**: re-read analyzer output with `read_file` to discover the true source, then attack the dependency graph (swap or drop a dependency) instead of config.
- **Why pivot was right**: mis-attribution wasted effort; attacking the actual heavy dependency yields larger, more reliable reduction.

### (c) Token Cost Per Turn (minimize, tokens)
- **Metric**: OpenAI token count per agent turn.
- **Stalled hypothesis class**: trimming prompt text.
- **Log showed**: micro-trims hit a floor because the fixed tool schema dominates total tokens.
- **Pivot taken**: change the *shape* of the interaction (one call instead of three, or defer a toolset) - a category change, not a size change.
- **Why pivot was right**: prompt size is secondary to interaction architecture; restructuring the workflow reduces total calls and thus tokens.

### (d) Flaky-Test Rate (minimize, %)
- **Metric**: percentage of tests that occasionally fail.
- **Stalled hypothesis class**: adding sleeps and retries to mitigate symptoms.
- **Log showed**: retries reduced visible failures but underlying race remained; plateau at 0.5%.
- **Pivot taken**: stop tuning the symptom and instrument the actual race (symptom-mitigation -> cause-detection). Add timing instrumentation to identify the race condition deterministically.
- **Why pivot was right**: symptom fixes only mask the problem; detecting the cause makes the flaky test reproducible and fixable.

## Applying the Playbook
When `stats` reports a plateau:
1. Review the last 5-10 log rows to identify the hypothesis class.
2. If the class is exhausted, choose a pivot from a different category.
3. Record the pivot decision as a `decision_log.py append` row (`tests: none`, `before: na`, `after: na`).
4. If no pivot categories remain, document the dead end: write a final log row with `tests: none`, `before: na`, `after: na`, and a detailed note proving the floor (algorithmic bound, external latency, etc.).
5. Continue the loop from the new baseline.