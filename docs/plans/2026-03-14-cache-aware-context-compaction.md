# Cache-Aware Context Compaction Design Note

> For Hermes: this note is a design/implementation sketch for revisiting prune-first compaction without optimizing token spend at the expense of prompt-cache stability.

Goal: reduce compression cost while keeping cache-break frequency as low as possible.

Architecture: keep Hermes' current invariant that conversation history is only mutated during context compression, then make prune-first compaction conservative enough that it only short-circuits when it buys meaningful runway. If pruning only gets us barely below threshold, fall through to the existing summary compaction immediately.

Tech Stack: `agent/context_compressor.py`, existing `call_llm()`-based summary path, pytest coverage in `tests/agent/test_context_compressor.py`.

---

## 1. Baseline behavior on current main

Today Hermes behaves like this:

1. Prompt crosses the compression threshold.
2. We mutate transcript history once by summarizing the middle region with an LLM.
3. We preserve role alternation and tool-call/tool-result integrity.
4. We continue the conversation from the compressed transcript.

This is expensive in two ways:
- an auxiliary summary call is often required
- the entire compressed middle region is rewritten even when the real problem was just a few huge old tool outputs

But it has one strong cache property:
- it tends to reclaim a lot of headroom per compression event, so the next compression is usually farther away

---

## 2. Why naive prune-first compaction is not enough

A naive prune-first policy says:
- prune old tool outputs
- if prompt is now below threshold, stop

This improves per-event token cost, but it can hurt cache economics:
- prune-only may reclaim less headroom than full compaction
- smaller headroom means the next compression may happen sooner
- each compression event is still a cache-breaking transcript mutation

So there is a real failure mode:
- fewer tokens per compression
- more compression events overall
- worse cache break cadence

That is exactly the tradeoff we want to avoid.

---

## 3. Cache-aware principle

Prune-first compaction should only short-circuit when it buys real runway, not when it merely dips under threshold.

Rule of thumb:
- compression frequency matters as much as compression size
- a smaller mutation is not automatically cheaper if it causes another mutation a few turns later

So the design target is:
- fewer auxiliary summary calls
- without materially increasing compression frequency

---

## 4. Conservative prototype policy

The conservative prototype keeps all existing compression invariants and only changes the acceptance rule for prune-only compaction.

### Phase 1: prune old middle tool outputs

Only prune tool outputs that are:
- in the compressible middle region
- not in protected head/tail windows
- not from protected tools (`read_file`, `memory`, `clarify`, `skill_view`, `todo`)

### Phase 2: require a low-water mark

Do not accept prune-only just because it lands below threshold.

Instead require:
- `post_prune_tokens <= prune_target_tokens`

Where:
- `prune_runway_tokens = max(prune_minimum_tokens, 15% of threshold_tokens)`
- `prune_target_tokens = threshold_tokens - prune_runway_tokens`

Interpretation:
- pruning must get us comfortably below threshold
- otherwise we immediately fall through to normal LLM summary compaction

Why this helps:
- protects cache by avoiding "micro-compactions" that would be followed by another compression shortly after
- still avoids the summary call when pruning truly buys useful runway

---

## 5. What the prototype currently does

The prototype branch currently:
- keeps prune-first compaction
- adds the low-water / runway requirement above
- preserves current main behavior for summary role alternation
- preserves the centralized `call_llm()` summary path
- keeps head/tail and tool-call/result integrity handling unchanged

This means the branch is no longer optimizing only for token reduction per event; it is explicitly biased toward fewer compression events.

---

## 6. Metrics we should evaluate before merging any future version

A serious cache-aware review should measure all of these, not just token savings:

1. Compression events per 100 conversation turns
2. Average turns between compressions
3. Auxiliary summary calls per session
4. Average tokens reclaimed per compression event
5. Total prompt+auxiliary tokens spent over a long session
6. Earliest changed message index during compression
7. Ratio of prune-only compressions to full summary compressions

The most important comparison is:
- baseline main vs conservative prune-first

Success is not:
- "fewer tokens in one compression"

Success is:
- "equal or better total session cost without increasing compression/cache-break cadence in a meaningful way"

---

## 7. Better long-term directions

If we want a stronger cache story than conservative prune-first, these are the real next-step options:

### A. Insertion-time trimming

Best cache-preserving option.

Idea:
- trim or summarize giant tool outputs before they become durable transcript history
- keep a compact representation from the start instead of mutating history later

Pros:
- avoids later cache-breaking rewrites for those blobs
- makes transcript size stable earlier

Cons:
- more invasive design change
- requires careful UX and provenance handling

### B. Provider/backend-aware compaction policy

Different providers may reward:
- preserving a longer stable prefix
- or simply reducing total prompt size

We may eventually want backend-specific heuristics for:
- prune runway targets
- compression thresholds
- when to prefer summary vs pruning

### C. Explicit compression telemetry

If compression remains a core feature, `ContextCompressor` should expose enough telemetry to understand real-world cadence:
- prune-only count
- full summary count
- average recovered tokens
- last compression mode

This is not required for the conservative prototype, but it would make future tuning much easier.

---

## 8. Recommended next steps

1. Keep the conservative prototype local for review.
2. Run targeted tests plus long-session manual trials.
3. If it looks promising, add telemetry before opening another PR.
4. If cache stability remains the top priority, pursue insertion-time trimming instead of further read-time pruning tweaks.

---

## 9. Review question for Teknium

The key product question is:

"Should Hermes optimize compression primarily for per-event token cost, or for minimizing the number of transcript mutations over the lifetime of a session?"

This prototype assumes the answer is:
- prioritize fewer transcript mutations unless pruning buys substantial runway.
