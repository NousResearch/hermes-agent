# Forgecode context compaction — analysis & takeaways for Hermes

**Date:** 2026-05-02
**Source:** [tailcallhq/forgecode](https://github.com/tailcallhq/forgecode) — Rust agent CLI, MIT
**Files read (commit on `main`):**
- `crates/forge_app/src/compact.rs` — `Compactor` (orchestration + summarization)
- `crates/forge_app/src/transformers/compaction.rs` — `SummaryTransformer` (deterministic pre-LLM pruning pipeline)
- `crates/forge_app/src/transformers/trim_context_summary.rs` — file/shell/search dedup by operation
- `crates/forge_domain/src/compact/compact_config.rs` — `Compact` config + `should_compact`
- `crates/forge_domain/src/compact/strategy.rs` — `CompactionStrategy::eviction_range` + tool-atomicity
- `crates/forge_domain/src/compact/result.rs` — `CompactionResult` (metrics)
- `crates/forge_app/src/hooks/compaction.rs` — `CompactionHandler` event hook
- `templates/forge-partial-summary-frame.md` — Handlebars summary frame
- `crates/forge_config/.forge.toml` — defaults

**Hermes side (cross-reference):**
- `agent/context_engine.py` — pluggable engine ABC
- `agent/context_compressor.py` — built-in compressor (1414 LOC), already substantial

---

## 1. Forgecode's compaction in one diagram

```
turn boundary
    │
    ▼
CompactionHandler.handle()                      [hooks/compaction.rs]
    │
    ├── token_count = context.token_count()
    ├── if Compact.should_compact(ctx, tokens):  ← multi-trigger
    │       (token | turn | message | on_turn_end)
    │
    └── Compactor::compact(ctx, max=false)       [compact.rs]
            │
            ├── strategy = if max { Retain } else { Min(Evict, Retain) }
            ├── (start, end) = strategy.eviction_range(ctx)
            │       ↳ "from first assistant msg to len-retain-1,
            │          adjusted to never split tool_call/tool_result"
            │
            └── compress_single_sequence(ctx, (start,end))
                    │
                    ├── filter out is_droppable() messages
                    ├── ctx_summary = ContextSummary::from(slice)
                    ├── ctx_summary = SummaryTransformer.transform(ctx_summary)
                    │       ↳ DropRole(System) → DedupeRole(User)
                    │         → TrimContextSummary (last-op-per-resource)
                    │         → StripWorkingDir
                    ├── summary_text = render("forge-partial-summary-frame.md", ...)
                    ├── extract LAST non-empty reasoning_details → inject into
                    │   first surviving assistant message
                    ├── accumulate Usage from compacted slice → attach to
                    │   summary message
                    └── splice (start..=end) ← single user message with summary_text
```

Compaction is **always a single-pass replacement of one contiguous range** with one synthetic user message. No sliding-window evictions, no per-message cherry-picking. That simplicity is load-bearing.

---

## 2. The configurable triggers (`Compact::should_compact`)

```rust
pub fn should_compact(&self, ctx: &Context, token_count: usize) -> bool {
    self.should_compact_due_to_tokens(token_count)   // absolute || %-of-window
        || self.should_compact_due_to_turns(ctx)     // count of user msgs
        || self.should_compact_due_to_messages(ctx)  // total msg count
        || self.should_compact_on_turn_end(ctx)      // last msg is user
}
```

Defaults from `crates/forge_config/.forge.toml`:

```toml
[compact]
eviction_window     = 0.2     # at most 20% of context summarized per pass
retention_window    = 6       # always keep last 6 messages raw
max_tokens          = 2000    # summary token budget
message_threshold   = 200     # 200+ messages → compact
token_threshold     = 100000  # 100K tokens → compact
on_turn_end         = false
```

Note: `token_threshold` is **combined with `token_threshold_percentage` by taking the lower** ("BUG 5" doc-comment in the test file: with a 128K model and a 100K threshold, three turns of tool output overshoot 128K before the threshold fires; capping at 70% of the window = 89.6K provides the safety margin).

---

## 3. The eviction-range algorithm (`strategy.rs`)

`CompactionStrategy` is a small algebra:

```rust
enum CompactionStrategy {
    Evict(f64),        // % of total tokens to summarize
    Retain(usize),     // preserve last N messages raw
    Min(Box<S>, Box<S>),
    Max(Box<S>, Box<S>),
}
```

Both `Evict` and `Retain` collapse to a `usize` "preserve_last_n" via `to_fixed`; then `find_sequence_preserving_last_n` picks the range. The two non-obvious correctness details:

**(a) Always start at the first assistant message.** This protects the leading user prompt verbatim. Any system/user prelude survives untouched no matter how many compactions stack.

**(b) Tool-call atomicity.** The end of the eviction range is adjusted so it never lands inside a `<tool_call, tool_result>` pair, and never splits parallel tool results:

```rust
if messages[end].has_tool_call() {
    if end == start { return None; }       // can't safely compact a singleton tool_call
    return Some((start, end - 1));         // back off
}
if messages[end].has_tool_result()
   && messages[end+1].has_tool_result() {  // parallel tool_results
    while messages[end].has_tool_result() { end -= 1; }
    end -= 1;                              // back off past the assistant tool_call too
}
```

Tests (`test_sequence_finding`) exercise this with a tiny DSL where `s/u/a/t/r` = system/user/assistant/tool-call/tool-result, e.g. `seq("sutrtrtra", 1) → "su[trtrtr]a"`.

---

## 4. The deterministic pre-LLM pipeline (`SummaryTransformer`)

Before summarizing, the slice goes through a **deterministic** transformer pipeline (no LLM call):

```rust
DropRole::new(Role::System)
    .pipe(DedupeRole::new(Role::User))
    .pipe(TrimContextSummary)               // ← last-op-per-resource
    .pipe(StripWorkingDir::new(cwd))
```

`TrimContextSummary` is the most interesting one. It builds an `Operation<'_>` enum keyed by **the resource the tool acts on**:

```rust
enum Operation<'a> {
    File(&'a str),       // FileRead / FileUpdate / FileRemove / Undo on same path
    Shell(&'a str),
    Search(&'a str),
    CodebaseSearch { queries: &'a [SearchQuery] },
    Fetch(&'a str),
    Followup(&'a str),
    Plan(&'a str),
    Skill(&'a str),
    Task(&'a str),
    Mcp(&'a str),
    Todo,
}
```

Then walks each assistant message's content and **drops the previous block if its operation matches the current one** (last-write-wins). On a session that did `read X → edit X → read X → edit X`, four blocks collapse to one.

This is the single most useful idea for local-model setups (Section 8 below).

---

## 5. Reasoning-chain preservation

For models with a separate reasoning channel (Anthropic extended-thinking, Qwen3 `<think>...</think>`), the convention is that the **first** assistant message after the system prompt establishes a reasoning chain that subsequent messages extend. If you naively splice a summary in, the first remaining assistant message has no reasoning block — chain broken.

Forgecode's fix:

```rust
// Get LAST non-empty reasoning from compacted slice
let reasoning_details = compaction_sequence.iter().rev()
    .find_map(|m| m.reasoning_details.filter(|rd| !rd.is_empty()).cloned());

// After splice, inject into the first surviving assistant if it lacks reasoning
if let Some(reasoning) = reasoning_details
    && let Some(first_assistant) = surviving.find_assistant()
    && first_assistant.reasoning_details.is_none_or(|rd| rd.is_empty()) {
    first_assistant.reasoning_details = Some(reasoning);
}
```

The "last non-empty" choice prevents reasoning from accumulating across multiple compactions (a test, `test_compress_single_sequence_no_reasoning_accumulation`, asserts exactly that: after two compactions, the surviving assistant still has exactly one reasoning block, not two).

---

## 6. Stacked summary frames (template)

`forge-partial-summary-frame.md` opens with:

> Use the following summary frames as the authoritative reference for all coding suggestions and decisions. Do not re-explain or revisit it unless I ask. **Additional summary frames will be added as the conversation progresses.**

So compaction N produces a single `[summary_N]` user message. On compaction N+1, the eviction range starts after `summary_N` (because `summary_N` is now part of the protected leading messages or the new "first assistant" baseline) and produces `[summary_{N+1}]`. The model sees a stack of structured frames, not a summary-of-a-summary-of-a-summary. Lossy compression that compounds gracefully.

---

## 7. What Hermes already has (so we don't reinvent)

`agent/context_compressor.py` (1414 LOC) is substantially more elaborate than forgecode's compactor on most axes:

| Capability | Hermes | Forgecode |
|---|---|---|
| Pluggable engine | ✅ `ContextEngine` ABC | ❌ single impl |
| Token-threshold trigger | ✅ `threshold_percent=0.75` | ✅ + absolute + % cap |
| Turn / message / on_turn_end triggers | ❌ | ✅ |
| Anti-thrashing (skip if last passes ineffective) | ✅ `_ineffective_compression_count` | ❌ |
| Tool-result pruning (cheap pre-pass) | ✅ `_prune_old_tool_results` | ✅ via `TrimContextSummary` |
| Last-op-per-file dedup | ⚠️ partial (`identical tool result dedup`) | ✅ explicit `Operation` keying |
| Tool-call atomicity in eviction range | ⚠️ unclear | ✅ explicit |
| Reasoning-chain preservation | ❌ | ✅ |
| Stacked summary frames | ⚠️ `_previous_summary` field exists | ✅ template-driven |
| Configurable summary model | ✅ aux model w/ failure recovery | ✅ `compact.model` |
| Focus-topic guided compression | ✅ `focus_topic` arg | ❌ |
| Manual `/compress` + preflight | ✅ `has_content_to_compress` | ✅ `:compact` CLI command |
| Usage accumulation through compaction | ⚠️ check via `usage_pricing.py` | ✅ explicit |
| Droppable-message filtering | ⚠️ | ✅ `is_droppable()` |

So this is an "augment Hermes' compressor with five specific ideas" exercise, not a "rewrite from scratch" exercise.

---

## 8. Takeaways prioritized for local Qwen

Local-model constraints that change the cost model vs. cloud:

- **No partial-prefix KV cache** for Qwen3.6 hybrid (Gated DeltaNet + Gated Attention). Every turn re-processes the full prompt from scratch. Documented in `~/llama-stack/README.md` and `~/llama-stack/config/moe.env`.
- **Decode ~60 tok/s (moe), ~33 tok/s (vision-tcq)** on a 3090. A 2K-token summary takes ~30s on moe, ~60s on vision-tcq. Compaction is an interactive-latency event.
- **256K (moe) / 128K (vision-tcq) windows**, but bigger ≠ free — re-processing 200K tokens of prompt at ~3200 tok/s (cold) is ~60s before the first decoded token.

That asymmetry — every turn pays for the full prompt, but compaction pays once — pushes the optimal trigger **earlier and more aggressive** than cloud defaults suggest, *and* makes deterministic pre-LLM pruning a much bigger win.

### P0 — high impact, low effort

**(a) Port the `Operation`-keyed last-op-per-resource dedup into Hermes' pre-pass.**
- Where: extend `agent/context_compressor.py` `_prune_old_tool_results` with an explicit "operation key per tool" map (`read_file/edit_file/delete_file → key=path`, `terminal → key=command_string`, `web_search → key=query`, `fetch → key=url`).
- Why local: a single agent loop that does `read X → edit X → read X → edit X` ten times can collapse 40 tool turns to 4. With no prefix caching, this directly shrinks the per-turn re-processing cost. The forgecode test `test_keeps_last_operation_per_path` is the spec.
- Caveat: don't dedup `terminal` blindly — `ls` followed by another `ls` after intervening writes is meaningful. Apply the dedup only if the resource state hasn't been mutated by an intervening write to the same path. Forgecode sidesteps this by deduping only within a *single* assistant message (consecutive blocks) — start there, expand later.

**(b) Add explicit tool-call ↔ tool-result atomicity to the eviction range.**
- Where: wherever Hermes computes the boundary between "summarize" and "keep raw" tail (look near `protect_first_n` / `protect_last_n` consumers in `context_compressor.py`).
- Why: corrupted `tool_call` without its matching `tool_result` confuses the model on the next turn and can cause it to re-issue the call. With local Qwen and presence_penalty 1.5 the re-issue often comes out slightly different, leading to stuck loops.
- Spec: forgecode's `find_sequence_preserving_last_n` lines 1586-1610 is the reference — back the boundary off by 1 if the boundary message has a `tool_call`, and walk back across consecutive `tool_result`s if there are parallel calls.

**(c) Combined absolute + %-of-window threshold (`min(absolute, ctx * pct)`).**
- Where: `agent/context_engine.py` `update_model()` — currently `threshold_tokens = int(context_length * threshold_percent)`. Add an absolute cap.
- Why local: with `qwen-instruct` at 256K and `threshold_percent=0.75`, compaction fires at ~192K — that's ~3 minutes of prompt re-processing per turn before any decode. A hard absolute cap (e.g., 80K) keeps interactive sessions snappy even though there's headroom. Cloud agents tolerate larger contexts because cloud has prefix caching; local doesn't.

### P1 — high impact, medium effort

**(d) Multi-trigger thresholds (turn + on_turn_end), not just tokens.**
- Where: `agent/context_compressor.py` `should_compress`.
- Why: token counts are computed *after* the prompt is built, which means you've already paid for tokenization. Turn count and "last message is user" are O(1) checks computable before re-tokenization. For long agentic loops, `turn_threshold = 30` is a useful early-warning that fires before the token threshold.
- `on_turn_end` is the killer for local: compact between user turns, not mid-tool-loop. The user is already typing → latency is hidden.

**(e) Reasoning-chain preservation for `qwen-thinking`.**
- Where: in the splicing logic. After compaction, find the first surviving assistant message. If it has no `<think>...</think>` block, extract the last non-empty `<think>` block from the compacted range and inject it into that message's content.
- Why: `qwen-thinking` (the `<|think_on|>` prelude variant) generates `<think>` blocks. Without preservation, the model produces a `<think>` block fresh on the post-compaction turn that has no continuity with the preceding (now-summarized) reasoning, often causing it to re-derive things it had already concluded. Forgecode's test `test_compress_single_sequence_no_reasoning_accumulation` is the anti-spec — preserve **only the last** to avoid stacking on repeated compactions.
- Hermes-specific: detect by `prelude == "<|think_on|>"` in the active model's provider config (`~/.hermes/config.yaml`). If absent, skip this step.

### P2 — nice to have

**(f) Cheaper compaction model.**
- Where: provider routing for the summarizer call. Hermes already supports an aux summary model; ensure the local config can point it at `qwen-instruct` even when the main session runs `qwen-thinking`.
- Why: `qwen-thinking`'s `<think>` reasoning is wasted on a summarization task. Same model file, different prelude, ~3-5x faster wall clock for the summary.
- Concrete config sketch (in `~/.hermes/config.yaml`):
  ```yaml
  context:
    engine: compressor
    summarizer:
      model: qwen-instruct        # always non-thinking for summaries
      max_tokens: 2000
  ```

**(g) Stacked summary frames with stable header.**
- Where: the summary template Hermes already uses (the `_previous_summary` field).
- Why: forgecode's "additional summary frames will be added as the conversation progresses" is a small prompt-engineering trick that lets the model reason about *which* summary is which. Verify whether Hermes' current `_previous_summary` is replaced or appended on subsequent compactions; if replaced, lossy compounding will erase early context.

**(h) Surface compaction metrics.**
- Forgecode returns a `CompactionResult { original_tokens, compacted_tokens, original_messages, compacted_messages }`. Hermes has `compression_count` and `last_*_tokens` but no per-event delta. Surfacing `compaction_savings_pct` per event would help diagnose the "anti-thrashing kicked in" path that Hermes already has — currently it logs but doesn't expose to the user.

---

## 9. Things to *not* port

- **Forgecode's whole eviction-strategy enum (`Min/Max/Evict/Retain`).** Hermes' `protect_first_n` + `protect_last_n` plus `threshold_percent` already covers the same shape. The enum is overhead in Python without static dispatch.
- **The Handlebars template wholesale.** Hermes' summary text already encodes structured tool calls. Take the *idea* of a stable framing header, not the template engine.
- **The `Compact::model` field on `Agent`.** Hermes already has aux-model config with failure recovery (`_last_aux_model_failure_error`); just route the summarizer through it.
- **`max=true` for manual `/compact`.** Hermes uses `focus_topic` for the same purpose (more aggressive + topic-guided). Don't add a second knob.

---

## 10. Suggested implementation order

If you do exactly one thing, do **(a) — operation-keyed dedup**. It's the highest leverage on local-model wall clock because it cuts work *before* the LLM call, and it's the single most-tested piece of the forgecode codebase (see `transformers/trim_context_summary.rs` test suite).

If you do three things, add **(b)** and **(c)** alongside it: tool-atomicity + absolute token cap. These are pure correctness/safety changes — no new tunables for users to misconfigure.

The reasoning-chain preservation **(e)** is worth its own small spec because it's `qwen-thinking`-specific and benefits from a focused test (build a fixture with two compactions across a `<think>`-bearing range and assert the preserved reasoning shows up exactly once).
