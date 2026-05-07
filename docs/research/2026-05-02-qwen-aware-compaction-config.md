# Qwen-Aware Compaction — Configuration Guide

When using Hermes with the local Qwen3.6 servers (`qwen-instruct`,
`qwen-thinking`, `qwen-27b-vision`), enable the `qwen_aware` extensions
in `~/.hermes/config.yaml` to reduce per-turn wall clock.

## Why local Qwen needs different defaults

- llama.cpp builds (b8994 and the buun fork) cannot do partial-prefix KV
  cache reuse for Qwen3.6's hybrid Gated DeltaNet + Gated Attention
  architecture. Every turn re-processes the full prompt from scratch.
- Decode is ~60 tok/s (35B-A3B MoE) and ~33 tok/s (27B vision-tcq).
  Compaction is a real interactive-latency event, so we want it to
  fire **earlier** (smaller threshold) and **less often** (deterministic
  pre-pass that shrinks the input).
- Qwen3.6 in default mode (no `preserve_thinking`) expects only the
  *latest* user-turn's `<think>` block in history. Stripping older
  `<think>` blocks both saves tokens and aligns with the model's
  trained-in expectations.

## Recommended config block

```yaml
compression:
  enabled: true
  threshold: 0.50              # existing default
  target_ratio: 0.20           # existing default
  protect_last_n: 20           # existing default
  qwen_aware:
    enabled: true              # master switch for everything below
    dedup_operations: true     # P0a: collapse consecutive read/edit/terminal on same key
    anchor_first_assistant: true # P0b: never start compaction at a user message
    threshold_absolute_max: 80000 # P0c: never grow the prompt past 80K tokens
    message_threshold: 200       # P1d: catch many-short-turn sessions
    turn_threshold: 30           # P1d: catch long back-and-forth conversations

auxiliary:
  compression:
    provider: local-qwen          # pin summarizer to the cheap, non-thinking model
    model: qwen-instruct
    base_url: http://127.0.0.1:8085/v1
```

## Per-feature toggles

Every feature can be turned off independently by setting its flag to
`false` (or omitting it). Master `qwen_aware.enabled: false` disables
everything regardless of the per-feature flags.

| Flag | What it does | Safe to disable? |
|---|---|---|
| `dedup_operations` | Collapses earlier same-resource tool calls into back-references | Yes — Pass 2 (`_summarize_tool_result`) still runs |
| `anchor_first_assistant` | Slides compress-start to the first assistant msg | Yes — `_align_boundary_*` still runs |
| `threshold_absolute_max` | Caps the trigger threshold at an absolute token count | Yes — `threshold_percent` still applies |
| `message_threshold` / `turn_threshold` | Extra compaction triggers | Yes — token threshold still fires |

## When NOT to use these flags

- **You're using SGLang/vLLM with RadixAttention instead of
  llama.cpp** — RadixAttention does prefix-cache reuse for hybrid
  models. The cost asymmetry that motivates `threshold_absolute_max`
  is much smaller. Either skip the absolute cap or raise it
  significantly (e.g., 200_000).
- **You're on a cloud model (Claude / GPT / Gemini)** — keep
  `qwen_aware.enabled: false`. None of these tradeoffs apply; the
  prompt cache amortizes prompt re-processing.

## What's NOT in the qwen_aware block (and why)

- **`<think>` block stripping** — Hermes' `_build_assistant_message`
  already strips reasoning tags at the storage boundary
  (`run_agent.py:8590-8602`). Adding a compaction-time strip would be
  redundant in `qwen-instruct` mode and harmful in `qwen-thinking`
  mode (where the local jinja template emits historical `<think>`
  blocks that the model is trained to use).
- **`on_turn_end` trigger** — Hermes' single `should_compress`
  callsite fires inside the tool loop after tool results are
  appended, so `messages[-1]` is never a user message. The trigger
  would never fire without an architectural change (a preflight
  callsite). Use `message_threshold` / `turn_threshold` instead.

## Verifying the flags landed

```bash
hermes dump | grep -A 12 "compression:"
```

Should show your `qwen_aware:` block populated with the values above.
