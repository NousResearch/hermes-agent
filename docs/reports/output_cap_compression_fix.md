# Output-Cap Compression Fix: Detailed Report

## The Bug

When a provider returns an "output cap too large" HTTP 400 (e.g. `max_tokens`
exceeds the space left in the context window), the retry path reduced
`max_tokens` by 64 tokens per attempt but never called `_compress_context()`.
The message compressor was not wired into this path at all.

Because the compressor never fired, each retry made almost no real progress:
input grew by ~65 tokens per attempt while the output cap shrank by 64, for a
net effect of ~1 token per attempt. The total footprint stayed just over the
context ceiling instead of coming back under it.

## What Happened (observed token math)

- Input grew per attempt: 134,465 → 134,530 → 134,595 → 134,660
- Output cap shrank per attempt: 65,535 → 65,471 → 65,406 → 65,341 → 65,276
- Net: ~1 token of growth per retry (65 − 64 = 1)
- Total stayed at 200,001 tokens — 1 over the 200,000 ceiling
- After the retry budget was exhausted, the run ended with
  `compression_exhausted=True` and the session dropped.

## How It Broke

The output-cap retry loop only adjusted `max_tokens` each pass. The flag it set
to continue — `restart_with_compressed_messages` — is named as though the
messages get compressed, but the messages were **not** actually compressed on
this branch. The compressor simply was not invoked, so nothing reduced the
growing footprint.

This path is old (~4 months, from a series of compaction/output-cap changes),
but the details of exactly which change first introduced it are less important
than the current behavior and the fix.

## Root Cause (factual)

The output-cap handling dates back to a compaction/output-cap change Apr 2026;
it was later refactored into `agent/conversation_loop.py` (May 2026) and its
cap calculation refined (Jul 2026). The defect is behavioral rather than
attributable to one edit: the output-cap branch set
`restart_with_compressed_messages = True` without ever calling
`_compress_context()`, so nothing actually shrank the in-flight footprint.

## Why It Surfaced Now (Aug 2026)

The bug is old (~4 months) but latent — it only becomes reachable when a
conversation actually fills or near-fills its context window **and** the
provider returns the specific "output cap too large" wording the parser
matches. That combination is rare; most sessions never get near the ceiling.

What changed leading up to the first real-world exposure in early Aug 2026:

1. Two token-estimator accuracy fixes reduced the reported token counts:
   - `530503a6a` (2026-07-28): exclude `reasoning_details` from the preflight
     estimate. Its own message notes that field inflated the rough estimate by
     ~4x and caused compression far too early (a measured session reported
     ~533K estimated vs ~140K real).
   - `e3bc51703` (2026-07-31): stop double-counting `api_content`.
2. The preflight auto-compression gate (`should_compress()` ->
   `should_compress_info()`) uses the same estimator, comparing
   `prompt_tokens >= threshold_tokens`, where
   `threshold_tokens ≈ (context_length − max_tokens) × 0.50`. With these fixes,
   the estimate dropped, so preflight judged sessions "under threshold" and
   deferred compression far longer than before.
3. A long-running session therefore reached the hard context ceiling for the
   first time, hit the provider's output-cap 400, and entered the output-cap
   retry branch — which (pre-fix) never compressed and death-looped.

In short: the Jul estimator fixes did not introduce the death-loop; they
unmasked it. They removed the (over-aggressive) early compression that had been
silently keeping sessions away from the ceiling, exposing the latent
never-compresses-on-output-cap defect.

## The Fix

Add `_compress_context()` to the output-cap retry path so the message history
is actually compressed before retrying:

1. The compressor drops the middle window, freeing a large fraction of tokens
   in a single pass.
2. If compression made real progress (message count dropped, or ≥5% token
   savings), the session continues under the ceiling.
3. If compression could not reduce the request, the loop retries with the already
   reduced `max_tokens` and, if the error keeps recurring, terminates through the
   existing max-attempts guard with a clear "cannot compress further" message.
4. Compression failures are non-fatal: if anything goes wrong during
   compression, the code falls back to retrying on `max_tokens` alone. The
   existing max-attempts guard still bounds the retry loop, so it cannot spin
   forever.

## Verification

- Both output-cap retry regression tests pass:
  `test_output_cap_retry_uses_provider_available_out`, and
  `test_output_cap_retry_with_large_api_only_content`.
- New `test_output_cap_retry_triggers_compression_and_recovers` locks in the
  fix and asserts that the retry sends the compressed history on the wire (not
  the original oversized request) and that `context_length` is untouched.
- New `test_output_cap_retry_compression_no_progress_terminates_bounded` locks
  in that a zero-progress compression terminates via the max-attempts guard
  rather than spinning forever.
- Full `tests/run_agent/test_run_agent.py`: 235 passed with the optional
  `anthropic` package installed. (Without it, the single
  `test_interruptible_anthropic_interrupt_never_closes_shared_client` fails on
  an ``ImportError: The 'anthropic' package is required`` — an environment
  issue, unrelated to this change.)
- `ruff check` passes on both modified files.

## Impact

- Sessions at the context ceiling now recover via message compression instead
  of exhausting retries.
- Applies across providers (vLLM, DashScope/Qwen, OpenRouter, LM Studio,
  Anthropic). Additive fix; no breaking changes.

## Affected Code

`agent/conversation_loop.py` — the output-cap retry block (added compression to
the retry path that previously only adjusted `max_tokens`).
