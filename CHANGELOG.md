# Changelog

## [v0.19.2] — 2026-08-03

### Fixed
- **context_compressor / conversation_loop**: Output-cap retry path now compresses message history (#55546)

  Previously, when the provider returned an "output cap too large" HTTP 400, the
  retry loop only reduced `max_tokens` by 64 tokens per attempt. The input grew by
  ~65 tokens each retry, leaving the total stuck at 200,001 tokens — 1 token over
  the 200,000 ceiling. After 3 retries, the session crashed with
  `compression_exhausted=True`.

  The fix adds `_compress_context()` to the output-cap retry path so the compressor
  drops the middle window, freeing a large fraction of tokens in one pass. If
  compression makes ≥5% token savings (or drops the message count), the session
  continues under the ceiling. If compression cannot reduce the request, the loop
  retries with the reduced `max_tokens` and, if the error keeps recurring,
  terminates through the existing max-attempts guard with a clear "cannot compress
  further" message — it no longer spins on `max_tokens` alone. Compression failures
  stay non-fatal and fall back to retrying on `max_tokens` alone.

  Affected providers: vLLM, DashScope (Qwen), OpenRouter, LM Studio, Anthropic.

### Testing
- `test_output_cap_parsing.py` — output-cap error parsing across providers
- `test_1630_context_overflow_loop.py` — context overflow heuristic and compression_exhausted flag
- `test_preflight_compression_cap_e2e.py` — preflight compression with configurable max_attempts
- `test_max_tokens_propagation.py` — max_tokens config propagation to gateway