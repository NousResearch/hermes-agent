# Doubleword Models and Pricing Reference

Use this reference when the user asks about cost, when model choice is
ambiguous, or before submitting a large or expensive job. Prices are per 1M
tokens, input / output, unless otherwise noted.

Realtime inference is available for most models at standard rates. Verify
current Doubleword pricing before submitting high-cost work or quoting exact
costs.

## Selection Heuristics

- Cheapest capable text generation: start with `GPT-OSS-20B`,
  `Qwen3-14B-FP8`, or `Qwen3.5-4B` depending on output-quality needs.
- Balanced inexpensive generation: use `DeepSeek-V4-Flash`,
  `Qwen3.6-35B-A3B-FP8`, `Qwen3.5-35B-A3B-FP8`, or `Gemma-4-31B`.
- Harder reasoning, long-form generation, difficult extraction, or eval
  judging: consider `DeepSeek-V4-Pro`, `Kimi-K2.6`,
  `Qwen3.5-397B-A17B`, `GLM-5.2-FP8`, or Nemotron large models.
- Vision-language tasks: use `Qwen3-VL-235B-A22B` or
  `Qwen3-VL-30B-A3B`.
- OCR: use a specialized OCR model unless the task requires custom reasoning
  over extracted text.
- Embeddings: use `Qwen3-Embedding-8B`.

## Text Generation Models

| Model | Async, 1h | Batch, 24h |
| --- | --- | --- |
| DeepSeek-V4-Pro | $1.31 / $2.75 | $1.05 / $2.20 |
| DeepSeek-V4-Flash | $0.10 / $0.20 | $0.07 / $0.14 |
| Kimi-K2.6 | $0.70 / $3.00 | $0.45 / $2.00 |
| GLM-5.2-FP8 | $1.05 / $3.30 | $0.70 / $2.20 |
| GLM-5.1-FP8 | $1.05 / $3.30 | $0.70 / $2.20 |
| Qwen3.5-397B-A17B | $0.30 / $1.80 | $0.15 / $1.20 |
| Qwen3.6-35B-A3B-FP8 | $0.07 / $0.30 | $0.05 / $0.20 |
| Qwen3.5-35B-A3B-FP8 | $0.07 / $0.30 | $0.05 / $0.20 |
| Qwen3.5-9B | $0.04 / $0.35 | $0.03 / $0.29 |
| Qwen3.5-4B | $0.05 / $0.08 | $0.04 / $0.06 |
| Gemma-4-31B | $0.11 / $0.30 | $0.07 / $0.20 |
| Nemotron-3-Ultra-550B-A55B | $0.37 / $1.87 | $0.25 / $1.25 |
| Nemotron-3-Super-120B-A12B | $0.23 / $0.56 | $0.15 / $0.38 |
| GPT-OSS-20B | $0.03 / $0.20 | $0.02 / $0.15 |
| Qwen3-VL-235B-A22B | $0.15 / $0.55 | $0.10 / $0.40 |
| Qwen3-VL-30B-A3B | $0.07 / $0.30 | $0.05 / $0.20 |
| Qwen3-14B-FP8 | $0.03 / $0.30 | $0.02 / $0.20 |

## Specialized Models

| Task | Model | Async | Batch |
| --- | --- | --- | --- |
| OCR | DeepSeek-OCR-2 | $0.08 / $0.08 | $0.05 / $0.05 |
| OCR | olmOCR-2-7B | $0.15 / $0.15 | $0.10 / $0.10 |
| OCR | LightOnOCR-2-1B | $0.08 / $0.08 | $0.05 / $0.05 |
| Embeddings | Qwen3-Embedding-8B | $0.03 input | $0.02 input |

## Batch Limits

- Maximum batch file size: 200 MB.
- Maximum requests per batch file: 50,000.
- Split larger workloads into numbered shards and submit each shard separately.
