---
name: doubleword
description: Route Doubleword LLM inference and data-processing jobs across realtime, async, and batch tiers using the dw CLI or OpenAI-compatible API.
version: 1.0.0
platforms: [linux, macos]
metadata:
  hermes:
    tags: [llm, inference, batch, data-processing, ocr, embeddings]
    category: ai
    requires_toolsets: [terminal]
    config:
      - key: doubleword.api_key
        description: Doubleword API key exposed to the shell as DOUBLEWORD_API_KEY.
        prompt: Enter your Doubleword API key.
---

# Doubleword Inference

## When to Use

Use this skill when a user asks to run generation, extraction, classification,
OCR, embeddings, evals, or dataset processing through Doubleword
(`https://api.doubleword.ai/v1`).

Prefer the native `dw` CLI for file, async, and batch workflows. Use the
OpenAI-compatible SDK only when programmatic request construction or streaming
control is clearer than CLI execution.

Load references only when needed:

- `references/models-and-pricing.md` for model choice, cost comparison, and
  task-specific model tables.
- `references/cli-recipes.md` for exact validation, submission, status,
  retrieval, resume, SDK fallback commands, and the official Doubleword command
  reference link.

## Procedure

1. Check local readiness before remote work:
   - confirm `DOUBLEWORD_API_KEY` is set;
   - with browser login, run `dw whoami` before uploading any file and stop if
     identity or organization verification fails;
   - with headless/API-key login, `dw whoami` may fail because it requires the
     admin API; instead, run local file validation plus a non-interactive,
     token-limited realtime probe such as
     `MODEL="${MODEL:-openai/gpt-oss-20b}"; dw realtime "$MODEL" "Reply with OK." --temperature 0 --max-tokens 2 --no-stream`
     before upload and stop if the inference probe fails.
2. Classify the task:
   - Realtime: single prompt, interactive lookup, or explicit immediate result.
   - Async: background work needed in the current session, medium datasets, or
     roughly 100+ requests.
   - Batch: large datasets, evals, overnight jobs, or no stated deadline.
3. Choose the slowest tier that satisfies the user's urgency:
   - Realtime for immediate answers;
   - Async with a 1h completion window for same-session background work;
   - Batch with a 24h completion window for lowest-cost large jobs.
4. Select a model:
   - respect explicit user choices unless incompatible with the task;
   - use specialized OCR or embedding models for those task types;
   - otherwise choose the cheapest capable model;
   - load `references/models-and-pricing.md` if cost/model detail matters.
5. Prepare JSONL input for multi-request jobs:
   - include stable per-row identifiers when possible;
   - keep each file under 200 MB and 50,000 requests;
   - split larger workloads into numbered shards.
6. Validate local payloads before upload:
   - run `dw files validate <path>`;
   - run `dw files stats <path>`;
   - fix validation errors locally before submitting.
7. Submit the job:
   - use realtime synchronously and return the output;
   - for Async or Batch, upload/create the job, capture the batch ID, and report
     it to the user.
8. For Async or Batch, do not idle in polling loops. Check status later with a
   discrete `dw batches get <batch_id>` command only when useful or requested.
9. Retrieve completed results with `dw batches results <batch_id> -o <file>` and
   validate the output shape before using it downstream.

## Pitfalls

- Do not run `while ... sleep ...` polling loops in Hermes; idle loops may be
  terminated.
- Do not upload hidden files, `.env` files, credentials, unrelated repository
  files, or data outside the user's requested scope.
- Do not print API keys, authorization headers, or signed URLs.
- Do not upload invalid JSONL. Remote validation failures waste time and may
  obscure row-level formatting issues.
- Do not assume 24h Batch is acceptable when the user needs results during the
  current working session.
- Do not rely on embedded pricing as authoritative for high-cost decisions;
  verify current Doubleword pricing when exact cost matters.

## Verification

For realtime jobs:

- output is returned to the user;
- the response names the selected model.

For Async or Batch jobs:

- either `dw whoami` succeeded before upload, or a headless/API-key readiness
  path succeeded with local file validation and a non-interactive,
  token-limited realtime inference probe;
- `dw files validate` and `dw files stats` succeeded for every uploaded JSONL
  shard;
- the user receives the selected mode, selected model, batch ID, source file or
  shard count, completion window, status-check command, and intended results
  path;
- completed results are downloaded and checked with `dw files stats` before
  downstream processing.
