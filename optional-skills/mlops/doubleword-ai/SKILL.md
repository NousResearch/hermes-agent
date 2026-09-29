---
name: doubleword
description: Run realtime, async, and batch inference jobs.
version: 1.0.0
author: aschkandw, Hermes Agent
license: MIT
platforms: [linux, macos]
required_environment_variables:
  - name: DOUBLEWORD_API_KEY
    prompt: Doubleword API key
    help: Create or copy a key from your account at https://doubleword.ai/.
    required_for: Doubleword inference and batch jobs
prerequisites:
  commands: [dw]
metadata:
  hermes:
    tags: [llm, inference, batch, data-processing, ocr, embeddings]
    category: ai
    requires_toolsets: [terminal]
---

# Doubleword Skill

Run inference and dataset-processing jobs through Doubleword with the `dw`
CLI. This skill selects realtime, async, or batch execution and covers
validation and result retrieval; it does not install the CLI, create
credentials, or wait indefinitely for remote jobs.

## When to Use

- The user requests generation, extraction, classification, OCR, embeddings,
  evals, or dataset processing through Doubleword.
- The workload needs cost-aware routing between immediate inference and
  deferred processing.
- A JSONL dataset needs validation, submission, status checking, or result
  retrieval.

Prefer the native `dw` CLI for file, async, and batch workflows. Use the
OpenAI-compatible SDK only when programmatic request construction or streaming
control is clearer than CLI execution.

Load references only when needed:

- `references/models-and-pricing.md` for model choice, cost comparison, and
  task-specific model tables.
- `references/cli-recipes.md` for exact validation, submission, status,
  retrieval, resume, SDK fallback commands, and the official Doubleword command
  reference link.

## Prerequisites

1. Install the Doubleword `dw` CLI and verify it is available with
   `dw --version`. See the
   [official command reference](https://doublewordai.github.io/dw/commands.html).
2. Create a Doubleword account and API key. On skill load, Hermes securely
   prompts for `DOUBLEWORD_API_KEY` and stores it in the active profile's
   `.env` file. For manual setup, add it to
   `${HERMES_HOME:-~/.hermes}/.env`; never store the key in `config.yaml`.
3. Ensure the `terminal` tool is available for invoking `dw`.

For headless/API-key authentication, pass the environment variable to the CLI
once with `dw login --api-key "$DOUBLEWORD_API_KEY"`. The CLI then stores the
credential in `~/.dw/credentials.toml` with restricted permissions. Use
`dw login` instead for browser authentication and access to admin commands.

Do not print the key or pass its literal value in a command. Browser login and
API-key login expose different account-management capabilities, as described
in the readiness procedure below.

## How to Run

Invoke all `dw` operations through the `terminal` tool. Start by checking the
installed CLI and active local account:

```bash
dw --version
dw account current
```

Browser-authenticated accounts can discover current models with
`dw models list`. API-key-only accounts should use the model selected by the
user or the model guidance in `references/models-and-pricing.md`, then confirm
it with the inference request under `## Verification`.

For detailed syntax, run `dw --help`, `dw <group> --help`, or load
`references/cli-recipes.md`.

## Quick Reference

| Task | Command |
| --- | --- |
| Configure API-key login | `dw login --api-key "$DOUBLEWORD_API_KEY"` |
| Check browser-login identity | `dw whoami` |
| List chat models (browser login) | `dw models list --type chat` |
| Validate JSONL | `dw files validate <path>` |
| Inspect JSONL | `dw files stats <path>` |
| Run realtime inference | `dw realtime <model> "<prompt>"` |
| Upload a file | `dw files upload <path>` |
| Create an async job | `dw batches create --file <file_id> --completion-window 1h` |
| Create a batch job | `dw batches create --file <file_id> --completion-window 24h` |
| Check status | `dw batches get <batch_id>` |
| Download results | `dw batches results <batch_id> -o <path>` |

## Procedure

1. Check local readiness before remote work:
   - run `dw --version`;
   - with browser login, run `dw login` when needed, then `dw whoami` before
     uploading any file; stop if identity or organization verification fails;
   - with headless/API-key login, run
     `dw login --api-key "$DOUBLEWORD_API_KEY"`, then the non-interactive,
     token-limited command from `## Verification`; stop if either fails.
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
   - with browser login, run `dw models list` to confirm availability;
   - with API-key-only login, confirm availability with the inference probe;
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
10. Report the selected mode and model. For deferred jobs, also report the
    batch ID, source path or shard count, completion window, status command,
    and intended result path.

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

Run one token-limited request through the `terminal` tool:

```bash
dw realtime "${MODEL:-openai/gpt-oss-20b}" "Reply with OK." --temperature 0 --max-tokens 2 --no-stream
```

The command must exit successfully and return a short response. It verifies
that the CLI, API key, endpoint, and selected model can perform inference.
