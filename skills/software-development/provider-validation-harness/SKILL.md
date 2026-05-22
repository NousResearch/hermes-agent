---
name: provider-validation-harness
description: Use when validating whether a provider, model, or OpenAI-compatible endpoint is ready for real Hermes agent-loop/tool-call use. Runs receipt-based `hermes providers validate` checks and frames results as deployment readiness, not a leaderboard.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [providers, validation, hermes, tool-calling, local-models, evaluation]
    related_skills: [hermes-agent, github-pr-workflow, requesting-code-review]
---

# Provider Validation Harness

## Overview

Use this skill when a user wants to know whether a model, provider, or local endpoint is **Hermes-ready**. A model that answers normal chat is not necessarily ready for Hermes. Hermes readiness means the model/provider can survive the real agent loop: tool schemas, actual tool calls, persisted sessions, recovery from failed tool calls, refusal/abstention boundaries, and user-visible output hygiene.

The harness is intentionally a deployment-readiness screen, not a benchmark. It runs real `hermes chat -Q` subprocess turns and scores the resulting session receipts. Direct `/v1/chat/completions` probes only prove that an endpoint responds; they do not prove agent-loop behavior.

## When to Use

Use this skill for:

- Validating a newly configured provider/model before recommending it for Hermes use.
- Checking an OpenAI-compatible local endpoint before promoting it into a Hermes lane.
- Comparing runtime variants such as MTP on/off, speculative decoding, quantization, prompt caching, context length, or inference server changes.
- Preparing or reviewing PRs that add provider-readiness validation to Hermes.
- Investigating claims like “this provider works” when only raw chat or curl tests have passed.

Do **not** use this as:

- a public model leaderboard;
- an MMLU/GSM8K/arena-style intelligence benchmark;
- a substitute for full task-specific evaluation;
- proof that all Hermes tools work under all contexts;
- permission to send private workflows or vault data through cloud providers.

## Core Principle

Provider validation asks:

> Can this model/provider preserve real Hermes agent-loop behavior with receipts?

It does **not** ask:

> Which model is smartest?

Correctness comes before speed. A faster endpoint is not better if it breaks tool calls, emits malformed tool arguments, leaks hidden reasoning markers, fabricates file/tool work, or fails recovery behavior.

## Current CLI Shape

The harness command lives under `hermes providers validate`:

```bash
hermes providers validate \
  --provider PROVIDER \
  --model MODEL \
  --toolsets file \
  --suite agent-readiness \
  --out /tmp/hermes-provider-validation \
  --timeout 120
```

Arguments:

- `--provider`: provider to validate. Omit to use the configured provider.
- `--model`: model to validate. Omit to use the configured model.
- `--toolsets`: comma-separated toolsets for validation turns. Default: `file`.
- `--suite`: validation suite. Current suite: `agent-readiness`.
- `--out`: directory for JSONL/JSON/Markdown receipts. If omitted, Hermes creates a temp dir.
- `--timeout`: per-case timeout in seconds. Default: `120`.

## What the Agent-Readiness Suite Checks

The current `agent-readiness` suite uses a temporary fixture directory and real `hermes chat -Q` subprocesses. It checks these behaviors:

- **No-tool abstention:** answer exactly without calling tools when tools are unnecessary.
- **Real file read:** call `read_file` and report a marker from an actual file.
- **Real file search:** call `search_files` and report a marker found under a fixture tree.
- **Failed read recovery:** attempt a missing file, recover by reading the correct file, and finish correctly.
- **Side-effect abstention:** avoid write/patch/terminal/code-execution tools when asked for a side effect that should be blocked.
- **Visible reasoning hygiene:** do not leak `<think>`, `<reasoning>`, scratchpad, or hidden reasoning markers into the final user-visible answer.

Scoring checks include:

- subprocess exit status;
- session id discovered in stdout/stderr;
- expected final text found;
- required tools called;
- forbidden tools absent;
- no tools called when tools were forbidden/unnecessary;
- visible reasoning markers absent from final text.

The suite reads persisted Hermes session messages to verify actual tool calls. Assistant text claiming a tool was used is not enough.

## Standard Workflow

1. **Confirm the question.** Decide whether you are validating a provider/model, a local endpoint, or a runtime optimization variant.
2. **Keep scope narrow.** Start with `--toolsets file` unless the user explicitly needs broader tool behavior.
3. **Run the harness.** Save receipts with `--out` when the result matters.
4. **Inspect the summary.** Read `summary.md` or `summary.json` in the output directory.
5. **Inspect failures.** For failed cases, check `results.jsonl` and `raw/<case>.session.json`, plus stdout/stderr.
6. **Classify the result.** Use the failure taxonomy below.
7. **Report with receipts.** Include provider, model, suite, pass count, output directory, and the specific failure reason.

Example:

```bash
OUT=/tmp/hermes-provider-validation-$(date +%Y%m%d-%H%M%S)
hermes providers validate \
  --provider openai-codex \
  --model gpt-5.5 \
  --toolsets file \
  --suite agent-readiness \
  --out "$OUT" \
  --timeout 120

sed -n '1,200p' "$OUT/summary.md"
```

## Local / OpenAI-Compatible Endpoint Pattern

For local endpoints, create or select a Hermes provider config that points at the endpoint, then run the same harness. Do not infer Hermes readiness from a direct curl request to `/v1/chat/completions`.

Example shape, assuming a configured custom provider:

```bash
hermes providers validate \
  --provider custom:local-qwen \
  --model Qwen3.6-27B \
  --toolsets file \
  --suite agent-readiness \
  --out /tmp/qwen36-agent-readiness
```

If the endpoint is only raw-API reachable and not yet configured in Hermes, first add the provider using normal Hermes config/provider setup. Keep secrets in `.env` or config-supported secret fields; do not hardcode credentials in commands, docs, or receipts.

## Runtime Optimization / Variant Evaluation

Use this lane when comparing the **same model** under different runtime settings:

- MTP enabled vs disabled;
- speculative decoding enabled vs disabled;
- quantization A vs B;
- prompt-cache on vs off;
- context-length settings;
- batch/concurrency settings;
- inference servers such as llama.cpp, vLLM, SGLang, TGI, Ollama, or TensorRT-LLM;
- GPU split/tensor-parallel settings.

Rules:

1. Hold the suite fixed.
2. Hold model identity as fixed as possible.
3. Change one runtime variable at a time.
4. Run the baseline first.
5. Compare correctness before speed.
6. Do not promote an optimization that breaks tool calling or receipt verification.
7. Report the optimization as passing only if it preserves agent behavior and improves the intended metric.

Example A/B run:

```bash
BASE=/tmp/hermes-qwen36-no-mtp
MTP=/tmp/hermes-qwen36-mtp

hermes providers validate \
  --provider custom:qwen36-no-mtp \
  --model qwen36-27b \
  --toolsets file \
  --suite agent-readiness \
  --out "$BASE"

hermes providers validate \
  --provider custom:qwen36-mtp \
  --model qwen36-27b \
  --toolsets file \
  --suite agent-readiness \
  --out "$MTP"
```

When reporting A/B results, include both receipt paths and state whether the variant preserved readiness before discussing latency or throughput.

## Reading Results

The output directory contains:

- `summary.md`: human-readable summary.
- `summary.json`: machine-readable suite summary.
- `results.jsonl`: one JSON result per case.
- `fixtures/`: temporary test files used by the suite.
- `raw/<case>.stdout`: captured subprocess stdout.
- `raw/<case>.stderr`: captured subprocess stderr.
- `raw/<case>.session.json`: persisted Hermes session messages when a session id was found.

A real pass requires both the right final text and the right receipts. For tool cases, inspect `tool_calls` in `summary.md`/`results.jsonl` or the raw session JSON.

## Failure Taxonomy

Use these buckets when reporting failures:

- **Provider unreachable:** subprocess exits nonzero before model behavior is evaluated; likely auth, base URL, model name, or service availability.
- **Session persistence failure:** no session id found or session messages cannot be loaded; readiness cannot be proven even if stdout looks good.
- **Tool-call failure:** required tool missing, wrong tool called, invalid tool schema, or tool result ignored.
- **Fabricated tool work:** final answer claims work that is not present in session receipts.
- **Recovery failure:** model cannot recover after an expected failed tool call.
- **Side-effect boundary failure:** model calls write/terminal/code tools when the suite required abstention.
- **Visible reasoning leak:** final user-visible text includes hidden-reasoning markers such as `<think>` or scratchpad text.
- **Harness/config failure:** command shape, local installation, or test fixture issue prevents evaluation.
- **Inconclusive:** partial receipts exist but not enough to classify model/provider behavior.

## PR / Upstream Framing

When upstreaming or reviewing related work, frame this as **provider/deployment readiness**, not benchmarking.

Good framing:

> This helps users avoid configuring providers that can answer normal chat but fail real Hermes tool-call workflows.

Avoid framing like:

> This ranks models.

Keep first PRs small and generic:

- read-only or tightly bounded suite;
- real Hermes subprocess/session receipts;
- docs for interpretation;
- focused tests for parsing, scoring, and command construction;
- no Rig-specific provider names, secrets, paths, or private workflows.

## Common Pitfalls

1. **Treating raw API success as Hermes readiness.** A direct chat-completions response does not prove tool schemas, persisted sessions, or recovery behavior.
2. **Trusting assistant claims.** If a model says it read a file, verify the session tool call and result.
3. **Optimizing speed before correctness.** MTP/speculative/quantization wins do not count if tool behavior regresses.
4. **Running too many toolsets first.** Broad toolsets add context and possible failure modes. Start with `file`, then expand deliberately.
5. **Confusing local endpoint viability with Hermes lane promotion.** A local model can be an operational endpoint while still failing loaded Hermes context or tool-call requirements.
6. **Leaking private workflows into validation prompts.** Keep upstream suites generic and synthetic.
7. **Overgeneralizing a pass.** `agent-readiness` passing means the provider passed this narrow suite, not all Hermes tasks.
8. **Ignoring stdout/stderr receipts.** Timeouts and auth failures often appear outside the final assistant text.
9. **Forgetting provider aliases.** Use the exact provider/model route Hermes will use in production, not a nearby alias.

## Verification Checklist

Before reporting success:

- [ ] `hermes providers validate --help` shows the expected command shape.
- [ ] The intended provider/model route is the one tested.
- [ ] The suite and toolsets are stated.
- [ ] `summary.md` or `summary.json` was inspected.
- [ ] Tool cases have actual `tool_calls` receipts.
- [ ] Failures are classified, not hand-waved.
- [ ] Runtime-variant comparisons hold suite/model constant and change one variable at a time.
- [ ] Output directory is included for reproducibility when useful.
- [ ] No secrets or private user data appear in prompts, paths, docs, or receipts.
- [ ] Upstream PR language says deployment-readiness, not benchmark or leaderboard.
