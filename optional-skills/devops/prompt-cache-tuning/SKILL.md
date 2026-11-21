---
name: prompt-cache-tuning
description: Diagnose cache misses from local usage evidence.
version: 1.0.0
author: Pedro Relvas (@PRELVAS94), Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [prompt-caching, performance, cost, diagnostics]
    category: devops
    related_skills: [hermes-agent]
---

# Prompt Cache Tuning Skill

Diagnose low prompt-cache reuse using local usage records and a controlled
comparison. This skill reports token reuse, not guaranteed savings, and does not
change configuration, send requests, or inspect credentials automatically.

## When to Use

- Cache reads disappear during an otherwise stable conversation.
- Input costs rise after a model, routing, prompt, or toolset change.
- Compare a proposed cache improvement against a recorded baseline.

## Prerequisites

- Python 3.10+; the helper uses only the standard library on all listed platforms.
- An explicitly selected, local UTF-8 JSONL file with one usage object per line,
  either bare or wrapped in `{"usage": {...}}`. Choose one API shape per file.
- Export only usage counters. Do not copy prompts, credentials, conversation IDs,
  or entire API responses into a shared report. No API key is needed here.
- Optional Hermes logs/configuration supplied by the user for diagnosis. Read only
  relevant non-secret settings; never read `.env` or credential stores.

## How to Run

Load this skill with `skill_view`. Use `terminal` to execute the helper from the
installed skill directory (use the directory returned by `skill_view`):

```text
python scripts/summarize_usage.py /path/to/usage.jsonl --format chat
python scripts/summarize_usage.py /path/to/usage.jsonl --format responses
python scripts/summarize_usage.py /path/to/usage.jsonl --format anthropic
```

Replace the placeholder input path with the user's file. Output is JSON on stdout;
input files are never modified. `-` reads stdin. Errors return status 2 without
printing input contents. The helper makes no network calls and needs no Hermes
installation. See [usage formats and limitations](references/usage-formats.md).

## Quick Reference

| Wire usage shape | Cache-read field | Prompt denominator |
| --- | --- | --- |
| Chat Completions | `prompt_tokens_details.cached_tokens` | `prompt_tokens` |
| Responses | `input_tokens_details.cached_tokens` | `input_tokens` |
| Native Anthropic Messages | `cache_read_input_tokens` | `input_tokens` + cache reads + cache writes |
| DeepSeek-style Chat | `prompt_cache_hit_tokens` | `prompt_tokens` |

`cache_read_fraction` is the sum of cache reads divided by the sum of prompt tokens
for measurable calls, not an average of per-call percentages. Missing counters
are unknown, not zero; `measured_calls` and `unmeasured_calls` expose coverage.
An explicit zero is retained. Confirm the chosen wire shape before comparing.

## Procedure

1. Establish evidence. Record model, provider/endpoint, API shape, session
   continuity, toolset, and idle gaps separately from the usage-only file. Use
   `read_file` or `search_files` on user-selected logs if necessary. Hermes API-call
   logs can include `cache=read/total`, `write=`, and `upstream=`; an omitted
   `cache=` field alone cannot distinguish zero reads from unavailable telemetry.
2. Run the helper on a baseline segment with several consecutive requests from
   one conversation. Separate cold starts, post-compression calls, idle gaps, and
   fallback routes from steady-state calls. Do not infer a cause from one zero.
3. Check prefix stability. Keep system text, memories, model slug, tool schemas,
   and historical messages stable during the comparison. Put newly retrieved
   content in new user messages or tool results, not a rewritten system prompt.
   Compression intentionally changes context; re-warming afterwards is expected.
4. Check session/routing continuity. Hermes derives cache scope and affinity from
   the conversation; do not invent or rotate IDs per request, or share one ID
   across unrelated conversations. OpenRouter's provider adapter sends a sticky
   `session_id` and provider preferences together. A configured `provider.order`
   is not, by itself, proof of broken caching. Compare the actual serving route;
   do not remove privacy, region, or endpoint restrictions to chase a hit rate.
5. Check provider requirements using `web_extract` on its current official docs.
   Minimum cacheable sizes, retention, accounting, and support differ by model
   and route. Hermes handles supported Anthropic-style markers; do not manually
   add them to arbitrary APIs. Check the installed version's cache policy before
   considering TTL changes; a longer write tier can cost more without paying off.
6. Propose one reversible change. Obtain user approval before changing provider,
   model, toolset, or cache settings. Keep a rollback copy and apply prompt/toolset
   changes between conversations. A smaller toolset can reduce prompt size but
   may also reduce capability; do not remove tools needed for the task.
7. Compare a similar post-change segment with the same helper. Record coverage,
   reuse fraction, call counts, and cold-start/idle conditions. Without provider
   rates and reported billing, do not convert token reuse into money saved.

## Pitfalls

- Treating every model, proxy, or OpenAI-compatible endpoint as cache-capable.
- Mixing native Anthropic counters with Chat totals: their denominators differ.
- Assuming a non-zero counter proves every provider's hit semantics. Hermes notes
  that MiniMax-M3 native usage can carry a constant cache-read floor; compare
  consecutive uncached input counts and provider documentation as well.
- Aggregating unrelated sessions, fallback providers, or auxiliary calls into one
  apparent warm-session hit rate. The helper cannot identify these boundaries.
- Editing SOUL, memories, system context, or tools during a measurement, then
  mistaking the expected cold prefix for a routing fault.
- Claiming a universal target percentage, TTL, discount, or guaranteed savings.

## Verification

- Check that `calls`, `measured_calls`, and `unmeasured_calls` match the export.
- Inspect `per_call` for cold/warm transitions. Missing counters remain `null`;
  an all-unmeasured or zero-denominator file has no reuse fraction.
- Re-run against the same input: the report must be identical. Verify that input
  contents and live configuration were not changed by the helper.
- Report what the evidence supports, what remains unknown, the proposed change,
  and its rollback. A local analysis is not a live provider-cache verification.

For architecture, use the existing Hermes docs on
[context compression and caching](https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching).
This skill complements that page with offline measurement and a diagnostic workflow.
