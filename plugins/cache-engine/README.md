# cache-engine

**Context Economy Engine.** A `llm_request` middleware that clips oversized tool
results before they enter the context window — while keeping the prompt prefix
byte-stable, so the provider's prompt cache keeps hitting.

```yaml
plugins:
  enabled:
    - cache-engine

cache_engine:
  mode: report      # report = measure only (default) | enforce = clip
```

## Why the prefix is sacred

A cache miss costs about **50×** a cache read ($0.30 vs $0.006 per million on
DeepSeek). Rewriting history mid-conversation — trimming old messages,
compacting in place, reordering — invalidates the cache from the edit point to
the end of the context and costs more than the bytes it saves. This plugin only
ever touches the messages **after the newest user message**.

## What it does

- Keeps head 60% + tail 40% of the allowed budget (exit codes and stack traces
  live at the edges), replaces the middle with one pointer line:

  ```
  … [38,412 chars / ≈12,804 tok omitted · sha256:9f2ac41b77de] …
  ```

- Per-tool caps (`web_extract`, `skill_view`, `session_search`,
  `memory_recall` get 900 tokens; `read_file` 2000; default 1200; hard ceiling
  6000), clamped to `[64, hard_cap]`.
- One JSONL stats line per request (`~/.hermes/cache-engine/stats.jsonl`).
  Stats writes are wrapped: they can never break a request.

## Measured

Replayed over the full 78 days of history in `state.db` (49,875 tool results, 917 sessions):

| | |
|---|---|
| tool content | 95.4M chars ≈ 31.8M tokens |
| results over cap | 5,516 (11%) |
| tokens never sent | **14.0M (44% of tool content)** |
| prefix messages rewritten | **0 of 85,630** |
| overhead | 0.56 µs median per result |

Reproduce with `scripts/cee_bench.py` in the standalone repo
([Hadi77io/hermes-cache-engine](https://github.com/Hadi77io/hermes-cache-engine)),
which replays `state.db` and asserts prefix invariance.

## Configuration

| key | default | meaning |
|---|---|---|
| `mode` | `report` | `report` measures, `enforce` clips |
| `default_cap` | 1200 | tokens kept per tool result |
| `hard_cap` | 6000 | nothing is clipped above this |
| `head_ratio` | 0.6 | head share of the kept window |
| `chars_per_token` | 3.0 | estimator; measured corpus value is 3.36 |
| `per_tool` | see `config.py` | per-tool caps |
| `stats_path` | `~/.hermes/cache-engine/stats.jsonl` | savings log |

Env overrides: `HERMES_CACHE_ENGINE_MODE`, `HERMES_CACHE_ENGINE_DEFAULT_CAP`.

## Tests

```bash
python -m pytest tests/plugins/test_cache_engine.py -q
```
