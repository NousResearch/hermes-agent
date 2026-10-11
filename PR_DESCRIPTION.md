fix(prompt_builder): trim skill index for cron jobs to avoid 403 context-limit errors

## Problem
When a large skill library is installed (e.g. 2890 skills), `build_skills_system_prompt()` renders the FULL sub-description index (~176K tokens) into every agent's system prompt — including cron jobs. Small/free models (e.g. `tencent/hy3:free`, context window ~128–262K) then hit a 403 context-limit error on the very first request, and the cron job never completes (last_status: error, no output delivered).

## Root Cause (corrected)
The previous revision of this PR inferred "cron-ness" from toolset equality (`_avail_set == cron_enabled_toolsets`). That heuristic was wrong on two counts, confirmed against the actual toolset wiring (`toolsets.py`, `model_tools.py`, `cron/scheduler.py`):

1. The cron agent is built by the scheduler with `platform="cron"` (`cron/scheduler.py:4111`). Whether its toolset list contains `skills` depends on `enabled_toolsets` — and the `skills` toolset is **not** included by `terminal`/`web`/`file` (each is a standalone toolset with `includes: []`). So the equality check either never fired (skills absent → `build_skills_system_prompt` not even called) or never fired for the real 403 case (skills present → set never equal to `[terminal,web,file]`). The fix never actually trimmed anything.
2. Inferring cron-ness from toolset membership is brittle in both directions: a cron job with one extra toolset (e.g. an MCP server) would keep the full index and still 403; a human chat configured with the same toolset set would silently lose skill discovery.

## Fix (this revision)
Use an **explicit cron signal** instead of a heuristic:

- `agent/system_prompt.py` gains `_cron_trim_signal(agent)` which returns `True` **only** when `agent.platform == "cron"` **and** `skills.cron_whitelist` is a non-empty configured list.
- `build_skills_system_prompt()` gains a `cron_trim: bool` parameter. When set, it renders ONLY the whitelisted skills as `name: [category-tag]` (~3–5K tokens) instead of the full sub-description index. When `None` (human chat), the full index is preserved for skill discovery.
- `load_config_readonly()` is read **once per cron build only** (it is already a cached fast-path keyed on file mtime/size, no deepcopy), so the human-chat hot path is untouched.

This is an **opt-in** feature: without `skills.cron_whitelist` configured, cron agents keep the full index (human-equivalent behaviour) — no silent feature drop, matching the project's "don't destroy the feature you secure" guideline.

## Config keys (now declared in DEFAULT_CONFIG)
```yaml
skills:
  cron_whitelist: []          # non-empty list enables cron trimming
  cron_whitelist_only: true   # when false, whitelist narrows descriptions instead of dropping
```
Declaring them in `DEFAULT_CONFIG` makes the opt-in discoverable via `hermes tools`/setup and config docs, and surfaces typos as config validation errors instead of silent no-ops.

## Test
Added `tests/agent/test_cron_skill_trim.py` and `test_cron_trim_renders_only_whitelist` / `test_human_chat_keeps_full_index` in `tests/agent/test_prompt_builder.py`. All pass:
- cron + whitelist → only whitelisted skills render (`name: [category]`), full descriptions dropped
- human chat (no cron_trim) → full sub-description index preserved
- cron without whitelist → no trim (feature preserved)

Before (full index): 178,196 tokens → 403 on tencent/hy3:free.
After (cron + whitelist): 8,719 tokens → cron job completes, last_status: ok.
