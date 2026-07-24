# Token-Minimization Telemetry — Study & Implementation Plan

Date: 2026-06-11

## Pause checkpoint for previous development line

The Team Operating Room / Mission Control development line is paused precisely after:

- Kanban Dashboard plugin version: `0.1.26`.
- Completed slice: **Targeted Rework & Focused Challenge v1**.
- Implemented cycle on a single task:
  `TEAM_SINGLE_TASK_CONFIRMED -> TEAM_RUNTIME_STARTED -> TEAM_PERSPECTIVE -> TEAM_CHALLENGE -> TEAM_SYNTHESIS_FINAL -> TEAM_REVIEW_DECISION`.
- Daniele review controls implemented:
  `approve`, `request_changes`, `deepen_tension`, `convert_to_task`, `park`.
- Latest added behavior:
  - `request_changes` writes `TEAM_REWORK_REQUEST` + `TEAM_SYNTHESIS_REWORK` on the original task.
  - `deepen_tension` writes `TEAM_FOCUSED_CHALLENGE` on the original task and parses focused pairs such as `finance vs mrv`.
- Latest regression: `82 passed` across:
  - `tests/plugins/test_kanban_mission_control_plugin.py`
  - `tests/plugins/test_subagent_control_plugin.py`
  - `tests/plugins/test_kanban_dashboard_outputs.py`
- Latest static checks passed:
  - `py_compile`
  - `node --check`
  - manifest JSON checks
  - `git diff --check`
- Latest smoke: task `t_9f1ec38b` returned `team_rework_requested`, `focused_challenge_requested`, focused pair `finance -> mrv`, `child_tasks_created=[]`, `dispatch_started=false`, `external_send_started=false`.

Resume this line by surfacing `TEAM_REWORK_REQUEST`, `TEAM_SYNTHESIS_REWORK`, and `TEAM_FOCUSED_CHALLENGE` in Subagent Control Team Operating Room as Rework requests / Focused challenges / Synthesis revisions.

## New objective

Daniele's new objective is **not** token accounting for its own sake.

The objective is:

> Use telemetry as a control loop to make Hermes consume fewer tokens by design.

Telemetry must therefore answer:

1. Which context blocks are expensive?
2. Which blocks are repeatedly resent without producing value?
3. Which tool schemas, memories, skills, summaries, attachments, or history segments create token bloat?
4. Which cacheable prefixes are not hitting cache?
5. Which workflows should be compressed, externalized, retrieved lazily, or cached differently?
6. Which agent/team/runtime patterns reduce total effective tokens per useful outcome?

## Current codebase findings

### Existing token/cost accounting

Relevant files:

- `agent/usage_pricing.py`
  - Defines `CanonicalUsage` with:
    - `input_tokens`
    - `output_tokens`
    - `cache_read_tokens`
    - `cache_write_tokens`
    - `reasoning_tokens`
  - `normalize_usage(...)` normalizes Anthropic, Codex Responses, and OpenAI Chat Completions usage shapes.
  - `estimate_usage_cost(...)` estimates costs and includes cache read/write pricing where known.

- `agent/conversation_loop.py`
  - Per LLM call:
    - estimates rough request tokens before call with `estimate_messages_tokens_rough(...)` and `estimate_request_tokens_rough(...)`.
    - invokes request middleware before execution.
    - normalizes response usage after call.
    - updates `agent.session_*` token counters.
    - writes token counters to `SessionDB.update_token_counts(...)`.
    - logs cache hit stats when provider reports them.

- `hermes_state.py`
  - `sessions` table already has aggregate token columns:
    - `input_tokens`
    - `output_tokens`
    - `cache_read_tokens`
    - `cache_write_tokens`
    - `reasoning_tokens`
    - `api_call_count`
    - cost fields.
  - Current granularity is session-level aggregate, not per-block / per-tool / per-request optimization data.

- `agent/turn_finalizer.py`
  - Final result includes total session counters and `last_prompt_tokens`.

- `plugins/observability/langfuse/__init__.py`
  - Already traces LLM calls and forwards usage/cost details to Langfuse.
  - Useful as external observability, but not sufficient for by-design minimization because it does not rewrite or optimize context.

### Existing context compression and caching

Relevant files:

- `agent/context_engine.py`
  - Defines pluggable context engine interface.
  - Engines decide when/how to compress and receive normalized usage data.
  - Important: this is the natural extension point for smarter context minimization.

- `agent/context_compressor.py` / `agent/conversation_compression.py`
  - Built-in lossy summarization engine.
  - Default config:
    - `compression.enabled=true`
    - threshold `0.50`
    - target ratio `0.20`
    - `protect_last_n=20`
    - `protect_first_n=3`
  - Already prunes old tool results before LLM summarization.
  - Uses auxiliary compression model and structured summary.

- `agent/prompt_caching.py`
  - Applies Anthropic-style prompt cache markers with `system_and_3` strategy:
    - system prompt
    - last 3 non-system messages
  - Cache TTL default: `5m`.

- `agent/agent_runtime_helpers.py::anthropic_prompt_cache_policy(...)`
  - Decides when to enable prompt caching for Anthropic/OpenRouter/Nous Portal/Claude, Qwen/Alibaba-family routes, MiniMax Anthropic-compatible routes, etc.

- `agent/conversation_loop.py`
  - Preserves stable system prompt bytes across turns for prompt cache friendliness.
  - Keeps plugin context out of system prompt intentionally so the cache prefix stays stable.
  - Normalizes whitespace and JSON tool-call arguments to improve prefix/KV cache reuse.

- `hermes_cli/config.py`
  - `prompt_caching.cache_ttl`
  - `openrouter.response_cache` / `response_cache_ttl`
  - compression config.

### Existing middleware/plugin hooks

Relevant files:

- `hermes_cli/middleware.py`
  - Supports `llm_request` middleware that can rewrite request payloads.
  - Supports `llm_execution` middleware that wraps provider execution.
  - This is the right place for **measure-only first**, and later guarded optimization middleware.

- `agent/conversation_loop.py`
  - Calls `apply_llm_request_middleware(...)` before plugins and before sending request.
  - Calls `pre_api_request` observer hook with sanitized request payload + rough token estimates.
  - Calls `run_llm_execution_middleware(...)` around provider execution.

## Gap analysis

Hermes already tracks aggregate usage and has compression/prompt caching.

What is missing for Daniele's goal:

1. **Per-block attribution**
   - Current session counters answer “how many tokens were used?”
   - They do not answer “which part of the prompt caused the tokens?”

2. **Optimization recommendations**
   - Current `/usage` and `/insights` are descriptive.
   - Need prescriptive output: what to remove, cache, summarize, externalize, retrieve lazily.

3. **Cache effectiveness control loop**
   - Cache read/write tokens are counted, but there is no policy feedback such as:
     - cache hit ratio too low
     - system prompt prefix changed
     - tool schemas dominate prompt
     - too many cache writes vs reads

4. **Context budget enforcement by class**
   - No budget by block type:
     - system prompt
     - tools schema
     - skills
     - memory
     - conversation history
     - tool results
     - attachments/images
     - compression summary
     - plugin context
     - prefill.

5. **Multi-agent token ROI**
   - Kanban/team agents can multiply token usage.
   - Need metric: useful outcome per token, not token usage per agent only.

6. **Optimization actions**
   - No explicit “token governor” that can apply safe reductions before an LLM call.

## Proposed architecture: Token Efficiency Control Loop

### Principle

Do not build “billing telemetry”. Build an optimization loop:

```text
measure -> attribute -> diagnose -> recommend -> optimize -> verify
```

### Layer 1 — Request shape telemetry, no mutation

Add a local token telemetry module, e.g.:

- `agent/token_telemetry.py`
- optional plugin/dashboard surface later.

At each LLM request, capture a bounded record with:

```json
{
  "schema": "hermes.token_efficiency.v1",
  "session_id": "...",
  "turn_id": "...",
  "api_request_id": "...",
  "platform": "cli|telegram|kanban|cron|dashboard",
  "provider": "...",
  "model": "...",
  "api_mode": "...",
  "message_count": 42,
  "tool_count": 18,
  "rough_request_tokens": 123456,
  "actual": {
    "input_tokens": 10000,
    "output_tokens": 1000,
    "cache_read_tokens": 80000,
    "cache_write_tokens": 12000,
    "reasoning_tokens": 500
  },
  "cache": {
    "hit_ratio": 0.80,
    "write_to_read_ratio": 0.15,
    "provider_reported": true
  },
  "blocks": [
    {"kind": "system", "rough_tokens": 18000, "cacheable": true, "stable_hash": "..."},
    {"kind": "tools_schema", "rough_tokens": 22000, "count": 18},
    {"kind": "skills", "rough_tokens": 9000},
    {"kind": "memory", "rough_tokens": 2500},
    {"kind": "summary", "rough_tokens": 6000},
    {"kind": "history_recent", "rough_tokens": 30000},
    {"kind": "tool_results", "rough_tokens": 14000},
    {"kind": "attachments", "rough_tokens": 4800}
  ]
}
```

Do not store raw prompts by default. Store hashes, sizes, categories, and safe previews only.

### Layer 2 — Per-block attribution

Implement a request classifier before provider call:

- System prompt = first system message.
- Tool schemas = `tools` payload, not messages.
- Conversation history = messages by role and age.
- Tool results = role `tool` messages.
- Compression summary = message content starting with `SUMMARY_PREFIX` / context compaction marker.
- Skills/memory/user profile = initially detected inside system prompt by section markers, later source-attributed in prompt builder.
- Attachments/images = multimodal parts; estimate with existing image token heuristic.
- Plugin/pre-LLM context = injected into user message; classify by marker/source where available.

Important implementation note: first pass can classify coarsely. A second pass should add source-level metadata in `agent/prompt_builder.py` rather than trying to regex parse the final system prompt forever.

### Layer 3 — Token efficiency metrics

Primary metrics should optimize behavior:

- **Effective paid input tokens**:
  `input_tokens + cache_write_tokens * write_cost_weight + cache_read_tokens * read_cost_weight`
  not raw prompt tokens.

- **Cache hit ratio**:
  `cache_read_tokens / (input_tokens + cache_read_tokens + cache_write_tokens)`.

- **Context churn**:
  stable prefix hash changed between turns? If yes, why?

- **Tool schema overhead**:
  tool schema tokens / request tokens.

- **Tool result bloat**:
  tool result tokens / request tokens.

- **Compression ROI**:
  tokens before compression vs after compression vs summary cost.

- **Subagent/team ROI**:
  tokens per finished task / review output / accepted synthesis.

- **Avoidable tokens** estimate:
  tokens likely removable via narrower toolsets, summary, lazy retrieval, output pruning, memory compaction, or skill selection.

### Layer 4 — Recommendations before mutation

Add a diagnostic command/surface before any automatic optimization:

- CLI/slash: `/token-efficiency` or extend `/usage`/`/insights`.
- Dashboard/Mission Control panel later: “Token Efficiency”.

Example output:

```text
Token efficiency: needs attention
- 43% of prompt is tool schemas; enable only web+file+terminal for this task.
- Cache hit ratio 12%; system prefix changed 4 turns in a row because plugin context entered system prompt.
- Tool results are 28K tokens; prefer search_files/read_file pagination or summarize tool output.
- Compression would save ~65K prompt tokens; run /compress or lower threshold for this profile.
```

### Layer 5 — Guarded optimizer / Token Governor

Only after measuring:

Config block proposal:

```yaml
token_efficiency:
  enabled: true
  mode: observe        # observe | recommend | optimize_safe
  store_raw_prompts: false
  local_db: true
  max_records_per_session: 200
  budgets:
    tool_schema_ratio_warn: 0.30
    tool_result_ratio_warn: 0.25
    cache_hit_ratio_warn: 0.40
    request_tokens_warn: 120000
  optimizer:
    narrow_toolsets: recommend   # recommend first, never silently remove tools mid-task
    prune_old_tool_results: safe
    compress_early: recommend
    lazy_retrieve_history: future
    skill_context_budget: recommend
```

Safe automatic optimizations later:

1. Prune/replace old bulky tool results more aggressively before LLM calls.
2. Prefer paginated file reads and search snippets, guided by tool middleware warnings.
3. Auto-trigger compression earlier only when compression ROI estimate is high and aux model is healthy.
4. Use context-engine retrieval for old details instead of replaying large summaries.
5. Route auxiliary tasks to cheaper models with adequate context.

Unsafe optimizations requiring explicit confirmation:

- Disabling toolsets mid-session.
- Dropping skills/memory from system prompt.
- Changing model/provider.
- Changing compression threshold globally.
- Changing team/kanban dispatch patterns.

## Recommended implementation phases

### Phase 0 — Baseline study and checkpoint

Status: done in this plan.

### Phase 1 — Local token telemetry store, observe-only

Implement:

- `agent/token_telemetry.py` with:
  - `class TokenTelemetryRecord`
  - `class TokenBlockEstimate`
  - `class TokenEfficiencyStore`
  - `class TokenEfficiencyAnalyzer`
- SQLite or JSONL under profile-aware path:
  - Prefer SQLite: `$HERMES_HOME/token_efficiency.db`.
  - Simpler first slice: `$HERMES_HOME/telemetry/token_efficiency.jsonl`.
- Hook into `agent/conversation_loop.py` around existing pre/post LLM call data.
- Store no raw prompts by default.

Tests:

- block attribution for system/tools/history/tool results/summary/images.
- no raw prompt content persisted when `store_raw_prompts=false`.
- actual usage merged into the pre-call record.
- cache hit ratio computed correctly.

### Phase 2 — `/token-efficiency` report

Implement CLI/slash report from local telemetry:

- Current session summary.
- Last request breakdown.
- Top avoidable token sources.
- Cache hit quality.
- Compression ROI estimate.
- Recommended safe next action.

Keep it read-only.

### Phase 3 — Token Efficiency dashboard panel

Add read-only Mission Control/diagnostic panel:

- session-level token budget cards.
- cache hit trend.
- biggest block classes.
- “recommended optimizations” list.

Do not add mutation controls here yet.

### Phase 4 — Safe optimizer middleware

Use `llm_request` middleware only after enough telemetry exists:

- `observe`: log only.
- `recommend`: annotate diagnostics only.
- `optimize_safe`: only safe, deterministic reductions that do not change task semantics.

Candidate first safe optimizer:

- If old tool result messages outside protected tail exceed threshold, replace full content with compact placeholder or generated extract, similar to compressor Phase 1 but pre-call and transparent.

### Phase 5 — Context-engine evolution

If metrics show summaries are too large or lossy, build/enable a new context engine:

- `plugins/context_engine/token_efficient_lcm/`
- store old context in a retrieval index/DAG.
- inject only relevant snippets by task/user query.
- expose tools like `context_search`, `context_expand`.

This is the deeper “consume less by design” step.

## First concrete slice I would implement next

**Token Efficiency Telemetry v1 — observe-only**

Status: implemented 2026-06-11.

Implemented files:

- `agent/token_telemetry.py`
- `tests/agent/test_token_telemetry.py`
- `agent/conversation_loop.py` integration

Current behavior:

1. Each successful LLM call with provider usage creates a local JSONL record at `$HERMES_HOME/telemetry/token_efficiency.jsonl`.
2. Records use schema `hermes.token_efficiency.v1`.
3. Records include coarse block attribution:
   - `system`
   - `tools_schema`
   - `history_recent`
   - `tool_results`
   - `summary`
   - `attachments`
4. Records merge post-call normalized usage:
   - `input_tokens`
   - `output_tokens`
   - `cache_read_tokens`
   - `cache_write_tokens`
   - `reasoning_tokens`
5. Records compute cache metrics:
   - provider-reported cache stats boolean
   - cache hit ratio
   - write/read ratio
6. Records include first diagnostics:
   - `tool_schema_overhead`
   - `tool_result_bloat`
   - `low_cache_hit_ratio`
7. Raw prompts, raw user content, raw tool outputs, and raw tool schemas are not persisted by default; only counts and stable hashes are stored.

Verification:

- RED test first: `ModuleNotFoundError: No module named 'agent.token_telemetry'`.
- Focused green: `3 passed in 0.39s`.
- Relevant regression: `53 passed, 1 warning in 1.77s` on:
  - `tests/agent/test_token_telemetry.py`
  - `tests/agent/test_usage_pricing.py`
  - `tests/plugins/test_langfuse_plugin.py`
- Static checks:
  - `python -m py_compile agent/token_telemetry.py agent/conversation_loop.py agent/usage_pricing.py`
  - `git diff --check`

Original definition of done:

1. Every LLM call produces one local safe telemetry record.
2. Record contains block-level rough token attribution and post-call actual usage.
3. No raw prompt text or tool arguments are persisted by default.
4. Tests prove cache hit ratio, block attribution, and no-raw-prompt behavior.
5. A simple CLI/dev function can summarize last N records.

Implementation note: v1 writes records after a successful provider response that includes usage metadata. A follow-up hardening slice should also emit no-usage/error records for providers or failures that do not return usage.

## Next concrete slice

**Token Efficiency Report v1**

Status: implemented 2026-06-11.

Implemented behavior:

- Added `render_token_efficiency_report(...)` in `agent/token_telemetry.py`.
- Added CLI slash command `/token-efficiency [N]` with alias `/tokens`.
- Command is read-only and CLI-only for v1.
- Report reads `$HERMES_HOME/telemetry/token_efficiency.jsonl` through `TokenEfficiencyStore`.
- Report shows:
  - records analyzed;
  - input/output/cache read/cache write tokens;
  - cache hit ratio;
  - top prompt blocks by rough tokens;
  - last request model/provider/request size/cache hit/diagnostics;
  - concrete recommendations.
- Recommendations currently include:
  - use narrower toolsets when `tools_schema` dominates;
  - prune or summarize old tool outputs when `tool_results` bloat is detected;
  - preserve stable prefixes when cache hit ratio is low;
  - estimate compression ROI for large summaries/history.
- Empty store behavior is explicit: "No token efficiency records yet. Run a model call..."
- Still no request mutation: no tools, context, model, or cache policy are changed.

Verification:

- RED test first: `render_token_efficiency_report` import failed until implemented.
- Focused green: `5 passed in 0.27s` for `tests/agent/test_token_telemetry.py`.
- Registry/CLI focused: `7 passed in 0.58s`.
- Relevant regression: `199 passed, 1 warning in 12.20s` on:
  - `tests/agent/test_token_telemetry.py`
  - `tests/agent/test_usage_pricing.py`
  - `tests/plugins/test_langfuse_plugin.py`
  - `tests/hermes_cli/test_commands.py`
- Static checks:
  - `python -m py_compile agent/token_telemetry.py agent/conversation_loop.py cli.py hermes_cli/commands.py`
  - `git diff --check`
- Local smoke report printed expected sections and recommendations for synthetic bloated telemetry.

## Next concrete slice

**Token Efficiency Hardening v1**

Status: implemented 2026-06-11.

Implemented behavior:

- Added privacy-safe no-usage records via `finalize_token_efficiency_no_usage(...)`.
- Added privacy-safe error/retry records via `finalize_token_efficiency_error(...)`.
- Added compression ROI events via `build_compression_efficiency_event(...)`.
- `conversation_loop.py` now appends:
  - `status: "no_usage"` when a provider returns a usable response without usage metadata;
  - `status: "error"` for API exceptions and invalid/malformed API responses;
  - `status: "compression"` after automatic context compression, with before/after rough tokens and saved token ROI.
- Error records store only error type, retry count, and will-retry flag. They do not store exception messages, raw prompt, raw payload, or raw tool outputs.
- Compression records store before/after/saved tokens and summary hash only. They do not store compressed summary text.
- `/token-efficiency` now surfaces:
  - no-usage record count;
  - error record count;
  - compression event count;
  - compression saved rough tokens;
  - recommendations for provider usage blind spots, retry/fallback overhead, and compression ROI tuning.

Verification:

- RED test first: helper imports failed until implemented.
- Focused green: `9 passed in 0.35s` for `tests/agent/test_token_telemetry.py`.
- Relevant regression: `203 passed, 1 warning in 13.87s` on:
  - `tests/agent/test_token_telemetry.py`
  - `tests/agent/test_usage_pricing.py`
  - `tests/plugins/test_langfuse_plugin.py`
  - `tests/hermes_cli/test_commands.py`
- Static checks:
  - `python -m py_compile agent/token_telemetry.py agent/conversation_loop.py cli.py hermes_cli/commands.py`
  - `git diff --check`
- Local smoke report showed no-usage/error/compression counts and recommendations without leaking private payload text.

## Next concrete slice

**Token Efficiency Dashboard Panel v1**

Status: implemented 2026-06-11.

Implemented behavior:

- Kanban Dashboard plugin version bumped to `0.1.27`.
- Added protected read-only endpoint:
  - `GET /api/plugins/kanban-dashboard/token-efficiency?limit=N`
- Added `token_efficiency` block inside:
  - `GET /api/plugins/kanban-dashboard/overview`
- Added Mission Control panel `Token Efficiency` in the plugin bundle, visible near the top after summary cards.
- Panel shows:
  - records;
  - cache hit;
  - error/no-usage counts;
  - compression saved rough tokens;
  - top context blocks;
  - recommendations.
- Guardrails returned in JSON:
  - `no_request_mutation=true`;
  - `no_auto_optimization=true`;
  - `no_raw_prompt=true`;
  - `no_raw_tool_output=true`;
  - `dashboard_panel_only=true`.
- UI copy is explicit: `observe-only`, `No auto-optimization`, `Riduzione token by design`.

Verification:

- RED backend/API test first: endpoint returned `404` and overview lacked `token_efficiency`.
- RED UI bundle test first: `Token Efficiency` strings absent from bundle.
- Focused green: `3 passed in 1.58s`.
- Relevant regression: `70 passed in 18.95s` on:
  - `tests/agent/test_token_telemetry.py`
  - `tests/plugins/test_kanban_mission_control_plugin.py`
  - `tests/plugins/test_kanban_dashboard_outputs.py`
- Static checks:
  - `python -m py_compile agent/token_telemetry.py agent/conversation_loop.py plugins/kanban-dashboard/dashboard/plugin_api.py`
  - `node --check plugins/kanban-dashboard/dashboard/dist/index.js`
  - `python -m json.tool plugins/kanban-dashboard/dashboard/manifest.json`
  - `git diff --check`
- Live dashboard restarted in tmux `hermes-dashboard` on `127.0.0.1:9130`.
- Smoke backend `9130`:
  - `/api/plugins/kanban-dashboard/token-efficiency` returned `200`, label `Token Efficiency`, `read_only=true`, mode `observe_only`.
  - `/api/plugins/kanban-dashboard/overview` returned `token_efficiency` block with `no_auto_optimization=true`.
- Smoke local proxy `9129` returned `200` for `/api/plugins/kanban-dashboard/token-efficiency`.

## Next concrete slice

**Token Efficiency Optimizer Recommendation Preview v1**

Status: implemented 2026-06-11.

Implemented behavior:

- Added `build_token_efficiency_optimizer_preview(records)` in `agent/token_telemetry.py`.
- Added Kanban Dashboard protected endpoint:
  - `GET /api/plugins/kanban-dashboard/token-efficiency/optimizer-preview?limit=N`
- Added `token_efficiency_optimizer_preview` block inside:
  - `GET /api/plugins/kanban-dashboard/overview`
- Added Mission Control UI component:
  - `TokenOptimizerPreviewPanel`
- Bumped Kanban Dashboard plugin:
  - `0.1.27` → `0.1.28`

Preview proposal keys:

- `narrow_toolsets`
- `compress_now`
- `prune_tool_results`
- `stabilize_prefix`
- `lazy_retrieve_history`

Every proposal returns:

- `mutation_available=false`
- `requires_approval=true`
- `auto_applied=false`
- `estimated_savings_tokens`
- `confidence`
- bounded evidence only, no prompt/tool-output text.

Top-level guardrails:

- `no_request_mutation=true`
- `no_context_mutation=true`
- `no_toolset_mutation=true`
- `no_model_change=true`
- `no_cache_policy_change=true`
- `no_auto_apply=true`
- `approval_required_for_future_optimizer=true`

Verification:

- RED helper test first: import failed because `build_token_efficiency_optimizer_preview` did not exist.
- RED backend/API test first: endpoint returned `404`; overview lacked `token_efficiency_optimizer_preview`.
- RED UI bundle test first: bundle lacked `Optimizer Recommendation Preview`, `TokenOptimizerPreviewPanel`, `Preview-only`, `auto_applied=false`.
- Focused green: `5 passed in 1.68s`.
- Relevant regression: `74 passed in 17.59s` on:
  - `tests/agent/test_token_telemetry.py`
  - `tests/plugins/test_kanban_mission_control_plugin.py`
  - `tests/plugins/test_kanban_dashboard_outputs.py`
- Static checks:
  - `python -m py_compile agent/token_telemetry.py agent/conversation_loop.py plugins/kanban-dashboard/dashboard/plugin_api.py`
  - `node --check plugins/kanban-dashboard/dashboard/dist/index.js`
  - `python -m json.tool plugins/kanban-dashboard/dashboard/manifest.json`
  - `git diff --check`
- Live dashboard restarted in tmux `hermes-dashboard` on `127.0.0.1:9130`.
- Smoke backend `9130` and local proxy `9129`:
  - `/api/plugins/kanban-dashboard/token-efficiency/optimizer-preview` returned `200`, label `Optimizer Recommendation Preview`, mode `preview_only`, `mutation_available=false`, `auto_applied=false`.
  - `/api/plugins/kanban-dashboard/overview` returned `token_efficiency_optimizer_preview` with `requires_approval=true`.

## Next concrete slice

**Token Efficiency Safe Optimizer Action Specs v1**

Status: implemented 2026-06-11.

Scope decision: kept deliberately small to avoid over-engineering. This slice only adds inspectable specs to existing optimizer-preview proposals. It does not add apply endpoints, background jobs, new optimizer middleware, or new workflow state.

Implemented behavior:

- Added compact `action_spec` objects to optimizer preview proposals in `agent/token_telemetry.py`.
- Proposal specs cover:
  - `narrow_toolsets` → `kind=toolset_scope`, preserved capabilities, lost capabilities to review, fallback.
  - `compress_now` → `kind=compression`, preserve recent decisions/open tasks, fallback.
  - `prune_tool_results` → `kind=tool_result_policy`, preserve recent tool results, fallback.
  - `stabilize_prefix` → `kind=cache_prefix`, preserve instruction priority, fallback.
  - `lazy_retrieve_history` → `kind=lazy_retrieval`, `retrieval_fallback=session_search_or_full_context`, fallback.
- Added function-first guardrails to optimizer preview:
  - `function_first=true`
  - `no_capability_degradation_without_review=true`
  - `avoid_over_engineering=true`
- Dashboard endpoint `/api/plugins/kanban-dashboard/token-efficiency/optimizer-preview` now returns action specs.
- Mission Control UI shows compact `Action spec · Function first` blocks, including kind, capability risk, what would change, fallback, and `apply_endpoint=null · auto_apply_allowed=false`.
- Kanban Dashboard plugin bumped to `0.1.29`.

Verification:

- RED core test first failed on missing `function_first`/`action_spec`.
- RED UI test first failed on missing `Action spec` text in bundle.
- Focused: `3 passed in 1.26s`.
- Relevant regression: `77 passed in 16.85s` on:
  - `tests/agent/test_token_telemetry.py`
  - `tests/plugins/test_kanban_mission_control_plugin.py`
  - `tests/plugins/test_kanban_dashboard_outputs.py`
- Static checks:
  - `python -m py_compile agent/token_telemetry.py agent/conversation_loop.py plugins/kanban-dashboard/dashboard/plugin_api.py`
  - `node --check plugins/kanban-dashboard/dashboard/dist/index.js`
  - `python -m json.tool plugins/kanban-dashboard/dashboard/manifest.json`
  - `git diff --check`
- Live dashboard restarted in tmux `hermes-dashboard` on `127.0.0.1:9130`.
- Live smoke backend `9130` and proxy `9129` returned `200` for optimizer preview with `function_first=true`, `avoid_over_engineering=true`, `mutation_available=false`. Real store had no hotspot proposals at smoke time, so proposal count was correctly `0`; synthetic tests cover populated specs.

## Next concrete slice

**Token Efficiency Low-Noise Closure v1**

Status: implemented 2026-06-11.

Purpose: close the token-efficiency loop by making the recent work functional in Mission Control, not noisier. Token-efficiency stays available, but only asks for cockpit attention when there is an actionable hotspot.

Implemented behavior:

- Added `visibility` to `build_token_efficiency_optimizer_preview(...)`:
  - no hotspots/proposals → `needs_attention=false`, `recommended_surface=diagnostics_collapsed`, `reason=no_actionable_token_hotspots`, `cockpit_card=null`.
  - actionable proposals with estimated savings → `needs_attention=true`, `recommended_surface=mission_control_review`, `reason=actionable_token_hotspots`, and a small `cockpit_card` with `autonomy_gate=review_only_no_auto_apply`.
- Mission Control UI now shows `Diagnostics collapsed` / `No actionable token hotspots` instead of a full empty optimizer panel when there is nothing useful to review.
- When attention is needed, UI shows `Token hotspots require review` and then the proposal/action-spec cards.
- Kanban Dashboard plugin bumped to `0.1.30`.

Verification:

- RED tests first failed on missing `visibility` and missing UI collapse copy.
- Focused: `3 passed in 1.01s`.
- Relevant regression: `80 passed in 19.53s` on:
  - `tests/agent/test_token_telemetry.py`
  - `tests/plugins/test_kanban_mission_control_plugin.py`
  - `tests/plugins/test_kanban_dashboard_outputs.py`
- Static checks:
  - `python -m py_compile agent/token_telemetry.py agent/conversation_loop.py plugins/kanban-dashboard/dashboard/plugin_api.py`
  - `node --check plugins/kanban-dashboard/dashboard/dist/index.js`
  - `python -m json.tool plugins/kanban-dashboard/dashboard/manifest.json`
  - `git diff --check`
- Live dashboard restarted in tmux `hermes-dashboard` on `127.0.0.1:9130`.
- Live smoke backend `9130` and proxy `9129` returned `200` with:
  - `needs_attention=false`
  - `recommended_surface=diagnostics_collapsed`
  - `reason=no_actionable_token_hotspots`
  - `mutation_available=false`
  - `function_first=true`

## Closure recommendation

The token-efficiency branch is now functionally closed for the current product stage:

- it observes;
- diagnoses;
- reports;
- survives blind spots;
- previews optimizations;
- explains action specs;
- stays low-noise unless useful;
- never applies changes automatically.

Do not proceed to `optimize_safe` yet. Next development should return to core Hermes capability / decision-cockpit value unless real telemetry produces recurring actionable hotspots.

## Design guardrails

- Token minimization must never silently degrade task quality.
- Never remove tools, memory, skills, or context without clear policy and, initially, user approval.
- Optimize stable prefixes and repeated context first.
- Prefer lazy retrieval over huge always-on context.
- Prefer source attribution over regex parsing long prompts.
- Keep telemetry local and privacy-safe by default.
- Treat cache hit ratio as a design metric, not just a billing stat.

## Open implementation questions

1. JSONL vs SQLite for v1 telemetry store.
   - JSONL is fastest to implement and inspect.
   - SQLite is better for dashboard queries and long-term trend analysis.
   - Recommendation: SQLite if implementing directly in core; JSONL only for spike.

2. Whether to implement as core module or plugin.
   - Core is better because the agent loop already has all request/usage data.
   - Plugin is safer for iteration but may duplicate classification logic.
   - Recommendation: core module for record construction + optional dashboard/plugin consumer.

3. How aggressive should future optimization be?
   - Start observe-only.
   - Then recommend-only.
   - Then safe optimize with exact, tested transformations.

4. How to attribute system prompt sections precisely.
   - Short term: parse section headers.
   - Better: modify prompt builder to emit section spans/metadata for telemetry, without changing prompt bytes.

## Final recommendation

Build token telemetry as a **Token Efficiency Control Loop**, not a billing dashboard.

The system should answer:

> What should Hermes stop resending, cache better, compress earlier, retrieve lazily, or move out of the always-on prompt?

Only after that should it mutate requests automatically.
