# Provider/Brain Path Diagnostics Pattern

## When to use this pattern

When a provider-backed brain path (DeepSeek, OpenRouter, OpenAI) is configured but always falls back to local_fallback. The brain returns `LOCAL_FALLBACK_ANSWER_READY` instead of `DEEPSEEK_ANSWER_READY` (or similar provider-specific verdict).

## Diagnostic checklist

### 1. Dotenv loading in CLI scripts

The server (`src/server.js`) loads `.env` via `dotenv.config()`, but CLI scripts (`scripts/openclaw-operator-cli.js`) typically do NOT. When the CLI spawns a brain ask command, it runs without provider env vars.

**Fix**: Add `dotenv` loading at the top of the CLI entry point:

```javascript
#!/usr/bin/env node
import dotenv from 'dotenv';
dotenv.config();
```

### 2. Provider URL path construction

The brain ask handler in `readonly-operator-cli-client.js` constructs the API request URL. A common bug: non-openrouter providers had their base URL path silently stripped.

```javascript
// BEFORE (BUG): for non-openrouter providers, pathname is dropped
const requestPath = brainConfig.provider === 'openrouter'
  ? `${basePath}/chat/completions`
  : '/chat/completions';  // /v1 dropped from https://api.deepseek.com/v1

// AFTER (FIX): preserve pathname for ALL providers
const basePath = String(parsedBase.pathname ?? '').replace(/\/$/, '');
const requestPath = `${basePath}/chat/completions`;
```

**Check**: Verify `DEEPSEEK_BASE_URL` (or equivalent) matches the actual API endpoint format. DeepSeek uses `https://api.deepseek.com/v1` (note the `/v1`). The code must construct `https://api.deepseek.com/v1/chat/completions`, not `https://api.deepseek.com/chat/completions`.

### 3. Timeout configuration

The default timeout for brain ask in `buildBrainAskResponse` is 15s. DeepSeek API latency can reach 28s for complex questions with project knowledge injection.

**Fix**: Increase the default from 15000 to 35000 (covers all 9 benchmark questions with margin):

```javascript
timeoutMs = 35000  // was 15000
```

**To verify**: Measure actual latency for each question via direct `buildBrainAskResponse` call with generous timeout.

### 4. Benchmark runner env inheritance

The benchmark runner (`cloe-operator-benchmark-runner.mjs`) spawns CLI commands via `spawnSync`. When run from within Hermes or restricted environments, `spawnSync` may fail with EPERM, triggering a direct `executeOpenClawOperatorCli()` fallback. This path uses the parent's `process.env` which may not have dotenv-loaded vars (unless the CLI module already loaded them via `dotenv.config()` at import time).

**Fix**: Ensure the CLI entry point calls `dotenv.config()` BEFORE any other imports that read env vars.

### 5. Data contract verification

When a new pipeline stage produces unexpected output, verify the DATA CONTRACTS — not just whether the data exists. Common mismatches:

| Tool | Actual format field | Bad assumption |
|------|-------------------|----------------|
| Review engine | `scenario_reviews[].reviewers[].average_score` | `scenarios[].turns[].behavior_review.naturality_score` |
| Review engine (governance) | `scenario_reviews[].reviewers[].compliance_score` | `scenario_reviews[].reviewers[].average_score` |
| Benchmark runner | `status: "useful"` (provider) or `"partial"` (fallback) | `status === "PASS"` |
| Evidence collector | `scenarios[].turns[].response` (text) | Review engine format (different shape) |

**Fix**: Read the actual output format first, then adapt the consumer. Never assume a field name or structure.

### 6. Benchmark leak pattern tuning for provider answers

The benchmark runner's `INTERNAL_LEAK_PATTERNS` may be too aggressive for provider-backed answers. Patterns like `project_knowledge`, `safety envelope`, and `cognitive context pack` match natural language phrases that a good provider answer would legitimately use.

**Fix**: Narrow patterns to match only actual JSON/internal artifact markers, not natural language:

```javascript
// Narrow patterns — only match actual JSON artifact markers
const INTERNAL_LEAK_PATTERNS = [
  'ReasoningContextEnvelope',
  'selected_collectors',
  'available_items',
  'compact_summary',
  'system prompt',
  'brain pack',
  '"cognitive_context_pack_leak"',  // very specific
  '"project_knowledge"',            // quoted JSON key only
  '"safety_envelope"'              // quoted JSON key only
];
```

## Verification

After applying fixes, run:

```bash
# 1. Verify env vars are loaded
node -e "import('dotenv').then(d => { d.config(); console.log('API_KEY:', !!process.env.DEEPSEEK_API_KEY); })"

# 2. Test direct brain ask
node -e "
import('dotenv').then(async d => {
  d.config();
  const mod = await import('./src/services/readonly-operator-cli-client.js');
  const r = await mod.buildBrainAskResponse({env: process.env, prompt: 'Test query', timeoutMs: 60000});
  console.log('Provider:', r.data?.llm_provider, 'Verdict:', r.data?.final_verdict);
});
"

# 3. Run benchmark
node tools/ci/cloe-operator-benchmark-runner.mjs 2>/dev/null

# 4. Run pipeline
node tools/ci/cloe-cognitive-answer-quality-stabilization.mjs 2>/dev/null
```
