# Context Guardrails

## Overview

Context guardrails prevent oversized prompts, retry storms, and runaway context growth **before** provider throttling occurs. This is a proactive advisory system that monitors token consumption and recommends mitigation strategies.

**Philosophy:** Detect pressure early, recommend action, never truncate automatically. The operator decides when and how to manage context.

## Architecture

```
User Request
    ↓
Estimate Token Count
    ↓
Check Historical Context Growth
    ↓
Detect Retry Amplification Patterns
    ↓
Calculate Context Pressure State
    ↓
Generate Advisory Recommendations
    ↓
Display to Operator (human decides)
```

## Context States

### SAFE (< 50% of provider limit)
**Characteristics:**
- Token budget: 0-50% of provider context window
- Growth rate: Normal (< 10% per turn)
- Retry count: 0-1 recent retries
- Compression ratio: < 1.5x original

**Advisory:** None. Operate normally.

**Thresholds (examples):**
- GPT-4 Turbo (128K): < 64,000 tokens
- Claude 3.5 Sonnet (200K): < 100,000 tokens
- Qwen2.5-Coder (32K): < 16,000 tokens

### LARGE (50-75% of provider limit)
**Characteristics:**
- Token budget: 50-75% of provider context window
- Growth rate: Elevated (10-25% per turn)
- Retry count: 2-3 recent retries
- Compression ratio: 1.5-2.5x original

**Advisory:** Monitor closely. Consider decomposition for next major task.

**Recommendations:**
- "Context approaching 75% of provider limit. Consider breaking next large task into smaller chunks."
- "Retry count: 3. If next attempt fails, context reset recommended."

### OVERSIZED (75-90% of provider limit)
**Characteristics:**
- Token budget: 75-90% of provider context window
- Growth rate: High (25-50% per turn)
- Retry count: 4-6 recent retries
- Compression ratio: 2.5-4x original

**Advisory:** Action recommended. High risk of provider rejection.

**Recommendations:**
- "⚠️ Context at 82% of provider limit. Recommend starting fresh session for next task."
- "Retry amplification detected (5 attempts in 10 minutes). Context reset recommended."
- "Compression ratio 3.2x. Original context likely too large to reconstruct efficiently."

### CRITICAL (> 90% of provider limit)
**Characteristics:**
- Token budget: > 90% of provider context window
- Growth rate: Extreme (> 50% per turn)
- Retry count: > 6 recent retries
- Compression ratio: > 4x original

**Advisory:** Immediate action required. Provider rejection imminent.

**Recommendations:**
- "🚨 CRITICAL: Context at 94% of provider limit. Start new session immediately."
- "Retry storm detected (8 attempts in 5 minutes). Stop current operation."
- "Giant prompt detected (estimated 180K tokens). Decompose or switch to longer-context provider."

## Retry Amplification

Retry storms compound context growth exponentially:

```
Turn 1: User message (500 tokens) + Context (10K) → API call fails
Turn 2: Retry with error message (1K) + Context (10K) → 11K context
Turn 3: Retry with debug info (2K) + Context (11K) → 13K context
Turn 4: Retry with full stack trace (5K) + Context (13K) → 18K context
Turn 5: Retry with workaround attempt (3K) + Context (18K) → 21K context
...
```

**Detection:**
- 3+ failed API calls in 10-minute window
- Exponential growth in context size
- Error message accumulation (each retry adds diagnostics)

**Mitigation:**
- "Retry amplification detected. Consider context reset or alternative approach."
- "5 retries with growing context. Switch to simpler model or decompose task."

## Token Estimation

### Counting Strategy

**Conservative estimation** (prefer overestimate):
- Characters × 0.3 = estimated tokens (English)
- Characters × 0.5 = estimated tokens (code, technical)
- Characters × 0.6 = estimated tokens (mixed, conversation)
- Tool results: actual size + 20% overhead for formatting

**Provider-specific adjustments:**
- OpenAI (GPT-4): tiktoken library (gold standard)
- Anthropic (Claude): 1 char ≈ 0.35 tokens
- DeepSeek/Qwen: 1 char ≈ 0.4 tokens (BPE tokenizer)
- Llama models: 1 char ≈ 0.3 tokens

### Components to Track

1. **System Prompt** (static, but can grow with skills/memory)
2. **User Message** (variable)
3. **Conversation History** (cumulative growth)
4. **Tool Results** (can be large: file reads, web scrapes)
5. **Reasoning Content** (o1/DeepSeek R1 models)
6. **Compression Artifacts** (summaries, compressions add tokens)

### Rolling Context Growth

Track growth rate over last N turns:

```bash
Turn N-5: 10,000 tokens
Turn N-4: 12,000 tokens (+20%)
Turn N-3: 15,000 tokens (+25%)
Turn N-2: 19,500 tokens (+30%)
Turn N-1: 26,000 tokens (+33%)
Turn N:   35,000 tokens (+35%) ← ACCELERATING

State: LARGE → OVERSIZED (projected CRITICAL in 2-3 turns)
```

## Giant Prompt Detection

**Definition:** Single turn that significantly exceeds provider's practical comfort zone.

**Thresholds:**
- Small models (< 32K context): > 8K tokens in single turn
- Medium models (32-128K context): > 32K tokens in single turn
- Large models (128K+ context): > 64K tokens in single turn

**Common causes:**
- Massive file reads (entire codebase)
- Bulk web scraping results
- Large log file dumps
- Uncompressed database exports
- Recursive directory listings

**Recommendations:**
- "Giant prompt detected: 85K token file read. Recommend pagination or targeted section."
- "File too large for efficient processing. Use search_files with patterns instead of full read."
- "Consider split-apply-combine: process file in chunks, merge results."

## Compression Ratio Estimation

When context grows beyond limits, agents often compress/summarize. Track effectiveness:

**Healthy compression:**
- Original: 50K tokens → Compressed: 10K tokens (5x ratio)
- Retains key information
- One-time operation

**Unhealthy compression:**
- Original: 50K tokens → Compressed: 40K tokens (1.25x ratio)
- Minimal benefit
- Repeated compressions (2nd, 3rd attempts)
- Loss of critical details

**Detection:**
- Compare pre-compression and post-compression sizes
- Track number of compression attempts in session
- Flag if compression ratio < 2x (ineffective)

**Recommendations:**
- "Compression ratio 1.3x. Context still oversized. Recommend session reset."
- "3rd compression attempt. Diminishing returns. Start fresh session."

## Provider Pressure Scoring

Combine metrics into single pressure score (0-100):

```python
pressure_score = (
    (token_usage_pct * 0.4) +          # 40% weight on absolute usage
    (growth_rate_pct * 0.3) +          # 30% weight on growth velocity
    (retry_count * 5.0) +              # 5 points per retry
    (compression_attempts * 10.0)      # 10 points per compression
)
```

**Interpretation:**
- 0-25: SAFE (green)
- 26-50: LARGE (yellow)
- 51-75: OVERSIZED (orange)
- 76-100: CRITICAL (red)

**Examples:**
- Token usage 40%, growth 5%, 0 retries → Score: 17.5 (SAFE)
- Token usage 70%, growth 20%, 2 retries → Score: 44.0 (LARGE)
- Token usage 85%, growth 30%, 5 retries → Score: 78.0 (CRITICAL)

## Operational Guidance

### Task Decomposition Recommendations

When context pressure is LARGE or higher:

**Bad (monolithic):**
- "Analyze all Python files in this repository and suggest improvements"

**Good (decomposed):**
1. "List all Python files, group by module"
2. "Analyze authentication module for security issues"
3. "Analyze API routes for performance issues"
4. "Analyze database layer for N+1 queries"
5. "Synthesize findings into summary report"

**Script suggestion:**
```bash
# Detect decomposable tasks
if [[ "$task" =~ "all"|"entire"|"whole" ]] && [[ "$context_state" == "LARGE" ]]; then
  echo "Advisory: Task spans multiple areas. Consider sequential processing."
fi
```

### Session Reset Timing

Recommend reset when:
- Pressure score > 60
- Retry count > 4 in last 15 minutes
- Compression attempts > 2
- Natural task boundary (completed feature, finished review)

**Good reset points:**
- After completing a PR review
- After deploying a feature
- After finishing a debugging session
- After a successful test run

**Bad reset points:**
- Mid-debugging (lose context on active issue)
- During multi-file refactor (lose cross-file dependencies)
- In middle of complex explanation

### Ollama Routing Integration

When context pressure is OVERSIZED and task is LOCAL_SAFE:

**Recommendation:**
```
⚠️ Context pressure: OVERSIZED (78% of provider limit)

This appears to be a LOCAL_SAFE workload (log summarization).

Options:
1. Reset session and restart task (recommended for cloud)
2. Switch to local Ollama (qwen2.5-coder:14b supports 32K context, 
   plenty of headroom for this task)
3. Decompose task into smaller chunks

Your choice - all are reasonable given current pressure.
```

**Integration logic:**
- Check context state (OVERSIZED or CRITICAL)
- Classify workload (LOCAL_SAFE, CLOUD_PREFERRED, etc.)
- Recommend local routing if safe + high pressure

### Cloud-Only Escalation

When context pressure is CRITICAL and task is CLOUD_REQUIRED or high-complexity:

**Recommendation:**
```
🚨 CRITICAL: Context pressure at 92%

This is a CLOUD_REQUIRED workload (production deployment review).

Recommended actions:
1. Start fresh session immediately (preserve current session for reference)
2. Use longer-context provider if available (GPT-4 Turbo 128K → Claude 200K)
3. Decompose into smaller review chunks

DO NOT: Attempt local routing (complexity too high for current Ollama models)
```

## Telegram Visibility

Display context state in status messages:

### Normal Operation (SAFE)
```
✓ Task complete

Context: 15.2K / 128K tokens (12%) [SAFE]
```

### Warning State (LARGE)
```
✓ Feature implemented

Context: 78.5K / 128K tokens (61%) [LARGE]
⚠️ Approaching context limit. Consider session reset before next major task.
```

### Alert State (OVERSIZED)
```
✓ Review complete

Context: 105K / 128K tokens (82%) [OVERSIZED]
⚠️ High context pressure. Recommend starting fresh session.
Retry count: 4 | Growth rate: +28% per turn
```

### Critical State (CRITICAL)
```
🚨 CRITICAL: Context at 118K / 128K tokens (92%)

Immediate action required:
• Start new session
• Current session preserved for reference
• Consider task decomposition

Retry storm detected: 7 attempts in 8 minutes
```

### Retry Amplification Warning
```
⚠️ Retry amplification detected
3 failed attempts in 6 minutes, context +35%

Recommended: Stop retries, reset session, alternative approach
```

## Safety Constraints

### What This System Does NOT Do
1. **No prompt rewriting** - Never modifies user input
2. **No automatic truncation** - Never silently drops context
3. **No provider switching** - Never changes configured provider
4. **No destructive actions** - Never deletes history without consent
5. **No secrets handling** - Never inspects credentials or keys

### What This System DOES Do
1. **Monitor token counts** - Track estimated usage
2. **Detect patterns** - Identify retry storms, giant prompts
3. **Calculate pressure** - Score context state (SAFE → CRITICAL)
4. **Generate advisories** - Actionable recommendations
5. **Log metrics** - Audit trail for analysis

## Monitoring and Metrics

Log to `~/.hermes/logs/context-metrics.jsonl`:

```json
{
  "timestamp": "2026-05-23T11:00:00Z",
  "session_id": "abc123...",
  "turn": 15,
  "estimated_tokens": 45000,
  "provider_limit": 128000,
  "usage_pct": 35.2,
  "growth_rate_pct": 12.5,
  "retry_count": 1,
  "compression_attempts": 0,
  "pressure_score": 18.3,
  "state": "SAFE",
  "advisory_shown": false
}
```

Track over time:
- Average pressure score per session
- Frequency of OVERSIZED/CRITICAL states
- Retry storm incidents
- Giant prompt occurrences
- Compression effectiveness

## Future Enhancements (Out of Scope for v1)

- **Automatic context pruning** - Smart removal of old tool results
- **Semantic compression** - LLM-powered summarization with verification
- **Multi-session linking** - Carry minimal context across resets
- **Provider auto-escalation** - Switch to longer-context models when needed
- **Token budget reservations** - Pre-allocate budget for known-large operations
- **Real-time tokenization** - Use actual tokenizer instead of estimates

## References

- [TOKEN_BUDGETING.md](./TOKEN_BUDGETING.md) - Budget allocation strategies
- [OLLAMA_ROUTING_STRATEGY.md](./OLLAMA_ROUTING_STRATEGY.md) - Local routing integration
- OpenAI tokenizer documentation
- Anthropic prompt engineering guide
