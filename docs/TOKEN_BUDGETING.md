# Token Budgeting

## Overview

Token budgeting is the practice of allocating context window capacity across conversation turns to maximize useful work while avoiding provider rejections. This document defines strategies for budget allocation, growth management, and pressure relief.

## Core Principles

1. **Measure before act** - Estimate token costs before making API calls
2. **Reserve headroom** - Never use 100% of available context
3. **Compress early** - Better to compress at 60% than truncate at 95%
4. **Decompose proactively** - Split large tasks before hitting limits
5. **Reset strategically** - Choose reset points that preserve productivity

## Budget Allocation Strategy

### Static Components (20-30% of budget)

**System Prompt:**
- Base instructions: ~2-4K tokens
- Skills loaded: ~1-3K tokens each (multiply by active count)
- Memory/user profile: ~1-2K tokens
- Project context (AGENTS.md): ~5-10K tokens

**Example (GPT-4 Turbo 128K):**
- Base system: 3K tokens
- 3 active skills: 6K tokens
- Memory: 1.5K tokens
- Project context: 8K tokens
- **Total static: 18.5K tokens (14.5% of budget)**

### Dynamic Components (50-70% of budget)

**Conversation history:**
- User messages: variable (500-2K per turn)
- Assistant responses: variable (1-5K per turn)
- Tool results: highly variable (100 bytes - 100KB+)
- Reasoning traces (o1/R1): 2-10x response size

**Growth pattern (typical):**
- Turn 1: 20K tokens (static + first exchange)
- Turn 3: 25K tokens (+5K)
- Turn 5: 32K tokens (+7K)
- Turn 10: 50K tokens (+18K)
- Turn 20: 85K tokens (+35K)

### Reserve Capacity (10-20% of budget)

**Emergency headroom:**
- Error recovery: room for stack traces, diagnostics
- Retry overhead: error messages, alternative attempts
- Compression metadata: summaries, annotations
- Response buffer: ensure model can generate full output

**Example reserves:**
- 128K context → reserve 15K tokens (12%)
- 200K context → reserve 30K tokens (15%)
- 32K context → reserve 5K tokens (16%)

## Budget Thresholds

### Conservative (Default)

| State | Threshold | Action |
|-------|-----------|--------|
| SAFE | 0-50% | Operate normally |
| LARGE | 50-70% | Monitor, prepare for decomposition |
| OVERSIZED | 70-85% | Recommend action (reset/decompose) |
| CRITICAL | 85-100% | Block new large operations |

**Use when:**
- Production operations
- High-stakes tasks (security, financial, legal)
- Unfamiliar models or providers
- Complex multi-turn workflows

### Aggressive (Advanced Users)

| State | Threshold | Action |
|-------|-----------|--------|
| SAFE | 0-60% | Operate normally |
| LARGE | 60-80% | Monitor |
| OVERSIZED | 80-92% | Recommend action |
| CRITICAL | 92-100% | Block |

**Use when:**
- Rapid prototyping / exploration
- Known-safe operations (read-only, local dev)
- Experienced operators who understand risks
- Models with proven stability at high utilization

### Model-Specific Adjustments

**Small context models (< 32K):**
- Shift all thresholds -10% (OVERSIZED at 60%, not 70%)
- Reason: Less room for error, tighter margins

**Large context models (> 128K):**
- Can use aggressive thresholds
- Reason: More headroom, forgiving of estimation errors

**Reasoning models (o1, R1):**
- Shift thresholds -20% (account for hidden reasoning tokens)
- Reason: Internal reasoning consumes significant capacity

## Growth Management

### Healthy Growth Patterns

**Linear growth (ideal):**
```
Turn 1:  20K tokens
Turn 5:  28K tokens (+8K)
Turn 10: 36K tokens (+8K)
Turn 15: 44K tokens (+8K)
```
- Predictable
- Sustainable for 20-30 turns
- Easy to manage

**Bounded growth (good):**
```
Turn 1:  20K tokens
Turn 5:  32K tokens (+12K)
Turn 10: 40K tokens (+8K)
Turn 15: 45K tokens (+5K)
```
- Initial spike (loading context)
- Stabilizes
- Sustainable with compression

### Unhealthy Growth Patterns

**Exponential growth (bad):**
```
Turn 1:  20K tokens
Turn 5:  35K tokens (+15K)
Turn 10: 62K tokens (+27K)
Turn 15: 110K tokens (+48K)
```
- Unsustainable
- Indicates retry storms or giant prompts
- Requires immediate intervention

**Staircase growth (warning):**
```
Turn 1:  20K tokens
Turn 2:  22K tokens (+2K)
Turn 3:  50K tokens (+28K) ← giant prompt
Turn 4:  52K tokens (+2K)
Turn 5:  55K tokens (+3K)
Turn 6:  85K tokens (+30K) ← another giant prompt
```
- Episodic large operations
- Risk of sudden overflow
- Consider decomposition

## Pressure Relief Strategies

### 1. Compression (First Line)

**When:** Context reaches 60-70% of budget

**Techniques:**
- Summarize old tool results (keep metadata, compress content)
- Condense conversation history (preserve key decisions)
- Remove redundant information (duplicate file reads)

**Effectiveness check:**
- Compression ratio should be > 2x (50%+ reduction)
- If < 2x, compression is ineffective → try different strategy

**Example compression:**
```
BEFORE (5K tokens):
User: "Read file X"
Assistant: "I'll read the file"
Tool: [5000 lines of code]
Assistant: "The file contains..."

AFTER (1K tokens):
[Compressed] File X read (5000 lines): authentication module, 
JWT handling, OAuth2 flow. Key findings: [summary]
```

### 2. Decomposition (Strategic)

**When:** Context at 70-80% and facing new large task

**Process:**
1. Identify task boundaries
2. Split into independent subtasks
3. Process sequentially (reset between if needed)
4. Synthesize results in fresh session

**Example:**
```
Monolithic: "Review entire codebase for security issues"
Decomposed:
  1. "Review auth module for security issues"
  2. "Review API endpoints for injection vulnerabilities"
  3. "Review database queries for SQL injection"
  4. "Synthesize security findings into report"
```

### 3. Session Reset (Nuclear Option)

**When:** Context at 85%+ or retry storm detected

**Strategy:**
1. Save current session reference (ID, summary)
2. Start fresh session
3. Load minimal context (task description only)
4. Link back to old session if needed

**Preserve between resets:**
- Task goals and requirements
- Key decisions made
- Current status / progress
- Blockers / open questions

**Don't preserve:**
- Full conversation history
- Old tool results (unless currently needed)
- Error messages / retries
- Intermediate reasoning

### 4. Provider Escalation

**When:** Current provider context insufficient for task complexity

**Options:**
- GPT-4 (8K) → GPT-4 Turbo (128K)
- Claude 3 Haiku (200K) → Claude 3.5 Sonnet (200K) [better at using full context]
- Qwen2.5-Coder (32K) → DeepSeek-V3 (128K)

**Trade-offs:**
- Cost increases (longer context = higher price per token)
- Latency increases (more tokens to process)
- May need API key for different provider

## Budget Estimation Examples

### Example 1: Code Review Task

**Task:** Review PR with 3 files, 500 lines total

**Budget breakdown:**
- System prompt: 18K tokens
- Task description: 500 tokens
- 3 file reads: 15K tokens (estimated)
- Review reasoning: 5K tokens (model generation)
- **Total: 38.5K tokens**

**Providers:**
- ✅ GPT-4 Turbo (128K): 30% utilization - SAFE
- ✅ Claude 3.5 Sonnet (200K): 19% utilization - SAFE
- ⚠️ Llama 3.1 (32K): 120% utilization - IMPOSSIBLE
- ✅ Qwen2.5-Coder (32K): 120% utilization - Need decomposition

**Recommendation:** Use cloud provider (GPT-4/Claude) without decomposition, or decompose into per-file reviews for Qwen.

### Example 2: Log Analysis Task

**Task:** Analyze 10MB error log, extract patterns

**Budget breakdown:**
- System prompt: 18K tokens
- Task description: 300 tokens
- Log file read: 150K tokens (!!)
- Analysis reasoning: 10K tokens
- **Total: 178K tokens**

**Providers:**
- ❌ GPT-4 Turbo (128K): 139% utilization - OVERSIZED
- ⚠️ Claude 3.5 Sonnet (200K): 89% utilization - HIGH
- ❌ All local models: IMPOSSIBLE

**Recommendation:** Decompose - don't read entire log. Use search_files with error patterns, process in chunks, or sample representative errors.

**Better approach:**
```bash
# Instead of: read_file("error.log")  # 10MB
# Do: search_files(pattern="ERROR|FATAL", path="error.log", limit=100)
# Budget: 18K + 0.3K + 5K (100 errors) + 5K = 28.3K tokens
```

### Example 3: Multi-File Refactor

**Task:** Refactor authentication across 10 files, 2K lines

**Budget breakdown (naive):**
- System prompt: 18K tokens
- Task description: 1K tokens
- 10 file reads: 60K tokens
- Refactor planning: 15K tokens
- **Total: 94K tokens**

**Providers:**
- ⚠️ GPT-4 Turbo (128K): 73% utilization - LARGE
- ✅ Claude 3.5 Sonnet (200K): 47% utilization - SAFE
- ❌ Qwen2.5-Coder (32K): 294% utilization - IMPOSSIBLE

**Better approach (decomposed):**
1. Read all 10 files, generate refactor plan: 94K tokens (LARGE)
2. Reset session
3. For each file: read file + apply plan: ~25K tokens (SAFE)
4. Result: 1 planning session + 10 implementation sessions, all SAFE

## Retry Storm Prevention

### Detection

Retry storm pattern:
- 3+ API failures in 10 minutes
- Each retry adds context (error messages, diagnostics)
- Context grows 10-30% per retry
- Pressure score accelerates

**Example timeline:**
```
10:00 - API call fails (network error) → +2K tokens (error message)
10:02 - Retry with diagnostics → +3K tokens
10:05 - Retry with workaround → +5K tokens
10:09 - Retry with alternative → +8K tokens
10:14 - Retry with full debug → +12K tokens
Total added: 30K tokens in 14 minutes
```

### Prevention

**Early intervention (after 2 failures):**
```
⚠️ 2 API failures detected in 4 minutes

Current context: 45K tokens (+7K from retries)

Recommendations:
1. Check provider status (cloud outage?)
2. Verify API keys and quotas
3. Consider simplified prompt (reduce complexity)
4. Reset session if 3rd attempt fails
```

**Hard stop (after 5 failures):**
```
🚨 Retry storm detected: 5 failures in 12 minutes

Context has grown 40% (35K → 49K tokens)

STOPPING retries. Recommended actions:
1. Reset session immediately
2. Verify provider availability
3. Check for systematic issues (auth, rate limits)
4. Consider alternative provider or local routing
```

## Tool Result Budgeting

Large tool results are the most common cause of budget overflow.

### Result Size Heuristics

| Tool | Typical Size | Max Size | Notes |
|------|--------------|----------|-------|
| read_file | 1-10K tokens | 100K+ | Entire files |
| search_files | 1-5K tokens | 50K+ | Many matches |
| terminal | 0.5-5K tokens | 500K+ | Long output (logs, compilation) |
| web_search | 2-10K tokens | 30K+ | Multiple results |
| browser_navigate | 5-20K tokens | 100K+ | Full page content |

### Budget-Aware Tool Use

**read_file:**
```bash
# Bad: Read entire large file
read_file("logs/app.log")  # 150K tokens

# Good: Read specific section
read_file("logs/app.log", offset=1, limit=500)  # 15K tokens

# Better: Search for patterns
search_files(pattern="ERROR", path="logs/app.log", limit=50)  # 5K tokens
```

**terminal:**
```bash
# Bad: Unconstrained output
terminal("pytest")  # Could be 100K+ tokens if verbose

# Good: Limit output
terminal("pytest -q")  # Quiet mode

# Better: Capture failures only
terminal("pytest --tb=short --failed-first")  # Minimal output
```

**browser_navigate:**
```bash
# Bad: Full page content
browser_navigate("https://docs.python.org")  # 50K tokens

# Good: Targeted extraction
browser_navigate("https://docs.python.org/3/library/json.html")  # 10K
# Then: Ask specific questions about visible content
```

## Integration with Ollama Routing

When context pressure is high AND task is LOCAL_SAFE:

**Decision matrix:**
| Context State | Task Type | Recommendation |
|---------------|-----------|----------------|
| SAFE | Any | Use configured provider |
| LARGE | LOCAL_SAFE | Monitor, consider local for next task |
| OVERSIZED | LOCAL_SAFE | Recommend local OR reset session |
| CRITICAL | LOCAL_SAFE | Block cloud, recommend local + reset |
| OVERSIZED | CLOUD_REQUIRED | Block operation, require reset |

**Example advisory:**
```
⚠️ Context pressure: OVERSIZED (78% of 128K budget)

Task: "Summarize error logs" (LOCAL_SAFE)

Options:
1. Reset session, continue with cloud provider
2. Switch to local Ollama (32K context is plenty for summaries)
3. Decompose: summarize by time period, merge results

Context pressure + safe task = local routing reasonable
```

## Monitoring and Reporting

### Real-Time Display (Telegram)

**Compact format (normal operations):**
```
✓ Task complete [Context: 45K/128K (35%) SAFE]
```

**Expanded format (pressure detected):**
```
✓ Feature implemented

Context Budget:
• Current: 78.5K / 128K tokens (61%)
• State: LARGE
• Growth: +12% per turn
• Retries: 1

⚠️ Approaching limit. Consider decomposition for next major task.
```

**Critical format:**
```
🚨 CRITICAL CONTEXT PRESSURE

• Current: 118K / 128K tokens (92%)
• Retry storm: 7 attempts in 8 minutes
• Growth: +35% per turn

IMMEDIATE ACTION REQUIRED:
1. Start new session
2. Preserve current session for reference
3. Decompose remaining work

Do not attempt large operations in current session.
```

### Metrics Logging

Log to `~/.hermes/logs/token-budget.jsonl`:

```json
{
  "timestamp": "2026-05-23T11:30:00Z",
  "session_id": "abc123",
  "turn": 15,
  "components": {
    "system_prompt": 18500,
    "conversation_history": 32000,
    "tool_results": 8500,
    "reserve": 15000
  },
  "total_estimated": 74000,
  "provider_limit": 128000,
  "utilization_pct": 57.8,
  "state": "LARGE",
  "growth_rate_pct": 12.5,
  "retry_count": 2,
  "recommendations": ["monitor", "prepare_for_decomposition"]
}
```

## References

- [CONTEXT_GUARDRAILS.md](./CONTEXT_GUARDRAILS.md) - Pressure detection and mitigation
- [OLLAMA_ROUTING_STRATEGY.md](./OLLAMA_ROUTING_STRATEGY.md) - Local routing integration
- OpenAI tokenizer documentation
- Anthropic prompt engineering guide
