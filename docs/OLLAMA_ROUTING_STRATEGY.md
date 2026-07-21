# Ollama Routing Strategy

## Overview

This document defines AI-HQ's advisory routing strategy for recommending local Ollama models when cloud providers are degraded or unavailable.

**Philosophy:** Advisory recommendations only. No automatic provider switching, no config writes, no inference proxying. The system provides intelligent recommendations; operators make decisions.

## Architecture

```
Cloud Provider Issue Detected
    ↓
Classify Workload Type
    ↓
Match Against Local Inference Policy
    ↓
Check Ollama Model Availability
    ↓
Generate Advisory Recommendation
    ↓
Display to Operator (human decides)
```

## Workload Classifications

### LOCAL_SAFE
Workloads suitable for local inference with minimal risk:
- Summarization tasks (logs, status, changelogs)
- Light maintenance operations (formatting, linting suggestions)
- Shell command generation (non-destructive operations)
- Documentation queries
- Code explanations (non-critical)
- Ticket/issue summarization

**Recommended Models:**
- `llama3.1:8b` - Fast, efficient for text tasks
- `qwen2.5-coder:14b` - Code-aware, reasoning capable

### CLOUD_PREFERRED
Workloads that benefit from cloud capabilities but can fall back to local:
- Code review (non-security-critical)
- Feature planning
- Test generation
- Refactoring suggestions
- API documentation generation

**Fallback Criteria:**
- Cloud outage > 5 minutes
- Rate limiting persistent
- User explicitly requests local routing

### CLOUD_REQUIRED
Workloads that must stay on cloud providers:
- Security-sensitive operations (auth, secrets, permissions)
- Production deployments
- Database migrations
- Financial calculations
- Legal/compliance analysis
- Customer-facing content generation

**No local fallback permitted.**

### FALLBACK_ONLY
Emergency-only local routing for basic operations:
- System health checks
- Log parsing (non-sensitive)
- Config validation (syntax-only)
- Environment diagnostics

## Decision Matrix

| Scenario | Provider Status | Workload | Recommendation |
|----------|----------------|----------|----------------|
| Normal ops | All healthy | Any | Use configured cloud provider |
| Minor degradation | 503 errors <5min | Any | Retry cloud, no local suggestion |
| Sustained outage | 503 >5min | LOCAL_SAFE | Recommend local Ollama |
| Sustained outage | 503 >5min | CLOUD_PREFERRED | Suggest waiting or local with caveat |
| Sustained outage | 503 >5min | CLOUD_REQUIRED | Block operation, suggest manual review |
| Rate limited | 429 persistent | LOCAL_SAFE | Recommend local Ollama |
| Ollama unavailable | Cloud healthy | Any | Use cloud (normal path) |
| Ollama unavailable | Cloud degraded | LOCAL_SAFE | Wait for cloud recovery |

## Safety Constraints

### What This System Does NOT Do
1. **No automatic provider switching** - Never changes user's configured provider
2. **No config writes** - Never modifies `config.yaml` or `.env`
3. **No inference proxying** - Never intercepts/redirects API calls
4. **No credential handling** - Never touches API keys or secrets
5. **No silent fallbacks** - Always explicit advisory messages

### What This System DOES Do
1. **Monitor provider health** - Track 503/429 errors, response times
2. **Classify workloads** - Analyze task descriptions for safety
3. **Check Ollama availability** - Verify local models are running
4. **Generate recommendations** - Provide actionable guidance to operators
5. **Log decisions** - Audit trail for routing suggestions

## Integration Points

### Provider Status Monitoring
- Enhanced error messages in credential_pool.py
- Recovery suggestions in error_classifier.py
- Status display in CLI status commands

### Workload Classification
- Analyze user message intent
- Detect keywords (deploy, security, summary, format)
- Pattern match common task types
- Conservative classification (prefer CLOUD_REQUIRED when ambiguous)

### Ollama Health Checks
- Poll http://localhost:11434/api/tags for available models
- Cache results (30-second TTL)
- Verify recommended models are present
- Check model sizes match expectations

## Example Advisory Messages

### Cloud Outage + LOCAL_SAFE Task
```
⚠️  Cloud provider (OpenRouter) unavailable (503 error, 7 minutes).

This appears to be a LOCAL_SAFE workload (log summarization).

Recommendation: Consider using local Ollama for this task.

Available models:
  • llama3.1:8b (4.7GB) - Fast, efficient for summaries
  
To use Ollama:
  hermes config set provider ollama
  hermes config set model llama3.1:8b
  
Or wait for cloud recovery (monitoring continues).
```

### Cloud Outage + CLOUD_REQUIRED Task
```
⚠️  Cloud provider (OpenRouter) unavailable (503 error, 3 minutes).

This appears to be a CLOUD_REQUIRED workload (database migration).

Recommendation: Wait for cloud recovery. Local inference not recommended
for this operation due to risk level.

Cloud status: Monitoring every 30s. Will notify on recovery.
```

### Rate Limit + CLOUD_PREFERRED Task
```
⚠️  Rate limit exceeded (429 error).

This appears to be a CLOUD_PREFERRED workload (code review).

You can:
  1. Wait ~15 minutes for rate limit reset
  2. Use local Ollama with qwen2.5-coder:14b (advisory: may miss edge cases)
  3. Switch to alternate provider (Anthropic, DeepSeek)

Your choice - all options are reasonable for this workload.
```

## Monitoring and Metrics

Track in logs (no telemetry):
- Classification decisions (workload type)
- Recommendation events (local suggested vs. not suggested)
- User actions (followed recommendation vs. alternative path)
- Ollama availability checks (success/failure rates)
- Cloud recovery times (outage duration)

## Future Considerations

**Out of scope for v1:**
- Automatic context-aware provider selection
- Multi-model ensemble routing
- Performance-based model recommendations
- Cost optimization routing
- A/B testing infrastructure

**Possible future work:**
- User preference learning (operator always prefers X for Y tasks)
- Workload complexity scoring (difficulty estimation)
- Model capability matrix (which models excel at which tasks)
- Hybrid routing (local for draft, cloud for refinement)

## References

- [LOCAL_INFERENCE_POLICY.md](./LOCAL_INFERENCE_POLICY.md) - Detailed policy rules
- `scripts/ollama-routing-policy.sh` - Classification logic implementation
- `scripts/ollama-model-health.sh` - Health check implementation
- `agent/credential_pool.py` - Provider error handling integration point
