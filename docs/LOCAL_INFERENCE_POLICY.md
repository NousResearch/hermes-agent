# Local Inference Policy

## Purpose

Define concrete rules for when local Ollama inference is safe, preferred, or prohibited based on workload characteristics and risk assessment.

## Core Principle

**Local inference is a fallback capability for low-risk operations when cloud providers are degraded. Security-sensitive, production-critical, or high-stakes operations always require cloud provider guarantees.**

## Classification Rules

### LOCAL_SAFE Detection

A workload is classified as `LOCAL_SAFE` if it matches ANY of these patterns:

**Text Summarization:**
- Keywords: "summarize", "summary", "tl;dr", "overview", "digest"
- Context: log files, changelogs, status reports, documentation
- Risk: Low (output is advisory, easily verified)

**Code Formatting:**
- Keywords: "format", "lint", "style", "prettier", "black", "indent"
- Context: non-production code, local files
- Risk: Low (deterministic, easily reverted)

**Shell Command Suggestions:**
- Keywords: "how to", "command for", "what's the command"
- Context: read-only operations (ls, grep, find, cat, ps, df)
- Risk: Low (user reviews before execution)

**Documentation Queries:**
- Keywords: "explain", "what is", "how does", "documentation"
- Context: concept explanation, API reference lookup
- Risk: Low (informational only)

**Non-Critical Code Explanation:**
- Keywords: "explain this code", "what does this do", "walk me through"
- Context: learning, onboarding, debugging assistance
- Risk: Low (no code changes)

**Ticket/Issue Management:**
- Keywords: "parse ticket", "extract issue", "summarize thread"
- Context: GitHub issues, Jira tickets, forum threads
- Risk: Low (information extraction)

### CLOUD_PREFERRED Detection

A workload is classified as `CLOUD_PREFERRED` if:

**Code Review:**
- Keywords: "review", "code review", "PR review", "check this code"
- Context: feature branches, non-security code
- Risk: Medium (quality matters, but not critical)
- Fallback threshold: 5+ minute cloud outage

**Feature Planning:**
- Keywords: "plan", "design", "architecture", "approach"
- Context: pre-implementation, brainstorming
- Risk: Medium (strategic but not executed immediately)
- Fallback threshold: Persistent rate limiting

**Test Generation:**
- Keywords: "write tests", "generate tests", "test cases"
- Context: unit tests, integration tests
- Risk: Medium (quality impacts coverage, but tests are verified)
- Fallback threshold: Sustained cloud degradation

**Refactoring Suggestions:**
- Keywords: "refactor", "improve", "optimize", "clean up"
- Context: non-production code, development branches
- Risk: Medium (changes code but in dev environment)
- Fallback threshold: User explicitly requests local

### CLOUD_REQUIRED Detection

A workload is classified as `CLOUD_REQUIRED` if it matches ANY of these patterns:

**Security Operations:**
- Keywords: "security", "vulnerability", "auth", "authentication", "authorization", "permissions", "secrets", "keys", "tokens", "credentials"
- Context: authentication logic, security reviews, access control
- Risk: **HIGH** (security mistakes have cascading consequences)
- **NO LOCAL FALLBACK EVER**

**Production Deployments:**
- Keywords: "deploy", "production", "release", "ship", "go live", "cutover"
- Context: production environments, customer-facing systems
- Risk: **HIGH** (availability, correctness critical)
- **NO LOCAL FALLBACK EVER**

**Database Operations:**
- Keywords: "migration", "schema change", "ALTER TABLE", "DROP", "database", "SQL", "query optimization"
- Context: production databases, customer data
- Risk: **HIGH** (data loss potential)
- **NO LOCAL FALLBACK EVER**

**Financial Calculations:**
- Keywords: "calculate", "invoice", "billing", "payment", "pricing", "revenue", "cost", "tax"
- Context: money-related operations
- Risk: **HIGH** (accuracy is legally required)
- **NO LOCAL FALLBACK EVER**

**Legal/Compliance:**
- Keywords: "legal", "compliance", "GDPR", "HIPAA", "SOC2", "audit", "regulatory"
- Context: legal documents, compliance reports
- Risk: **HIGH** (regulatory implications)
- **NO LOCAL FALLBACK EVER**

**Customer-Facing Content:**
- Keywords: "customer", "client", "user-facing", "public", "marketing", "announcement"
- Context: emails, blog posts, documentation, support responses
- Risk: **HIGH** (brand reputation, customer trust)
- **NO LOCAL FALLBACK EVER**

### FALLBACK_ONLY Detection

Emergency-only scenarios (cloud completely unavailable, urgent need):

**System Diagnostics:**
- Keywords: "check", "verify", "validate", "status", "health"
- Context: non-production systems, local development
- Risk: Low (informational, no changes)
- Threshold: Cloud unavailable >15 minutes

**Log Parsing:**
- Keywords: "parse logs", "extract errors", "find pattern"
- Context: non-sensitive logs, development logs
- Risk: Low (read-only, no decisions)
- Threshold: Cloud unavailable >15 minutes

**Config Validation:**
- Keywords: "validate config", "check syntax", "verify format"
- Context: YAML/JSON syntax checking (no semantic analysis)
- Risk: Low (syntax-only, no execution)
- Threshold: Cloud unavailable >15 minutes

## Edge Cases and Overrides

### Ambiguous Workloads
If classification is uncertain, **default to CLOUD_REQUIRED**. False negatives (blocking safe operations) are preferable to false positives (allowing risky operations locally).

### User Override
The classification system is advisory. If an operator explicitly configures Ollama as their provider, respect that choice. Do not show routing recommendations in this case (they've already decided).

### Mixed Workloads
If a single request contains both LOCAL_SAFE and CLOUD_REQUIRED components:
1. Classify as CLOUD_REQUIRED
2. Suggest decomposition: "This request mixes safe (summary) and critical (deploy) operations. Consider splitting into separate requests."

### Development vs. Production Context
Context matters:
- "deploy to staging" → CLOUD_PREFERRED (if staging is low-risk)
- "deploy to production" → CLOUD_REQUIRED (always)
- "review PR for feature X" → CLOUD_PREFERRED
- "review security patch for CVE-2024-XXXX" → CLOUD_REQUIRED

Detection heuristic: Look for explicit environment mentions. If absent, assume production.

## Implementation Notes

### Classification Pipeline

```bash
# Pseudocode
1. Normalize user message (lowercase, trim)
2. Check for CLOUD_REQUIRED keywords (blocking check)
3. If found → return CLOUD_REQUIRED immediately
4. Check for CLOUD_PREFERRED patterns
5. If found → return CLOUD_PREFERRED
6. Check for LOCAL_SAFE patterns
7. If found → return LOCAL_SAFE
8. Default → CLOUD_REQUIRED (conservative)
```

### Keyword Lists

Maintained in `scripts/ollama-routing-policy.sh`:
- `CLOUD_REQUIRED_KEYWORDS` (array)
- `CLOUD_PREFERRED_KEYWORDS` (array)
- `LOCAL_SAFE_KEYWORDS` (array)

Update as patterns emerge in real usage.

### Logging

Every classification decision logs:
- User message (redacted if sensitive)
- Detected keywords
- Classification result
- Reasoning (why this classification)
- Timestamp

Example log entry:
```json
{
  "timestamp": "2026-05-23T05:45:12Z",
  "message_hash": "a3b2c1...",
  "classification": "LOCAL_SAFE",
  "matched_keywords": ["summarize", "changelog"],
  "reasoning": "Summarization task with non-sensitive context",
  "provider_status": "degraded",
  "recommendation_shown": true
}
```

## Model Selection for Local Inference

When local routing is recommended:

| Workload Type | Primary Model | Fallback Model | Rationale |
|---------------|---------------|----------------|-----------|
| Text summarization | `llama3.1:8b` | `llama3.2:3b` | Fast, efficient for text |
| Code explanation | `qwen2.5-coder:14b` | `qwen2.5-coder:7b` | Code-aware, reasoning |
| Shell commands | `llama3.1:8b` | `qwen2.5-coder:14b` | General purpose works well |
| Documentation | `llama3.1:8b` | `llama3.2:3b` | Text generation strength |
| Log parsing | `llama3.1:8b` | `qwen2.5-coder:7b` | Pattern extraction |

Model selection criteria:
1. **Speed** - Faster is better for advisory tasks
2. **Size** - Smaller models reduce memory pressure
3. **Specialization** - Code tasks prefer code-tuned models
4. **Availability** - Fall back to smaller variants if primary unavailable

## Testing and Validation

### Test Cases

Verify classification for:
- [x] "Summarize the git log" → LOCAL_SAFE
- [x] "Format this Python file" → LOCAL_SAFE
- [x] "Explain this function" → LOCAL_SAFE
- [x] "Review this PR" → CLOUD_PREFERRED
- [x] "Write tests for authentication" → CLOUD_REQUIRED
- [x] "Deploy to production" → CLOUD_REQUIRED
- [x] "Calculate invoice total" → CLOUD_REQUIRED
- [x] "Generate blog post for customers" → CLOUD_REQUIRED
- [x] "Check system status" → FALLBACK_ONLY
- [x] "Parse error logs" → FALLBACK_ONLY

### False Positive/Negative Analysis

**False Positive** (incorrectly allowed local):
- "Summarize security vulnerabilities" should be CLOUD_REQUIRED (security context)
- "Format production config" should be CLOUD_REQUIRED (production context)

**Fix:** Add compound detection for sensitive contexts.

**False Negative** (incorrectly blocked from local):
- "Explain JWT authentication in general terms" might be flagged as CLOUD_REQUIRED due to "authentication"
- "Summarize changelog for security release" might be flagged due to "security"

**Fix:** Educational/informational queries about security topics are LOCAL_SAFE. Add context awareness.

## Maintenance

Review classification rules quarterly:
1. Analyze false positives/negatives from logs
2. Add new keyword patterns as use cases emerge
3. Refine compound detection rules
4. Update model recommendations based on Ollama releases

## References

- [OLLAMA_ROUTING_STRATEGY.md](./OLLAMA_ROUTING_STRATEGY.md) - Overall architecture
- OpenAI Moderation API categories (reference for risk levels)
- OWASP Top 10 (security-sensitive operation patterns)
