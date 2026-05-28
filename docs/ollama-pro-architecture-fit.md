# Ollama Pro / Cloud Models — Architecture Fit Analysis
**Date**: 2026-05-25 | **Author**: Autonomous analysis pass

## 1. What Ollama Pro Actually Is

Ollama Pro ($20/mo) and Max ($100/mo) are managed-inference tiers that give you cloud-hosted GPU access *through the same Ollama API surface you already run locally*. Key data points:

| Feature | Free | Pro ($20/mo) | Max ($100/mo) |
|---------|------|-------------|---------------|
| Local models | Unlimited | Unlimited | Unlimited |
| Cloud models | 1 concurrent | 3 concurrent | 10 concurrent |
| Cloud usage | Light (eval only) | 50x Free | 5x Pro (250x Free) |
| Usage model | GPU-time based, not token-counted | Same + included balance | Same + more balance |
| Privacy | N/A (no cloud) | No logging, no training, zero retention | Same |
| Host regions | N/A | US primary, EU/SG overflow | Same |
| Auth | N/A | `ollama signin` or `OLLAMA_API_KEY` | Same |

## 2. Available Cloud Models (Relevant to Hermes)

| Cloud Model | Params | Context | Use Case for Hermes |
|-------------|--------|---------|---------------------|
| `qwen3-coder:480b-cloud` | 480B | 128K+ | **Primary reasoning model** — replaces/augments Ring-2.6-1t in consult chain |
| `gpt-oss:120b-cloud` | 120B | 128K+ | Secondary reasoning, Red team / critique phase |
| `gpt-oss:20b-cloud` | 20B | 128K+ | Fast cloud fallback (bridges gap between local 8B and heavy models) |
| `deepseek-v3.1:671b-cloud` | 671B | 128K+ | Ultra-deep reasoning — overkill for most tasks but available |
| `kimi-k2.5:cloud` | ~2T (MoE) | 128K+ | **Fills the Kimi gap** — works natively via Ollama cloud, solves the 401 problem |
| `minimax-m2.7:cloud` | — | 128K+ | Multilingual specialist (deferred) |
| `glm-4.7:cloud` | — | — | General purpose (deferred) |

## 3. API Compatibility — Why This Matters

Ollama's cloud models expose via the **same OpenAI-compatible REST API** as local models:
```
POST http://localhost:11434/api/chat
{
  "model": "qwen3-coder:480b-cloud",
  "messages": [{"role": "user", "content": "..."}],
  "stream": false
}
```

**This means zero code changes to `gateway/run.py`'s AIAgent or `model_routing.py` to use cloud models.** The models just appear as additional options in the existing provider chain. Authentication flows through `OLLAMA_API_KEY` environment variable — fits our `.env` vault pattern identically.

## 4. Where Each Cloud Model Fits in Hermes Architecture

### 4.1 Revised Fallback/Routing Chain

```
TIER 0 — Identity/System:     context-architect.md (never trimmed)
TIER 1 — Active Task:         Pinned in working context
TIER 2 — Recent Important:    Pinned in working context  
TIER 3 — Semantic Memory:     Memory Palace SQLite
TIER 4 — Background:          Available for reload on-demand
TIER 5 — Tool Output:         Trimmed first (compression candidate)
TIER 6 — Conversation:        Trimmed first (deletion candidate)

MODEL SELECTION (task-routed):
───────────────────────────────────────────────────
User-facing calls (default):
  Mac M2:  qwen3:8b (fast, free) → qwen3-coder:30b-a3b (reasoning)
  Linux:   qwen3:8b (pending DO droplet)

Agent-internal calls (consult/merge/critique):
  Auto-route to premium via effective_budget=9999.0
  Phase 1 (parse/analyze):  deepseek-v4-flash (fast, cheap)
  Phase 2 (critique/red-team): grok-4.20-reasoning OR gpt-oss:120b-cloud
  Phase 3 (quality gate/merge): ring-2.6-1t OR qwen3-coder:480b-cloud

Cloud overflow (local unavailable or exceeds VRAM):
  qwen3-coder:480b-cloud (via Ollama Pro) ← NEW TIER
  gpt-oss:20b-cloud (via Ollama Pro) ← NEW FAST TIER
  
Emergency degradation:
  All cloud dead → local-only (8B trim + 30B reasoning) + Telegram alert
```

### 4.2 Solving the Kimi Problem

Kimi K2.5 is available natively as `kimi-k2.5:cloud` through Ollama cloud. This eliminates:
- The missing key problem (direct moonshot.cn API still 401)
- The Io Net proxy dependency
- Cold standby status

**Action**: Sign into Ollama cloud → `OLLAMA_API_KEY` in `.env` → Kimi becomes `kimi-k2.5:cloud` accessible via localhost like any other model.

### 4.3 Solving the DeepSeek Activation Problem

DeepSeek v3.1 is available as `deepseek-v3.1:671b-cloud` via Ollama Pro. This provides a backup path if the direct DeepSeek API key continues to have issues, though the current new key (`sk-bca...5661`) is validated and live.

### 4.4 Elevating the Consult/Merge Chain

Current: DeepSeek v4-pro → Grok-4.20 → Ring-2.6-1t (130B/262K context)

Proposed (with Ollama Pro):
- **Option A**: Keep current chain, add `qwen3-coder:480b-cloud` as Ring alternative / escalation  
- **Option B**: Replace Ring with `qwen3-coder:480b-cloud` (480B > Ring's model capacity, same API pattern)
- **Option C**: Use Ring for fast quality gate, `qwen3-coder:480b-cloud` for deep reasoning pass

**Recommendation**: Option C — Ring for real-time quality checks (fast, 262K context), `qwen3-coder:480b-cloud` for final deep merge/deliberation pass. Both accessible via same Ollama API.

## 5. Pro Tier Recommendation

**Pro ($20/mo) is the right tier for Hermes:**
- 3 concurrent cloud models covers our immediate needs (qwen3-coder:480b + kimi-k2.5 + one buffer)
- 50x usage over Free is sufficient for agent-internal deliberation calls
- GPU-time billing means efficient caching = lower cost (shared context)
- US/EU/SG hosting provides geographic diversity
- Native OpenAI API compatibility = zero integration cost

**Max ($100/mo) is premature** until we have multiple concurrent agent pipelines running in production. Revisit at scale.

## 6. Authentication Setup

Required `.env` additions:
```bash
# Ollama Cloud (Pro tier)
OLLAMA_API_KEY=ollama-p-...
```

The `key_guardian.py` should validate this key daily alongside existing cloud keys.

## 7. Impact on Platform/Launcher Architecture

The Mac Launcher (Gatekeeper/Jamf/MDM) should:
1. Install Ollama v0.12+ (required for cloud model support)
2. Run `ollama signin` with service account credentials during enrollment
3. Pre-pull `qwen3-coder:480b-cloud` and `kimi-k2.5:cloud` manifests
4. Cloud models auto-route through localhost:11434 — no proxy/firewall changes needed

## 8. OpenClaw Context

Ollama acquired OpenClaw (Feb 2026) and integrated it as `ollama launch openclaw`. Notable:
- OpenClaw skills are static files; Hermes skills are autonomously created/refined by the agent
- The Ollama-integrated version is more tightly coupled to their cloud ecosystem
- Our `skill_engine.py` approach (JSON-defined, 3 trigger types, caching, param validation) is architecturally superior for our use case
- **No action needed** — our skill engine replaces OpenClaw entirely as designed

## 9. Risks and Considerations

| Risk | Mitigation |
|------|-----------|
| Cloud dependency for premium models | All cloud models accessible via single localhost port; swap providers by changing model name only |
| Ollama cloud outage | Fallback chain still has 4 other cloud providers (DeepSeek, Grok, Ring via OpenRouter) |
| Data in transit to Ollama cloud | Zero retention policy, but add TLS verification in production |
| Pro cost creep with heavy usage | GPU-time billing is efficient; set usage alerts at 90% threshold |
| Vendor lock-in to Ollama | All models are open-weight; can migrate to direct API calls if needed |

## 10. Decision Matrix — Compression vs. Deletion

Applying Ollama Pro context to the trimming philosophy question:

- **Local models (8B/30B)**: Limited context window = **deletion** is correct. T5/T6 blocks get dropped, not compressed. Compression wastes tokens rephrasing low-value content.
- **Cloud models (480B/671B)**: Massive context window = **compression** becomes valuable. When routing to qwen3-coder:480b-cloud, the cost of re-processing long context is low relative to the value of retaining information.
- **Hybrid approach**: The context_orchestrator should be model-aware. When routing to a cloud model with >100K context, compress T4/T5 instead of deleting. When routing to local 8B, hard-delete T5/T6.

**This means the context_orchestrator.py needs a `target_model` parameter to adjust trim strategy based on destination model's context budget.**