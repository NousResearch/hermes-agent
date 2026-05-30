# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search, reranking, and automatic deduplication.

## Requirements

- `pip install mem0ai`
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)

## Setup

```bash
hermes memory setup    # select "mem0"
```

Or manually:
```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
```

## Config

Config file: `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `user_id` | `hermes-user` | User identifier on Mem0 |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | Enable reranking for recall |

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |

## Time-Series Awareness

Memories are enriched with **Ebbinghaus decay-based freshness signals**, enabling agents to reason about memory recency and prioritize recent/actively-used memories.

### Domain Profiles

Memories are classified into three domains with different decay rates:

| Domain | Half-life | Use Cases | Beta (reinforcement) |
|--------|-----------|-----------|---------------------|
| `volatile` | 3 days | Market data, real-time metrics | 0.8 |
| `normal` | 7 days | Projects, tools, general facts | 0.6 |
| `stable` | 30 days | User preferences, infrastructure | 0.4 |

### Enriched Fields

Both `mem0_profile` and `mem0_search` results include these additional fields:

| Field | Type | Description |
|-------|------|-------------|
| `age_days` | int | Days since the memory was created |
| `domain` | str | Detected domain: `volatile`, `normal`, or `stable` |
| `retention_score` | float | Ebbinghaus forgetting-curve score (0.0–1.0) |
| `lifecycle_state` | str | `active`, `warm`, `cold`, `archive`, or `forgotten` |
| `freshness_tag` | str | Visual indicator: `""`, `⏳`, `⚠️`, or `🔴` |
| `suggested_bit_width` | int | Quantization downgrade: 32→8→4→2 |
| `n_use` | int | Usage count (from `access_count` or `n_use`) |
| `eligible_for_promotion` | bool | `True` if n_use ≥ 5 and age ≤ 14 days |

### Formula

```
score(t) = (n_use)^β · e^(-λ·Δt) · s

where:
  λ = ln(2) / half_life
  λ_eff = λ · (1 + κ·(1 - trust))    (trust-weighted acceleration, κ=2.0)
  s = strength (default per domain)
```

### Lifecycle Thresholds

| State | Retention Range | Bit Width | Tag |
|-------|----------------|-----------|-----|
| active | > 0.8 | 32 | — |
| warm | 0.5–0.8 | 8 | ⏳ |
| cold | 0.2–0.5 | 4 | ⚠️ |
| archive | threshold–0.2 | 2 | 🔴 |

### Customization

Domain parameters can be overridden at the module level before the provider is initialized. For example, to make `normal` memories decay faster:

```python
from plugins.memory.mem0 import _EBBINGHAUS_PARAMS
_EBBINGHAUS_PARAMS["normal"]["half_life_days"] = 3
```
