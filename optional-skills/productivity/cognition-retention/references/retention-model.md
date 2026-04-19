# Cognition Retention Model — Technical Reference

## The Weibull Forgetting Curve

Cognition predicts recall probability using:

```
R(t) = exp(-(t/S)^β)
```

| Parameter | Meaning | Range |
|---|---|---|
| `R(t)` | Probability of recall at time `t` | 0.0 – 1.0 |
| `t` | Time since last practice (days) | ≥ 0 |
| `S` | Stability — how slowly you forget | 0.1 – 365 days |
| `β` | Shape — individual learning style | 0.35 – 2.5 |

When `β = 1`, this reduces to exponential decay: `R(t) = exp(-t/S)`.
When `β > 1`, forgetting accelerates over time (cliff-like).
When `β < 1`, forgetting decelerates (long tail).

## How Stability Is Computed

Unlike traditional spaced-repetition systems (Anki, SuperMemo) that fit `S` from population data, Cognition derives `S` from the **spectral gap** of your personal knowledge graph:

```
S = 1 / λ₁(L)
```

where `λ₁(L)` is the smallest nonzero eigenvalue of the graph Laplacian `L` of your concept network.

**Intuition:** Concepts that are well-connected to things you already know (high spectral gap) have higher stability and decay slower. Isolated concepts (low spectral gap) decay faster.

## Review Update Dynamics

After each practice event, the model updates:

1. **Stability** increases based on grade, spacing bonus, and practice weight
2. **Difficulty** adjusts based on prediction error (predicted R vs observed recall)
3. **Uncertainty** decreases with more reviews (the model becomes more confident)
4. **Calibration gap** tracks systematic bias (are we over/under-predicting for this concept?)

## Practice Weights

| Weight | Multiplier | Stability gain |
|---|---|---|
| `active` | 1.0× | Full — wrote code, debugged, explained |
| `passive` | 0.5× | Half — read docs, watched tutorial |
| `reference` | 0.2× | Minimal — skimmed index, bulk import |

## Optimal Review Interval

The next review is scheduled when R(t) drops to the target threshold (default 0.9):

```
interval = S × (-ln(R_target))^(1/β)
```

For `R_target = 0.9`, `β = 1`: `interval ≈ 0.105 × S`
(e.g., if `S = 30 days`, review at `t ≈ 3.15 days`)

## Threshold for Review Urgency

| Retention | Status | Action |
|---|---|---|
| ≥ 80% | Solid | No action needed |
| 60–79% | Slipping | Review within a few days |
| < 60% | At risk | Review now |

## MCP Protocol

All scripts communicate with the Cognition server via MCP (Model Context Protocol) over HTTP:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "log_learning",
    "arguments": {
      "concept": "react-hooks",
      "topic": "React",
      "score": 0.85,
      "practice_weight": "active",
      "source_integration": "hermes_agent"
    }
  }
}
```

Auth: `Authorization: Bearer cog_me_YOUR_KEY`
Accept: `application/json, text/event-stream`

## Available MCP Tools

| Tool | Purpose |
|---|---|
| `get_session_context` | Retention state + weak concepts + teammate nudges |
| `get_user_retention` | Overall retention + topic breakdown |
| `get_weak_topics` | Concepts below threshold |
| `suggest_review` | Review recommendations ranked by urgency |
| `log_learning` | Record a single learning event |
| `log_learning_batch` | Batch ingest (max 200/call) |
| `notify_concept_practiced` | Stronger signal with proof string |
| `get_concept_suggestions` | Context-aware suggestions based on current file/topic |
| `list_indexed_sources` | What integrations are already indexed |

## Further Reading

- [Cognition documentation](https://cognitionus.com/docs)
- Wozniak, P. "Two components of long-term memory." *Acta Neurobiologiae Experimentalis* (1995)
- Murre, J.M.J. & Dros, J. "Replication and analysis of Ebbinghaus' forgetting curve." *PLOS ONE* (2015)
