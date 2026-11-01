# Decision Quality Index (DQI)

## Formula

```
For each claim:
  weight = confidence × evidence_multiplier × rebuttal_multiplier

DQI = sum(weights) / count(weights)
```

### Evidence multipliers

| Strength     | Multiplier |
|--------------|------------|
| direct       | 1.0        |
| inferred     | 0.7        |
| speculative  | 0.4        |

### Rebuttal multipliers

| Status      | Multiplier |
|-------------|------------|
| unrebutted  | 1.0        |
| conceded    | 0.7        |
| unresolved  | 0.5        |

## Thresholds

| DQI Range     | Assessment | Action                           |
|---------------|------------|----------------------------------|
| ≥ 0.8         | high       | Auto-execute (no minority report)|
| 0.6 – 0.8     | moderate   | Human review flagged             |
| < 0.6         | low        | Round 3 convergence required     |

## Notes

- Position changes (Round 2) do not enter the DQI formula directly but signal healthy debate.
- If an agent produces no claims (timeout or error), they contribute zero weight, lowering DQI.
- DQI ≥ 0.9 after Round 1 warrants a groupthink check — optionally inject adversarial probe round.
