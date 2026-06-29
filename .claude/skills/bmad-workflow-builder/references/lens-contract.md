# Lens Contract

The return mechanics every scan lens shares. Your own spec file gives you the lane and the bar; this file is how the work comes back.

You receive compact pre-pass JSON and the skill path from the parent. Read the metrics first and open a raw file only for judgment a metric cannot settle. Return your findings to the parent in-context: never write a file or a per-subagent analysis document. The parent merges all lens returns and renders the report itself.

Return exactly this JSON and nothing else:

```json
{
  "lens": "<your lens name>",
  "verdict": "<one line for this lens>",
  "findings": [
    {
      "id": "<lens>-<n>",
      "severity": "critical | high | medium | low",
      "title": "<short>",
      "location": "<file:region or file>",
      "evidence": "<what was observed>",
      "recommendation": "<the fix>"
    }
  ]
}
```

- `id` numbers sequentially within your lens (`<lens>-1`, `<lens>-2`), so every finding stays traceable after the merge.
- The leanness lens alone adds `proposed_smallest` and `predicted_delta` to its defend-against-absence findings; every other lens and every other finding omits those keys.
- If you find nothing, return an empty `findings` array with a verdict saying the skill passes your lens. Do not pad the list to look thorough — a weak finding that would not survive a real run is worse than no finding.
