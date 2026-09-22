# skill_suggestion evals

Measures whether the agent loads the right skill from the index, and whether a
pre-turn suggestion (the `skill-suggest` user plugin) improves on the agent's
own selection.

## What was measured (2026-09-17, glm-5.3:cloud via ollama-launch, 234-skill live index)

| Arm | wrong_load (35 covered) | needless_load (12 uncovered) |
|---|---|---|
| agent alone (index descriptions ≤160 chars) | 20.0% (strict) / 25.7% (exact) | 33.3% |
| agent with skill-suggest hint | see `suggestion_arm.py` | see `suggestion_arm.py` |
| oracle (handed the answer) | not measured locally | not measured locally |

- "strict" credits category-path loads (`email/himalaya`) that the real
  `skill_view` resolves to the same skill; "exact" requires the bare name.
- Every wrong pick in the baseline went to an OMH catch-all workflow skill
  (`omh-apps`, `omh-live-info`, `omh-web-research`, `omh-decision-prototype`) —
  the failure class the plugin's `excluded` list targets.

## Files

- `skill_baseline.py` — gold-labelled request set (35 covered + 12 negative)
  against the live `build_skills_system_prompt()` index; scores the FIRST
  response's first `skill_view` call. Resumable (JSONL cache).
- `suggestion_arm.py` — same requests with the skill-suggest plugin's
  `<skill_relevance>` block prepended, as `pre_llm_call` injects it.
- `score_baseline.py` — exact vs strict scoring of the baseline JSONL.
- `judge_shootout.py` — accuracy/latency shootout for the plugin's judge model
  (candidates: local llama32, deepseek-v4-flash, kimi, gemini flashes).
- `README.md` — this file.

## Running

```bash
# from the repo root, with the agent's provider endpoint reachable
./venv/bin/python evals/skill_suggestion/skill_baseline.py
./venv/bin/python evals/skill_suggestion/suggestion_arm.py
```

Both scripts are resumable: delete the `*_results.jsonl` caches to re-run live.
The request set is the harness — extend COVERED/NEGATIVES rather than editing
the scoring. Gold labels come from each skill's own description; before adding
a gold label, confirm the skill exists on the target platform (a macOS-only
skill never appears in a Linux session's index and is not a valid gold).

## Judge selection log (2026-09-17)

| Judge model | correct | latency | verdict |
|---|---|---|---|
| llama32-agent (local) | 3/6 | 1.5s/case | false-negatives everything — unusable |
| deepseek-v4-flash:cloud | 6/6 | 2.1-3.1s/case | **selected** |
| kimi-k2.6:cloud | 6/6 | 8.5s/case | accurate but 3-4x slower |
| gemini-2.5-flash | 3/6 | 0.2s/case | too conservative under catch-all rules |
| gemini-3.6-flash | 6/6 | 3.2s/case | verified backup; needs `gemini` in the plugin's allowed_providers |

Thresholds are roster-specific. Re-run `judge_shootout.py` after any large
roster change or exclusion-list edit before trusting a new judge.