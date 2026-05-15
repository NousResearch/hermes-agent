# OpenRouter Model Refresh — Architecture Decisions (030526)

## Three-Location Sync Problem

`refresh_openrouter_models.py` must exist in three places:

| Location | Purpose | How updated |
|---|---|---|
| `~/.hermes/scripts/refresh_openrouter_models.py` | Cron executes this | `cp` from feat branch after every edit |
| `~/.hermes/hermes-agent-feat/scripts/refresh_openrouter_models.py` | Feat branch source | Edit here first |
| `subagent-model-routing/SKILL.md` WHITELISTS section | Human-readable mirror | Patched atomically with script |

**Why not a symlink:** The cron sandbox (`scheduler.py → _run_job_script()`) calls `path.resolve()` which follows symlinks. A symlink pointing outside `~/.hermes/scripts/` is blocked with a path traversal error. Real file copies only.

**Why the feat branch is source of truth:** The script is a declared dependency of PR #12794. Whitelist changes must land there to keep the PR current. Post-merge, the copy step targets `~/.hermes/hermes-agent/scripts/` instead.

## Price Cache (added 030526)

`~/.hermes/caches/openrouter_prices_last.json` — written at end of each run, read at start of next for delta comparison. Schema:

```json
{
  "_run_date": "2026-05-03 20:42 UTC",
  "google/gemini-2.5-flash": {"in": 0.0000003, "out": 0.0000025},
  ...
}
```

Threshold: 20% change triggers a flag. First run seeds the cache with no delta output ("delta tracking starts next week").

**Why not parse the skill's pricing tables:** The skill tables are human-readable narrative, not a reliable machine-parseable baseline. They have inconsistent formats across sections and aren't exhaustive. The JSON sidecar is stdlib, self-healing (resets with every approved update), and zero-dependency.

## CRON_PROMPT_TEMPLATE removed (030526)

The template was removed from the script because:
- The live prompt lives in `jobs.json` (job `6d0271a4d5cb`) — that's the real one
- The template was a diverging copy with no enforcement
- Any agent reading the script would get a misleading picture of what the cron actually says
- The script's MAINTENANCE docstring already explains the full architecture

To inspect the live prompt: `python3 -c "import json; jobs=json.load(open('/Users/jj/.hermes/cron/jobs.json')); print(next(j['prompt'] for j in jobs['jobs'] if j['id']=='6d0271a4d5cb'))"`

## Whitelist Tier Exclusivity

`budget / standard / premium` are mutually exclusive. `coding` may overlap any of them. Enforced at script import time:

```python
_EXCLUSIVE_TIERS = {"budget", "standard", "premium"}
# raises ValueError on overlap with clear error message naming the offending model
# coding is explicitly exempt
```

## Overlap Validation Rationale

Before 030526 the same model appeared in up to 4 lists simultaneously (e.g. `mistralai/devstral-small` in full/standard/coding/budget). The auto-router uses these lists for routing decisions — duplicates caused ambiguous precedence and made the lists meaningless as tiers. The restructure assigns each model a single home except coding (intentional overlap for specialist access from any tier).
