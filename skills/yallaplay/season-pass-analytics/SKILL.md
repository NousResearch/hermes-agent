---
name: season-pass-analytics
description: 'Use for Season Pass analytics: reward collection, pass level-up, premium/free
  track, reward-collect, seasonpass-init, seasonpass-level-up, engagement, pass progression,
  and Season Pass A/B reads.'
version: 1.0.0
author: YallaPlay
license: Proprietary
metadata:
  hermes:
    tags:
    - season-pass
    - analytics
    - snowflake
    - rewards
    - engagement
    - yallaplay
    related_skills:
    - analytics
    - knowledge
  generated_from:
    wiki_path: yallaplay-wiki/products/core/season_pass/analytics.md
    section: Agent quick context
    source_sha256: 125efb689f2031772724d7485096609c049f14a7bcd072ef91ad30ddadaef1d5
    built: '2026-07-02'
---

# Season Pass Analytics — Compiled Agent Context

<!-- BEGIN GENERATED FROM WIKI AGENT QUICK CONTEXT -->

> Generated from `yallaplay-wiki/products/core/season_pass/analytics.md` → `## Agent quick context`.
> Do not hand-edit this compiled body; update the wiki section and rerun `python3 tools/compile_wiki_skills.py`.

Use this compiled context when the prompt mentions **Season Pass**, **pass engagement**, **reward collect**, **unclaimed rewards**, **pass level-up**, **free/premium track**, or the event names below.

### Fast path

- Source table: `EVENTS.TRANSACTION` for core pass exposure, progression, purchases, and reward claims.
- Current active season at this snapshot: `summer_1`; verify season keys when answering future-dated questions.
- Report `spades` and `rummy` separately unless the user explicitly asks for a combined cross-app read.
- Use `EVENT_DATE` for event windows and `TRANSACTION_METADATA_SEASON_KEY` to isolate one season.
- Join pass events by `(APP, USER_ID, TRANSACTION_METADATA_SEASON_KEY)`.
- Prefer inline `tools/snowflake_query.py "WITH ..."` for quick toplines; write SQL/CSV artifacts only when the user asks to persist, chart, or reuse the result.

### Core event semantics

| `TRANSACTION_NAME` | Meaning | Key metadata |
|---|---|---|
| `seasonpass-init` | User has the feature running / initialized for the season | `TRANSACTION_METADATA_SEASON_KEY` |
| `seasonpass-level-up` | Season-pass XP crossed a level threshold | `TRANSACTION_METADATA_SEASON_KEY` only |
| `reward-collect` | User claimed a season-pass reward | `TRANSACTION_METADATA_SEASON_KEY`, `TRANSACTION_METADATA_SEASON_TRACK`, `TRANSACTION_METADATA_SEASON_LEVEL` |
| `chest-redeem` | Season chest payout opened | `TRANSACTION_METADATA_SEASON_KEY`, `TRANSACTION_METADATA_SEASON_TRACK`, `TRANSACTION_METADATA_SEASON_LEVEL` |

### Canonical denominators

- **Feature exposure / running:** users with `seasonpass-init` for the app + season.
- **Reward-claim engagement:** users with zero `reward-collect` for the same app + season.
- **Users likely to have claimable rewards:** prefer users with `COUNT(DISTINCT TRANSACTION_SEQUENCE)` on `seasonpass-level-up` **>= 2**. In the 2026-07-02 `summer_1` read, `>= 1 seasonpass-level-up` equalled all initialized users, so the first level-up appears initialization-adjacent and is too broad for “has rewards to claim.”
- **Free-track collection rates:** free rewards exist on only 25 of 40 levels in `spring_2`; do not divide by all levels unless verifying reward-bearing levels for the current season.
- **Premium-track collection:** filter `reward-collect` on `TRANSACTION_METADATA_SEASON_TRACK = 'premium'`; only premium buyers can claim it.

### Known traps

- `TRANSACTION_METADATA_SEASON_TRACK` and `TRANSACTION_METADATA_SEASON_LEVEL` exist on `reward-collect` / `chest-redeem`, not on `seasonpass-init` or `seasonpass-level-up`.
- Spades is a persistent 50/50 A/B: arm B has the pass, arm A is holdout. Raw pass event counts are treatment/pass users, not full Spades population.
- Rummy has no holdout, so causal reads need pre/post or matched controls and remain confounded.
- Season key matters: do not mix `spring_2`, `summer_1`, or staff/test seasons.
- Watch right-censoring for current-season reads; show a mature cohort cut such as initialized or first-level-up at least 7 days ago when interpreting unclaimed reward rates.

### Minimal no-reward-collect query shape

```sql
WITH level_uppers AS (
  SELECT APP, USER_ID, TRANSACTION_METADATA_SEASON_KEY AS season_key,
         COUNT(DISTINCT TRANSACTION_SEQUENCE) AS pass_level_ups
  FROM EVENTS.TRANSACTION
  WHERE EVENT_DATE >= :season_start
    AND TRANSACTION_METADATA_SEASON_KEY = :season_key
    AND TRANSACTION_NAME = 'seasonpass-level-up'
    AND APP IN ('spades', 'rummy')
  GROUP BY APP, USER_ID, TRANSACTION_METADATA_SEASON_KEY
), collectors AS (
  SELECT DISTINCT APP, USER_ID, TRANSACTION_METADATA_SEASON_KEY AS season_key
  FROM EVENTS.TRANSACTION
  WHERE EVENT_DATE >= :season_start
    AND TRANSACTION_METADATA_SEASON_KEY = :season_key
    AND TRANSACTION_NAME = 'reward-collect'
    AND APP IN ('spades', 'rummy')
)
SELECT l.APP, COUNT(*) AS eligible_users,
       COUNT_IF(c.USER_ID IS NULL) AS zero_collect_users,
       ROUND(100.0 * COUNT_IF(c.USER_ID IS NULL) / NULLIF(COUNT(*), 0), 2) AS pct_zero_collect
FROM level_uppers l
LEFT JOIN collectors c
  ON c.APP = l.APP AND c.USER_ID = l.USER_ID AND c.season_key = l.season_key
WHERE l.pass_level_ups >= 2
GROUP BY l.APP
ORDER BY l.APP;
```

<!-- END GENERATED FROM WIKI AGENT QUICK CONTEXT -->

## Maintenance

- Canonical source: `yallaplay-wiki/products/core/season_pass/analytics.md`.
- If this skill and the wiki disagree, the wiki wins; patch the wiki, then regenerate this skill.
- Keep this skill symlinked into the active Hermes profile so `/skills` and prompt routing see the git-tracked copy.
