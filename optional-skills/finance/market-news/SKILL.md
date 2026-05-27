---
name: market-news
description: Turn market news into trading risk and regime signals.
version: 0.1.0
author: GitHub Copilot, Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [finance, market, news, crypto, forex, stocks, trading]
    category: finance
    related_skills: [stocks]
---

# Market News Skill

Turn market-moving headlines into risk and regime signals instead of vague
sentiment chatter. The skill is read-only: it polls RSS/Atom feeds, dedupes
items across runs, scores likely impact, and recommends a trading posture such
as `trade_halt`, `cooldown`, `size_down`, or `watch`.

Use it to decide how cautious the strategy should be around headlines. Do not
use it as a blind direction oracle.

## When to Use

- User asks whether news matters for crypto, forex, or stocks trading.
- User wants a recurring feed watcher that outputs trading risk gates.
- User wants a quick digest of macro, regulatory, earnings, or exchange risk.
- A cron job should watch headlines and stay silent when nothing new appears.

## Prerequisites

- Python 3.9+ stdlib only. No API key, no pip install.
- Invoke through the `terminal` tool.
- Once installed, the helper script lives at:
  `~/.hermes/skills/finance/market-news/scripts/market_news_watch.py`

State is stored under `~/.hermes/watcher-state/market-news-<profile>.json`
unless you override `--name`.

## How to Run

```bash
python3 ~/.hermes/skills/finance/market-news/scripts/market_news_watch.py \
  --profile crypto --emit-initial --json
```

The script baselines on first run and prints nothing unless you pass
`--emit-initial`. After that, only unseen items are emitted.

## Quick Reference

```bash
python3 market_news_watch.py --profile crypto --emit-initial
python3 market_news_watch.py --profile forex --emit-initial --json
python3 market_news_watch.py --profile stocks --max-items 4
python3 market_news_watch.py --profile macro --emit-initial
python3 market_news_watch.py --profile crypto --url https://www.coindesk.com/arc/outboundfeeds/rss/
```

## Procedure

### 1. Pick the right profile

- `crypto` - exchange risk, stablecoins, regulation, ETFs, hacks, macro
- `forex` - central banks, CPI, NFP, rate decisions, intervention
- `stocks` - earnings, guidance, filings, mergers, halts, downgrades
- `macro` - rates, inflation, yields, tariffs, recessions, liquidity

### 2. Run the watcher

```bash
python3 ~/.hermes/skills/finance/market-news/scripts/market_news_watch.py \
  --profile crypto --emit-initial
```

The script fetches the configured feeds, dedupes against its state file, then
prints only the new items.

### 3. Read the recommendation as a risk instruction

- `trade_halt` - severe event risk; stand down until the market stabilizes
- `cooldown` - avoid fresh aggression around the event window
- `size_down` - trade smaller and assume worse slippage / spread conditions
- `watch` - relevant context exists but does not justify a hard gate yet

Treat the action as a risk-layer input. The point is to change sizing,
cooldowns, and execution assumptions, not to force a long/short opinion.

### 4. Wire it into cron

For recurring automation, schedule it as a no-agent script job. Example prompt:

> Every 30 minutes, run the market-news script with the `crypto` profile. If it
> emits new items, summarize them into trading risk gates and deliver them. If
> it prints nothing, stay silent.

### 5. Override feeds when needed

Pass one or more `--url` values to ignore the built-in profile feeds and use
your own sources for a strategy or market niche.

## Pitfalls

- News is usually better as a risk/regime filter than a direction signal.
- RSS is imperfect. Feeds can lag, change format, or disappear.
- First run creates the baseline and stays silent unless `--emit-initial` is
  set.
- A headline can be market-moving even when the text does not contain your
  favorite keyword. Review the feed mix periodically.
- `trade_halt` is a caution signal, not a substitute for explicit strategy kill
  switches.

## Verification

```bash
python3 ~/.hermes/skills/finance/market-news/scripts/market_news_watch.py \
  --profile crypto --emit-initial --json
```

Expected result: JSON with a top-level `action`, `reason`, and `items` list.