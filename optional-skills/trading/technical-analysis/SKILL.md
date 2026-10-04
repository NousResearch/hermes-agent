---
name: technical-analysis
description: |
  Read a price chart and explain it — trend, support/resistance, common
  indicators (RSI, MACD, moving averages, volume), and chart patterns. Turns
  raw OHLCV data into a plain-English read. Education only, NOT financial advice.
version: 0.1.0
author: HeLLGURD
license: MIT
platforms: [linux, macos, windows]
category: trading
triggers:
  - "technical analysis of [asset]"
  - "analyze this chart"
  - "what's the trend on [asset]"
  - "read this chart for me"
  - "support and resistance for [asset]"
  - "what is RSI / MACD saying"
  - "is [asset] overbought"
toolsets:
  - terminal
  - web
  - file
metadata:
  hermes:
    tags: [Trading, Technical-Analysis, Charts, Crypto, Stocks, Indicators, TA]
    related_skills: [stocks, token-research]
---

# Technical Analysis

Translate a price chart into a clear read: what the trend is, where the key
levels are, and what the common indicators are saying — explained so both a
beginner and an experienced trader get value. Works for crypto, stocks, FX, or
any asset with OHLCV (open/high/low/close/volume) data.

> ⚠️ **Education and analysis only. This is NOT financial advice and NOT a
> buy/sell signal.** Technical analysis describes probabilities, not
> certainties. Indicators lag, patterns fail, and markets are driven by factors
> no chart shows. Never trade more than you can afford to lose.

Uses `web` to fetch public price data when needed. No trading, no order
placement — this skill is analysis only.

---

## When to Use

- User wants a technical read on an asset's chart
- User asks what an indicator (RSI, MACD, etc.) is signaling
- User wants support/resistance levels or trend direction explained
- User pastes OHLCV data or points at a ticker and wants analysis

Do NOT use for:
- Placing trades or managing orders — analysis only, by design
- Fundamental analysis (earnings, tokenomics) — different domain
  (see `stocks` / `token-research`)
- Price predictions with specific targets/dates — that's speculation

---

## Hard Guardrails

1. **No financial advice.** Never say "buy" or "sell." Describe what the chart
   shows and what scenarios it suggests; the decision is the user's.
2. **Probabilities, not certainties.** Frame everything as "suggests",
   "indicates", "often precedes" — never "will".
3. **Disclaimer every time.** Every analysis ends with the not-financial-advice
   note.
4. **No leverage encouragement.** If the user mentions high leverage, note the
   liquidation risk neutrally; don't cheerlead.

---

## Prerequisites

- `web` toolset to fetch price data if the user gives only a ticker.
- If the user pastes OHLCV data directly, no network access is needed.

---

## Procedure

### Step 1 — Get the data

- **Ticker given:** fetch recent OHLCV from a public source (CoinGecko for
  crypto, a public stock API) — confirm the timeframe the user cares about
  (intraday, daily, weekly).
- **Data/screenshot description given:** work from what's provided.
- Always confirm the **timeframe** — a "downtrend" on the 1h can be a pullback
  in a daily uptrend. Multi-timeframe context matters.

### Step 2 — Identify the trend

The single most important read. Determine:
- **Direction:** uptrend (higher highs + higher lows), downtrend (lower highs +
  lower lows), or range (sideways between two levels).
- **Strength:** steep and steady vs choppy and weak.
- **Stage:** early, mature, or showing exhaustion signs.

Trend is the backdrop for everything else — indicators mean different things in
a trend vs a range.

### Step 3 — Mark support and resistance

- **Support:** price levels where buying has repeatedly stepped in (bounces).
- **Resistance:** levels where selling has repeatedly capped advances.
- **Round numbers** and prior swing highs/lows often act as levels.
- Note that **broken resistance often becomes support** (and vice versa).
- Give specific price levels, not vague zones, when the data supports it.

### Step 4 — Read the indicators

Explain only the indicators relevant to the question. For each, give the
reading AND what it means in context:

**Moving Averages (MA / EMA):**
- Price above a rising MA = bullish structure; below a falling MA = bearish.
- MA crossovers (e.g. 50 crossing 200 — "golden/death cross") are slow,
  trend-confirming signals, not entries.

**RSI (Relative Strength Index, 0–100):**
- > 70 = overbought (can stay there in strong trends — not an automatic sell).
- < 30 = oversold (can stay there in strong downtrends).
- **Divergence** (price makes a new high, RSI doesn't) is the higher-value
  signal — flag it when present.

**MACD:**
- MACD line crossing above signal line = bullish momentum shift; below =
  bearish.
- Histogram shrinking = momentum fading even if price still moves.

**Volume:**
- Moves on rising volume are more credible than moves on thin volume.
- A breakout on low volume is suspect (prone to fakeout).

### Step 5 — Note chart patterns (if clearly present)

Only call a pattern when it's genuinely there — don't force it:
- **Continuation:** flags, pennants, triangles (often resolve in trend
  direction).
- **Reversal:** double top/bottom, head-and-shoulders, rounding.
- Always pair a pattern with its **invalidation level** — the price that proves
  the pattern wrong.

### Step 6 — Deliver the read

```
Technical Analysis: BTC/USD — Daily

Trend:        Uptrend, mature stage (higher highs + higher lows since March)

Key levels:
  Resistance  $72,000 (prior swing high), $75,000 (round number)
  Support     $66,000 (broken resistance, now support), $61,000 (50-day MA)

Indicators:
  RSI 68      Approaching overbought; not yet divergent
  MACD        Positive but histogram shrinking — momentum cooling
  50-day MA   Price above and MA rising — structure intact
  Volume      Declining on recent push — watch for a weak-volume breakout

Pattern:      Ascending triangle, resistance at $72k.
              Invalidation below $66k support.

Read:         Uptrend intact but momentum is cooling near resistance. A
              decisive close above $72k on strong volume would suggest
              continuation; a rejection there with the volume divergence
              could mean a pullback toward $66k support.

⚠️ Education only — NOT financial advice. TA describes probabilities, not
certainties. Indicators lag and patterns fail. Never trade more than you can
afford to lose.
```

---

## Calibrating to the Reader

- **Beginner** ("what is RSI?") → explain the concept before the reading, avoid
  unexplained jargon.
- **Experienced** (uses TA terms fluently) → skip definitions, lead with the
  levels and the read.

---

## Indicator Cheat Sheet

| Indicator | Bullish | Bearish | Best for |
|---|---|---|---|
| Price vs MA | Above rising MA | Below falling MA | Trend direction |
| RSI | Rising from < 30 | Falling from > 70 | Momentum extremes, divergence |
| MACD | Cross above signal | Cross below signal | Momentum shifts |
| Volume | High on trend moves | High on reversals | Confirming/doubting moves |

---

## Edge Cases

**Low-liquidity / micro-cap asset:** TA is far less reliable — thin volume
means levels and patterns are easily manipulated. Say so explicitly.

**Very short timeframe (1m–5m):** noise dominates; most patterns are
unreliable. Recommend zooming out for context.

**Conflicting signals** (e.g. bullish trend but bearish divergence): present
both honestly rather than forcing a single conclusion. Conflicting signals are
information too.

**News-driven spike:** charts don't show catalysts. If a move looks
news-driven, note that TA can't explain or predict it.

---

## What This Skill Does NOT Cover

- Placing or managing trades — analysis only, by design
- Specific price targets with dates — speculation, not analysis
- Fundamental analysis (earnings, tokenomics, on-chain) — see `stocks` /
  `token-research`
- Backtesting strategies or automated signals — different tooling
