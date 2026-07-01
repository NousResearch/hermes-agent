---
name: stock-sentiment
description: "Sentiment and smart money for US stocks: the SentiSense Score, sentiment polarity, market mood (fear/greed), insider and congressional and 13F flows, analyst actions, AI insights, and sentiment-tagged news via a read-only API."
version: 0.1.0
author: SentiSense
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [Sentiment, Stocks, Finance, Market, Investing]
    category: finance
    related_skills: [stocks, dcf-model, comps-analysis]
required_environment_variables:
  - name: SENTISENSE_API_KEY
    prompt: SentiSense API key
    help: "Free key from Settings > Developer Console at https://sentisense.ai"
    required_for: sentiment, smart-money, and AI-insight data
---
# Stock Sentiment Skill

The sentiment and smart-money layer for US equities. A quote skill tells you the price; this skill reads what the market feels about a stock (the SentiSense Score, sentiment polarity, mentions, share of voice), where the smart money is moving (insider, congressional, and institutional flows plus analyst actions), and what the AI read of the tape is (per-stock and market-wide insights, sentiment-tagged news), all through the read-only SentiSense API.

Read-only educational data interface. Output is informational context, never a personalized buy or sell recommendation.

## When to Use

Reach for this skill when the question is about perception, positioning, or signal rather than raw price:

- "What is the sentiment on $NVDA?" or "Is the mood on $TSLA bullish or bearish?"
- "What is the smart money doing this week?" (insider cluster-buys, congressional trades, 13F flows, and analyst upgrades converging on the same tickers).
- "What is the overall market mood today, fear or greed?"
- "Is sentiment diverging from price on $COIN?" (price up while sentiment falls, or the reverse).
- "What is the pre-earnings sentiment setup on $AAPL?"
- "What is the AI insight on $MSFT, and what are people saying in the news?"

This skill complements the rest of the finance catalog rather than competing with it. The `stocks` skill covers real-time quote, search, history, and compare; `dcf-model` and `comps-analysis` cover intrinsic and relative valuation. This skill adds the sentiment, smart-money, and AI-insight dimension none of those carry. A useful pairing: value a name with `dcf-model` or `comps-analysis`, pull its quote with `stocks`, then read this skill for the sentiment and positioning context around it.

Do not use it for order entry, portfolio management, or personalized advice. It has no write, trading, or wallet surface; every endpoint is a GET.

## Prerequisites

- Python 3.8+ using only the standard library (`urllib`, `json`); no third-party packages required. Any HTTP client or plain `curl` works too.
- A free `SENTISENSE_API_KEY`. Generate one from Settings > Developer Console at https://sentisense.ai. The key is required on every call; anonymous requests return `401 api_key_required`.
- Network access to `https://app.sentisense.ai`.
- Read-only scope. Every endpoint here is a GET. Nothing this skill does can place a trade, move money, or modify account state.

Tiers:

| Tier | Quota | Rate |
|------|-------|------|
| Free | 1,000 requests/month | 30 requests/min |
| PRO ($15/mo) | Unlimited | 300 requests/min |

The free tier exercises every workflow below. Preview-gated endpoints return a truncated but real slice on a free key (for example the top 3 insights); PRO removes the monthly cap and returns full history and full lists.

## How to Run

This skill is invoked through the agent's terminal or shell tool: issue HTTP GET requests to the SentiSense API and synthesize the JSON into a concise, sourced answer. The base URL is `https://app.sentisense.ai`. Authenticate every request with the `X-SentiSense-API-Key` header; keep the key in the shell environment and never place it in a query string or in user-facing output.

```bash
curl -s -H "X-SentiSense-API-Key: $SENTISENSE_API_KEY" \
  "https://app.sentisense.ai/api/v2/metrics/entity/NVDA/metric/sentiment"
```

An anonymous call returns `401 api_key_required`. A rate-limited call returns `429` with a `Retry-After` header; back off for the indicated seconds rather than retrying immediately or serving a stale value.

On Windows, use the bundled Python client (cross-platform) and reference the key as `%SENTISENSE_API_KEY%` (cmd) or `$env:SENTISENSE_API_KEY` (PowerShell) rather than the POSIX `$SENTISENSE_API_KEY` shown above.

Two response envelopes exist; unwrap correctly before reading fields:

- Read FLAT (top-level, no `.data`): `stocks/price`, `stocks/prices`, `stocks/chart`, `stocks/popular`, `stocks/{T}/profile`, `market-mood`, and the metric series (`sentiment`, `sentisense`, `mentions`, and `social_dominance` are bare arrays). `institutional/quarters` is also a bare array.
- Read WRAPPED as `{ isPreview, previewReason, data }` (use `.data`): `insider/*`, `politicians/*`, `institutional/holders`, `analyst/*`, `insights/*`, and `calendar/earnings` (here `data` is a dict, so read `data.earnings[]`).
- `documents/ticker` has its own shape `{ documents, totalCount }`; read `.documents[]`.

When unsure, accept both: `rows = raw if isinstance(raw, list) else raw.get("data", raw)`.

An optional stdlib helper, `scripts/sentiment_client.py`, wraps all of this: it injects the auth header, prepends the base URL, and normalizes both envelopes (including the nested sentiment scalar) so the agent reasons over clean values. Use it or plain `curl`, whichever fits the host. The core of the helper is small enough to inline:

```python
#!/usr/bin/env python3
"""Minimal stdlib client for the read-only SentiSense API."""
import json, os, urllib.parse, urllib.request

BASE = "https://app.sentisense.ai"

def get(path, **params):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url, headers={"X-SentiSense-API-Key": os.environ["SENTISENSE_API_KEY"]})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)

def rows(raw):
    """Wrap-vs-flat: some endpoints return a bare array, others {isPreview, data}."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "data" in raw:
        return raw["data"]
    return raw

def latest_sentiment(ticker):
    """Latest sentiment polarity in [-1, 1]; the scalar is nested at metricValue.value.value."""
    series = get(f"/api/v2/metrics/entity/{ticker}/metric/sentiment")
    if not series:
        return None
    return float(series[-1]["metricValue"]["value"]["value"])
```

```bash
python scripts/sentiment_client.py sentiment NVDA
python scripts/sentiment_client.py mood
```

## Quick Reference

All paths are relative to `https://app.sentisense.ai` and are GET. Every call requires the `X-SentiSense-API-Key` header. `{T}` is an uppercase ticker, `{slug}` a member slug, `{id}` a story id. Full schema: https://sentisense.ai/skill.md.

```
SENTIMENT & MOOD
  GET /api/v2/metrics/entity/{T}/metric/sentiment?startTime={epochMs}&endTime={epochMs}
        Sentiment polarity time series. Omit params for the server default 7-day window.
        Bare array; latest scalar is series[-1].metricValue.value.value (a float in [-1, 1]).
  GET /api/v2/metrics/entity/{T}/metric/sentisense
        The SentiSense Score (unbounded composite; report as-is, never normalize to 0-100).
  GET /api/v2/metrics/entity/{T}/metric/mentions
        Mention-volume time series (how much a ticker is being talked about).
  GET /api/v2/metrics/entity/{T}/metric/social_dominance
        Share-of-conversation time series (a ticker's dominance of the chatter).
  GET /api/v2/market-mood
        Composite fear/greed plus sub-signals and per-sector breakdowns. Flat, but the
        composite is nested: market.currentScore, market.phase, market.weeklyChange,
        market.signals[]; sectors.{SectorName}.{currentScore, phase, weeklyChange}.

SMART MONEY  (wrapped in {isPreview, previewReason, data}; free key returns a preview slice)
  GET /api/v1/insider/cluster-buys?lookbackDays=N         Tickers with multiple insider buys.
  GET /api/v1/insider/trades/{T}?lookbackDays=N           Form 4 rows; transactionType BUY|SELL.
  GET /api/v1/politicians/activity?lookbackDays=N         Congressional trades; PURCHASE|SALE.
  GET /api/v1/politicians/filings/{T}?lookbackDays=N      Per-ticker congressional filings.
  GET /api/v1/politicians/member/{slug}                   Member profile (data.recentTrades[]).
  GET /api/v1/institutional/quarters                      Call FIRST; bare array, [0].reportDate is latest.
  GET /api/v1/institutional/holders/{T}?reportDate={Q}    Top 13F holders (data.holders[], largest first).
  GET /api/v1/analyst/{T}/consensus                       Price-target band (consensus.consensusLabel).
  GET /api/v1/analyst/{T}/actions?lookbackDays=N          Recent rating changes for one ticker.
  GET /api/v1/analyst/{T}/estimates                       EPS band (estimateLow/Mean/High,
                                                          numberOfAnalysts) + surprises[]; no revenue.
  GET /api/v1/analyst/activity?lookbackDays=N             Market-wide actions (filter actionType client-side).

AI INSIGHTS  (wrapped; batch, carry generatedAt)
  GET /api/v1/insights/stock/{T}         Per-stock signals ranked by importance; data[0].insightText is the headline. Free preview top 3.
  GET /api/v1/insights/stock/{T}/types   Available insight types (no auth, no quota cost).
  GET /api/v1/insights/market            Top market-wide signals (data[], insightText; ticker embedded in insightText).

NEWS & STORIES
  GET /api/v1/documents/ticker/{T}?limit=N          Sentiment-tagged feed ({documents, totalCount}); each doc
                                                   has url, source, published (epoch seconds), averageSentiment; no title.
  GET /api/v1/documents/stories?limit=N             Pre-clustered stories; cluster.title is SentiSense-authored and safe to show.
  GET /api/v1/documents/stories/ticker/{T}?limit=N  Stories for one ticker.
  GET /api/v1/documents/stories/{id}                Story detail (PublicStoryDetailDto; aspectPerspectives[], bullishView/bearishView).
  GET /api/v1/documents/search?query=...            Topical document search.

SUPPORTING  (price, prices, chart are real-time; profile, popular, calendar, market-summary are reference or batch)
  GET /api/v1/stocks/price?ticker={T}                       price.currentPrice, price.changePercent.
  GET /api/v1/stocks/prices?tickers=A,B,C                   Batch quotes.
  GET /api/v1/stocks/{T}/profile                            profile.name, sector, industry.
  GET /api/v1/stocks/chart?ticker={T}&timeframe=1D|5D|1W|1M|3M|6M|1Y|ALL   Bars; read each point's timestamp (Unix ms).
  GET /api/v1/stocks/popular                                Candidate universe for screens.
  GET /api/v1/calendar/earnings?ticker={T}                  data.earnings[]; next date + consensus EPS + confirmed.
  GET /api/v1/market-summary                                Market-wide narrative headline.
```

Sentiment is polarity: a float in [-1, 1] where the sign is the direction (negative is bearish and meaningful, positive is bullish) and the magnitude is conviction. Represent the sign unmistakably; do not map it onto a 0-100 scale. The SentiSense Score is a separate, unbounded composite; report it as-is. Mentions and social dominance are their own metric series on the same `/metric/{metricType}` endpoint (`mentions` for talk volume, `social_dominance` for share of the conversation); all four series (`sentiment`, `sentisense`, `mentions`, `social_dominance`) are Public with no quota cost. A separate `/api/v2/metrics/entity/{T}/distribution/{metricType}` endpoint breaks a metric down by source (share of voice, a "where this signal came from" view, not per-source sentiment values).

## Workflows

Opinionated recipes. Each fans out its independent calls in parallel, then synthesizes; none recommends buying or selling. Frame every result as educational context on positioning and mood.

### 1. Sentiment read on a ticker

Answer "what is the market feeling about $T" in a few dense lines. Fire these in parallel:

1. `GET /api/v2/metrics/entity/{T}/metric/sentiment` for the polarity trend (server default 7-day window; the latest scalar is `series[-1].metricValue.value.value`, a float in [-1, 1]).
2. `GET /api/v2/metrics/entity/{T}/metric/sentisense` for the composite score.
3. `GET /api/v1/documents/ticker/{T}?limit=8` for mention volume (`totalCount`) and the sentiment-tagged feed.
4. `GET /api/v1/insights/stock/{T}` for the top AI insight (`data[0].insightText`, with `generatedAt` for freshness).

Synthesize as educational context, leading with the differentiated sentiment read, not the price: "$NVDA sentiment +0.42 over 7d and rising; SentiSense Score elevated; mention volume heavy; latest AI insight: 'Data-center demand commentary firming' (as of the batch time)." Show the `generatedAt` age so the reader knows these are batch metrics.

### 2. Market mood (fear and greed)

Answer "what is the overall market mood today."

1. `GET /api/v2/market-mood`.

The response is flat, but the composite is nested under `market`, not the root: `market.currentScore`, `market.phase` (for example Greed or Fear), `market.weeklyChange`, and `market.signals[]` (each sub-gauge with its value and change). Per-sector readings live at `sectors.{SectorName}.{ currentScore, phase, weeklyChange }`; `sectors` is a string-keyed dict, not an array, and has overlapping GICS labels (`Technology` and `Information Technology`, `Healthcare` and `Health Care`), so dedupe those before ranking top and bottom sectors. Report as context: "Market mood 62 (Greed), +4 over the week. Greed leaders: Technology, Communications. Fear: Energy, Utilities." Optionally pair with `GET /api/v1/market-summary` for the narrative headline and `GET /api/v1/insights/market` for the top market-wide signals.

### 3. Smart-money convergence screen

Find tickers where insider buying, congressional purchases, and analyst upgrades line up in the same window; convergence is the signal a quote feed cannot produce.

1. `GET /api/v1/insider/cluster-buys?lookbackDays=7`.
2. `GET /api/v1/politicians/activity?lookbackDays=7`, keeping rows with `transactionType == "PURCHASE"`.
3. `GET /api/v1/analyst/activity?lookbackDays=7`, filtering client-side to `actionType == "UPGRADE"` (there is no server-side type filter).

All three are wrapped: read `.data`. Intersect the three ticker lists and report names appearing in two or more buckets, ranked by total signal count, with a one-liner each: "$NVDA: 4 insiders bought, 1 congressional purchase, 2 analyst upgrades (7d)." If a 7-day bucket returns an empty array (common on quiet weeks; `isPreview:false`, disclosure lag, not an error), widen that specific call to `lookbackDays=30` and note the wider window rather than showing a blank result. For one ticker's full flow, run `insider/trades/{T}`, `politicians/filings/{T}`, `institutional/quarters` then `institutional/holders/{T}?reportDate={Q}`, and `analyst/{T}/actions`. Present as observed positioning, never as advice.

### 4. Pre-earnings sentiment check

Read the sentiment and positioning into an earnings print.

1. `GET /api/v1/calendar/earnings?ticker={T}` for the next report date and consensus (`data.earnings[0].earningsDate`, `confirmed`); an empty response means the name is outside the forward window, so ask the user for the date instead of guessing.
2. `GET /api/v2/metrics/entity/{T}/metric/sentiment?startTime={now-30d}&endTime={now}` (epoch milliseconds) for the 30-day sentiment trend.
3. `GET /api/v1/insider/trades/{T}?lookbackDays=60` for recent insider activity (`transactionType` BUY or SELL).
4. `GET /api/v1/analyst/{T}/estimates` for the EPS band and `surprises[]` beat/miss history.
5. `GET /api/v1/analyst/{T}/actions?lookbackDays=30` for recent rating changes.
6. `GET /api/v1/insights/stock/{T}` for the current AI read.

Synthesize the setup as educational context: "$AAPL earnings in 5d: sentiment +0.22 over 30d and trending up; insiders net sellers (2 sells, 0 buys); EPS consensus $1.52 (range $1.48 to $1.55, 28 analysts), beat in 3 of the last 4 quarters; 3 upgrades in 30d. Setup reads mixed-to-constructive." Do not tell the user how to trade the print.

### 5. Sentiment-versus-price divergence

Surface names where perception and price disagree; a bullish gap (price down, sentiment up) and a bearish gap (price up, sentiment down) are the two shapes of interest.

1. `GET /api/v1/stocks/popular` for the candidate list.
2. For each candidate, in parallel: `GET /api/v1/stocks/chart?ticker={T}&timeframe=1M` (a bare array of intraday bars; filter to `timestamp >= now-7d` and compare the first versus last bar for the 7-day move) and `GET /api/v2/metrics/entity/{T}/metric/sentiment` (server default 7-day window; measure the trend across the returned series).
3. Rank by the absolute gap between the price move and the sentiment move; report the top few in each direction.

Frame the result as an observed divergence, not a signal to act: "Bullish divergence: $TSLA price -8% while sentiment +0.11 over 7d. Bearish divergence: $COIN price +14% while sentiment -0.09." Keep the real-time price and the batch sentiment labeled with their own freshness; do not blend them into one implied "now."

## Pitfalls

- **Batch, not real time.** Sentiment, the SentiSense Score, mentions, share of voice, news clustering, and AI insights are batch metrics computed on a schedule; only quote, price, and chart points are real time. State a batch value with its `generatedAt` age and never label it "real time."
- **Empty smart-money windows are normal.** The 7-day insider and congressional feeds often return empty arrays on quiet weeks (disclosure lag, `isPreview:false`, not an error). Widen that specific call to `lookbackDays=30` and note the wider window rather than showing a blank result.
- **Preview gating is data, not failure.** On the free tier, preview-gated endpoints return `isPreview:true` with a real truncated slice (for example the top 3 insights, the current earnings week, a sliced holder list). Render the slice as the answer and tag it `(preview)`. Mention PRO only when the truncation is materially limiting the answer.
- **Wrap versus flat differs by endpoint.** Reading `.data` on a flat endpoint (or the reverse) yields nothing. Flat: `stocks/price`, `stocks/prices`, `stocks/chart`, `stocks/popular`, `stocks/{T}/profile`, `market-mood`, the `sentiment`, `sentisense`, `mentions`, and `social_dominance` series, and `institutional/quarters`. Wrapped under `.data`: `insider/*`, `politicians/*`, `institutional/holders`, `analyst/*`, `insights/*`, and `calendar/earnings`. When unsure, accept both.
- **The sentiment scalar is nested.** The series is a bare array and the float lives at `series[i].metricValue.value.value`; `series[i].metricValue.value` is itself a dict, so there is no top-level `series[i].value` shortcut.
- **Congress and insider use different verbs.** Insider rows carry `transactionType` BUY or SELL; congressional rows carry PURCHASE or SALE. Filter each with its own vocabulary.
- **Always fetch quarters first.** Call `institutional/quarters` and pass `[0].reportDate` to `institutional/holders`; never hardcode a quarter.
- **Documents carry no article title.** The document feed returns URLs, `source`, `published` (epoch seconds), and `averageSentiment`, not the publisher's headline. Pre-clustered story titles (`cluster.title`) are SentiSense-authored and safe to display verbatim; prefer stories when a readable title is needed.
- **No invented endpoints.** There is no options flow, no dark pool, and no `/congress` (congressional data lives under `/politicians`). The earnings calendar is `/api/v1/calendar/earnings`.
- **No advice.** When asked "should I buy," return data-grounded synthesis (sentiment, smart-money flow, analyst consensus, AI insight) framed as educational context, not a personal recommendation.

## Verification

Confirm the skill is wired correctly before trusting a synthesis:

1. **Reachability and auth.** `GET /api/v1/insights/stock/AAPL/types` needs no authentication and costs no quota; a `200` with a JSON list confirms the base URL and network. Then repeat one authenticated call, for example `curl -s -o /dev/null -w "%{http_code}" -H "X-SentiSense-API-Key: $SENTISENSE_API_KEY" "https://app.sentisense.ai/api/v2/market-mood"`; a `200` confirms the header and key. A `401 api_key_required` means the header or `SENTISENSE_API_KEY` is missing or wrong; a `429` means the per-minute rate was exceeded, so honor the `Retry-After` hint.
2. **Sentiment parses.** Fetch `/api/v2/metrics/entity/AAPL/metric/sentiment`, confirm a non-empty array, and read `series[-1].metricValue.value.value`; it should be a float in [-1, 1]. A value outside that range means the wrong nesting was read.
3. **Mood nests as expected.** Fetch `/api/v2/market-mood` and confirm `market.currentScore`, `market.phase`, and `market.weeklyChange` are present (not at the root), and that `sectors` is a populated dict.
4. **Envelope check.** Confirm `institutional/quarters` parses as a bare array and `insider/cluster-buys?lookbackDays=30` parses as `{ isPreview, data }` with `data` an array (an empty array on a quiet window is a valid result, not a failure).
5. **Freshness is surfaced.** Any batch value presented to the user carries its `generatedAt`; if a synthesis omits the age on a sentiment or insight figure, or describes a batch surface as real time, it is not verified.

A run passes when every quoted number traces to a `200` response read this turn, batch and real-time surfaces are labeled distinctly, and the output reads as educational context rather than a recommendation.