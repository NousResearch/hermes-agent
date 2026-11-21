# Ideas, claims and papers

Use `terminal` for the commands below, with the project Python environment active and the
study directory as the working directory. Use `read_file` for files; examples are not permission
to execute unreviewed code or open a sealed holdout.

## From a sentence to a spec

A claim is testable once each of these has one answer. Write them in `STUDY.md` before any run.

| Part | Question | Example ("funding flips predict reversals") |
|---|---|---|
| Hypothesis | What happens, in one sentence? | When funding turns from positive to negative, price rises over the next 24 h |
| Universe | Which markets, picked how? | `x.top_n(25)`: point in time by 30-day volume |
| Event or signal | What exactly is measured, at what time? | `h.funding_rate` hourly, sign change versus 8 h earlier |
| Horizon | How long is it held? | 24 h, `x.time_stop("24h")` or a daily clock |
| Sizing | How much, and against what? | Equal weight across markets with a flip, long only, gross 1 |
| Costs | What does it pay? | Engine taker and maker fees and funding |
| Window | Over what, ending where? | 2025-03-22 to the holdout |
| Kill | What result ends it? | Mean 24 h return after flips not above other hours', or gauntlet below `holds out of sample` |

Where the claim is vague ("funding matters"), pick the most literal reading, write it down as your
reading, and test that one. Ask the user only when two readings would lead to different studies.

A tweet's chart, a screenshot or a quoted number is the claim, not evidence. Don't reuse its window or
its parameters as if they were neutral: they were likely chosen after the fact. Test them, and also the
neighbours a fair person would have picked.

## Cheap evidence: the event study

Before a strategy, measure the effect directly:

```python
import oxarchive_backtest as oxb

src = oxb.Source("data/hyperliquid")
rows = src.rows("BTC", "2025-03-22", "2026-07-01", columns=["close", "funding_rate"])
```

`Source.rows` returns minute rows as dicts (`pandas.DataFrame(rows)` for analysis) and respects the
study's seal. Resample to the event's clock, mark the events, and compare the forward return after
events with the forward return at all other times, per market and pooled, with a count of events and a
t-stat or a bootstrap interval. Event studies are looks, not trials: they aren't in the ledger, so keep
them few, and list each one in `STUDY.md`.

Overlapping windows inflate t-stats: with a 24 h horizon on hourly events, count independent events, not
hours. Say how many independent events there were. Under 30 is an anecdote.

## Papers

Extract the rules exactly as the paper states them, with its universe, rebalance, signal and costs, and
note every place our data or venue differs (a perp, not a stock; hourly funding; a 19-month sample).
Test the paper's version first, then at most one adaptation, and report both. A paper's out-of-sample
result on another market is a prior, not evidence about this one.

## Writing the strategy

Use the pieces in `oxarchive_backtest.x` and the reads on `h` ([strategy-api.md](strategy-api.md)). Parameters are the arguments of `def strategy(...)`. Read only through `h`,
keep state in `ctx.state`, and run `0xbt check FILE` before the first backtest.
