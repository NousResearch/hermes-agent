# Writing a strategy (oxarchive_backtest)

Use `terminal` for the commands below, with the project Python environment active and the
study directory as the working directory. Use `read_file` for files; examples are not permission
to execute unreviewed code or open a sealed holdout.

The strategy API in brief, from the engine's own documentation. `help(oxb.Strategy)` and `help(x)` in
Python have the rest. Checks (nulls, split, walk-forward, deflated Sharpe) are in gauntlet.md.

## The model

A strategy is a **clock** plus a chain of **steps**, each optional and each swappable:

```python
import oxarchive_backtest as oxb
from oxarchive_backtest import x

vol = lambda h: h.shift(1).close.vol("30d")

strategy = oxb.Strategy(
    universe = x.top_n(25),                          # which markets, point in time
    clock    = x.every("7d", anchor="2025-03-22 01:00"),
    signal   = lambda h: h.close.ret("7d") / vol(h),  # a score per market
    weights  = x.long_short(0.2, by=lambda h: 1 / vol(h)),
    # risk   = [x.cap(0.1)],                          # adjust the weights
    # execute = x.rebalance(),                        # the default: trade to the weights
)
```

At each tick of the clock the run calls, in order:

| Step | Gets | Returns | Default |
|---|---|---|---|
| `universe` | (picked ahead, point in time) | the markets `h` holds | required: a list of names or a rule |
| `signal` | `h` | a `Series` of scores | none |
| `weights` | scores, `h` | a `Series` of target weights (fraction of equity, signed), or None to skip this tick | none |
| `risk` | weights, `h`, `ctx` | adjusted weights | none |
| `execute` | `ctx`, weights, `h` | places orders | `x.rebalance()` |

A strategy that doesn't fit "trade to target weights" writes its own last step instead: `step=lambda ctx, h: ...` replaces signal, weights, risk and execute and places orders itself (`ctx.limit`, `ctx.stop`, ...). With `clock="1m"` it runs every minute. There is one model; the low level is its finest setting, not a second API.

**When it trades.** `clock` is a duration (`"1h"`), `"daily"`, `"weekly"` or `"monthly"`, or a clock piece: `x.every("7d", anchor=)`, `x.daily("08:00", tz="America/New_York")`, `x.weekly("mon", at="08:00")`, `x.monthly(-1)` (the last day), `x.at_open("us", after="15m")`, `x.at_close("us", before="15m")`. Calendar clocks follow the zone's daylight saving; session clocks tick only on the exchange's trading days. `delay="2h"` works the weights out at the tick and trades them two hours later. `hold="1h"` trades back to the last target weights every hour between ticks, so they stay fixed instead of drifting with prices (the default).

A user function can take fewer arguments than the table gives (`weights=lambda s: ...`); the run passes what it asks for.

**Parameters** are the arguments of a function that returns the strategy. A file defines `strategy`, either a `Strategy` or a function returning one; a sweep varies that function's arguments.

```python
def strategy(lookback="7d", quantile=0.2):
    return oxb.Strategy(..., signal=lambda h: h.close.ret(lookback) / vol(h),
                        weights=x.long_short(quantile, by=lambda h: 1 / vol(h)))
```

## History: `h`

`h` holds the last bars of every market in the universe, resampled from the minute rows to `bars` (default `"1h"`) and closed on the bar boundary, so it never shows an unfinished bar.

- `h.close`, `h.quote_volume`, `h.open_interest`, ...: any dataset column, as a field.
- On a field: `last()`, `ago(n)`, `ret(n)`, `mean(n)`, `sum(n)`, `std(n)`, `min(n)`, `max(n)`, `vol(n)`, `window(n)`. `n` is a number of bars or a duration (`"7d"`). Each returns a `Series` (`window` returns lists).
- `logret(n)`, `ewma(halflife, window=None)`.
- `beta(n, to="BTC")`, `corr(n, to=)`, `resid(n, to=, fit="60d")`: against a market or `"universe"` (the members' equal-weight mean return), returns counted as in `vol`. The run loads `to` even when it isn't in the universe. `resid` sums each bar's return less its beta (fitted on the `fit` returns before it) times `to`'s: residual momentum is `h.close.resid("15d", fit="60d") / vol(h)`.
- `seasonal(n, period="1d", ahead=1)`: the typical one-bar return at this point of the period, over the last n periods. `ahead=1` is the coming bar's (a signal: trade the hour); `ahead=0` the bar just closed (a control: `ret(1) - seasonal(n, ahead=0)` is the move net of its usual size).
- `h.shift(k)`: the same history as of k bars earlier. `h.bars("1d")`: the same markets on other bars, each size folded from the minute rows on its own; its latest bar is the last to close (at a 13:00 tick, yesterday's day).
- `h.markets`, `h.time`, and `h.now`: the time as a `Moment` (below).

## Time: `h.now`

A `Moment` reads the latest bar's close as a calendar, for seasonality signals, filters and controls.

- UTC fields: `year`, `month`, `day`, `hour`, `minute`, `weekday` (0 = Monday), `weekend`, `quarter`, `days_to_month_end`, `days_to_quarter_end`; `local(zone)` for another zone's wall clock (UTC, America/New_York, America/Chicago, Europe/London, Asia/Tokyo).
- Sessions: `open(s)`, `minutes_since_open(s)`, `minutes_to_close(s)`, `minutes_to_open(s)`, `minutes_since_close(s)`, `holiday(s)`. `"us"` is NYSE's regular session with its holidays and 13:00 early closes (rule-based, checked against NYSE's published lists for 2024 to 2027); `"europe"` (London 08:00 to 16:30) and `"asia"` (Tokyo, with its lunch break) are weekdays only, without holiday lists.
- Venue events: `minutes_to_funding` (Hyperliquid settles on the hour), `expiry` (today's 08:00 UTC options expiry by Deribit's schedule: daily, weekly, monthly or quarterly) and `hours_to_expiry(kind)`.

Rules:

- **Bars:** open is the first open, close the last close, high and low the extremes, volumes and counts are sums, state columns (open interest, funding rate, mark) are the last value.
- **Missing prices carry forward** up to `carry` (default `"6h"`) and are None after that.
- **Bad prints:** `repair=True` drops a price more than 2.5 times from the median close of the 48 bars before it (trailing, so it never looks ahead); the close then carries forward. Off by default.
- **Reads are cached** until a bar closes or the universe changes, so reading `vol(h)` in both the signal and the weights, or every minute on hourly bars, computes it once.
- **Not enough history:** an op that needs more bars than the run has seen returns None for every market. A strategy therefore never trades on a half-filled window.
- **`vol(n)`** is the sample standard deviation of the last n one-bar returns, a missing return counting as 0 and a move of 50% or more as a bad print (also 0).
- All sums are plain loops in Rust, so results match on every platform.

## Series

What signals and weights pass around: one value per market, None where unknown, in universe order.

- Arithmetic with another Series or a number (`+ - * /`, unary `-`, `abs`); None propagates and division by zero gives None.
- `rank()` (percentile, ties averaged, as pandas' `rank(pct=True)`), `zscore()`, `top(k)`, `bottom(k)`, `dropna()`, `sum()`, `mean()`, `log()`, `exp()`, `sqrt()`, `sign()`.
- `demean(by=None)` (within groups when `by` maps markets to groups), `neutralize(*exposures)`: the residual of a cross-sectional regression on the exposures, such as `s.neutralize(h.close.beta("60d"))`.
- `x.blend(a, b, weights=None)`: signals combined by the mean of their pct ranks.
- Reads like a dict: `s["BTC"]`, `s.items()`, `len(s)`. `to_pandas()` for research.

## Pieces (`oxarchive_backtest.x`)

| Kind | Pieces |
|---|---|
| Universe | a list of names; `x.top_n(n, window, min_history, every, anchor, by="volume", min_volume=0, exclude=(), sectors=None, per_sector=0)`; `x.rule(select, every, anchor)` for your own, where `select(view)` reads `view.closes`, `volume`, `open_interest`, `funding`, `volatility`, `history_hours` and `listed` from before the pick |
| Clock | `"1m"`, `"1h"`, `"daily"`, `"weekly"`, `"monthly"`; `x.every(duration, anchor)`, `x.daily(at, tz, weekdays)`, `x.weekly(day, at, tz)`, `x.monthly(day, at, tz)`, `x.at_open(session, after/before)`, `x.at_close(session, before/after)` |
| Weights | `x.long_short(quantile, by=None)`, `x.long_only(top=None, quantile=None, by=None)`, `x.sign(scale=1.0, short=True)` |
| Risk | `x.cap(market, gross=, net=)`, `x.gross(target)`, `x.book(gross=, net=)`, `x.beta_neutral(to, n)`, `x.hedge(to, n)`, `x.vol_target(target, n, max_scale)`, `x.turnover(max)`, `x.only(session or condition, outside="hold")` |
| Execute | `x.rebalance(exposure=1.0, band=1e-6, max_cost=None)` |
| Exits | `x.stop_loss(pct)`, `x.take_profit(pct)`, `x.trailing_stop(pct)`, `x.time_stop(after)`, `x.drawdown_stop(pct, cooldown=None)` |

**Exits** (`exits=[...]` on the Strategy) are checked by the run every minute, whatever the clock, in Rust. A position's entry is its average fill price since it opened; once a minute's high or low reaches a level, a reduce-only market order closes it at the next minute's open (a stop counts before a target in the same bar), and the market stays flat until the strategy trades it again. `x.drawdown_stop(0.2)` cancels everything, closes every position and stops calling the strategy once equity is 20% off its peak, for good or for `cooldown`. `Report.exits` lists each one: (time, market or None, why).

**`on_universe=lambda ctx, joined, left: ...`** is called at the minute the universe's members change, with the names that joined and left.

`by` weights a leg by a function of `h` (inverse volatility: `by=lambda h: 1 / vol(h)`); without it the leg is equal weight. Each leg of `long_short` is half the book; at a quantile of 0.5 or more a market can be in both legs and gets both.

## What the run works out for itself

Because steps are called through `h`, the run checks the strategy once before starting (it calls the steps on an empty history and records what they read) and derives:

- **Warm-up:** the longest window read on any bar size, in time, and the universe rule's own lookback. A backtest from `start` reads data from `start - warm-up`; the strategy's first tick is at `start`, with full windows.
- **Columns:** the fields read plus the ones the fill model needs. Nothing else is decoded.
- **When to call Python:** only at clock ticks, `delay` and `hold` times (and `on_fill` when there are fills). A weekly strategy over 19 months is called about 80 times, not once a minute.
- **Markets:** the universe's members over the window, plus any market read against (`beta(to="BTC")`). The rule is part of the strategy, so a deploy picks live with the same code, and the source hash in the receipt covers it.

The check calls each step once, with every read None and `ctx.time` 0 (a step that returns None hands the next one zero weights, so every step is seen), so keep state between ticks in `ctx.state` (which the check doesn't keep) rather than in your own variables. A step that reads something only on some branches can miss the check; the run then stops with an error naming the field, and `history="30d"` or `fields=[...]` on the Strategy declares it.

## Running

```python
report = oxb.backtest(strategy, "2025-07-01", "2025-10-01", source="data/hyperliquid",
                      config=oxb.Config(starting_cash=100_000))
report.stats        # return, annual_return, volatility, sharpe, sortino, max_drawdown, calmar,
                    # hit_rate, trades, win_rate, turnover, exposure, fills, fees, funding...
report.attribution  # {market: {pnl, fees, funding, volume, trades, wins}}
report.equity       # [(ts, equity)] from start
report.daily_returns
report.receipt(source_text, params)
```

## Ctx

`ctx.time`, `ctx.equity`, `ctx.cash`, `ctx.positions`, `ctx.price(m)`, `ctx.universe`, `ctx.markets`, `ctx.open_orders`, `ctx.state` (a dict kept between ticks), and orders: `ctx.market(m, size)`, `ctx.limit(m, size, price)`, `ctx.stop(m, size, trigger)`, `ctx.cancel(id)`, `ctx.cancel_all()`, `ctx.rebalance(weights, exposure, band)`, and `ctx.cost(weights, exposure, band)`: the fill model's estimate, in quote currency, of the fees and slippage of trading to those weights now.

In a portfolio, a market order that the leverage cap would cut waits until the minute's other orders have filled (fill model `bar-v2`), so a rebalance's sells free margin for its buys whatever the order of the markets.

## Paper trading

`oxb.Session(strategy, markets, start=...)` runs the same loop on rows as they arrive:
`session.warm_up(source)` replays the history the strategy reads, then `session.step({"BTC": row, ...})`
once a minute with each closed minute's row, and `session.report()` gives fills and stats. Fed the
dataset's rows, a session gives the backtest's exact result hash. It places no orders: live trading is out of
scope for this skill.
