# Porting Pine Script, TS indicators and other frameworks

Use `terminal` for the commands below, with the project Python environment active and the
study directory as the working directory. Use `read_file` for files; examples are not permission
to execute unreviewed code or open a sealed holdout.

Port the logic, not the platform's fill behaviour. The engine's fills, fees, funding and leverage
limits are the ones that count here; say where they differ from the original's.

## Pine Script to oxb

| Pine | oxb |
|---|---|
| `close`, `open`, `high`, `low`, `volume` | `h.close.last()`, ..., `h.volume.last()` (a `Series`, one value per market) |
| `close[n]` | `h.close.ago(n)` or `h.shift(n).close.last()` |
| `ta.sma(close, n)` | `h.close.mean(n)` |
| `ta.ema(close, n)` | `h.close.ewma(halflife)`: Pine's decay per bar is `1 - 2 / (n + 1)`, so halflife = `ln 0.5 / ln(1 - 2 / (n + 1))` bars (about `(n + 1) * 0.35`). Pine's EMA also seeds from an SMA; use `window(n)` and your own loop to match it exactly |
| `ta.stdev(close, n)` | `h.close.std(n)` |
| `ta.highest(high, n)`, `ta.lowest(low, n)` | `h.high.max(n)`, `h.low.min(n)` |
| `ta.roc(close, n)` | `h.close.ret(n)` (a fraction, not a percent) |
| `ta.change(close, n)` | `h.close.last() - h.close.ago(n)` |
| `ta.rsi`, `ta.atr`, `ta.macd` | build from `window(n)` in plain Python loops (deterministic math only: no numpy transcendental functions) |
| `ta.crossover(a, b)` | `a(h) > b(h)` and `a(h.shift(1)) <= b(h.shift(1))`, per market |
| `timeframe.period` | `bars="1h"`, `"4h"`, `"1d"`; `h.bars("1d")` for a second timeframe |
| `request.security(..., "D", ...)` | `h.bars("1d")`: its latest bar is the last one to close, so no lookahead |
| `strategy.entry` / `strategy.close` | `weights=` for target positions, or `step=` with `ctx.market` |
| `strategy.exit(stop=, limit=)` | `exits=[x.stop_loss(p), x.take_profit(p)]` or `ctx.stop`, `ctx.limit` |
| `strategy.position_size` | `ctx.positions[m]` |
| `calc_on_every_tick`, `process_orders_on_close` | not modelled: the engine trades from the next minute after a bar closes |
| Pine `na` | `None`; a read with too little history is `None` for every market |

Pine's defaults that commonly inflate results, to check and state: zero commission, fills at the bar's
close of the signal bar, `pyramiding`, and repainting with `request.security` on an unconfirmed bar.

## Other frameworks

- **backtrader / zipline / freqtrade:** the `next()` or `populate_*` logic is the signal; the broker
  settings are the costs. Map `self.data.close[0]` to `h.close.last()`, `self.order_target_percent` to
  weights.
- **vectorbt / pandas:** a signal frame of `entries`/`exits` usually means "trade on this bar's close
  with this bar's information". Shift it by one bar when porting, or the port won't match and the
  original was looking ahead.
- **A notebook:** find where the signal is computed and where returns are computed from it; the shift
  between them, the cost line and the universe are what to check (audit.md).

## Checking the port

Inspect the original with `read_file` before any execution. Do not run an uncleared original
strategy, notebook, dependency installer or imported module. Only after explicit user approval of
the inspected code and its isolated project environment, run it on the same market and window
and compare the trade list for the first
week: entries and exits should match bar for bar except for the documented fill difference. If the
original can't run here, compare the signal values on a few dated bars the user can check by hand.
Report the match before any verdict.
