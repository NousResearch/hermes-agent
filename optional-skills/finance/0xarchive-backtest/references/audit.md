# Auditing someone else's backtest

Use `terminal` for the commands below, with the project Python environment active and the
study directory as the working directory. Use `read_file` for files; examples are not permission
to execute unreviewed code or open a sealed holdout.

The question is "what would this have made, honestly?" Find every place the original's number is
higher than an honest test would give, then run the honest test.

## Checklist

Go through every item, mark each found / not found / can't tell, and quote the line for each found.

1. **Lookahead.** Signals computed from the bar they trade on (`close` used to decide and to fill at
   the same close); a rolling window that includes the current bar; `shift(-1)`; a daily value used
   during its own day; normalising with the full-sample mean or std; future funding or OI joined on the
   wrong timestamp.
2. **Survivorship.** A universe of markets picked today (the current top 20) applied to the past;
   delisted markets dropped. The fix is a point-in-time universe (`x.top_n`, `x.rule`).
3. **Costs.** No fees, maker fees assumed on taker orders, no slippage or spread, no funding on perps.
4. **Fills.** Filling at a price that didn't trade (a limit at the low), filling the whole size at the
   touch in a thin market, ignoring the minimum order size, no leverage limit.
5. **Compounding and leverage.** Returns summed across markets as if each had its own full account;
   leverage above the venue's limit; no liquidation.
6. **Snooping.** Parameters that look tuned (a 13-period, a 0.37 threshold), a window that starts just
   after a crash, a grid search reported as its best point, and no out-of-sample period.
7. **Accounting.** Sharpe from per-trade returns annualised as if daily; a return series with gaps
   treated as zero-volatility days; equity in quote units mixed with base.

## Then run it honestly

Port it (ports.md) with the found items fixed, run the gauntlet in a study, and report two numbers side
by side: the original's claim, and the engine's after the fixes, each fix's contribution where it can be
isolated (run with and without it; each run is a trial and is counted).

Audits make good posts. Keep them factual: what the code does, line by line, and what the honest number
is. Don't speculate about the author's intent.
