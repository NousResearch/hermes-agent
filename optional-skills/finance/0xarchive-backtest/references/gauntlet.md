# The gauntlet, the study and the verdict

Use `terminal` for the commands below, with the project Python environment active and the
study directory as the working directory. Use `read_file` for files; examples are not permission
to execute unreviewed code or open a sealed holdout.

## Study commands

```bash
0xbt study init [--holdout 2026-07-01] [--data data/hyperliquid]   # once per question
0xbt study status                                                    # trials, checks, holdout state
0xbt study commit FILE -p name=value [--data DIR]                    # opens the holdout, once
```

- A study is `.0xarchive/` (`study.json` and `trials.jsonl`). It is found like git finds `.git`: the
  nearest one at or above the current directory, or `$OXS_STUDY`. Run everything from inside it.
- **Trial:** a distinct result hash among the runs you asked for (`0xbt backtest`, `0xbt sweep`, the
  gauntlet's grid, `oxb.backtest` and `oxb.sweep` in Python). The same strategy over another window is
  another trial, because choosing the window is a try too.
- **Check:** a run a check makes for itself (nulls, the benchmark, the re-selections inside split and
  walk-forward). Recorded, not counted.
- **Sealed:** any engine read reaching past the holdout's start raises `Sealed` (backtests, sweeps,
  universe picks, `Session.warm_up`, `Source.rows`). Reading the Parquet some other way gets past it,
  which rule 2 of the skill forbids.
- **Commit** runs one file and params from the holdout's start to the end of the data, records the
  result, and opens the study. Every run after that is marked `after_reveal` and the card says the
  holdout is spent. It works once per study.

## The gauntlet

```bash
0xbt gauntlet FILE --start 2025-03-22 --end 2026-07-01 -g lookback=3d,7d -g quantile=0.1,0.2 --card card.md --json gauntlet.json
```

Options: `--at DATE` (the split; default two thirds in), `--seeds N` (random and stale nulls each,
default 10), `--train 180d --test 60d` (walk-forward), `--benchmark BTC` (or `equal`), plus backtest's
`--data`, `-p`, `-c`, `--max-leverage`. In Python: `oxb.gauntlet(make, grid, start, end, source, config)`.

It runs, in order:

1. **The grid over the whole window**, as trials. The best Sharpe is the candidate. With no `-g`, the
   strategy as written is the only point.
2. **Costs:** the candidate's stats with fees and funding in, against the benchmark
   (alpha, beta, information ratio).
3. **Nulls** (`oxb.null_test`): the flipped signal; random signals (what the universe, weighting and
   costs earn alone); stale signals (its own scores at random earlier times: the scores without their
   timing). A strategy written as one `step` has no signal, so its nulls are skipped with a note.
4. **Out of sample:** split at `--at` (pick on the first part, run on the rest), then walk-forward when
   the window holds a train window and two test windows.
5. **Deflated Sharpe** (Bailey and Lopez de Prado): the probability the Sharpe beats the best that luck
   would show over every trial in the study's ledger. Without a study it counts only this grid, and the
   card says so.
6. **Robustness:** the share of grid points with a positive Sharpe, the weakest neighbour of the best
   point as a ratio of its Sharpe, and the top market's share of gains.

Expect about `grid + 2 * seeds + 1 + grid * (folds + 1) + folds + 2` backtests: a 4-point grid over 16
months is about 60, roughly 10 minutes on a laptop with the 62-market dataset. Run it in the background.

## The ladder

The verdict is the first rung whose rule fails. The rules are code (`oxb.LADDER`), not judgment.

| Level | Rung | Reached when |
|---|---|---|
| 1 | `loses after costs` | the candidate's Sharpe is 0 or below, or none |
| 2 | `fails nulls` | flipped does as well, or more than 5% of random or of stale runs match or beat it |
| 3 | `in-sample only` | the split's out-of-sample Sharpe, or walk-forward's, is 0 or below |
| 4 | `holds out of sample` | the deflated Sharpe is below 0.95 |
| 5 | `passes deflated Sharpe` | no passed holdout for this candidate yet |
| 6 | `passes sealed holdout` | the committed candidate is this file and params, with a holdout Sharpe above 0 |

## Notes the card can carry, and what to say about each

- **under a year in sample:** every test is weak; say so before the verdict.
- **a peak, not a plateau:** a neighbouring setting has under half the best Sharpe. The best point is
  probably fitted to noise even if it passes.
- **one market:** without its top market the strategy made nothing. It is a bet on that market.
- **no study:** the deflated Sharpe ignores earlier tries. Start a study and rerun before reporting.
- **nulls not run:** a `step` strategy. Say the rung was reached without nulls.
- **the holdout is open:** later results were not tested on unseen data.

## Reading the card for the user

Lead with the rung and the trial count. Then the one reason that decided it, then what would move it
up a rung (a longer window, fewer tries, a broader universe), stated as a test to run, not a promise.
Quote numbers from the card, with the result hash. Never recompute them by hand.
