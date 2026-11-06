# Reverse engineer a trader

Use `terminal` for the commands below, with the project Python environment active and the
study directory as the working directory. Use `read_file` for files; examples are not permission
to execute unreviewed code or open a sealed holdout.

The question has three parts, answered in this order:

1. **What is their record, honestly?** Their own book through our engine, with our fees and funding.
2. **Is it skill or luck?** Their book through the gauntlet: does it beat its nulls?
3. **What rules produce it?** A strategy written from their style, scored by how much of their book it
   reproduces, then put through the gauntlet itself.

The helper is `scripts/trader.py` beside this skill; the mirror strategy is `templates/mirror.py`.
Copy both into the study directory so the ledger records the copies you ran.

## 1. Pull the book

```bash
python trader.py fetch 0xADDRESS --start 2025-06-01 --end 2026-07-01 --data data/hyperliquid
python trader.py profile trader.json
```

- `fetch` reads `GET /v1/hyperliquid/wallets/{address}/positions/changes`: one row per fill that changed
  a position, signed (shorts negative), from 2025-05-25 on Hyperliquid core. `--venue hip3` and
  `--venue lighter` (account index) read the same route on those venues, but the backtest dataset is
  Hyperliquid core only, so the mirror covers core markets alone.
- It bills like trades, one credit per 1,000 rows. A busy wallet over a year can be hundreds of thousands
  of rows: try one month first and tell the user the row count before pulling a year.
- The Free plan reads only the last 30 days. A `historical_depth_exceeded` error means the window needs
  a paid plan; say so rather than shortening the window silently.
- Wallet data is public venue data, but it is still about a person. Analyse the address the user gave;
  don't go looking for who owns it, and don't build lists of other people's wallets.

`profile` gives the style from the fills alone: markets, round trips, median and 90th percentile holding
time, long share of held time, largest gross position, maker share, when it opens (UTC hour, weekday),
realized PnL and fees. `not_in_dataset` lists markets the backtest can't trade; report what share of the
book they were.

## Reconstruction limits

The helper reconstructs changes in timestamp order and estimates `max_gross_usd` using each
market's most recently observed fill price. It is not an account-equity history or a continuously
marked exposure series. The profile measures held time between observed changes, omits time after
the last change, and cannot reconstruct positions already open before the window. Do not call its
long/short share a complete-window exposure measure. A book without a positive observed position
value cannot be mirrored; the helper reports that condition rather than dividing by zero.

Keep wallet fetches, profiles and feature fitting within the unsealed training window. Wallet JSON
is not protected by the engine's Parquet seal; fetching later fills is not permission to inspect
holdout behavior. Other-venue symbols matching core-market names do not prove equivalent instruments:
report unsupported coverage rather than treating a HIP-3 or Lighter history as a core-perp backtest.

## 2. Their record, and skill or luck

```bash
0xbt study init --holdout 2026-07-01
0xbt backtest mirror.py -p path=trader.json --start 2025-06-01 --end 2026-07-01
0xbt gauntlet mirror.py -p path=trader.json --start 2025-06-01 --end 2026-07-01 --card trader-card.md
```

The mirror holds their signed position value in each market, scaled by their largest gross position so
the book never passes 1x, and samples it each hour (`-p clock=1m` follows within a minute and runs
slower). Its nulls read as questions about the trader:

- **flipped:** fading them. If fading does as well, the book has no direction to speak of.
- **random:** random positions in the same markets on the same clock: what the markets alone paid.
- **stale:** their own positions at random earlier times. Beating every stale run means their timing,
  not just their exposure, made the money. This is the "skill or luck" answer.

The mirror's PnL is not their PnL: it trades up to a clock tick late, at our fill model's prices, at a
fixed scale. Compare it with `realized_pnl` from the profile and say how far apart they are. A big gap
usually means fast trading the hourly mirror can't follow; rerun with `clock=1m`.

## 3. Find the rules

Look at where their entries sit before guessing rules. For each opening fill in `trader.json`, read the
minutes around it with `oxb.Source(DIR).rows(market, t - 6h, t + 1h, columns=[...])` and tabulate what
was true at entry against all other hours: recent return, funding rate, open interest change,
taker flow (`cvd_delta`), and the hour of day. Check `data.md` before selecting features: book,
depth and liquidation-level fields are unavailable in API-built minute files and cannot be filled
in or treated as zero. A feature that separates their entries from ordinary hours is a candidate
rule; one that doesn't is not, however plausible it sounds.

Write the rules as a strategy file, then score it:

```bash
python trader.py replicate trader.json candidate.py --start 2025-06-01 --end 2026-07-01 -p ...
```

It reports, over the market hours they held a position, the share where the candidate was on the same
side, the other side, or flat; the market hours the candidate held and they didn't; the correlation of
the two books' daily returns; and the share of their profitable markets where the candidate also made
money. The books run at different sizes, so compare returns, not PnL. Both runs are trials in the ledger.

As a guide: the strategy that made a book scores about 1.0 same side and a high return correlation
against it; an unrelated strategy scores near 0 same side and mostly flat.

Each revision of the rules is a try. Keep them few and say how many there were. A candidate that copies
their book closely is a description of what they did, not yet a strategy: put it through the gauntlet
like any other, because rules fitted to one trader's year are fitted to that year.

## What to report

1. Their record in the engine (Sharpe, return, drawdown, against BTC), and how it compares with their
   realized PnL.
2. Skill or luck: the mirror's rung, and the stale-null result in a sentence.
3. Their style in three or four facts from the profile.
4. The rules found, with the replication numbers (same side, return correlation) and the candidate's own rung.
5. What the data couldn't show: markets outside the dataset, a hold time shorter than the clock, rules
   that read data we don't have (their other venues, their spot book, off-exchange hedges).
