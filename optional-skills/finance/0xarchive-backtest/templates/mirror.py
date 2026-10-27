# A trader's own book as a strategy: the positions `trader.py fetch` pulled for one wallet,
# held through the engine's fills, fees and funding.
#
#     0xbt backtest mirror.py -p path=trader.json --start 2025-06-01 --end 2026-07-01
#     0xbt gauntlet mirror.py -p path=trader.json --start ... --end ...
#
# The signal at each tick is the trader's signed position value in each market, divided by
# their largest gross position value over the file, so the book's gross never passes 1.
# That makes the gauntlet's nulls mean something for a trader:
#   flipped  - fading them;
#   random   - random positions in the same markets on the same clock (the machinery);
#   stale    - their own positions at random earlier times: their book without its timing.
# A trader whose record beats every stale run has timing, not just exposure.
#
# The book is sampled at each clock tick and traded from the next minute, so with the
# default hourly clock it trades up to an hour after them. clock="1m" follows closely and
# runs slower. A weight is traded only when it moves more than `band` (a fraction of
# equity) from what is held, so price drift alone doesn't trade. Markets the dataset lacks
# are left out; trader.py profile lists them.
import bisect
import json

import oxarchive_backtest as oxb
from oxarchive_backtest import Series, x


def strategy(path="trader.json", clock="1h", band=0.01):
    with open(path, encoding="utf-8") as f:
        book = json.load(f)
    paths = {}
    for c in book["changes"]:
        ts, sizes = paths.setdefault(c["symbol"], ([], []))
        ts.append(c["ts"])
        sizes.append(c["end_position"])
    scale = book["max_gross_usd"]
    if not scale or scale < 0:
        raise ValueError("the book has no positive observed position value to mirror")
    markets = sorted(m for m in paths if m in set(book.get("in_dataset", paths)))

    def held(market, t):
        ts, sizes = paths[market]
        i = bisect.bisect_right(ts, t) - 1
        return sizes[i] if i >= 0 else 0.0

    def signal(h):
        px = h.close.last()
        t = h.time
        out = {}
        for m in h.markets:
            p = px[m]
            out[m] = None if t is None or p is None else held(m, t) * p / scale
        return Series(out)

    return oxb.Strategy(
        universe=markets,
        clock=clock,
        signal=signal,
        weights=lambda s: s,
        # Random nulls draw each market's weight from [-1, 1); keep their book at 1x too.
        risk=[x.cap(1.0, gross=1.0)],
        execute=x.rebalance(band=band),
    )
