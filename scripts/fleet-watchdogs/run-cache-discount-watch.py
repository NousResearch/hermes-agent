#!/usr/bin/env python3
"""Cache-DISCOUNT watchdog (companion to cache-hit-watch).

Why this exists
---------------
cache-hit-watch answers "are tokens being cached?" via the OpenRouter API.
This script answers the question that actually bills: "did the cache save
money, per (model, upstream provider), on our real recorded traffic?"

It reads the session_model_usage rows that the 2026-09-01 provider-threading
change persists on every call: provider_name (the ACTUAL upstream host, not
'openrouter'), native_tokens_prompt / native_tokens_cached, cache_discount
(USD saved) and total_cost (USD billed). Those fields are why the DeepInfra
11x overspend was invisible: every row used to say only 'openrouter'.

`no_agent` cron script: no LLM call, no tokens, $0.
Silent unless something is wrong (empty stdout = silent tick).
Recommendation-only by policy: it never edits config.
"""

import os
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/.hermes/scripts"))
from comms_style import clean  # noqa: E402

HERMES_HOME = Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")

WINDOW_HOURS = 24
MIN_CALLS = 8              # don't judge a provider on a handful of cold starts
MIN_PROMPT_TOKENS = 200_000
CACHED_SHARE_FLOOR = 0.40  # "the provider claims it cached a lot"
SAVING_RATE_FLOOR = 0.05   # ...but discounted less than 5% of gross -> Parasail trap
SPREAD_RATIO = 3.0         # same model, one host >= 3x another's cost per prompt token
UNATTRIBUTED_SHARE = 0.50  # over half of fresh traffic missing provider_name


def state_dbs():
    yield HERMES_HOME / "state.db"
    profiles = HERMES_HOME / "profiles"
    if profiles.is_dir():
        for p in sorted(profiles.iterdir()):
            db = p / "state.db"
            if db.is_file():
                yield db


def read_rows(db_path, since):
    """Rows from one state.db; silently skip DBs that predate the migration."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        cols = {r[1] for r in conn.execute("PRAGMA table_info(session_model_usage)")}
        if "provider_name" not in cols:
            conn.close()
            return []
        rows = list(conn.execute(
            "SELECT model, provider_name, api_call_count, native_tokens_prompt,"
            " native_tokens_cached, cache_discount, total_cost"
            " FROM session_model_usage WHERE last_seen >= ?", (since,)))
        conn.close()
        return rows
    except sqlite3.Error:
        return []


def main():
    since = time.time() - WINDOW_HOURS * 3600
    agg = {}   # (model, provider_name) -> totals
    unattributed_calls = 0
    total_calls = 0
    for db in state_dbs():
        for r in read_rows(db, since):
            calls = r["api_call_count"] or 0
            total_calls += calls
            name = (r["provider_name"] or "").strip()
            if not name:
                unattributed_calls += calls
                continue
            key = (r["model"], name)
            a = agg.setdefault(key, {"calls": 0, "prompt": 0, "cached": 0,
                                     "discount": 0.0, "billed": 0.0})
            a["calls"] += calls
            a["prompt"] += r["native_tokens_prompt"] or 0
            a["cached"] += r["native_tokens_cached"] or 0
            a["discount"] += r["cache_discount"] or 0.0
            a["billed"] += r["total_cost"] or 0.0

    problems = []

    # 1. Threading regression: recorded traffic with no upstream attribution.
    if total_calls >= MIN_CALLS and total_calls and \
            unattributed_calls / total_calls > UNATTRIBUTED_SHARE:
        problems.append(
            "Provider attribution is missing on %d of %d calls in the last %dh.\n"
            "The provider_name threading may have regressed, or the gateway is "
            "running pre-migration code." % (unattributed_calls, total_calls, WINDOW_HOURS))

    # 2. Cache HIT without cache SAVING (the Parasail trap).
    for (model, prov), a in sorted(agg.items()):
        if a["calls"] < MIN_CALLS or a["prompt"] < MIN_PROMPT_TOKENS:
            continue
        cached_share = a["cached"] / a["prompt"] if a["prompt"] else 0.0
        gross = a["discount"] + a["billed"]
        saving_rate = a["discount"] / gross if gross > 0 else 0.0
        if cached_share >= CACHED_SHARE_FLOOR and saving_rate < SAVING_RATE_FLOOR:
            problems.append(
                "%s on %s: %.0f%% of prompt tokens marked cached but only %.1f%% "
                "of gross cost was discounted (%d calls, billed $%.4f, saved $%.4f).\n"
                "A cache hit is not a cache saving - this host is charging full rate."
                % (model, prov, cached_share * 100, saving_rate * 100,
                   a["calls"], a["billed"], a["discount"]))

    # 3. Same model, wildly different billed cost per prompt token across hosts.
    by_model = {}
    for (model, prov), a in agg.items():
        if a["calls"] >= MIN_CALLS and a["prompt"] > 0 and a["billed"] > 0:
            by_model.setdefault(model, []).append((prov, a["billed"] / a["prompt"], a))
    for model, entries in sorted(by_model.items()):
        if len(entries) < 2:
            continue
        entries.sort(key=lambda e: e[1])
        cheap, dear = entries[0], entries[-1]
        if cheap[1] > 0 and dear[1] / cheap[1] >= SPREAD_RATIO:
            problems.append(
                "%s: %s bills %.1fx more per prompt token than %s over the same "
                "window (%s $%.4f for %dk tokens vs %s $%.4f for %dk).\n"
                "Recommendation only - verify with a billed probe before repinning."
                % (model, dear[0], dear[1] / cheap[1], cheap[0],
                   dear[0], dear[2]["billed"], dear[2]["prompt"] // 1000,
                   cheap[0], cheap[2]["billed"], cheap[2]["prompt"] // 1000))

    if problems:
        body = "Cache discount watch (last %dh of recorded billed traffic)\n\n" % WINDOW_HOURS
        body += "\n\n".join(problems)
        print(clean(body))


if __name__ == "__main__":
    main()
