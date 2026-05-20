#!/usr/bin/env python3
"""Permutation test for donation/contract timing correlation (stdlib-only).

For each (donor, vendor) pair, compute the mean number of days between each
donation and the nearest contract award. Then shuffle contract award dates
N times within the observation window and compute the same statistic. The
one-tailed p-value is the fraction of permutations whose mean is <= the
observed mean (smaller distance = tighter clustering).

Adapted from ShinMegamiBoson/OpenPlanter (MIT). Differences:
  - Pure stdlib (no pandas / numpy)
  - Domain-agnostic (no snow-vendor / CRITICAL-politician filter)
  - Configurable column names via flags
  - Optional --seed for reproducibility
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%m-%d-%Y", "%Y%m%d")


def parse_date(raw: str) -> dt.date | None:
    if not raw:
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _read(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _nearest_distance(donation_date: dt.date, awards: list[dt.date]) -> int:
    """Absolute days to nearest award date."""
    return min(abs((donation_date - a).days) for a in awards)


def _permute(
    awards_count: int,
    donations: list[dt.date],
    date_min: dt.date,
    date_max: dt.date,
    rng: random.Random,
) -> float:
    """One permutation: draw uniform random award dates, compute mean nearest-distance."""
    span_days = (date_max - date_min).days or 1
    rand_awards = [
        date_min + dt.timedelta(days=rng.randint(0, span_days))
        for _ in range(awards_count)
    ]
    distances = [_nearest_distance(d, rand_awards) for d in donations]
    return statistics.mean(distances)


def analyze(
    donations_path: str,
    donation_date_col: str,
    donation_amount_col: str,
    donation_donor_col: str,
    donation_recipient_col: str,
    contracts_path: str,
    contract_date_col: str,
    contract_vendor_col: str,
    cross_links_path: str | None,
    n_permutations: int = 1000,
    min_donations: int = 3,
    p_threshold: float = 0.05,
    seed: int | None = None,
    out_path: str = "timing.json",
) -> dict:
    rng = random.Random(seed)

    donations = _read(donations_path)
    contracts = _read(contracts_path)

    # Allow optional join through cross_links — donor (left) ↔ vendor (right).
    # When present, donor strings get mapped to matched vendor names so the
    # vendor-date index lookup actually finds the contracts.
    matched_pairs: set[tuple[str, str]] | None = None
    donor_to_vendors: dict[str, set[str]] = defaultdict(set)
    if cross_links_path:
        matched_pairs = set()
        for row in _read(cross_links_path):
            left = row.get("left_name", "")
            right = row.get("right_name", "")
            matched_pairs.add((left, right))
            donor_to_vendors[left].add(right)

    # Index contract dates by vendor name.
    vendor_to_award_dates: dict[str, list[dt.date]] = defaultdict(list)
    all_award_dates: list[dt.date] = []
    for row in contracts:
        d = parse_date(row.get(contract_date_col, ""))
        if not d:
            continue
        vendor_to_award_dates[row.get(contract_vendor_col, "").strip()].append(d)
        all_award_dates.append(d)

    if not all_award_dates:
        raise SystemExit(f"No parseable dates in {contracts_path}/{contract_date_col}")
    global_min = min(all_award_dates)
    global_max = max(all_award_dates)

    # Group donations by (donor, recipient).
    grouped: dict[tuple[str, str], list[tuple[dt.date, float]]] = defaultdict(list)
    for row in donations:
        donor = row.get(donation_donor_col, "").strip()
        recip = row.get(donation_recipient_col, "").strip()
        d = parse_date(row.get(donation_date_col, ""))
        try:
            amt = float(row.get(donation_amount_col, "0") or 0)
        except ValueError:
            amt = 0.0
        if not (donor and recip and d):
            continue
        grouped[(donor, recip)].append((d, amt))

    results = []
    skipped = 0
    for (donor, recip), records in grouped.items():
        if len(records) < min_donations:
            skipped += 1
            continue
        # Only test if donor appears in cross-links (when provided). The
        # (donor, candidate) tuple itself is NOT what's in matched_pairs —
        # cross_links pairs are (donor, vendor). We use the cross-link to
        # map donor → vendor name(s) so the vendor-date index resolves.
        if matched_pairs is not None and donor not in donor_to_vendors:
            skipped += 1
            continue
        # Try direct donor→awards first, then go through cross-link vendor names.
        award_dates = list(vendor_to_award_dates.get(donor, []))
        if not award_dates:
            award_dates = list(vendor_to_award_dates.get(recip, []))
        if not award_dates and donor_to_vendors.get(donor):
            for vendor_name in donor_to_vendors[donor]:
                award_dates.extend(vendor_to_award_dates.get(vendor_name, []))
        if not award_dates:
            skipped += 1
            continue

        donation_dates = [d for (d, _) in records]
        observed = statistics.mean(
            _nearest_distance(d, award_dates) for d in donation_dates
        )

        permuted_means = [
            _permute(len(award_dates), donation_dates, global_min, global_max, rng)
            for _ in range(n_permutations)
        ]
        p_value = sum(1 for m in permuted_means if m <= observed) / n_permutations
        null_mean = statistics.mean(permuted_means)
        null_std = statistics.pstdev(permuted_means) or 1.0
        effect_size = (null_mean - observed) / null_std

        results.append(
            {
                "donor": donor,
                "recipient": recip,
                "n_donations": len(records),
                "n_award_dates": len(award_dates),
                "observed_mean_days": round(observed, 2),
                "null_mean_days": round(null_mean, 2),
                "p_value": round(p_value, 4),
                "effect_size_sd": round(effect_size, 2),
                "significant": p_value < p_threshold,
                "total_donation_amount": round(sum(a for (_, a) in records), 2),
            }
        )

    results.sort(key=lambda r: r["p_value"])

    payload = {
        "metadata": {
            "n_permutations": n_permutations,
            "min_donations": min_donations,
            "p_threshold": p_threshold,
            "seed": seed,
            "n_pairs_tested": len(results),
            "n_pairs_skipped": skipped,
            "n_significant": sum(1 for r in results if r["significant"]),
            "observation_window": [global_min.isoformat(), global_max.isoformat()],
        },
        "results": results,
    }

    Path(out_path).write_text(json.dumps(payload, indent=2))
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--donations", required=True)
    p.add_argument("--donation-date-col", required=True)
    p.add_argument("--donation-amount-col", required=True)
    p.add_argument("--donation-donor-col", required=True)
    p.add_argument("--donation-recipient-col", required=True)
    p.add_argument("--contracts", required=True)
    p.add_argument("--contract-date-col", required=True)
    p.add_argument("--contract-vendor-col", required=True)
    p.add_argument(
        "--cross-links",
        help="Optional cross_links.csv to restrict (donor, vendor) pairs",
    )
    p.add_argument("--permutations", type=int, default=1000)
    p.add_argument("--min-donations", type=int, default=3)
    p.add_argument("--p-threshold", type=float, default=0.05)
    p.add_argument("--seed", type=int)
    p.add_argument("--out", default="timing.json")
    a = p.parse_args()

    payload = analyze(
        donations_path=a.donations,
        donation_date_col=a.donation_date_col,
        donation_amount_col=a.donation_amount_col,
        donation_donor_col=a.donation_donor_col,
        donation_recipient_col=a.donation_recipient_col,
        contracts_path=a.contracts,
        contract_date_col=a.contract_date_col,
        contract_vendor_col=a.contract_vendor_col,
        cross_links_path=a.cross_links,
        n_permutations=a.permutations,
        min_donations=a.min_donations,
        p_threshold=a.p_threshold,
        seed=a.seed,
        out_path=a.out,
    )
    meta = payload["metadata"]
    print(
        f"Tested {meta['n_pairs_tested']} pairs ({meta['n_pairs_skipped']} skipped). "
        f"Significant (p<{meta['p_threshold']}): {meta['n_significant']}. "
        f"Wrote {a.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
