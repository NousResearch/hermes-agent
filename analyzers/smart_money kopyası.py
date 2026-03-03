"""
analyzers/smart_money.py
------------------------
Smart Money Tracker — single source of truth.

Evaluates whether a wallet exhibits "smart money" behaviour and returns
results using the **spec-canonical key names**:

    smart_entry     bool  – True if wallet meets smart-money criteria
    avg_roi         float – average ROI % across completed buy→sell round-trips
    active_months   int   – months between first and latest transaction

Smart money criteria (at least 2 of 3 must be satisfied)
---------------------------------------------------------
1. avg_roi >= SMART_ROI_THRESHOLD  (default 8 %)
2. active_months >= SMART_DURATION_MONTHS  (default 3 months)
3. avg_roi > 0  (softer signal used when data is sparse)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.helpers import ts_to_dt, average, months_between

# ── Thresholds ────────────────────────────────────────────────────────────────
SMART_ROI_THRESHOLD   = 8.0   # percent; avg ROI >= this → smart signal
SMART_DURATION_MONTHS = 3     # months; active for at least this long


class SmartMoneyTracker:
    """
    Evaluates smart money signals for a wallet.

    Parameters
    ----------
    wallet : str
        The Solana wallet address being analyzed (base-58, 32–44 chars).
    transactions : list[dict]
        Helius enhanced-transaction objects for the wallet.

    Usage
    -----
    >>> result = SmartMoneyTracker(wallet, txs).analyze()
    >>> result.keys()
    dict_keys(['wallet', 'smart_entry', 'avg_roi', 'active_months'])
    """

    def __init__(self, wallet: str, transactions: List[Dict]) -> None:
        self.wallet       = wallet
        self.transactions = transactions

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """
        Run smart money analysis and return a result dict.

        Returns
        -------
        dict with spec-canonical keys:
            wallet, smart_entry, avg_roi, active_months
        """
        roi_values    = self._compute_roi_per_nft()
        avg_roi       = round(average(roi_values), 1)
        active_months = self._activity_duration_months()
        smart_entry   = self._is_smart_wallet(avg_roi, active_months)

        return {
            "wallet":        self.wallet,
            "smart_entry":   smart_entry,
            "avg_roi":       avg_roi,
            "active_months": active_months,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _nft_sale_events(self) -> List[Dict]:
        """Extract NFT sale events (NFT_SALE / SALE) from the transaction list."""
        events: List[Dict] = []
        for tx in self.transactions:
            ts = tx.get("timestamp") or tx.get("blockTime")
            dt = ts_to_dt(ts)
            for ev in tx.get("events", {}).get("nft", []):
                ev_type = (ev.get("type") or "").upper()
                if ev_type not in ("NFT_SALE", "SALE"):
                    continue
                events.append({
                    "type":       ev_type,
                    "mint":       (
                        ev.get("mint")
                        or (ev.get("nfts") or [{}])[0].get("mint", "")
                    ),
                    "timestamp":  dt,
                    "buyer":      ev.get("buyer"),
                    "seller":     ev.get("seller"),
                    "amount_sol": (ev.get("amount") or 0) / 1e9,
                })
        return events

    def _compute_roi_per_nft(self) -> List[float]:
        """
        For each mint that was both *bought* and *sold* by this wallet at
        known prices, compute ROI = (sell − buy) / buy × 100.

        Only mints where both a buy price and a sell price are on record are
        included; free mints (price = 0) are excluded as buyers.
        """
        sales = self._nft_sale_events()

        buy_price:  Dict[str, float] = {}
        sell_price: Dict[str, float] = {}

        for ev in sales:
            mint = ev["mint"]
            if not mint:
                continue
            if ev["buyer"] == self.wallet and ev["amount_sol"] > 0:
                buy_price[mint] = ev["amount_sol"]
            elif ev["seller"] == self.wallet and ev["amount_sol"] > 0:
                sell_price[mint] = ev["amount_sol"]

        roi_list: List[float] = []
        for mint in sell_price:
            if mint in buy_price and buy_price[mint] > 0:
                roi = (sell_price[mint] - buy_price[mint]) / buy_price[mint] * 100.0
                roi_list.append(roi)

        return roi_list

    def _activity_duration_months(self) -> int:
        """
        Return the number of whole months between the wallet's earliest and
        most recent transactions (using a 30-day month approximation).
        """
        timestamps = [
            ts_to_dt(tx.get("timestamp") or tx.get("blockTime"))
            for tx in self.transactions
        ]
        dts = [dt for dt in timestamps if dt is not None]
        if len(dts) < 2:
            return 0
        return months_between(min(dts), max(dts))

    def _is_smart_wallet(self, avg_roi: float, active_months: int) -> bool:
        """
        Classify the wallet as smart money when it meets >= 2 of 3 criteria:
          1. avg_roi >= SMART_ROI_THRESHOLD
          2. active_months >= SMART_DURATION_MONTHS
          3. avg_roi > 0  (soft positive-ROI signal)
        """
        criteria = sum([
            avg_roi >= SMART_ROI_THRESHOLD,
            active_months >= SMART_DURATION_MONTHS,
            avg_roi > 0,
        ])
        return criteria >= 2


# ── Standalone example ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import datetime as _dt
    from datetime import timezone as _tz

    WALLET = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
    OTHER  = "BotWallet1111111111111111111111111111111111"
    BASE   = int(_dt.datetime(2024, 1, 1, tzinfo=_tz.utc).timestamp())

    sample_txs = []
    for i in range(30):
        mint = f"Mint{i:040d}"
        sample_txs.append({
            "signature": f"buy{i}",
            "timestamp": BASE + i * 86400,
            "type": "NFT_SALE",
            "events": {"nft": [{"type": "NFT_SALE", "mint": mint,
                                 "buyer": WALLET, "seller": OTHER,
                                 "amount": 1_000_000_000}]},
            "tokenTransfers": [],
        })
        sample_txs.append({
            "signature": f"sell{i}",
            "timestamp": BASE + i * 86400 + 86400 * 10,
            "type": "NFT_SALE",
            "events": {"nft": [{"type": "NFT_SALE", "mint": mint,
                                 "seller": WALLET, "buyer": OTHER,
                                 "amount": 1_200_000_000}]},
            "tokenTransfers": [],
        })

    result = SmartMoneyTracker(WALLET, sample_txs).analyze()
    print("\n── analyzers/smart_money.py standalone example ──")
    print(json.dumps(result, indent=2))

    expected = {"wallet", "smart_entry", "avg_roi", "active_months"}
    missing  = expected - result.keys()
    extra    = result.keys() - expected
    if not missing and not extra:
        print("\n  ✓ All spec keys present")
    else:
        if missing: print(f"  ✗ MISSING: {missing}")
        if extra:   print(f"  ! EXTRA:   {extra}")
