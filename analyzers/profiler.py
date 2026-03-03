"""
analyzers/profiler.py
---------------------
Wallet Profiler — single source of truth.

Builds a comprehensive behavioural fingerprint for an NFT wallet and returns
results using the **spec-canonical key names**:

    total_nfts              int   – NFTs minted by this wallet
    avg_flip_duration       float – mean hold-time in days before resale
    high_risk_collections   int   – collections with repeat fast-flip activity
    net_roi                 float – net ROI % across all completed buy→sell trips

High-risk collection heuristic
-------------------------------
A collection prefix (first 8 chars of the mint address) is flagged when the
wallet made >= HIGH_RISK_SALE_COUNT flips (hold time <= FLIP_DAYS) within it.

Detection thresholds (adjust here, nowhere else)
-------------------------------------------------
FLIP_DAYS            = 14   max hold-time (days) that counts as a flip
HIGH_RISK_SALE_COUNT = 2    min flips per collection prefix to flag as high-risk
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from utils.helpers import ts_to_dt, average

# ── Thresholds ────────────────────────────────────────────────────────────────
FLIP_DAYS            = 14
HIGH_RISK_SALE_COUNT = 2


class WalletProfiler:
    """
    Profiles an NFT wallet's overall activity.

    Parameters
    ----------
    wallet : str
        The Solana wallet address being profiled (base-58, 32–44 chars).
    transactions : list[dict]
        Helius enhanced-transaction objects for the wallet.
    owned_assets : list[dict], optional
        Current NFT assets from DAS ``getAssetsByOwner``.
        Pass ``[]`` or omit if the DAS call was skipped.

    Usage
    -----
    >>> result = WalletProfiler(wallet, txs, assets).analyze()
    >>> result.keys()
    dict_keys(['wallet', 'total_nfts', 'avg_flip_duration',
               'high_risk_collections', 'net_roi'])
    """

    def __init__(
        self,
        wallet: str,
        transactions: List[Dict],
        owned_assets: Optional[List[Dict]] = None,
    ) -> None:
        self.wallet       = wallet
        self.transactions = transactions
        self.owned_assets = owned_assets or []

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """
        Build a wallet profile and return a result dict.

        Returns
        -------
        dict with spec-canonical keys:
            wallet, total_nfts, avg_flip_duration,
            high_risk_collections, net_roi
        """
        total_nfts           = self._count_mints()
        flip_durations       = self._compute_flip_durations()
        avg_flip_duration    = round(average(flip_durations), 1) if flip_durations else 0
        high_risk_collections = self._count_high_risk_collections(flip_durations)
        net_roi              = round(self._compute_net_roi(), 1)

        return {
            "wallet":               self.wallet,
            "total_nfts":           total_nfts,
            "avg_flip_duration":    avg_flip_duration,
            "high_risk_collections": high_risk_collections,
            "net_roi":              net_roi,
        }

    # ── Event extraction ──────────────────────────────────────────────────────

    def _all_nft_events(self) -> List[Dict]:
        """Flatten all NFT-related events from the transaction list."""
        events: List[Dict] = []
        for tx in self.transactions:
            ts  = tx.get("timestamp") or tx.get("blockTime")
            dt  = ts_to_dt(ts)
            sig = tx.get("signature", "")

            for ev in tx.get("events", {}).get("nft", []):
                ev_type = (ev.get("type") or "").upper()
                mint    = (
                    ev.get("mint")
                    or (ev.get("nfts") or [{}])[0].get("mint", "")
                )
                events.append({
                    "type":       ev_type,
                    "mint":       mint,
                    "timestamp":  dt,
                    "signature":  sig,
                    "buyer":      ev.get("buyer"),
                    "seller":     ev.get("seller"),
                    "amount_sol": (ev.get("amount") or 0) / 1e9,
                })

        return events

    # ── Calculations ──────────────────────────────────────────────────────────

    def _count_mints(self) -> int:
        """
        Count distinct NFTs minted by this wallet.

        Uses two signals:
        1. MINT / NFT_MINT events in the transaction history.
        2. Owned assets where the wallet is a verified creator.
        """
        minted: set = set()

        for ev in self._all_nft_events():
            if ev["type"] in ("MINT", "NFT_MINT") and ev["mint"]:
                minted.add(ev["mint"])

        for asset in self.owned_assets:
            for creator in asset.get("creators") or []:
                if creator.get("address") == self.wallet and creator.get("verified"):
                    mint_id = asset.get("id", "")
                    if mint_id:
                        minted.add(mint_id)

        return len(minted)

    def _acquire_sell_maps(self) -> tuple:
        """
        Return (acquire_dt, sell_dt, sell_price, buy_price) maps keyed by mint.

        Used by both _compute_flip_durations and _compute_net_roi to avoid
        iterating over events twice.
        """
        events = self._all_nft_events()

        acquire_dt: Dict[str, datetime] = {}
        sell_dt:    Dict[str, datetime] = {}
        buy_price:  Dict[str, float]    = {}
        sell_price: Dict[str, float]    = {}

        for ev in events:
            mint = ev["mint"]
            if not mint:
                continue

            if ev.get("buyer") == self.wallet:
                if ev["timestamp"]:
                    acquire_dt[mint] = ev["timestamp"]
                if ev["amount_sol"] > 0:
                    buy_price[mint] = ev["amount_sol"]

            if ev.get("seller") == self.wallet:
                if ev["timestamp"]:
                    sell_dt[mint] = ev["timestamp"]
                if ev["amount_sol"] > 0:
                    sell_price[mint] = ev["amount_sol"]

        return acquire_dt, sell_dt, buy_price, sell_price

    def _compute_flip_durations(self) -> List[float]:
        """
        For each NFT acquired then sold by this wallet, return the hold time
        in fractional days (sell_dt - acquire_dt).
        """
        acquire_dt, sell_dt, _, _ = self._acquire_sell_maps()

        durations: List[float] = []
        for mint in sell_dt:
            if mint in acquire_dt and sell_dt[mint] >= acquire_dt[mint]:
                hold = (sell_dt[mint] - acquire_dt[mint]).total_seconds() / 86400.0
                durations.append(hold)

        return durations

    def _flip_mint_addresses(self) -> List[str]:
        """
        Return mint addresses in the same order as ``_compute_flip_durations()``.
        Used by ``_count_high_risk_collections`` to pair mints with durations.
        """
        acquire_dt, sell_dt, _, _ = self._acquire_sell_maps()

        return [
            mint
            for mint in sell_dt
            if mint in acquire_dt and sell_dt[mint] >= acquire_dt[mint]
        ]

    def _count_high_risk_collections(
        self, flip_durations: Optional[List[float]] = None
    ) -> int:
        """
        Count collection prefixes where the wallet made >= HIGH_RISK_SALE_COUNT
        fast flips (hold time <= FLIP_DAYS).
        """
        if flip_durations is None:
            flip_durations = self._compute_flip_durations()

        mint_addresses = self._flip_mint_addresses()
        by_collection: Dict[str, List[float]] = defaultdict(list)

        for mint, hold in zip(mint_addresses, flip_durations):
            if hold <= FLIP_DAYS:
                by_collection[mint[:8]].append(hold)

        return sum(
            1 for flips in by_collection.values()
            if len(flips) >= HIGH_RISK_SALE_COUNT
        )

    def _compute_net_roi(self) -> float:
        """
        Compute net ROI as (total_revenue - total_cost) / total_cost × 100.

        Only mints where both a buy price and a sell price are known are
        included; free mints (price = 0) on the buy side are excluded.
        """
        _, _, buy_price, sell_price = self._acquire_sell_maps()

        total_cost    = 0.0
        total_revenue = 0.0
        for mint in sell_price:
            if mint in buy_price and buy_price[mint] > 0:
                total_cost    += buy_price[mint]
                total_revenue += sell_price[mint]

        if total_cost == 0:
            return 0.0
        return (total_revenue - total_cost) / total_cost * 100.0


# ── Standalone example ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import datetime as _dt
    from datetime import timezone as _tz

    WALLET = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
    OTHER  = "BuyerWallet22222222222222222222222222222222"
    BASE   = int(_dt.datetime(2024, 3, 1, tzinfo=_tz.utc).timestamp())

    sample_txs   = []
    owned_assets = []

    for i in range(42):
        mint      = f"MintAddr{i:036d}"
        ts_mint   = BASE + i * 7200
        hold_days = 3 if i % 5 == 0 else 30   # every 5th = fast flip
        ts_sell   = ts_mint + hold_days * 86400

        sample_txs.append({
            "signature": f"mintSig{i}", "timestamp": ts_mint, "type": "MINT",
            "events": {"nft": [{"type": "MINT", "mint": mint,
                                 "buyer": WALLET, "seller": None, "amount": 0}]},
            "tokenTransfers": [],
        })
        sample_txs.append({
            "signature": f"saleSig{i}", "timestamp": ts_sell, "type": "NFT_SALE",
            "events": {"nft": [{"type": "NFT_SALE", "mint": mint,
                                 "seller": WALLET, "buyer": OTHER,
                                 "amount": int(1_000_000_000 * 1.18)}]},
            "tokenTransfers": [],
        })
        owned_assets.append({
            "id": mint,
            "creators": [{"address": WALLET, "verified": True}],
        })

    result = WalletProfiler(WALLET, sample_txs, owned_assets).analyze()
    print("\n── analyzers/profiler.py standalone example ──")
    print(json.dumps(result, indent=2))

    expected = {"wallet", "total_nfts", "avg_flip_duration",
                "high_risk_collections", "net_roi"}
    missing  = expected - result.keys()
    extra    = result.keys() - expected
    if not missing and not extra:
        print("\n  ✓ All spec keys present")
    else:
        if missing: print(f"  ✗ MISSING: {missing}")
        if extra:   print(f"  ! EXTRA:   {extra}")
