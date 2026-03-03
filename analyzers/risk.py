"""
analyzers/risk.py
-----------------
Wallet Risk Analyzer — single source of truth.

Detects high-risk behavioral patterns in a Solana NFT wallet and returns
results using the **spec-canonical key names** used throughout the skill:

    past_tx_count         int   – total transactions fetched for the wallet
    risk_score            str   – "LOW" | "MEDIUM" | "HIGH"
    mint_dump_pattern     bool  – minted then sold within MINT_DUMP_DAYS
    wash_trading_signals  int   – count of A→B→A transfer loops
    fast_flips            bool  – rapid resales within the same collection

Detection thresholds (adjust here, nowhere else)
-------------------------------------------------
MINT_DUMP_DAYS   = 7    days after mint before a sale counts as a dump
FAST_FLIP_DAYS   = 3    max hold-time (days) to flag as a fast flip
WASH_WINDOW_DAYS = 7    window (days) to detect A→B→A wash-trade loops
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from utils.helpers import ts_to_dt, average   # noqa: F401 (average unused here but imported for consistency)

# ── Detection thresholds ──────────────────────────────────────────────────────
MINT_DUMP_DAYS   = 7
FAST_FLIP_DAYS   = 3
WASH_WINDOW_DAYS = 7


class WalletRiskAnalyzer:
    """
    Analyzes wallet transaction history and returns a risk profile.

    Parameters
    ----------
    wallet : str
        The Solana wallet address being analyzed (base-58, 32–44 chars).
    transactions : list[dict]
        Helius enhanced-transaction objects for the wallet.

    Usage
    -----
    >>> result = WalletRiskAnalyzer(wallet, txs).analyze()
    >>> result.keys()
    dict_keys(['wallet', 'past_tx_count', 'risk_score',
               'mint_dump_pattern', 'wash_trading_signals', 'fast_flips'])
    """

    def __init__(self, wallet: str, transactions: List[Dict]) -> None:
        self.wallet       = wallet
        self.transactions = transactions

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self) -> Dict[str, Any]:
        """
        Run all risk checks and return a structured result dict.

        Returns
        -------
        dict with spec-canonical keys:
            wallet, past_tx_count, risk_score,
            mint_dump_pattern, wash_trading_signals, fast_flips
        """
        past_tx_count        = len(self.transactions)
        mint_dump_pattern    = self._detect_mint_dump()
        wash_trading_signals = self._detect_wash_trading()
        fast_flips           = self._detect_fast_flips()
        risk_score           = self._compute_risk_score(
            past_tx_count, mint_dump_pattern,
            wash_trading_signals, fast_flips,
        )

        return {
            "wallet":               self.wallet,
            "past_tx_count":        past_tx_count,
            "risk_score":           risk_score,
            "mint_dump_pattern":    mint_dump_pattern,
            "wash_trading_signals": wash_trading_signals,
            "fast_flips":           fast_flips,
        }

    # ── Event extraction ──────────────────────────────────────────────────────

    def _nft_events(self) -> List[Dict]:
        """
        Flatten all NFT-related events from the transaction list.

        Reads both the ``events.nft`` array (Helius enhanced TX format) and
        ``tokenTransfers`` as a fallback for plain SPL transfers.
        """
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
                    "source":     ev.get("source", ""),
                })

            # tokenTransfers fallback
            for tt in tx.get("tokenTransfers", []):
                mint = tt.get("mint", "")
                if not mint:
                    continue
                events.append({
                    "type":       "TRANSFER",
                    "mint":       mint,
                    "timestamp":  dt,
                    "signature":  sig,
                    "buyer":      tt.get("toUserAccount"),
                    "seller":     tt.get("fromUserAccount"),
                    "amount_sol": 0.0,
                    "source":     "TOKEN_TRANSFER",
                })

        return events

    # ── Detection logic ───────────────────────────────────────────────────────

    def _detect_mint_dump(self) -> bool:
        """
        Return ``True`` if the wallet minted an NFT and sold / transferred it
        within ``MINT_DUMP_DAYS`` days.
        """
        events = self._nft_events()

        mint_times: Dict[str, datetime] = {}
        for ev in events:
            mint = ev["mint"]
            if mint and ev["type"] == "MINT" and ev["timestamp"]:
                mint_times.setdefault(mint, ev["timestamp"])

        for ev in events:
            mint = ev["mint"]
            if not mint or ev["type"] not in ("NFT_SALE", "SALE", "TRANSFER"):
                continue
            if mint in mint_times and ev["timestamp"]:
                delta = (ev["timestamp"] - mint_times[mint]).days
                if 0 <= delta <= MINT_DUMP_DAYS:
                    return True

        return False

    def _detect_wash_trading(self) -> int:
        """
        Count NFTs whose ownership returned to the original sender within
        ``WASH_WINDOW_DAYS`` (A→B→A pattern).

        Returns
        -------
        int
            Number of distinct wash-trade signals detected.
        """
        events = self._nft_events()

        by_mint: Dict[str, List[Dict]] = defaultdict(list)
        for ev in events:
            if ev["mint"] and ev["type"] in ("TRANSFER", "NFT_SALE", "SALE"):
                by_mint[ev["mint"]].append(ev)

        signals = 0
        for evs in by_mint.values():
            sorted_evs = sorted(
                [e for e in evs if e["timestamp"]],
                key=lambda e: e["timestamp"],
            )
            for i, ev_a in enumerate(sorted_evs):
                sender_a = ev_a.get("seller")
                if not sender_a:
                    continue
                for ev_b in sorted_evs[i + 1:]:
                    delta = (ev_b["timestamp"] - ev_a["timestamp"]).days
                    if delta > WASH_WINDOW_DAYS:
                        break
                    if ev_b.get("buyer") == sender_a:
                        signals += 1
                        break  # count once per mint per window

        return signals

    def _detect_fast_flips(self) -> bool:
        """
        Return ``True`` if the wallet bought and resold an NFT within
        ``FAST_FLIP_DAYS`` in the same collection.

        Collection identity is approximated by the first 8 characters of the
        mint address when on-chain collection metadata is unavailable.
        """
        events = self._nft_events()

        buy_times:  Dict[str, datetime] = {}
        sell_times: Dict[str, datetime] = {}

        for ev in events:
            mint = ev["mint"]
            if not mint or not ev["timestamp"]:
                continue
            if ev["type"] in ("NFT_SALE", "SALE") and ev.get("buyer") == self.wallet:
                buy_times[mint] = ev["timestamp"]
            elif ev["type"] in ("NFT_SALE", "SALE") and ev.get("seller") == self.wallet:
                sell_times[mint] = ev["timestamp"]
            elif ev["type"] == "TRANSFER" and ev.get("buyer") == self.wallet:
                buy_times.setdefault(mint, ev["timestamp"])
            elif ev["type"] == "TRANSFER" and ev.get("seller") == self.wallet:
                sell_times.setdefault(mint, ev["timestamp"])

        collection_flips: Dict[str, int] = defaultdict(int)
        for mint in buy_times:
            if mint in sell_times:
                buy_dt  = buy_times[mint]
                sell_dt = sell_times[mint]
                if sell_dt >= buy_dt:
                    hold_days = (sell_dt - buy_dt).days
                    if hold_days <= FAST_FLIP_DAYS:
                        collection_flips[mint[:8]] += 1

        return any(v >= 2 for v in collection_flips.values()) or bool(collection_flips)

    # ── Scoring ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_risk_score(
        past_tx_count:        int,
        mint_dump_pattern:    bool,
        wash_trading_signals: int,
        fast_flips:           bool,
    ) -> str:
        """
        Compute a categorical risk score from the individual signals.

        Scoring matrix
        --------------
        +2  mint-dump pattern detected
        +2  per wash-trading signal (capped at +6)
        +1  fast flips detected
        +1  past_tx_count > 200 (high-volume wallet)

        Thresholds:  score >= 4 → HIGH | >= 2 → MEDIUM | else → LOW
        """
        score = 0
        if mint_dump_pattern:
            score += 2
        score += min(wash_trading_signals * 2, 6)
        if fast_flips:
            score += 1
        if past_tx_count > 200:
            score += 1

        if score >= 4:
            return "HIGH"
        if score >= 2:
            return "MEDIUM"
        return "LOW"


# ── Standalone example ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import datetime as _dt
    from datetime import timezone as _tz

    WALLET = "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"
    OTHER  = "BotWallet1111111111111111111111111111111111"
    BASE   = int(_dt.datetime(2024, 1, 1, tzinfo=_tz.utc).timestamp())

    sample_txs = []
    for i in range(50):
        mint = f"Mint{i:040d}"
        sample_txs.append({
            "signature": f"mintSig{i}", "timestamp": BASE + i * 3600,
            "type": "MINT",
            "events": {"nft": [{"type": "MINT", "mint": mint,
                                 "buyer": WALLET, "seller": None, "amount": 0}]},
            "tokenTransfers": [],
        })
        sample_txs.append({
            "signature": f"saleSig{i}",
            "timestamp": BASE + i * 3600 + 86400 * 2,  # sold 2 days later
            "type": "NFT_SALE",
            "events": {"nft": [{"type": "NFT_SALE", "mint": mint,
                                 "seller": WALLET, "buyer": OTHER,
                                 "amount": 1_500_000_000}]},
            "tokenTransfers": [],
        })

    result = WalletRiskAnalyzer(WALLET, sample_txs).analyze()
    print("\n── analyzers/risk.py standalone example ──")
    print(json.dumps(result, indent=2))

    expected = {"wallet", "past_tx_count", "risk_score",
                "mint_dump_pattern", "wash_trading_signals", "fast_flips"}
    missing  = expected - result.keys()
    extra    = result.keys() - expected
    if not missing and not extra:
        print("\n  ✓ All spec keys present")
    else:
        if missing: print(f"  ✗ MISSING: {missing}")
        if extra:   print(f"  ! EXTRA:   {extra}")
