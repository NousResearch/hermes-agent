import sqlite3

from walletdb.bundles.helius_phase1 import (
    build_bundle_candidates,
    detect_bundle_alerts,
    ingest_enhanced_transactions,
)


def main() -> None:
    conn = sqlite3.connect(":memory:")
    txs = [
        {
            "signature": "sig-1",
            "slot": 1,
            "timestamp": 1700000000,
            "type": "TRANSFER",
            "tokenTransfers": [
                {
                    "fromUserAccount": "wallet-a",
                    "toUserAccount": "wallet-b",
                    "mint": "mint-1",
                    "tokenAmount": 100,
                    "tokenDecimals": 6,
                    "symbol": "TKN",
                },
                {
                    "fromUserAccount": "wallet-b",
                    "toUserAccount": "wallet-c",
                    "mint": "mint-1",
                    "tokenAmount": 50,
                    "tokenDecimals": 6,
                },
            ],
        },
        {
            "signature": "sig-2",
            "slot": 2,
            "timestamp": 1700000100,
            "type": "TRANSFER",
            "tokenTransfers": [
                {
                    "fromUserAccount": "wallet-c",
                    "toUserAccount": "wallet-d",
                    "mint": "mint-1",
                    "tokenAmount": 25,
                    "tokenDecimals": 6,
                },
                {
                    "fromUserAccount": "wallet-d",
                    "toUserAccount": "wallet-e",
                    "mint": "mint-1",
                    "tokenAmount": 25,
                    "tokenDecimals": 6,
                },
            ],
        },
    ]

    ingested = ingest_enhanced_transactions(conn, txs)
    candidates = build_bundle_candidates(conn, token_mint="mint-1")
    alerts = detect_bundle_alerts(conn, min_wallets=5, min_edges=4, max_hop=2)

    print({
        "ingested": ingested,
        "candidates": len(candidates),
        "alerts": len(alerts),
    })


if __name__ == "__main__":
    main()
