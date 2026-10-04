# Common Binance Trading Pairs (CCXT Symbols)

Ready-to-use list for `fetch_cex_transactions(symbols=...)`. Covers the ~70
pairs we've exercised. Pass as-is to the plugin — it skips pairs with no
user trades silently.

```python
COMMON_SYMBOLS = [
    # Major majors
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
    "POL/USDT", "LINK/USDT", "LTC/USDT", "SHIB/USDT", "TRX/USDT",
    "UNI/USDT", "ATOM/USDT", "FIL/USDT", "ETC/USDT", "AAVE/USDT",
    "ALGO/USDT", "ARB/USDT", "OP/USDT", "NEAR/USDT", "APT/USDT",
    "SUI/USDT", "SEI/USDT", "PEPE/USDT", "BONK/USDT", "WIF/USDT",
    "FET/USDT", "RNDR/USDT", "IMX/USDT", "MKR/USDT", "SNX/USDT",
    "CRV/USDT", "COMP/USDT", "SUSHI/USDT", "YFI/USDT", "BAL/USDT",
    "LDO/USDT", "RPL/USDT", "TIA/USDT", "INJ/USDT", "JUP/USDT",
    "WLD/USDT", "TON/USDT", "ICP/USDT", "VET/USDT", "HBAR/USDT",
    "XLM/USDT", "NEO/USDT", "EOS/USDT", "THETA/USDT", "FTM/USDT",
    "MANA/USDT", "SAND/USDT", "AXS/USDT", "GALA/USDT", "APE/USDT",

    # BUSD (still traded on some pairs)
    "BTC/BUSD", "ETH/BUSD", "BNB/BUSD", "SOL/BUSD",

    # USDC
    "BTC/USDC", "ETH/USDC",

    # FDUSD
    "BTC/FDUSD", "ETH/FDUSD",

    # Fiat-direct
    "BTC/USD", "ETH/USD", "SOL/USD",
]
```

## Coverage notes

- **USDT** pairs = most liquid, catch the bulk of user activity.
- **BUSD** pairs are being sunset by Binance; keep for legacy history but
  don't expect them on new accounts post-2024.
- **USDC/FDUSD** pairs matter for US-based users who prefer fiat-pegged
  non-Tether stables.
- **BTC/USD, ETH/USD, SOL/USD** — only relevant if the user has directly
  traded with deposited fiat (not a swap from a stable).

## Expanding the list

If the user reports missing assets, append to the list rather than falling
back to a full-exchange scan (which times out). Common omissions from
above that users hold: `DOT`, `FIL`, `NEAR`, `APT`, `SUI`, `SEI`, `STX`,
`RUNE`, `GRT`, `ENS`, `1INCH`, `DYDX` — all end in USDT.
