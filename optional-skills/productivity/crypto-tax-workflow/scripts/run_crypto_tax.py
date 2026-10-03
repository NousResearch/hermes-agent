#!/usr/bin/env python3
"""
Bootstrap runner: fetch Binance trades via the crypto_tax plugin, compute
FIFO gains, and dump raw JSON for downstream analysis.

Usage:
    python3 run_crypto_tax.py                      # uses BINANCE_* from .hermes/.env
    TAX_YEAR=2023 python3 run_crypto_tax.py        # override tax year
    ANNUAL_INCOME_USD=120000 python3 ...           # override bracket calc

Output:
    crypto_tax_raw.json   — full plugin return (canonical artifact)
    stdout                — summary + per-event ledger preview

Requirements (system python user-site):
    pip3 install --user --break-system-packages ccxt python-dotenv pandas
"""
import sys
import os
import json
from pathlib import Path

# --- 1. Find & load the crypto_tax plugin ---------------------------------
PLUGIN_ROOT = Path.home() / ".hermes" / "plugins" / "crypto_tax"
if not PLUGIN_ROOT.exists():
    sys.exit(f"Plugin not found: {PLUGIN_ROOT}")
sys.path.insert(0, str(PLUGIN_ROOT))

# --- 2. Load .env BEFORE imports so os.getenv sees credentials ----------
from dotenv import load_dotenv
env_path = Path.home() / ".hermes" / ".env"
if not env_path.exists():
    sys.exit(f".env not found: {env_path}")
load_dotenv(env_path, override=True)

from tools import fetch_cex_transactions, calculate_crypto_gains

# --- 3. Symbol list (see references/common-symbols.md in the skill) ------
# Inline short copy so this script is self-contained. For the full list,
# read references/common-symbols.md.
COMMON_SYMBOLS = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT","ADA/USDT",
    "DOGE/USDT","AVAX/USDT","DOT/USDT","MATIC/USDT","POL/USDT","LINK/USDT",
    "LTC/USDT","SHIB/USDT","TRX/USDT","UNI/USDT","ATOM/USDT","FIL/USDT",
    "ETC/USDT","AAVE/USDT","ALGO/USDT","ARB/USDT","OP/USDT","NEAR/USDT",
    "APT/USDT","SUI/USDT","SEI/USDT","PEPE/USDT","BONK/USDT","WIF/USDT",
    "FET/USDT","RNDR/USDT","IMX/USDT","MKR/USDT","SNX/USDT","CRV/USDT",
    "COMP/USDT","SUSHI/USDT","YFI/USDT","BAL/USDT","LDO/USDT","RPL/USDT",
    "TIA/USDT","INJ/USDT","JUP/USDT","WLD/USDT","TON/USDT","ICP/USDT",
    "VET/USDT","HBAR/USDT","XLM/USDT","NEO/USDT","EOS/USDT","THETA/USDT",
    "FTM/USDT","MANA/USDT","SAND/USDT","AXS/USDT","GALA/USDT","APE/USDT",
    "BTC/BUSD","ETH/BUSD","BNB/BUSD","SOL/BUSD",
    "BTC/USDC","ETH/USDC","BTC/FDUSD","ETH/FDUSD","BTC/USD","ETH/USD","SOL/USD",
]

# --- 4. Fetch --------------------------------------------------------------
print("Fetching Binance trades via crypto_tax plugin...")
binance_key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_KEY")
binance_secret = os.getenv("BINANCE_SECRET") or os.getenv("BINANCE_API_SECRET")
result = fetch_cex_transactions(
    exchange_id="binance",
    api_key=binance_key,
    secret=binance_secret,
    symbols=COMMON_SYMBOLS,
)
if "error" in result:
    sys.exit(f"fetch_cex_transactions error: {result['error']}")

txs = result.get("transactions", [])
print(f"Fetched {result['total_fetched']} transactions across {len(COMMON_SYMBOLS)} pairs.")
if not txs:
    sys.exit("No trades found. Check API key permissions.")

# --- 5. Compute gains -----------------------------------------------------
tax_year = int(os.getenv("TAX_YEAR", "2024"))
annual_income = float(os.getenv("ANNUAL_INCOME_USD", "80000"))

tax_result = calculate_crypto_gains(
    transactions=txs,
    method="FIFO",
    jurisdiction="US",
    annual_income_usd=annual_income,
)

# --- 6. Persist raw JSON -------------------------------------------------
out_path = Path.cwd() / "crypto_tax_raw.json"
with open(out_path, "w") as f:
    json.dump(tax_result, f, indent=2, default=str)
print(f"\nRaw tax result: {out_path}")

# --- 7. Quick summary to stdout -----------------------------------------
summary = {k: v for k, v in tax_result.items() if k not in ("sales_detail", "transactions", "dispositions")}
print("\n" + "=" * 70)
print(f"  CRYPTO TAX REPORT (FIFO · US · Tax Year {tax_year})")
print("=" * 70)
for k, v in summary.items():
    print(f"  {k:<30}: {v}")

# Per-event ledger preview (first 30)
detail = tax_result.get("sales_detail") or tax_result.get("dispositions") or []
if detail:
    print(f"\n--- Disposals ({len(detail)} in ledger, showing first 30) ---")
    for d in detail[:30]:
        lt = "LTCG" if d.get("long_term") else "STCG"
        print(f"  {d.get('sell_date','')[:10]}  {d.get('asset',''):>6s}  "
              f"qty={d.get('sold_amount',0):>10.6f}  "
              f"proceeds=${d.get('proceeds',0):>10,.2f}  "
              f"basis=${d.get('cost_basis',0):>10,.2f}  "
              f"gain=${d.get('gain_loss',0):>+10,.2f}  {lt}")
    if len(detail) > 30:
        print(f"  ... and {len(detail) - 30} more")
