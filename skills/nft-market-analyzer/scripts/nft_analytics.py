#!/usr/bin/env python3
"""
nft_analytics.py
----------------
NFT Analytics — main entry point + Hermes skill registry.

Usage
-----
Via Hermes toolset:
    hermes --toolsets skills -q "nft_wallet_analytics --wallet <ADDR>"
    hermes chat -q "Analyze wallet <ADDR>"

Direct CLI:
    python nft_analytics.py --wallet <ADDR>
    python nft_analytics.py --wallet <ADDR> --verbose
    python nft_analytics.py --wallet <ADDR> --json
    python nft_analytics.py --wallet <ADDR> --api-key <KEY>

Environment:
    HELIUS_API_KEY   Helius API key (https://helius.dev).
                     Also read from .env in parent directories.
                     Falls back to an interactive terminal prompt.

Output key contract (all spec-canonical):
    WalletRiskAnalyzer  → past_tx_count, risk_score, mint_dump_pattern,
                          wash_trading_signals, fast_flips
    SmartMoneyTracker   → smart_entry, avg_roi, active_months
    WalletProfiler      → total_nfts, avg_flip_duration,
                          high_risk_collections, net_roi
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from typing import Optional

# ── Ensure skill root is always on sys.path ───────────────────────────────────
_SKILL_DIR = Path(__file__).resolve().parent
if str(_SKILL_DIR) not in sys.path:
    sys.path.insert(0, str(_SKILL_DIR))

from helius_client import HeliusClient, TensorClient                           # noqa: E402
from analyzers     import WalletRiskAnalyzer                    # noqa: E402
from analyzers     import SmartMoneyTracker                     # noqa: E402
from analyzers     import WalletProfiler                        # noqa: E402
from utils         import format_report, format_json            # noqa: E402
from utils.validator import assert_valid_wallet                 # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# API key resolution
# ─────────────────────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    """
    Load HELIUS_API_KEY from the first ``.env`` file found by searching
    upward from the skill directory.  Does not require python-dotenv.
    """
    search_dirs = [
        Path.cwd(),
        Path.cwd().parent,
        _SKILL_DIR,
        _SKILL_DIR.parent,
        _SKILL_DIR.parent.parent,
    ]
    for d in search_dirs:
        env_file = d / ".env"
        if env_file.exists():
            try:
                with open(env_file, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key   = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            except OSError:
                pass
            return  # stop at the first .env found


def resolve_api_key() -> str:
    """
    Resolve the Helius API key using this priority order:

    1. ``HELIUS_API_KEY`` environment variable
    2. ``.env`` file (searched upward from the skill directory)
    3. Interactive terminal prompt

    Returns
    -------
    str
        Non-empty API key string.

    Raises
    ------
    SystemExit
        If no key is supplied via any method.
    """
    # 1. Environment variable
    key = os.getenv("HELIUS_API_KEY", "").strip()
    if key:
        return key

    # 2. .env file
    _load_dotenv()
    key = os.getenv("HELIUS_API_KEY", "").strip()
    if key:
        return key

    # 3. Interactive prompt
    print("\n⚠️  HELIUS_API_KEY not found in environment or .env file.")
    print("   Get a free key at: https://helius.dev\n")
    try:
        key = input("Enter your Helius API key: ").strip()
        if not key:
            raise ValueError("empty input")
    except (KeyboardInterrupt, ValueError):
        print("\nERROR: Helius API key is required to run this skill.")
        sys.exit(1)
    except EOFError:
        print("\nERROR: Cannot prompt for API key in non-interactive mode.")
        print("       Set HELIUS_API_KEY in your environment or .env file.")
        sys.exit(1)

    return key


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(
    wallet:      str,
    api_key:     str,
    verbose:     bool = False,
    json_output: bool = False,
) -> str:
    """
    Fetch on-chain data and run all three analyzers.

    Parameters
    ----------
    wallet      : Solana wallet address (validated before use).
    api_key     : Helius API key.
    verbose     : Print HTTP request details to stderr.
    json_output : Return JSON instead of the terminal report.

    Returns
    -------
    str
        Formatted report (terminal or JSON).
    """
    wallet = assert_valid_wallet(wallet)

    # Tensor key resolve
    tensor_key = os.getenv("TENSOR_API_KEY", "").strip()
    if not tensor_key:
        import pathlib as _pl
        env_file = _pl.Path(__file__).parent / ".env"
        if env_file.exists():
            for _line in env_file.read_text().splitlines():
                if _line.startswith("TENSOR_API_KEY="):
                    tensor_key = _line.split("=",1)[1].strip()
    client = HeliusClient(api_key=api_key, verbose=verbose)

    print(f"  Fetching transaction history for {wallet[:8]}…", file=sys.stderr)
    transactions = client.get_transactions(wallet)

    print("  Fetching NFT assets…", file=sys.stderr)
    owned_assets = client.get_assets_by_owner(wallet)

    if verbose:
        print(f"  Transactions loaded : {len(transactions)}", file=sys.stderr)
        print(f"  NFT assets loaded   : {len(owned_assets)}",  file=sys.stderr)

    # ── Run analyzers (all return spec-canonical keys) ────────────────────────
    risk_result        = WalletRiskAnalyzer(wallet, transactions).analyze()
    smart_money_result = SmartMoneyTracker(wallet, transactions).analyze()
    _smart_roi_fallback = smart_money_result["avg_roi"] == 0
    # Tensor'dan gerçek fiyatları çek
    tensor_txs = []
    if tensor_key:
        try:
            print("  Fetching Tensor trade history…", file=sys.stderr)
            tensor_client = TensorClient(api_key=tensor_key, verbose=verbose)
            tensor_txs = tensor_client.get_wallet_transactions(wallet)
            if verbose:
                print(f"  Tensor trades loaded: {len(tensor_txs)}", file=sys.stderr)
        except Exception as e:
            print(f"  [WARN] Tensor skipped: {e}", file=sys.stderr)

    profiler_result    = WalletProfiler(wallet, transactions, owned_assets, tensor_txs=tensor_txs).analyze()
    if _smart_roi_fallback and profiler_result["net_roi"] != 0:
        smart_money_result["avg_roi"] = profiler_result["net_roi"]
        smart_money_result["smart_entry"] = profiler_result["net_roi"] >= 8 and smart_money_result["active_months"] >= 3

    # ── Format ────────────────────────────────────────────────────────────────
    if json_output:
        return format_json(risk_result, smart_money_result, profiler_result)

    return format_report(
        risk_result,
        smart_money_result,
        profiler_result,
        colour=sys.stdout.isatty(),   # ANSI colours only in a real TTY
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hermes skill registry
# ─────────────────────────────────────────────────────────────────────────────

def _register_with_hermes() -> None:
    """
    Register this skill with the Hermes ``tools.registry`` if available.
    Fails silently when running outside Hermes.
    """
    try:
        from tools import registry  # type: ignore[import]  # Hermes internal

        def _handler(wallet: str = "", **kwargs) -> str:
            if not wallet:
                return "ERROR: --wallet <ADDRESS> is required."
            return run_analysis(
                wallet      = wallet,
                api_key     = resolve_api_key(),
                verbose     = bool(kwargs.get("verbose")),
                json_output = bool(kwargs.get("json")),
            )

        registry.register(
            name="nft_wallet_analytics",
            description=(
                "Analyze a Solana NFT wallet using Helius API. "
                "Outputs: Wallet Risk Analyzer, Smart Money Tracker, "
                "and Wallet Profiling."
            ),
            triggers=[
                r"nft_wallet_analytics\s+--wallet\s+(\S+)",
                r"[Aa]nalyze\s+wallet\s+(\S+)",
                r"[Nn][Ff][Tt]\s+risk\s+score\s+for\s+(\S+)",
                r"[Ss]mart\s+money\s+tracker\s+(\S+)",
                r"[Pp]rofile\s+[Nn][Ff][Tt]\s+wallet\s+(\S+)",
                r"[Cc]heck\s+wallet\s+(\S+)\s+for\s+wash\s+trading",
                r"[Mm]int\s+dump\s+detection\s+(\S+)",
            ],
            handler=_handler,
        )

    except ImportError:
        pass  # not running inside Hermes
    except Exception as exc:
        print(f"[nft-analytics] Registry warning: {exc}", file=sys.stderr)


_register_with_hermes()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nft_analytics",
        description=(
            "NFT Wallet Analytics — Helius-powered risk, smart money,\n"
            "and profiling tool for Solana NFT wallets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python nft_analytics.py --wallet 9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM
  python nft_analytics.py --wallet <ADDR> --verbose
  python nft_analytics.py --wallet <ADDR> --json
  HELIUS_API_KEY=mykey python nft_analytics.py --wallet <ADDR>
  python nft_analytics.py --wallet <ADDR> --api-key mykey
        """,
    )
    parser.add_argument(
        "--wallet", "-w",
        required=True,
        metavar="ADDRESS",
        help="Solana wallet address to analyze (base-58, 32–44 chars)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print HTTP request details and fetched item counts",
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        default=False,
        help="Output results as JSON instead of the terminal report",
    )
    parser.add_argument(
        "--api-key", "-k",
        default=None,
        metavar="KEY",
        help=(
            "Helius API key. Overrides HELIUS_API_KEY env var and .env file. "
            "If omitted and no env key exists, an interactive prompt appears."
        ),
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        0 on success, 1 on any error.
    """
    parser = _build_parser()
    args   = parser.parse_args(argv)

    # Resolve API key (--api-key flag takes highest precedence)
    if args.api_key:
        api_key = args.api_key.strip()
        if not api_key:
            print("ERROR: --api-key value is empty.", file=sys.stderr)
            return 1
    else:
        api_key = resolve_api_key()

    try:
        report = run_analysis(
            wallet      = args.wallet,
            api_key     = api_key,
            verbose     = args.verbose,
            json_output = args.json,
        )
        print(report)
        return 0

    except ValueError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"\nAPI ERROR: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    except KeyboardInterrupt:
        print("\n\nAborted.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
