#!/usr/bin/env python3
"""
Bittensor Network Status Script
Checks wallet balance, staking info, and subnet list via btcli.
"""

import subprocess
import sys
import shutil
import argparse
from typing import Optional


def run_btcli(*args: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run a btcli command and return (returncode, stdout, stderr)."""
    cmd = ["btcli"] + list(args)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", f"Command timed out after {timeout}s: {' '.join(cmd)}"
    except FileNotFoundError:
        return 1, "", "btcli not found in PATH"


def check_btcli_installed() -> bool:
    """Return True if btcli is available in PATH."""
    return shutil.which("btcli") is not None


def get_btcli_version() -> Optional[str]:
    """Return btcli version string, or None on failure."""
    rc, stdout, _ = run_btcli("--version")
    if rc == 0 and stdout.strip():
        return stdout.strip()
    return None


def show_wallet_balance(wallet_name: str) -> None:
    """Print wallet balance for the given wallet name."""
    print(f"\n{'='*50}")
    print(f"Wallet Balance: {wallet_name}")
    print("=" * 50)
    rc, stdout, stderr = run_btcli("wallet", "balance", f"--wallet.name={wallet_name}")
    if rc == 0:
        print(stdout)
    else:
        print(f"[ERROR] {stderr or 'Failed to retrieve balance'}")


def show_stake_info(wallet_name: str) -> None:
    """Print staking information for the given wallet name."""
    print(f"\n{'='*50}")
    print(f"Staking Info: {wallet_name}")
    print("=" * 50)
    rc, stdout, stderr = run_btcli("stake", "show", f"--wallet.name={wallet_name}")
    if rc == 0:
        print(stdout)
    else:
        print(f"[ERROR] {stderr or 'Failed to retrieve stake info'}")


def list_subnets() -> None:
    """Print the list of active Bittensor subnets."""
    print(f"\n{'='*50}")
    print("Active Subnets")
    print("=" * 50)
    rc, stdout, stderr = run_btcli("subnet", "list")
    if rc == 0:
        print(stdout)
    else:
        print(f"[ERROR] {stderr or 'Failed to list subnets'}")


def show_metagraph(netuid: int) -> None:
    """Print metagraph for a specific subnet."""
    print(f"\n{'='*50}")
    print(f"Metagraph: Subnet {netuid}")
    print("=" * 50)
    rc, stdout, stderr = run_btcli(
        "subnet", "metagraph", f"--netuid={netuid}", timeout=60
    )
    if rc == 0:
        print(stdout)
    else:
        print(f"[ERROR] {stderr or f'Failed to retrieve metagraph for netuid {netuid}'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bittensor network status — wallet balance, stake, and subnet info"
    )
    parser.add_argument(
        "--wallet",
        default="default",
        metavar="NAME",
        help="Wallet name to query (default: 'default')",
    )
    parser.add_argument(
        "--netuid",
        type=int,
        default=None,
        metavar="N",
        help="Subnet UID to query metagraph for (optional)",
    )
    parser.add_argument(
        "--skip-subnets",
        action="store_true",
        help="Skip the subnet list (faster)",
    )
    args = parser.parse_args()

    # Check installation
    if not check_btcli_installed():
        print("[ERROR] btcli is not installed or not in PATH.")
        print("Install it with: pip install bittensor")
        return 1

    version = get_btcli_version()
    print(f"btcli version: {version or 'unknown'}")

    show_wallet_balance(args.wallet)
    show_stake_info(args.wallet)

    if not args.skip_subnets:
        list_subnets()

    if args.netuid is not None:
        show_metagraph(args.netuid)

    return 0


if __name__ == "__main__":
    sys.exit(main())
