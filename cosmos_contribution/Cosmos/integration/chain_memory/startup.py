"""
Chain Memory Startup - Prompt users to load chain memories on launch.

When enabled, this prompts users at startup to either:
1. Enter TX IDs manually
2. Load all memories from their wallet address
3. Skip and start fresh
"""

import os
import asyncio
import logging
from typing import List, Optional, Callable
from pathlib import Path

logger = logging.getLogger("chain_memory.startup")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment variable to enable chain memory on startup
CHAIN_MEMORY_ENABLED = os.getenv("CHAIN_MEMORY_ENABLED", "false").lower() == "true"

# Default RPC
DEFAULT_RPC = os.getenv("MONAD_RPC", "https://rpc.monad.xyz")


# =============================================================================
# INTERACTIVE PROMPT
# =============================================================================

def prompt_memory_load(
    bot_type: str = "cosmos",
    auto_detect_wallet: bool = True
) -> Optional[List[str]]:
    """
    Interactive prompt to load chain memories on startup.

    This should be called at the start of your bot's main() function.

    Args:
        bot_type: "cosmos" or "HermesAgent"
        auto_detect_wallet: If True, checks for MONAD_PRIVATE_KEY

    Returns:
        List of memory IDs to load, or None to skip

    Usage:
        # In your bot's main.py:
        from Cosmos.integration.chain_memory import prompt_memory_load

        def main():
            memories_to_load = prompt_memory_load()
            if memories_to_load:
                # Load memories before starting
                asyncio.run(load_chain_memories(memories_to_load))

            # Continue with normal startup...
    """
    if not CHAIN_MEMORY_ENABLED:
        return None

    print("\n" + "=" * 60)
    print("  CHAIN MEMORY - On-Chain AI Memory Storage")
    print("=" * 60)
    print(f"\n  Bot Type: {bot_type.upper()}")
    print("  Network: Monad Blockchain")

    # Check for wallet
    wallet_key = os.getenv("MONAD_PRIVATE_KEY")
    wallet_address = None

    if wallet_key and auto_detect_wallet:
        try:
            from eth_account import Account
            wallet_address = Account.from_key(wallet_key).address
            print(f"  Wallet: {wallet_address[:10]}...{wallet_address[-8:]}")
        except Exception:
            pass

    print("\n" + "-" * 60)
    print("\n  How would you like to load your memories?\n")
    print("  [1] Enter TX IDs manually")
    print("  [2] Load ALL memories from my wallet")
    print("  [3] Import from shared memory file")
    print("  [4] Skip - Start fresh")

    if not wallet_address:
        print("\n  Note: Option 2 requires MONAD_PRIVATE_KEY to be set")

    print("\n" + "-" * 60)

    while True:
        try:
            choice = input("\n  Enter choice (1-4): ").strip()

            if choice == "1":
                return _prompt_tx_ids()

            elif choice == "2":
                if not wallet_address:
                    print("\n  ERROR: MONAD_PRIVATE_KEY not set!")
                    print("  Please set this environment variable or choose another option.")
                    continue
                return _prompt_wallet_load(wallet_address)

            elif choice == "3":
                return _prompt_import_file()

            elif choice == "4":
                print("\n  Starting fresh without chain memories.\n")
                return None

            else:
                print("  Invalid choice. Please enter 1, 2, 3, or 4.")

        except KeyboardInterrupt:
            print("\n\n  Cancelled. Starting fresh.\n")
            return None


def _prompt_tx_ids() -> Optional[List[str]]:
    """Prompt for manual TX ID entry."""
    print("\n  Enter transaction IDs (one per line).")
    print("  Enter a blank line when done:\n")

    tx_ids = []
    while True:
        try:
            tx = input("  TX: ").strip()
            if not tx:
                break
            if tx.startswith("0x") and len(tx) == 66:
                tx_ids.append(tx)
                print(f"       Added ({len(tx_ids)} total)")
            else:
                print("       Invalid TX format. Should be 0x + 64 hex chars")
        except KeyboardInterrupt:
            break

    if tx_ids:
        print(f"\n  Will load memory from {len(tx_ids)} transactions.\n")
        return tx_ids

    print("\n  No valid TX IDs entered. Starting fresh.\n")
    return None


def _prompt_wallet_load(wallet_address: str) -> Optional[List[str]]:
    """Prompt to load all memories from wallet."""
    print(f"\n  Will scan blockchain for all memories uploaded by:")
    print(f"  {wallet_address}")
    print("\n  This may take a moment...")

    confirm = input("\n  Continue? (y/n): ").strip().lower()
    if confirm != 'y':
        return None

    # Return special marker to indicate "load all"
    return [f"WALLET:{wallet_address}"]


def _prompt_import_file() -> Optional[List[str]]:
    """Prompt to import from shared file."""
    print("\n  Enter the path to the shared memory file:")
    print("  (This is the JSON export from someone else's memory)\n")

    try:
        file_path = input("  Path: ").strip()
        file_path = file_path.strip('"').strip("'")

        if not os.path.exists(file_path):
            print(f"\n  File not found: {file_path}")
            return None

        with open(file_path, 'r') as f:
            import json
            data = json.load(f)

        if data.get("format") == "cosmos_chain_memory_v1":
            tx_hashes = data.get("tx_hashes", [])
            print(f"\n  Found memory: {data.get('title', 'Unknown')}")
            print(f"  {len(tx_hashes)} transactions")
            return tx_hashes
        else:
            print("\n  Invalid file format.")
            return None

    except Exception as e:
        print(f"\n  Error reading file: {e}")
        return None


# =============================================================================
# ASYNC LOADING
# =============================================================================

async def auto_load_memories(
    tx_ids_or_wallet: List[str],
    bot_type: str = "cosmos",
    memory_path: Optional[str] = None,
    on_progress: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Automatically load chain memories.

    Args:
        tx_ids_or_wallet: Either list of TX IDs or ["WALLET:0x..."]
        bot_type: "cosmos" or "HermesAgent"
        memory_path: Path to memory directory
        on_progress: Callback for progress updates

    Returns:
        True if memories were loaded successfully
    """
    from .memory_manager import ChainMemory

    def log_progress(msg: str):
        if on_progress:
            on_progress(msg)
        else:
            print(f"  {msg}")

    try:
        cm = ChainMemory(bot_type=bot_type)

        # Check if this is a wallet scan request
        if len(tx_ids_or_wallet) == 1 and tx_ids_or_wallet[0].startswith("WALLET:"):
            wallet_address = tx_ids_or_wallet[0].split(":", 1)[1]
            log_progress(f"Scanning blockchain for {wallet_address}...")

            packages = await cm.pull_all_memories(
                wallet_address=wallet_address,
                on_progress=lambda c, t, s: log_progress(s)
            )

            if not packages:
                log_progress("No memories found for this wallet.")
                return False

            log_progress(f"Found {len(packages)} memories. Loading...")

            for package in packages:
                if bot_type == "cosmos":
                    cm.load_into_cosmos(package, memory_path, merge=True)
                else:
                    cm.load_into_HermesAgent(package, memory_path, merge=True)

            log_progress(f"Loaded {len(packages)} memory packages!")
            return True

        else:
            # Direct TX ID list
            log_progress(f"Downloading from {len(tx_ids_or_wallet)} transactions...")

            package = await cm.pull_memory(
                tx_ids=tx_ids_or_wallet,
                on_progress=lambda c, t, s: log_progress(s)
            )

            if bot_type == "cosmos":
                cm.load_into_cosmos(package, memory_path, merge=True)
            else:
                cm.load_into_HermesAgent(package, memory_path, merge=True)

            stats = cm.memvid.get_memory_stats(package)
            log_progress(f"Loaded {stats['total_chunks']} memory chunks!")
            return True

    except Exception as e:
        log_progress(f"ERROR: {e}")
        logger.exception("Failed to load chain memories")
        return False


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point for chain memory management."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Chain Memory - On-Chain AI Memory Storage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Push current memory to chain
  python -m cosmos.integration.chain_memory push --title "My Bot Memory"

  # Pull memory by TX IDs
  python -m cosmos.integration.chain_memory pull --tx 0x123... 0x456...

  # Pull all memories for a wallet
  python -m cosmos.integration.chain_memory pull --wallet 0xYourAddress

  # List local memories
  python -m cosmos.integration.chain_memory list

  # Export memory for sharing
  python -m cosmos.integration.chain_memory export --id abc123
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Push command
    push_parser = subparsers.add_parser("push", help="Push memory to chain")
    push_parser.add_argument("--title", default="Bot Memory", help="Memory title")
    push_parser.add_argument("--bot", default="cosmos", choices=["cosmos", "HermesAgent"])
    push_parser.add_argument("--path", help="Path to memory directory")

    # Pull command
    pull_parser = subparsers.add_parser("pull", help="Pull memory from chain")
    pull_parser.add_argument("--tx", nargs="+", help="Transaction hashes")
    pull_parser.add_argument("--wallet", help="Wallet address to scan")
    pull_parser.add_argument("--id", help="Memory ID from local records")
    pull_parser.add_argument("--bot", default="cosmos", choices=["cosmos", "HermesAgent"])
    pull_parser.add_argument("--path", help="Path to memory directory")

    # List command
    list_parser = subparsers.add_parser("list", help="List local memories")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export memory for sharing")
    export_parser.add_argument("--id", required=True, help="Memory ID to export")
    export_parser.add_argument("--output", "-o", help="Output file path")

    # Import command
    import_parser = subparsers.add_parser("import", help="Import shared memory")
    import_parser.add_argument("--file", required=True, help="Path to import file")

    # Estimate command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate push cost")
    estimate_parser.add_argument("--bot", default="cosmos", choices=["cosmos", "HermesAgent"])
    estimate_parser.add_argument("--path", help="Path to memory directory")

    args = parser.parse_args()

    if args.command == "push":
        asyncio.run(_cli_push(args))
    elif args.command == "pull":
        asyncio.run(_cli_pull(args))
    elif args.command == "list":
        _cli_list(args)
    elif args.command == "export":
        _cli_export(args)
    elif args.command == "import":
        _cli_import(args)
    elif args.command == "estimate":
        _cli_estimate(args)
    else:
        parser.print_help()


async def _cli_push(args):
    """Handle push command."""
    from .memory_manager import ChainMemory

    print("\n=== PUSHING MEMORY TO CHAIN ===\n")

    cm = ChainMemory(bot_type=args.bot)

    def progress(current, total, status):
        print(f"  [{current}/{total}] {status}")

    record = await cm.push_memory(
        title=args.title,
        memory_path=args.path,
        on_progress=progress
    )

    print("\n=== SUCCESS ===")
    print(f"  Memory ID: {record.memory_id}")
    print(f"  TX Count: {len(record.tx_hashes)}")
    print(f"  Total Size: {record.total_size:,} bytes")
    print("\n  First 3 TX hashes:")
    for tx in record.tx_hashes[:3]:
        print(f"    {tx}")
    if len(record.tx_hashes) > 3:
        print(f"    ... and {len(record.tx_hashes) - 3} more")
    print()


async def _cli_pull(args):
    """Handle pull command."""
    from .memory_manager import ChainMemory

    print("\n=== PULLING MEMORY FROM CHAIN ===\n")

    cm = ChainMemory(bot_type=args.bot)

    def progress(current, total, status):
        print(f"  [{current}/{total}] {status}")

    if args.wallet:
        packages = await cm.pull_all_memories(
            wallet_address=args.wallet,
            on_progress=progress
        )
        for package in packages:
            if args.bot == "cosmos":
                cm.load_into_cosmos(package, args.path)
            else:
                cm.load_into_HermesAgent(package, args.path)
        print(f"\n=== LOADED {len(packages)} MEMORIES ===\n")

    elif args.tx:
        package = await cm.pull_memory(
            tx_ids=args.tx,
            on_progress=progress
        )
        if args.bot == "cosmos":
            cm.load_into_cosmos(package, args.path)
        else:
            cm.load_into_HermesAgent(package, args.path)
        print("\n=== MEMORY LOADED ===\n")

    elif args.id:
        package = await cm.pull_memory(
            memory_id=args.id,
            on_progress=progress
        )
        if args.bot == "cosmos":
            cm.load_into_cosmos(package, args.path)
        else:
            cm.load_into_HermesAgent(package, args.path)
        print("\n=== MEMORY LOADED ===\n")

    else:
        print("  ERROR: Specify --tx, --wallet, or --id")


def _cli_list(args):
    """Handle list command."""
    from .memory_manager import ChainMemory

    cm = ChainMemory()
    memories = cm.list_local_memories()

    print("\n=== LOCAL MEMORY RECORDS ===\n")

    if not memories:
        print("  No memories found.\n")
        return

    for mem in memories:
        print(f"  ID: {mem.memory_id}")
        print(f"  Title: {mem.title}")
        print(f"  Bot: {mem.bot_type}")
        print(f"  Chunks: {mem.total_chunks}")
        print(f"  Uploaded: {mem.uploaded_at}")
        print()


def _cli_export(args):
    """Handle export command."""
    from .memory_manager import ChainMemory

    cm = ChainMemory()
    export_str = cm.export_tx_list(args.id)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(export_str)
        print(f"\n  Exported to: {args.output}\n")
    else:
        print("\n" + export_str + "\n")


def _cli_import(args):
    """Handle import command."""
    from .memory_manager import ChainMemory

    with open(args.file, 'r') as f:
        export_str = f.read()

    cm = ChainMemory()
    record = cm.import_tx_list(export_str)

    print(f"\n  Imported: {record.memory_id}")
    print(f"  Title: {record.title}")
    print(f"  TX Count: {len(record.tx_hashes)}\n")


def _cli_estimate(args):
    """Handle estimate command."""
    from .memory_manager import ChainMemory

    cm = ChainMemory(bot_type=args.bot)
    estimate = cm.estimate_push_cost(args.path)

    print("\n=== COST ESTIMATE ===\n")
    print(f"  Estimated chunks: {estimate['num_chunks']}")
    print(f"  Total gas: {estimate['total_gas']:,}")
    print(f"  Gas price: {estimate['gas_price_gwei']:.2f} gwei")
    print(f"  Cost: {estimate['total_cost_mon']:.4f} MON")
    print(f"  Cost: ${estimate['total_cost_usd']:.2f} USD")
    print(f"  (at ${estimate['mon_price_usd']}/MON)\n")


if __name__ == "__main__":
    main()
