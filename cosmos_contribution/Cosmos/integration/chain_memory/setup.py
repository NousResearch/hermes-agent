#!/usr/bin/env python3
"""
Chain Memory Setup Wizard

Interactive setup for on-chain memory storage.
Collects user wallet info and verifies COSMOS token holdings.

Requirements:
- Solana wallet with 100k+ COSMOS tokens
- Monad wallet with MON for gas fees
"""

import os
import sys
import asyncio
import getpass
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Cosmos.integration.chain_memory.config import (
    ChainMemoryConfig,
    verify_cosmos_holdings,
    check_monad_balance,
    get_monad_address,
    MIN_COSMOS_REQUIRED,
    COSMOS_TOKEN_MINT,
    CONFIG_DIR,
)


def print_banner():
    """Print setup banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║       ⛓️  CHAIN MEMORY - On-Chain AI Memory Storage  ⛓️          ║
║                                                                  ║
║       Store your bot's memory permanently on Monad blockchain    ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  REQUIREMENTS:                                                   ║
║  ✓ Solana wallet with 100,000+ COSMOS tokens                     ║
║  ✓ Monad wallet with MON for gas fees                           ║
║                                                                  ║
║  COSMOS Token: 9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS      ║
║  Buy COSMOS: https://pump.fun/coin/9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_step(num: int, total: int, text: str):
    """Print a step header."""
    print(f"\n{'─' * 60}")
    print(f"  Step {num}/{total}: {text}")
    print(f"{'─' * 60}\n")


def get_input(prompt: str, default: str = None, secret: bool = False) -> str:
    """Get user input with optional default."""
    if default:
        prompt = f"{prompt} [{default}]: "
    else:
        prompt = f"{prompt}: "

    if secret:
        value = getpass.getpass(prompt)
    else:
        value = input(prompt).strip()

    return value if value else default


def setup_wizard():
    """Run the interactive setup wizard."""
    print_banner()

    print("\nThis wizard will help you set up Chain Memory for your bot.")
    print("Your bot's entire state will be backed up to Monad blockchain.")
    print("\nPress Ctrl+C at any time to cancel.\n")

    input("Press Enter to begin setup...")

    config = ChainMemoryConfig()

    # -------------------------------------------------------------------------
    # Step 1: Bot Type
    # -------------------------------------------------------------------------
    print_step(1, 5, "Select Your Bot")

    print("  Which bot are you using?\n")
    print("  [1] Cosmos (recommended - full support)")
    print("  [2] ClawwBot / HermesAgent")
    print("  [3] Claude Code")
    print("  [4] Kimi")
    print("  [5] Other")

    while True:
        choice = get_input("\n  Enter choice (1-5)", "1")
        bot_types = {
            "1": "cosmos",
            "2": "clawwbot",
            "3": "claude",
            "4": "kimi",
            "5": "other"
        }
        if choice in bot_types:
            config.bot_type = bot_types[choice]
            break
        print("  Invalid choice. Please enter 1-5.")

    if config.bot_type != "cosmos":
        print(f"""
  ⚠️  WARNING: You selected {config.bot_type.upper()}

  Chain Memory is optimized for Cosmos and may not function
  properly with other bots. Features that may not work:

  - Automatic state capture
  - Evolution history backup
  - Personality state sync
  - Auto-save on crash

  For best results, use with Cosmos:
  https://github.com/cosmos-ai/cosmos

  Continue anyway? (y/n)
""")
        if get_input("  ", "n").lower() != 'y':
            print("\n  Setup cancelled. Please install Cosmos first.\n")
            return

    # -------------------------------------------------------------------------
    # Step 2: COSMOS Token Verification
    # -------------------------------------------------------------------------
    print_step(2, 5, "Verify COSMOS Holdings")

    print(f"""
  To use Chain Memory, you must hold at least {MIN_COSMOS_REQUIRED:,} COSMOS tokens.

  This requirement:
  - Supports the COSMOS ecosystem
  - Prevents spam/abuse
  - Ensures committed users

  COSMOS Token Address (Solana):
  {COSMOS_TOKEN_MINT}

  Buy COSMOS: https://pump.fun/coin/{COSMOS_TOKEN_MINT}
""")

    while True:
        solana_address = get_input("  Enter your Solana wallet address")

        if not solana_address:
            print("  Solana address is required.")
            continue

        if len(solana_address) < 32 or len(solana_address) > 44:
            print("  Invalid Solana address format.")
            continue

        print("\n  Verifying COSMOS holdings...")

        try:
            result = asyncio.run(verify_cosmos_holdings(solana_address))

            if result['verified']:
                print(f"""
  ✅ VERIFIED!

  Wallet: {solana_address[:8]}...{solana_address[-8:]}
  COSMOS Balance: {result['balance']:,} COSMOS
  Required: {MIN_COSMOS_REQUIRED:,} COSMOS

  You are eligible to use Chain Memory!
""")
                config.solana_wallet_address = solana_address
                config.cosmos_verified = True
                config.cosmos_balance = result['balance']
                break

            else:
                print(f"""
  ❌ VERIFICATION FAILED

  {result.get('error', 'Unknown error')}

  You need at least {MIN_COSMOS_REQUIRED:,} COSMOS tokens.
  Current balance: {result.get('balance', 0):,} COSMOS

  Buy COSMOS: https://pump.fun/coin/{COSMOS_TOKEN_MINT}
""")
                retry = get_input("  Try a different wallet? (y/n)", "y")
                if retry.lower() != 'y':
                    print("\n  Setup cancelled. Please acquire COSMOS tokens first.\n")
                    return

        except Exception as e:
            print(f"\n  Error verifying: {e}")
            print("  Please check your internet connection and try again.")

    # -------------------------------------------------------------------------
    # Step 3: Monad Wallet Setup
    # -------------------------------------------------------------------------
    print_step(3, 5, "Monad Wallet Setup")

    print("""
  Chain Memory stores data on Monad blockchain.
  You need a Monad wallet with MON tokens for gas fees.

  ⚠️  SECURITY WARNING:
  Your private key will be stored in environment variables.
  Never share your private key with anyone.
""")

    while True:
        monad_key = get_input("  Enter your Monad private key (0x...)", secret=True)

        if not monad_key:
            print("  Private key is required.")
            continue

        if not monad_key.startswith('0x'):
            monad_key = '0x' + monad_key

        if len(monad_key) != 66:
            print("  Invalid private key format (should be 64 hex chars after 0x).")
            continue

        try:
            address = get_monad_address(monad_key)
            print(f"\n  Wallet address: {address}")
            config.monad_private_key = monad_key

            # Check RPC
            rpc_url = get_input("\n  Monad RPC URL", "https://rpc.monad.xyz")
            config.monad_rpc = rpc_url

            # Check balance
            print("\n  Checking MON balance...")
            balance_info = check_monad_balance(monad_key, rpc_url)

            print(f"""
  Wallet: {balance_info['address']}
  Balance: {balance_info['balance_mon']:.4f} MON
""")

            if not balance_info['has_funds']:
                print("""
  ⚠️  WARNING: Low balance!

  You need MON tokens for gas fees.
  Estimated cost: ~0.01-0.1 MON per backup

  Get MON: https://monad.xyz
""")
                cont = get_input("  Continue anyway? (y/n)", "y")
                if cont.lower() != 'y':
                    continue

            break

        except Exception as e:
            print(f"\n  Error: {e}")
            print("  Please check your private key and try again.")

    # -------------------------------------------------------------------------
    # Step 4: Auto-Save Settings
    # -------------------------------------------------------------------------
    print_step(4, 5, "Auto-Save Settings")

    print("""
  Chain Memory can automatically backup your bot's state.

  Auto-save will:
  - Backup state every X minutes
  - Backup before detected crashes
  - Only backup if you have sufficient MON balance
""")

    enable_auto = get_input("  Enable auto-save? (y/n)", "y")
    config.auto_save_enabled = enable_auto.lower() == 'y'

    if config.auto_save_enabled:
        interval = get_input("  Auto-save interval (minutes)", "60")
        try:
            config.auto_save_interval_minutes = int(interval)
        except Exception:
            config.auto_save_interval_minutes = 60

    # Chunk size
    chunk_size = get_input("\n  Chunk size in KB (default 80, proven)", "80")
    try:
        config.chunk_size_kb = int(chunk_size)
    except Exception:
        config.chunk_size_kb = 80

    # -------------------------------------------------------------------------
    # Step 5: Save Configuration
    # -------------------------------------------------------------------------
    print_step(5, 5, "Save Configuration")

    print(f"""
  Configuration Summary:
  ─────────────────────────────────────────────────────────
  Bot Type:        {config.bot_type}
  Solana Wallet:   {config.solana_wallet_address[:8]}...{config.solana_wallet_address[-8:]}
  COSMOS Balance:   {config.cosmos_balance:,} COSMOS ✓
  Monad Wallet:    {get_monad_address(config.monad_private_key)}
  Monad RPC:       {config.monad_rpc}
  Auto-Save:       {'Enabled' if config.auto_save_enabled else 'Disabled'}
  Chunk Size:      {config.chunk_size_kb} KB
  ─────────────────────────────────────────────────────────
""")

    confirm = get_input("  Save this configuration? (y/n)", "y")
    if confirm.lower() != 'y':
        print("\n  Setup cancelled.\n")
        return

    # Save config
    config.save()

    # Create env file template
    env_file = CONFIG_DIR / ".env.chain_memory"
    env_content = f"""# Chain Memory Environment Variables
# Add these to your shell profile or .env file

export MONAD_PRIVATE_KEY="{config.monad_private_key}"
export MONAD_RPC="{config.monad_rpc}"
export CHUNK_SIZE_KB="{config.chunk_size_kb}"
export CHAIN_MEMORY_ENABLED="true"
"""

    with open(env_file, 'w') as f:
        f.write(env_content)

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                    ✅ SETUP COMPLETE!                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

  Configuration saved to: {CONFIG_DIR / 'config.json'}

  IMPORTANT: Add these environment variables to your shell:

  export MONAD_PRIVATE_KEY="0x..."
  export MONAD_RPC="{config.monad_rpc}"
  export CHAIN_MEMORY_ENABLED="true"

  Or source the generated file:
  source {env_file}

  ─────────────────────────────────────────────────────────────────

  USAGE:

  # Push current state to chain
  python -m cosmos.integration.chain_memory push

  # Pull and restore state
  python -m cosmos.integration.chain_memory pull --wallet YOUR_ADDRESS

  # List your backups
  python -m cosmos.integration.chain_memory list

  ─────────────────────────────────────────────────────────────────

  For help: python -m cosmos.integration.chain_memory --help

""")


def main():
    """Main entry point."""
    try:
        setup_wizard()
    except KeyboardInterrupt:
        print("\n\n  Setup cancelled by user.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
