"""
Hermes Web3 Tools Ecosystem
============================
Modular Web3 tools for NFT minting workflow and automation.
Extends Hermes AI agent with blockchain capabilities.

IMPORTANT SAFETY DEFAULTS:
- DRY_RUN=true by default (no auto-send transactions)
- All transactions require explicit approval
- Private keys are NEVER logged or printed
- Multi-wallet support is for user's own burner wallets only

Modules:
- check_wallet: ETH balance checker with multi-chain RPC
- nft_contract_check: ERC721 contract info
- check_token_owner: ownerOf tokenId checker
- unminted_scanner: Scan unminted token IDs
- contract_analyzer: ABI loader and mint function detector
- wallet_manager: Burner wallet management with encryption
- mint_planner: Prepare mint transactions (dry run)
- approval_queue: SQLite approval queue
- mint_executor: Execute approved transactions only
- batch_multi_wallet_executor: Multi-wallet batch mint
- telegram_gateway: Telegram approval bot (future)
"""

__version__ = "0.1.0"
__author__ = "Hermes Web3 Tools"

# Safety defaults
DRY_RUN = True
REQUIRE_APPROVAL = True
