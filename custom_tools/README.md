# Hermes Web3 Tools - NFT Mint Automation Ecosystem

Modular Web3 tools extending [Hermes Agent](https://github.com/NousResearch/hermes-agent) with NFT minting workflow and automation capabilities.

## Architecture

```
hermes-agent/          (AI core - UNTOUCHED)
custom_tools/          (Web3 extension modules)
  check_wallet.py
  nft_contract_check.py
  check_token_owner.py
  unminted_scanner.py
  contract_analyzer.py
  wallet_manager.py
  mint_planner.py
  approval_queue.py
  mint_executor.py
  batch_multi_wallet_executor.py
  telegram_gateway/
    bot.py
```

## Safety First

**CRITICAL DEFAULTS:**
- `DRY_RUN=true` - No transactions sent by default
- All transactions require explicit approval
- Private keys are NEVER logged, printed, or exposed
- Multi-wallet is for YOUR OWN burner wallets only

**NEVER:**
- Bypass allowlist/captcha/anti-bot
- Bypass signature protection
- Expose or print private keys
- Auto-send without approval

## Quick Start

### 1. Install Dependencies

```bash
cd hermes-agent
pip install -r custom_tools/requirements.txt
```

### 2. Configure Environment

```bash
cp custom_tools/.env.example .env
# Edit .env with your RPC URLs and API keys
```

### 3. Load Environment

```bash
# Using python-dotenv (automatic in scripts)
# Or manually:
export $(grep -v '^#' .env | xargs)
```

## Module Usage

### check_wallet.py - Balance Checker

```bash
# Check single address
python -m custom_tools.check_wallet 0xYourAddress

# Check on specific chain
python -m custom_tools.check_wallet 0xYourAddress --chain base

# Check all configured chains
python -m custom_tools.check_wallet 0xYourAddress --all-chains
```

### nft_contract_check.py - Contract Info

```bash
# Check ERC721 contract
python -m custom_tools.nft_contract_check 0xContractAddress

# Check on Base chain
python -m custom_tools.nft_contract_check 0xContractAddress --chain base
```

### check_token_owner.py - Token Ownership

```bash
# Check single token
python -m custom_tools.check_token_owner 0xContract 42

# Check range
python -m custom_tools.check_token_owner 0xContract 1-100

# Check specific IDs
python -m custom_tools.check_token_owner 0xContract 1,5,10,42
```

### unminted_scanner.py - Find Unminted Tokens

```bash
# Scan range 1-1000
python -m custom_tools.unminted_scanner 0xContract 1 1000

# With custom delay and chain
python -m custom_tools.unminted_scanner 0xContract 1 5000 --chain base --delay 0.1

# Export results
python -m custom_tools.unminted_scanner 0xContract 1 1000 --csv results.csv --json-out results.json
```

### contract_analyzer.py - Detect Mint Functions

```bash
# Full analysis
python -m custom_tools.contract_analyzer 0xContractAddress

# Save analysis
python -m custom_tools.contract_analyzer 0xContractAddress --json-out analysis.json
```

### wallet_manager.py - Manage Burner Wallets

```bash
# Create new burner wallet
python -m custom_tools.wallet_manager create --label "burner1"

# List all wallets
python -m custom_tools.wallet_manager list

# Check balance
python -m custom_tools.wallet_manager balance --label "burner1" --chain ethereum

# Import from CSV (format: label,private_key)
python -m custom_tools.wallet_manager import-csv --file wallets.csv
```

### mint_planner.py - Plan Mint Transaction

```bash
# Plan mint (DRY_RUN by default)
python -m custom_tools.mint_planner 0xContract --wallet burner1

# With quantity
python -m custom_tools.mint_planner 0xContract --wallet burner1 --quantity 3

# Override function and price
python -m custom_tools.mint_planner 0xContract --wallet burner1 --function publicMint --price-wei 50000000000000000
```

### approval_queue.py - Transaction Approval

```bash
# List all entries
python -m custom_tools.approval_queue list

# List pending only
python -m custom_tools.approval_queue list --status pending

# Approve
python -m custom_tools.approval_queue approve --id 1

# Reject
python -m custom_tools.approval_queue reject --id 1 --reason "Too expensive"
```

### mint_executor.py - Execute Approved Transactions

```bash
# Execute single (requires DRY_RUN=false)
python -m custom_tools.mint_executor --id 1

# Execute all approved
python -m custom_tools.mint_executor --all
```

### batch_multi_wallet_executor.py - Multi-Wallet Batch

```bash
# Plan batch mint
python -m custom_tools.batch_multi_wallet_executor \
  --contract 0xAddress \
  --wallets burner1,burner2,burner3

# From CSV
python -m custom_tools.batch_multi_wallet_executor \
  --contract 0xAddress \
  --wallet-csv wallets.csv \
  --quantity 2

# With auto-approve (DANGEROUS - use with caution)
python -m custom_tools.batch_multi_wallet_executor \
  --contract 0xAddress \
  --wallets burner1,burner2 \
  --auto-approve \
  --concurrency 2
```

### telegram_gateway - Approval Bot (Future)

```bash
# Start bot (placeholder)
python -m custom_tools.telegram_gateway.bot
```

## Workflow Example

```bash
# 1. Analyze contract
python -m custom_tools.contract_analyzer 0xNFTContract --chain base

# 2. Scan for unminted tokens
python -m custom_tools.unminted_scanner 0xNFTContract 1 1000 --chain base

# 3. Create burner wallet
python -m custom_tools.wallet_manager create --label "mint_wallet_1"

# 4. Fund wallet (manually send ETH)

# 5. Plan mint transaction
python -m custom_tools.mint_planner 0xNFTContract --wallet mint_wallet_1 --chain base

# 6. Review and approve
python -m custom_tools.approval_queue approve --id 1

# 7. Execute (set DRY_RUN=false)
DRY_RUN=false python -m custom_tools.mint_executor --id 1
```

## PM2 / Systemd Startup

### PM2 (for Telegram bot - future)

```json
// ecosystem.config.js
{
  "apps": [{
    "name": "hermes-telegram-bot",
    "script": "python",
    "args": "-m custom_tools.telegram_gateway.bot",
    "cwd": "/path/to/hermes-agent",
    "env": {
      "TELEGRAM_BOT_TOKEN": "your-token",
      "TELEGRAM_ALLOWED_USERS": "123456789"
    }
  }]
}
```

```bash
pm2 start ecosystem.config.js
pm2 save
pm2 startup
```

### Systemd

```ini
# /etc/systemd/system/hermes-telegram.service
[Unit]
Description=Hermes Telegram Approval Bot
After=network.target

[Service]
Type=simple
User=hermes
WorkingDirectory=/path/to/hermes-agent
EnvironmentFile=/path/to/hermes-agent/.env
ExecStart=/usr/bin/python -m custom_tools.telegram_gateway.bot
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable hermes-telegram
sudo systemctl start hermes-telegram
```

## Security Notes

1. **Private Keys**: Stored encrypted with Fernet. Never in plaintext, never in logs.
2. **Wallet Files**: Stored in `.wallets/` with 0600 permissions.
3. **Approval Queue**: SQLite in `.data/` - all actions logged with timestamps.
4. **Telegram**: Only `TELEGRAM_ALLOWED_USERS` can approve transactions.
5. **Git Safety**: Add `.wallets/`, `.data/`, `.env` to `.gitignore`.

## Future Roadmap

- [ ] Full Telegram bot implementation (approve/reject via inline buttons)
- [ ] OWS (Open Wallet Standard) integration
- [ ] Multi-chain batch operations
- [ ] Gas optimization strategies
- [ ] MEV protection integration
- [ ] Autonomous workflow scheduler
- [ ] Web dashboard for approval queue
