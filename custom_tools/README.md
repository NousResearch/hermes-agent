# Hermes Web3 Tools - NFT Mint Automation Ecosystem

Modular Web3 tools extending [Hermes Agent](https://github.com/NousResearch/hermes-agent) with NFT minting workflow and automation capabilities.

## Architecture

```
hermes-agent/              (AI core - UNTOUCHED)
custom_tools/              (Web3 extension modules)
  __init__.py
  check_wallet.py          - Multi-chain ETH balance checker
  nft_contract_check.py    - ERC721 contract info
  check_token_owner.py     - ownerOf / minted detection
  unminted_scanner.py      - Bulk unminted token scanner
  contract_analyzer.py     - ABI loader, mint fn/price detection
  wallet_manager.py        - Encrypted burner wallet management
  mint_planner.py          - Tx planner + auto-queue pending
  approval_queue.py        - SQLite approval queue
  mint_executor.py         - Execute approved-only transactions
  batch_multi_wallet_executor.py - Multi-wallet batch mint
  telegram_gateway/
    __init__.py
    bot.py                 - Telegram approval bot (full)
  ecosystem.config.js      - PM2 startup config
  requirements.txt
  .env.example
```

## Safety First

**CRITICAL DEFAULTS:**
- `DRY_RUN=true` — No transactions sent by default
- All transactions require explicit approval (CLI or Telegram)
- Private keys are NEVER logged, printed, or exposed
- Multi-wallet is for YOUR OWN burner wallets only
- Telegram bot does NOT auto-execute (approval only)

**NEVER:**
- Bypass allowlist/captcha/anti-bot/signature protection
- Expose or print private keys
- Auto-send transactions without explicit approval

---

## Quick Setup

### 1. Install Dependencies

```bash
cd hermes-agent
pip install -r custom_tools/requirements.txt
```

### 2. Configure Environment

```bash
cp custom_tools/.env.example .env
# Edit .env with your values
```

### 3. Load Environment

```bash
export $(grep -v '^#' .env | xargs)
```

---

## Full Workflow

### Step 1: Create Burner Wallets

```bash
python -m custom_tools.wallet_manager create --label "burner1"
python -m custom_tools.wallet_manager create --label "burner2"
python -m custom_tools.wallet_manager create --label "burner3"
python -m custom_tools.wallet_manager list
```

### Step 2: Analyze NFT Contract

```bash
# Plan mint and auto-save to approval queue (default --queue)
python -m custom_tools.mint_planner 0xContract --wallet burner1

# With quantity
python -m custom_tools.mint_planner 0xContract --wallet burner1 --quantity 3

# Override function and price
python -m custom_tools.mint_planner 0xContract --wallet burner1 --function mint --price-wei 0

# Preview only, do NOT save to queue
python -m custom_tools.mint_planner 0xContract --wallet burner1 --no-queue
```

> **Note:** By default, `mint_planner` saves the plan to the approval queue
> with `status=pending` and prints the generated approval ID. Use `--no-queue`
> for preview-only mode.

### approval_queue.py - Transaction Approval

```bash
python -m custom_tools.approval_queue list
python -m custom_tools.approval_queue list --status pending
```

### Step 6: Approve (CLI or Telegram)

```bash
# CLI
python -m custom_tools.approval_queue approve --id 1

# Or via Telegram bot:
# /approve 1
```

### Step 7: Execute Approved Transaction

```bash
# Dry run (default - simulates)
python -m custom_tools.mint_executor --id 1

# Real execution
DRY_RUN=false python -m custom_tools.mint_executor --id 1

# Execute all approved
DRY_RUN=false python -m custom_tools.mint_executor --all
```

---

## Batch Multi-Wallet Flow

### Create wallet CSV (`wallets.csv`):
```csv
burner1
burner2
burner3
```

### Plan + Queue for all wallets:
```bash
python -m custom_tools.batch_multi_wallet_executor \
  --contract 0xNFTContract \
  --wallet-csv wallets.csv \
  --chain base \
  --function mint \
  --price-wei 0 \
  --quantity 1
```
This queues each wallet as a separate PENDING approval entry.

### Approve all via CLI:
```bash
python -m custom_tools.approval_queue approve --id 1
python -m custom_tools.approval_queue approve --id 2
python -m custom_tools.approval_queue approve --id 3
```

### Or auto-approve + execute:
```bash
python -m custom_tools.batch_multi_wallet_executor \
  --contract 0xNFTContract \
  --wallet-csv wallets.csv \
  --chain base \
  --function mint \
  --price-wei 0 \
  --auto-approve \
  --execute \
  --concurrency 2 \
  --min-delay 1.0 \
  --max-delay 3.0 \
  --report batch_report.json
```

---

## Telegram Approval Bot

### Setup:
1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Get your Telegram user ID via [@userinfobot](https://t.me/userinfobot)
3. Set in `.env`:
```env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_ALLOWED_USERS=123456789,987654321
```

### Start Bot:
```bash
python -m custom_tools.telegram_gateway.bot
```

### Bot Commands:
| Command | Description |
|---------|-------------|
| `/start` | Welcome + status |
| `/pending` | List pending approvals with Approve/Reject buttons |
| `/approve <id>` | Approve entry by ID |
| `/reject <id> [reason]` | Reject entry with optional reason |
| `/status <id>` | Check entry status |

### Inline Buttons:
- ✅ **Approve** — Approve the entry
- ❌ **Reject** — Reject the entry
- 👁 **Dry Run Preview** — Show full preview without action

> **Note:** The Telegram bot only approves/rejects. It does NOT execute transactions.
> Use `mint_executor` to execute after approval.

---

## PM2 Startup

### Start with PM2:
```bash
# 1. Analyze contract
python -m custom_tools.contract_analyzer 0xNFTContract --chain base

# 2. Scan for unminted tokens
python -m custom_tools.unminted_scanner 0xNFTContract 1 1000 --chain base

# 3. Create burner wallet
python -m custom_tools.wallet_manager create --label "test1"

# 4. Fund wallet (manually send ETH)

# 5. Plan mint transaction (auto-queues as PENDING)
python -m custom_tools.mint_planner 0xNFTContract --wallet test1 --function mint --price-wei 0
#    -> Shows preview + prints: "Queued as PENDING approval ID #1"

# 6. Verify it's in the queue
python -m custom_tools.approval_queue list

# 7. Approve
python -m custom_tools.approval_queue approve --id 1

# 8. Execute (set DRY_RUN=false)
DRY_RUN=false python -m custom_tools.mint_executor --id 1
```

### Key Rules:
- `mint_planner` **plans and queues** but NEVER sends transactions.
- `mint_executor` **only executes entries with status=approved**.
- `DRY_RUN=true` is the default — executor will simulate even if approved.
- Set `DRY_RUN=false` explicitly to send real transactions.

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

### PM2 Commands:
```bash
pm2 status                          # Check running processes
pm2 logs hermes-telegram-bot        # View bot logs
pm2 restart hermes-telegram-bot     # Restart bot
pm2 stop hermes-telegram-bot        # Stop bot
pm2 delete hermes-telegram-bot      # Remove from PM2
```

### Systemd Alternative:
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
sudo journalctl -u hermes-telegram -f
```

---

## Module Reference

### check_wallet.py
```bash
python -m custom_tools.check_wallet 0xAddress
python -m custom_tools.check_wallet 0xAddress --chain base
python -m custom_tools.check_wallet 0xAddress --all-chains
```

### nft_contract_check.py
```bash
python -m custom_tools.nft_contract_check 0xContract --chain base
```

### check_token_owner.py
```bash
python -m custom_tools.check_token_owner 0xContract 42
python -m custom_tools.check_token_owner 0xContract 1-100
```

### unminted_scanner.py
```bash
python -m custom_tools.unminted_scanner 0xContract 1 1000 --chain base --delay 0.1
python -m custom_tools.unminted_scanner 0xContract 1 5000 --csv results.csv --json-out results.json
```

### contract_analyzer.py
```bash
python -m custom_tools.contract_analyzer 0xContract --chain base --json-out analysis.json
```

### wallet_manager.py
```bash
python -m custom_tools.wallet_manager create --label "burner1"
python -m custom_tools.wallet_manager list
python -m custom_tools.wallet_manager balance --label "burner1" --chain ethereum
python -m custom_tools.wallet_manager import-csv --file wallets_import.csv
```

### mint_planner.py
```bash
python -m custom_tools.mint_planner 0xContract --wallet burner1 --function mint --price-wei 0
python -m custom_tools.mint_planner 0xContract --wallet burner1 --no-queue
```

### approval_queue.py
```bash
python -m custom_tools.approval_queue list
python -m custom_tools.approval_queue list --status pending
python -m custom_tools.approval_queue approve --id 1
python -m custom_tools.approval_queue reject --id 1 --reason "Too expensive"
```

### mint_executor.py
```bash
python -m custom_tools.mint_executor --id 1
DRY_RUN=false python -m custom_tools.mint_executor --id 1
DRY_RUN=false python -m custom_tools.mint_executor --all
```

### batch_multi_wallet_executor.py
```bash
python -m custom_tools.batch_multi_wallet_executor --contract 0xAddr --wallets w1,w2,w3
python -m custom_tools.batch_multi_wallet_executor --contract 0xAddr --wallet-csv wallets.csv
python -m custom_tools.batch_multi_wallet_executor --contract 0xAddr --wallet-csv wallets.csv --auto-approve --execute
```

---

## Key Rules

1. `mint_planner` **plans and queues** but NEVER sends transactions.
2. `mint_executor` **only executes entries with status=approved**.
3. `DRY_RUN=true` is the default — executor simulates even if approved.
4. Set `DRY_RUN=false` explicitly to send real transactions.
5. Telegram bot **only approves/rejects** — never executes.
6. Private keys stored encrypted, never in logs.

---

## Security Notes

| Concern | Protection |
|---------|-----------|
| Private Keys | Fernet encrypted in `.wallets/`, 0600 permissions |
| Approval | SQLite queue, all actions logged with timestamps |
| Telegram | Only `TELEGRAM_ALLOWED_USERS` can approve/reject |
| Execution | DRY_RUN=true default, requires explicit override |
| Git Safety | `.wallets/`, `.data/`, `.env` in `.gitignore` |

---

## Future Roadmap

- [ ] Telegram: notify on new pending entries
- [ ] Telegram: execute approved from bot (with confirmation)
- [ ] OWS (Open Wallet Standard) integration
- [ ] Multi-chain batch operations
- [ ] Gas optimization / EIP-1559 support
- [ ] MEV protection integration
- [ ] Web dashboard for approval queue
- [ ] Autonomous workflow scheduler
