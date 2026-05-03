# Veil skill

Wraps the [@veil-cash/sdk](https://github.com/veildotcash/veildotcash-sdk) CLI and optionally uses Bankr Agent API to sign & submit unsigned deposit/register transactions. Supports **ETH and USDC** privacy pools on Base.

## Assumptions

- **Veil SDK** is installed via one of these methods:

  **Option A: Global npm install (recommended)**
  ```bash
  npm install -g @veil-cash/sdk
  ```
  This makes the `veil` CLI available globally.

  **Option B: Clone from GitHub**
  ```bash
  mkdir -p ~/.openclaw/workspace/repos
  cd ~/.openclaw/workspace/repos
  git clone https://github.com/veildotcash/veildotcash-sdk.git
  cd veildotcash-sdk
  npm ci && npm run build
  ```

- Bankr skill is configured:
  - `~/.clawdbot/skills/bankr/config.json`

- Veil secrets are stored outside git:
  - `~/.clawdbot/skills/veil/.env.veil` (chmod 600)
  - `~/.clawdbot/skills/veil/.env` for `RPC_URL` (recommended — Veil queries a lot of blockchain data, so public RPCs will likely hit rate limits)

## Usage

```bash
cd veil

# Generate keypair
scripts/veil-init.sh

# Print keypair JSON
scripts/veil-keypair.sh

# Ask Bankr for address
scripts/veil-bankr-prompt.sh "What is my Base wallet address? Respond with just the address."

# Check balances (ETH pool — default)
scripts/veil-balance.sh --address 0x...

# Check balances (USDC pool)
scripts/veil-balance.sh --address 0x... --pool usdc

# Deposit via Bankr — ETH (build unsigned tx + submit)
scripts/veil-deposit-via-bankr.sh ETH 0.011 --address 0x...

# Deposit via Bankr — USDC (auto-handles approve + deposit)
scripts/veil-deposit-via-bankr.sh USDC 100 --address 0x...

# Withdraw / transfer / merge (local VEIL_KEY required)
scripts/veil-withdraw.sh ETH 0.007 0x...
scripts/veil-withdraw.sh USDC 50 0x...
scripts/veil-transfer.sh ETH 0.001 0x...
scripts/veil-transfer.sh USDC 25 0x...
scripts/veil-merge.sh ETH 0.001
scripts/veil-merge.sh USDC 100
```

## Notes

- `veil-bankr-prompt.sh` implements the same submit/poll loop as the Bankr skill, but localized here so this skill is self-contained.
- For USDC deposits via Bankr, `veil-deposit-via-bankr.sh` automatically submits the ERC20 approval transaction first, then the deposit transaction.
- All action scripts take asset as the first argument: `ETH` or `USDC`.
