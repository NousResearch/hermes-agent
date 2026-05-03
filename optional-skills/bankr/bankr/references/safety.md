# Safety & Access Control Reference

Comprehensive safety guidance for building agents and integrations with the Bankr API and CLI. Covers API key types, access controls, wallet separation, rate limits, and operational best practices.

## API Key Types & Separation

Bankr uses a single key format (`bk_...`) with **capability flags** that control what each key can access. You can optionally configure a separate key for the LLM Gateway.

### Capability Flags

Each API key has independent toggles managed at [bankr.bot/api](https://bankr.bot/api):

| Flag | Controls Access To | Default |
|------|-------------------|---------|
| `walletApiEnabled` | `/wallet/*` write endpoints (transfer, sign, submit) | true |
| `agentApiEnabled` | `/agent/*` AI endpoints (prompt, job status, profile) | false |
| `tokenLaunchApiEnabled` | Token deployment (`/token-launches/deploy`) and agent deploy tool | true |
| `llmGatewayEnabled` | LLM Gateway at `llm.bankr.bot` (chat completions, model access) | false |
| `readOnly` | When true, restricts Wallet/Agent API to read-only tools | true |

A single key can have multiple capabilities enabled (e.g., both Agent API and LLM Gateway).

### Agent API Key vs LLM Gateway Key

For most users, **one key works for both** the Agent API and LLM Gateway. However, you can configure a separate LLM key when you want different permissions or rate limits for each:

| Config | Agent API Key | LLM Gateway Key |
|--------|--------------|-----------------|
| Environment variable | `BANKR_API_KEY` | `BANKR_LLM_KEY` (falls back to `BANKR_API_KEY`) |
| CLI config key | `apiKey` | `llmKey` (falls back to `apiKey`) |
| Used by | `bankr agent`, `/agent/*` endpoints | `bankr llm claude`, `llm.bankr.bot` |

**When to use separate keys:**
- Your agent API key is read-only but your LLM key needs no such restriction (LLM calls are inherently read-only)
- You want to revoke LLM access without affecting agent operations (or vice versa)
- Different keys for different team members or environments

**Setting a separate LLM key:**
```bash
bankr login --api-key bk_AGENT_KEY --llm-key bk_LLM_KEY   # during login
bankr config set llmKey bk_LLM_KEY                         # after login
```

For full LLM Gateway setup details, see [llm-gateway.md](llm-gateway.md).

## API Key Access Control

Bankr API keys support granular access control configured at [bankr.bot/api](https://bankr.bot/api). Two key security features: **read-only mode** and **IP whitelisting**.

### Read-Only API Keys

When an API key has `readOnly: true`, all write tools are filtered from the agent session. The agent receives a system directive explaining the restriction and will inform users accordingly.

**Behavior by endpoint:**

| Endpoint | Read-Only Behavior |
|----------|-------------------|
| `POST /agent/prompt` | Works — but only read tools are available (balances, prices, analytics, portfolio, research) |
| `POST /agent/sign` | Blocked — returns 403 |
| `POST /agent/submit` | Blocked — returns 403 |
| `GET /agent/job/{jobId}` | Works — unaffected |
| `POST /agent/job/{jobId}/cancel` | Works — unaffected |

**403 error responses:**

For `/agent/sign`:
```json
{
  "error": "Read-only API key",
  "message": "This API key has read-only access and cannot sign messages or transactions. Update your API key permissions at https://bankr.bot/api"
}
```

For `/agent/submit`:
```json
{
  "error": "Read-only API key",
  "message": "This API key has read-only access and cannot submit transactions. Update your API key permissions at https://bankr.bot/api"
}
```

**Write tool categories filtered in read-only mode:**

| Category | Examples |
|----------|----------|
| Swaps | Token buy/sell/swap across all chains |
| Transfers | Send tokens, NFTs |
| NFT Operations | Purchase, mint NFTs |
| Staking | Stake/unstake operations |
| Orders | Limit orders, stop losses |
| Token Launches | Deploy ERC20/SPL tokens |
| Leverage | Open/close/modify positions |
| Polymarket | Place/redeem bets |
| Claims | Claim rewards, fees |

The agent receives a system directive and will explain the restriction if a user requests a write operation:

> *"This session has READ-ONLY API access. You can retrieve information (balances, prices, analytics, portfolio data, market research) but CANNOT execute any transactions."*

### IP Whitelisting

API keys support an `allowedIps` whitelist with both individual IPs and CIDR ranges. When configured, requests from non-whitelisted IPs are rejected at the authentication layer before reaching any endpoint.

- **Empty array** (`[]`) = all IPs allowed (default)
- **Non-empty array** = only listed IPs/ranges can use the key
- **CIDR notation** supported (e.g., `10.0.0.0/24`, `192.168.1.0/16`)

**403 error response:**
```json
{
  "error": "IP address not allowed",
  "message": "IP address not allowed for this API key"
}
```

### Configuring Access Control

Manage API key settings at [bankr.bot/api](https://bankr.bot/api):

| Field | Type | Description |
|-------|------|-------------|
| `readOnly` | boolean | When true, only read tools are available |
| `allowedIps` | string[] | IP/CIDR whitelist (e.g., `["1.2.3.4", "10.0.0.0/24"]`, empty = all allowed) |
| `walletApiEnabled` | boolean | Whether `/wallet/*` write endpoints are accessible |
| `agentApiEnabled` | boolean | Whether `/agent/*` AI endpoints are accessible |
| `tokenLaunchApiEnabled` | boolean | Whether token deployment is accessible |
| `llmGatewayEnabled` | boolean | Whether LLM Gateway endpoints are accessible |

## CLI Security

The Bankr CLI (`@bankr/cli`) stores credentials locally and provides its own safety considerations alongside the REST API.

### Credential Storage

The CLI stores keys in `~/.bankr/config.json`:

```json
{
  "apiKey": "bk_...",
  "llmKey": "bk_...",
  "apiUrl": "https://api.bankr.bot",
  "llmUrl": "https://llm.bankr.bot"
}
```

**Safety rules for CLI credentials:**
- Add `~/.bankr/` to your global `.gitignore` — never commit this directory
- On shared machines, restrict file permissions: `chmod 600 ~/.bankr/config.json`
- Use `bankr logout` to clear stored credentials when done on a shared machine
- For CI/CD, prefer environment variables (`BANKR_API_KEY`, `BANKR_LLM_KEY`) over config files

### Non-Interactive Login

When running the CLI in automated scripts or AI agent environments where interactive prompts aren't possible:

```bash
# Direct key login — no prompts
bankr login --api-key bk_YOUR_KEY

# With separate LLM key
bankr login --api-key bk_AGENT_KEY --llm-key bk_LLM_KEY

# Verify it worked
bankr whoami
```

### CLI vs REST API Access Controls

Access controls (read-only, IP whitelist) apply identically whether you use the CLI or REST API — they are enforced server-side on the API key itself. The CLI is a convenience wrapper; it submits the same requests as direct API calls.

```bash
# These two are equivalent — same access controls apply
bankr agent "What is my balance?"
curl -X POST "https://api.bankr.bot/agent/prompt" \
  -H "X-API-Key: bk_YOUR_KEY" \
  -d '{"prompt": "What is my balance?"}'
```

## Dedicated Agent Wallet

When building autonomous agents that execute transactions, use a **separate Bankr account** as the agent's wallet rather than your personal account. This limits blast radius — if an agent key is compromised or the agent misbehaves, only the dedicated wallet's funds are at risk.

### Why Separate Wallets

- **Limited exposure**: A compromised agent key only exposes the agent wallet's funds, not your main holdings
- **Clear accounting**: Agent transactions are isolated from personal activity
- **Independent controls**: Apply stricter access controls (read-only, IP whitelist) without affecting personal use
- **Easy revocation**: Disable the agent account without disrupting your primary wallet

### Setup Steps

1. **Create a new Bankr account** — Sign up at [bankr.bot/api](https://bankr.bot/api) with a different email. This provisions fresh EVM and Solana wallets automatically.
2. **Generate an API key** — Enable **Agent API** access for the key
3. **Configure access controls** — Set `readOnly`, `allowedIps`, or both as appropriate for your use case
4. **Fund with limited amounts** — Transfer only what the agent needs for its operations

### Recommended Funding

Fund the agent wallet with enough for gas and intended operations, not more:

| Chain | Gas Buffer | Trading Capital |
|-------|-----------|-----------------|
| Base | 0.01 - 0.05 ETH | As needed for trades |
| Polygon | 5 - 10 MATIC | As needed for trades |
| Ethereum | 0.05 - 0.1 ETH | As needed for trades |
| Solana | 0.1 - 0.5 SOL | As needed for trades |

Replenish periodically rather than pre-loading large amounts.

### Access Control Combinations

Choose the right combination based on your agent's purpose:

| Use Case | readOnly | allowedIps | Funding Level |
|----------|----------|------------|---------------|
| Monitoring / analytics bot | Yes | Yes (server IP) | None needed |
| Trading bot (server-side) | No | Yes (server IP) | Limited trading capital |
| Development / testing | No | No | Minimal (test amounts) |
| Read-only research agent | Yes | No | None needed |

## Rate Limits

### Daily Message Limits

The `/agent/prompt` endpoint enforces daily message limits per account:

| Tier | Daily Limit |
|------|-------------|
| Standard | 100 messages/day |
| Bankr Club | 1,000 messages/day |
| Custom | Set per API key |

**429 response when limit exceeded:**
```json
{
  "error": "Daily limit exceeded",
  "message": "You have reached your daily API limit of 100 messages. Upgrade to Bankr Club for 1000 messages/day. Resets at 2025-01-15T12:00:00.000Z",
  "resetAt": 1736942400000,
  "limit": 100,
  "used": 100
}
```

The reset window is **24 hours from the first message** (rolling window), not a fixed midnight reset. The `resetAt` field in the response tells you exactly when the counter resets.

### General API Rate Limits

| Scope | Limit | Window |
|-------|-------|--------|
| Public endpoints | 100 requests | 15 minutes per IP |
| General API | 120 requests | 1 minute per IP |
| External orders | 10 requests | 1 second per API key |

For error response handling, retry strategies, and exponential backoff guidance, see [error-handling.md](error-handling.md).

## Transaction Safety

Blockchain transactions are **irreversible** once confirmed. Key safety rules:

- **Test first** — Always test with small amounts before scaling up. Use Base or Polygon for low-cost testing.
- **Verify recipients** — Double-check addresses before transfers. See [transfers.md](transfers.md) for address resolution details.
- **Gas buffer** — Keep enough native tokens for gas on each chain you operate on. See the funding table above for recommended minimums.
- **Wait for confirmation** — Use `waitForConfirmation: true` with `/agent/submit` to ensure transactions are confirmed before proceeding. See [sign-submit-api.md](sign-submit-api.md).
- **Immediate execution** — `/agent/submit` executes transactions immediately with no confirmation prompt. For safety with the prompt API, the AI agent may ask for confirmation on large or unusual operations.
- **Understand calldata** — When using arbitrary transactions, verify the calldata source is trusted. See [arbitrary-transaction.md](arbitrary-transaction.md).

## Key Management

### Storage

- **Environment variables** — Store API keys in `BANKR_API_KEY` and LLM keys in `BANKR_LLM_KEY`, never in source code
- **CLI config** — The CLI stores keys in `~/.bankr/config.json`. Ensure this directory is in `.gitignore` and has restricted permissions
- **Never commit secrets** — Add `~/.bankr/`, `.env`, and credential files to `.gitignore`. Use `bankr logout` to clear CLI credentials on shared machines

### Rotation & Revocation

- **Rotate periodically** — Rotate keys via the dashboard at [bankr.bot/api](https://bankr.bot/api) or programmatically via the API key rotation endpoint. Rotation atomically generates a new key and deactivates the old one. After rotating, update both env vars and CLI config (`bankr login --api-key NEW_KEY`)
- **Revoke immediately** — If any key (API or LLM) is leaked, deactivate it immediately at the dashboard
- **One key per purpose** — Use separate keys for different agents, environments, and services (Agent API vs LLM Gateway) so you can revoke individually without disrupting unrelated systems

### Best Practices

- Prefer environment variables for server-side agents and CI/CD; use CLI config for local development
- If you use separate API and LLM keys, rotate them independently
- When revoking a compromised key, check both `BANKR_API_KEY` and `BANKR_LLM_KEY` — if the same key was used for both, both need updating

For the full API key setup and authentication workflow, see [api-workflow.md](api-workflow.md).

## Safety by Feature

Each feature has specific safety considerations documented in its reference file:

| Feature | Key Safety Points | Reference |
|---------|-------------------|-----------|
| Leverage Trading | Risk warnings, liquidation, position sizing | [leverage-trading.md](leverage-trading.md) |
| Transfers | Verify recipient address, ENS resolution | [transfers.md](transfers.md) |
| NFT Operations | Collection verification, floor price checks | [nft-operations.md](nft-operations.md) |
| Polymarket | Responsible betting, position limits | [polymarket.md](polymarket.md) |
| Token Deployment | Legal considerations, rate limits | [token-deployment.md](token-deployment.md) |
| Automation | Monitoring active orders, execution conditions | [automation.md](automation.md) |
| Arbitrary Transactions | Trust calldata source, verify contract targets | [arbitrary-transaction.md](arbitrary-transaction.md) |
| Sign & Submit API | Immediate execution, no confirmation prompt | [sign-submit-api.md](sign-submit-api.md) |

## Checklist

Before deploying an agent or integration:

- [ ] Use a **dedicated agent wallet** — not your personal account
- [ ] Fund the agent wallet with **limited amounts** appropriate to its purpose
- [ ] Set API key to **read-only** if the agent only needs to query data
- [ ] Configure **IP whitelisting** for server-side agents with known IPs
- [ ] Store keys in **environment variables** (`BANKR_API_KEY`, `BANKR_LLM_KEY`), never in source code or version control
- [ ] If using the CLI, ensure `~/.bankr/` is in `.gitignore` and has restricted file permissions
- [ ] Use **separate keys** for Agent API vs LLM Gateway if they need independent access controls or revocation
- [ ] **Test with small amounts** on low-cost chains (Base, Polygon) before production use
- [ ] Verify **recipient addresses** in any transfer logic before execution
- [ ] Implement **error handling** for rate limits (429) and access control errors (403)
- [ ] Monitor the agent's **daily message usage** against your tier limit
- [ ] Review and **rotate all keys** (API and LLM) periodically; revoke immediately if compromised
