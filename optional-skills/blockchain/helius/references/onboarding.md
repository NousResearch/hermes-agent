# Onboarding — Account Setup, API Keys & Plans

## What This Covers

Getting users set up with Helius: creating accounts, obtaining API keys, understanding plans, and managing billing. There are three paths to get an API key, plus SDK-based signup for applications.

## MCP Tools

| MCP Tool | What It Does |
|---|---|
| `setHeliusApiKey` | Configure an existing API key for the session (validates against `getBlockHeight`) |
| `generateKeypair` | Generate or load a Solana keypair for agentic signup (persists to `~/.helius-cli/keypair.json`) |
| `checkSignupBalance` | Check if the signup wallet has sufficient SOL + USDC |
| `agenticSignup` | Create a Helius account, pay with USDC, auto-configure API key |
| `getAccountStatus` | Check current plan, credits remaining, rate limits, billing cycle, burn-rate projections |
| `getHeliusPlanInfo` | View plan details — pricing, credits, rate limits, features |
| `compareHeliusPlans` | Compare plans side-by-side by category (rates, features, connections, pricing, support) |
| `previewUpgrade` | Preview upgrade pricing with proration before committing |
| `upgradePlan` | Execute a plan upgrade (processes USDC payment) |
| `payRenewal` | Pay a renewal payment intent |

## Getting an API Key

### Path A: Existing Key (Fastest)

If the user already has a Helius API key from the dashboard:

1. Use the `setHeliusApiKey` MCP tool with their key
2. The tool validates the key against `getBlockHeight`, then persists it to shared config
3. All Helius MCP tools are immediately available

If the environment variable `HELIUS_API_KEY` is already set, no action is needed — tools auto-detect it.

### Path B: MCP Agentic Signup (For AI Agents)

The fully autonomous signup flow, no browser needed:

1. **`generateKeypair`** — generates a new Solana keypair (or loads an existing one from `~/.helius-cli/keypair.json`). Returns the wallet address.
2. **User funds the wallet** with:
   - ~0.001 SOL for transaction fees
   - 1 USDC for the basic plan (or more for paid plans: $49 Developer, $499 Business, $999 Professional)
3. **`checkSignupBalance`** — verifies SOL and USDC balances are sufficient
4. **`agenticSignup`** — creates the account, processes USDC payment, returns API key + RPC endpoints + project ID
   - API key is automatically configured for the session and saved to shared config
   - If the wallet already has an account, it detects and returns existing credentials (no double payment)

**Parameters for `agenticSignup`:**
- `plan`: `"basic"` (default, $1), `"developer"`, `"business"`, or `"professional"`
- `period`: `"monthly"` (default) or `"yearly"` (paid plans only)
- `email`, `firstName`, `lastName`: required for paid plans
- `couponCode`: optional discount code

Here, paid plans refers to `"developer"`, `"business"`, and `"professional"`

### Path C: Helius CLI

The `helius-cli` provides the same autonomous signup from the terminal:

```bash
# Generate keypair (saved to ~/.helius-cli/keypair.json)
helius keygen

# Fund the wallet, then sign up (pays 1 USDC for basic plan)
helius signup --json

# List projects and get API keys
helius projects --json
helius apikeys <project-id> --json

# Get RPC endpoints
helius rpc <project-id> --json
```

**CLI exit codes** (for error handling in scripts):
- `0`: success
- `10`: not logged in (run `helius login`)
- `11`: keypair not found (run `helius keygen`)
- `20`: insufficient SOL
- `21`: insufficient USDC

Always use the `--json` flag for machine-readable output when scripting.

### SDK In-Process Signup

For applications that need to create Helius accounts programmatically:

```typescript
const helius = createHelius({ apiKey: '' }); // No key yet — signing up

const keypair = await helius.auth.generateKeypair();
const address = await helius.auth.getAddress(keypair);

// Fund the wallet (user action), then sign up
const result = await helius.auth.agenticSignup({
  secretKey: keypair.secretKey,
  plan: 'developer',
  period: 'monthly',
  email: 'user@example.com',
  firstName: 'Jane',
  lastName: 'Doe',
});
// result.apiKey, result.projectId, result.endpoints, result.jwt
```

## Plans and Pricing

The agentic signup flow uses these plan tiers (all paid in USDC):

| | Basic | Developer | Business | Professional |
|---|---|---|---|---|
| **Price** | $1 USDC | $49/mo | $499/mo | $999/mo |
| **Credits** | 1M | 10M | 100M | 200M |
| **Extra credits** | N/A | $5/M | $5/M | $5/M |
| **RPC RPS** | 10 | 50 | 200 | 500 |
| **sendTransaction** | 1/s | 5/s | 50/s | 100/s |
| **DAS** | 2/s | 10/s | 50/s | 100/s |
| **WS connections** | 5 | 150 | 250 | 250 |
| **Enhanced WS** | No | No | 100 conn | 100 conn |
| **LaserStream** | No | Devnet | Devnet | Full (mainnet + devnet) |
| **Support** | Discord | Chat (24hr) | Priority (12hr) | Slack + Telegram (8hr) |

The dashboard shows a "Free" tier at $0 — that is the same plan as Basic, but agentic signup charges $1 USDC to create the account on-chain.

### Credit Costs

- **0 credits**: Helius Sender (sendSmartTransaction, sendJitoBundle)
- **1 credit**: Standard RPC calls, sendTransaction, Priority Fee API, webhook events
- **3 credits**: per 0.1 MB streamed (LaserStream, Enhanced WebSockets)
- **10 credits**: getProgramAccounts, DAS API, historical data
- **100 credits**: Enhanced Transactions API, Wallet API, webhook management

### Feature Availability by Plan

| Feature | Minimum Plan |
|---|---|
| Standard RPC, DAS, Webhooks, Sender | Basic |
| Standard WebSockets | Basic |
| Enhanced WebSockets | Business |
| LaserStream (devnet) | Developer |
| LaserStream (mainnet) | Professional |
| LaserStream data add-ons | Professional ($500+/mo) |

Use the `getHeliusPlanInfo` or `compareHeliusPlans` MCP tools for current details.

## Managing Accounts

### Check Account Status

The `getAccountStatus` tool provides three tiers of information:

1. **No auth**: Tells the user how to get started (set key or sign up)
2. **API key only** (no JWT): Confirms auth but can't show credit usage — suggests calling `agenticSignup` to detect existing account
3. **Full JWT session**: Shows plan, rate limits, credit usage breakdown (API/RPC/webhooks/overage), billing cycle with days remaining, and burn-rate projections with warnings

Call `getAccountStatus` before bulk operations to verify sufficient credits.

### Upgrade Plans

1. **`previewUpgrade`** — shows pricing breakdown: subtotal, prorated credits, discounts, coupon status, amount due today
2. **`upgradePlan`** — executes the upgrade, processes USDC payment from the signup wallet
   - Requires `email`, `firstName`, `lastName` for first-time upgrades (all three or none)
   - Supports `couponCode` for discounts

### Pay Renewals

`payRenewal` takes a `paymentIntentId` from a renewal notification and processes the USDC payment.

## Environment Configuration

```bash
# Required — set one of these:
HELIUS_API_KEY=your-api-key          # Environment variable
# OR use setHeliusApiKey MCP tool    # Session + shared config
# OR use agenticSignup               # Auto-configures

# Optional
HELIUS_NETWORK=mainnet-beta          # or devnet (default: mainnet-beta)
```

### Shared Config

The MCP persists API keys and JWTs to shared config files so they survive across sessions:
- **API key**: saved to shared config path (accessible by both MCP and CLI)
- **Keypair**: saved to `~/.helius-cli/keypair.json`
- **JWT**: saved to shared config for authenticated session features

### Installing the MCP

```bash
claude mcp add helius npx helius-mcp@latest
```

## Choosing the Right Setup Path

| Scenario | Path |
|---|---|
| User has a Helius API key | `setHeliusApiKey` (Path A) |
| User has `HELIUS_API_KEY` env var set | No action needed — auto-detected |
| AI agent needs to sign up autonomously | `generateKeypair` -> fund -> `agenticSignup` (Path B) |
| Script/CI needs to sign up | `helius keygen` -> fund -> `helius signup --json` (Path C) |
| Application needs programmatic signup | SDK `agenticSignup()` function |
| User wants full account visibility | `agenticSignup` (detects existing accounts) then `getAccountStatus` |
| User needs a higher plan | `previewUpgrade` then `upgradePlan` |

## Common Mistakes

- Calling `agenticSignup` without first calling `generateKeypair` — there's no wallet to sign with
- Not funding the wallet before calling `agenticSignup` — the USDC payment will fail
- Assuming `agenticSignup` charges twice for existing accounts — it detects and returns existing credentials
- Using `getAccountStatus` without a JWT session — call `agenticSignup` first to establish the session (it detects existing accounts for free)
- Forgetting that paid plan signup requires `email`, `firstName`, and `lastName` — all three are required together
