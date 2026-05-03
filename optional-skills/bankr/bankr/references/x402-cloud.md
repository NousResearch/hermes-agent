# x402 Cloud Reference

x402 Cloud lets you deploy paid API endpoints that agents and developers pay for automatically using the x402 payment protocol. Write a handler, set a price, deploy with one command — callers pay in USDC on Base per request.

**Base URL:** `https://x402.bankr.bot`

**Dashboard:** [bankr.bot/x402](https://bankr.bot/x402)

**Docs:** [docs.bankr.bot/x402-cloud/overview](https://docs.bankr.bot/x402-cloud/overview)

## Pricing

| Plan | Platform Fee | Requests |
|------|-------------|----------|
| Free | 0% | Up to 1,000/month |
| Pro | 5% | Unlimited |
| Enterprise | 3% | Contact sales |

No credit card required. First 1,000 settled requests each month are free. Payments settle on-chain in USDC on Base — your share goes directly to your wallet.

## CLI Commands

All commands use the Bankr CLI (`bankr` or `bun run --cwd packages/cli start`).

```bash
bankr x402 init                     # Scaffold x402/ folder + bankr.x402.json
bankr x402 add <name>               # Add a new service
bankr x402 configure <name>         # Interactive pricing/description setup
bankr x402 deploy [name]            # Deploy all or a single service
bankr x402 list                     # List deployed services
bankr x402 logs <name>              # View request logs
bankr x402 pause <name>             # Pause a service
bankr x402 resume <name>            # Resume a service
bankr x402 delete <name>            # Delete a service
bankr x402 revenue [name]           # View earnings
bankr x402 env set KEY=VALUE        # Set encrypted env var
bankr x402 env list                 # List env var names
bankr x402 env unset KEY            # Remove env var
```

## Writing a Handler

Handlers are standard `Request → Response` functions. No framework needed.

```typescript
// x402/<service-name>/index.ts
export default async function handler(req: Request): Promise<Response> {
  return Response.json({ message: "Hello from x402!" });
}
```

### Reading Inputs

```typescript
// Query parameters
const url = new URL(req.url);
const city = url.searchParams.get("city") ?? "default";

// Request headers
const lang = req.headers.get("Accept-Language");

// JSON body (POST)
const body = await req.json();

// Form data
const form = await req.formData();
```

### Returning Responses

```typescript
// JSON (most common)
return Response.json({ data: "value" });
return Response.json({ error: "not found" }, { status: 404 });

// HTML
return new Response("<h1>Hello</h1>", {
  headers: { "Content-Type": "text/html" },
});

// Image or binary
return new Response(imageBuffer, {
  headers: { "Content-Type": "image/png" },
});

// Plain text
return new Response("Hello, world!");
```

### Using Environment Variables

```bash
bankr x402 env set API_KEY=sk_...
```

```typescript
const key = process.env.API_KEY;
```

### Using the LLM Gateway

x402 handlers can call the Bankr LLM Gateway for AI-powered features:

```typescript
const res = await fetch("https://llm.bankr.bot/v1/chat/completions", {
  method: "POST",
  headers: {
    Authorization: `Bearer ${process.env.BANKR_API_KEY}`,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "claude-sonnet-4-6",
    messages: [{ role: "user", content: "Analyze this text" }],
  }),
});
const data = await res.json();
```

Set the API key: `bankr x402 env set BANKR_API_KEY=<your-key>`

## Configuration

Service config lives in `bankr.x402.json` at the project root:

```json
{
  "network": "base",
  "services": {
    "my-service": {
      "description": "What this service does",
      "price": "0.001",
      "methods": ["GET"],
      "schema": {
        "input": { "query": "string" },
        "output": { "result": "string" }
      },
      "category": "data",
      "tags": ["keyword1", "keyword2"]
    }
  }
}
```

- **price**: USD per request (e.g. "0.001" = $0.001)
- **methods**: HTTP methods accepted (default: all). Use `["POST"]` for body-based endpoints.
- **schema**: Input/output schema for agent discovery. Agents use this to understand how to call your endpoint.
- **category/tags**: Improve discoverability for agents searching for services.

## Calling x402 Endpoints

### Via Bankr Agent (recommended for agent users)

The Bankr agent has built-in tools to discover and call x402 endpoints with automatic payment handling:

```bash
# Discover endpoints in the Bankr registry
bankr agent prompt "Find x402 endpoints for sentiment analysis"

# Call an endpoint — agent handles payment signing automatically
bankr agent prompt "Call the sentiment analysis endpoint on x402 with text 'Bitcoin is pumping'"

# Inspect an endpoint's pricing and schema before calling
bankr agent prompt "What does the x402 weather endpoint cost?"
```

The agent uses USDC on Base for all x402 payments. Maximum payment per request is $10. The agent will always confirm payment amount before calling.

### With x402-fetch (for developers)

```typescript
import { createWalletClient, http } from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { base } from "viem/chains";
import { wrapFetchWithPayment } from "x402-fetch";

const account = privateKeyToAccount("0x..." as `0x${string}`);
const wallet = createWalletClient({ account, chain: base, transport: http() });
const paidFetch = wrapFetchWithPayment(fetch, wallet, BigInt(1_000_000));

// GET
const res = await paidFetch("https://x402.bankr.bot/0xOwner/service?param=value");

// POST
const res = await paidFetch("https://x402.bankr.bot/0xOwner/service", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "hello" }),
});
```

### Inspecting Payment Requirements

```bash
curl -s https://x402.bankr.bot/0xOwner/service | jq .
```

Returns `{ x402Version, accepts: [{ scheme, network, maxAmountRequired, asset, payTo }] }`.

## Endpoint URL Format

```
https://x402.bankr.bot/<walletAddress>/<serviceName>[/path]
```

- `walletAddress`: Your Bankr wallet address (the deployer)
- `serviceName`: Name from your config
- `/path`: Optional sub-path passed to your handler

## How Payment Works

1. Client calls your endpoint — gets 402 with payment requirements
2. Client's wallet signs a USDC payment on Base
3. Client retries with `X-PAYMENT` header containing the signed payment
4. Payment is verified, your handler runs, payment settles on-chain
5. Your share goes to your wallet, platform fee (if any) goes to Bankr

Payments only settle if your handler returns a successful response (status < 400). Failed requests are never charged.

## Limits

| Resource | Limit |
|----------|-------|
| Bundle size | 5 MB |
| Memory | 256 MB |
| Execution time | 30 seconds |
| Deploys per hour | 20 |
| Services per account | Unlimited |
| Env var value size | 4 KB |
